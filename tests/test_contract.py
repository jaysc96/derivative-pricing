"""U4: one shape for every method, and no grid on the caller's surface.

Before this unit, five of the six methods returned a scalar and the sixth
returned a vector across bounds the caller had to invent. Anything pricing a
chain had to know which method it was holding. These tests fix the new contract
in place:

* every method returns a :class:`PriceResult` for the same contract;
* the finite-difference price at spot still matches what the old caller-supplied
  grid produced there, so deriving the grid internally did not move the answer;
* doubling the resolution barely moves it, so the derived grid is fine enough;
* finite-difference gamma is real, which a re-priced bump could not deliver once
  the grid follows spot.

The last one is the subtle one. Moving spot now moves the grid with it, so a
bump smaller than a cell reprices to the same interpolated number and gamma
comes back as exactly zero. Delta and gamma therefore come from differences
across the solver's own nodes (KTD5).
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest
from cases import CASES, option

from pricing import AMERICAN_METHODS, EUROPEAN_METHODS, PriceResult
from pricing.contracts import MIN_FD_STEPS, fd_grid

FD_SPOT_BASELINE = json.loads((Path(__file__).parent / "fd_spot_baseline.json").read_text())
REPO = Path(__file__).parent.parent

# Trees and Monte Carlo still need a discretization; finite differences no
# longer do, which is the point of the unit.
DISCRETIZATION = {
    "BT": lambda o: o.setTreeSteps(200),
    "TT": lambda o: o.setTreeSteps(200),
    "MC": lambda o: o.setSeedVariables(42, 20000, 1.0 / 252.0),
    "LSMC": lambda o: o.setSeedVariables(42, 20000, 1.0 / 252.0),
}

# Finite differences against the closed form is a discretization comparison, not
# a bug hunt — U5 owns the convergence tolerances. These are loose enough to pass
# on a 400-node grid and tight enough that a wrong answer would not.
PRICE_ABS, PRICE_REL = 0.01, 0.0025
RESOLUTION_ABS, RESOLUTION_REL = 0.01, 0.0025


def tolerance(reference, abs_floor, rel):
    """A cent, or a fraction of the price, whichever is more forgiving."""
    return max(abs_floor, rel * abs(reference))


# --------------------------------------------------------------------------
# Scenario 1: one shape, no caller-supplied grid
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "style,method",
    [("european", m) for m in EUROPEAN_METHODS] + [("american", m) for m in AMERICAN_METHODS],
)
def test_every_method_returns_a_scalar_price_for_the_same_contract(style, method):
    opt = option(style, "atm_call_1y", method)
    if method in DISCRETIZATION:
        DISCRETIZATION[method](opt)

    res = opt.priceOption()

    assert isinstance(res, PriceResult)
    for field in ("price", "delta", "gamma", "theta", "vega", "rho"):
        value = getattr(res, field)
        assert isinstance(value, float), f"{field} is {type(value).__name__}, not a scalar"
        assert value == value, f"{field} is nan"
    assert res.price > 0


def test_finite_differences_price_without_any_configuration():
    """No setter call at all — the grid comes from the contract."""
    res = option("european", "atm_call_1y", "FD").priceOption()
    assert res.price > 0


def _js_method_array(field: str) -> set[str]:
    """Extract a bare JS string-array literal like ``const X = ["A", "B"];``."""
    source = (REPO / "client" / "src" / "methodRules.js").read_text()
    match = re.search(rf'{field}\s*=\s*\[([^\]]*)\]', source)
    assert match, f"{field} not found in methodRules.js"
    return {item.strip().strip('"').strip("'") for item in match.group(1).split(",") if item.strip()}


def test_the_client_method_lists_agree_with_the_library_it_calls():
    """`client/src/methodRules.js` hand-copies these two lists rather than
    reading them from the API — nothing else enforces agreement, and a
    method added or removed on one side without the other either hides a
    real capability from the form or lets the form offer one the server will
    reject on every submit."""
    assert _js_method_array("EUROPEAN_METHODS") == set(EUROPEAN_METHODS)
    assert _js_method_array("AMERICAN_METHODS") == set(AMERICAN_METHODS)


def test_no_caller_supplied_grid_survives_anywhere():
    """The old bounds are gone from the library, the app, and the form."""
    opt = option("european", "atm_call_1y", "FD")
    assert not hasattr(opt, "setFDVariables")
    assert not hasattr(opt, "S_min")
    assert not hasattr(opt, "S_max")

    banned = re.compile(r"setFDVariables|stock_min_price|stock_max_price")
    searched = [
        REPO / "app.py",
        REPO / "api" / "pricing_routes.py",
        REPO / "client" / "src" / "App.jsx",
        *REPO.glob("pricing/*.py"),
    ]
    offenders = [p for p in searched if banned.search(p.read_text())]
    assert not offenders, f"price grid still reaches the caller in {offenders}"


# --------------------------------------------------------------------------
# Scenario 2: the derived grid agrees with the grid it replaced
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(k for k in FD_SPOT_BASELINE if k != "_meta"))
def test_derived_grid_stays_close_to_the_grid_it_replaced(key):
    """Same answer at spot, off a grid the caller no longer chooses.

    The recorded values come from the pre-U4 solver on its S_min=1, S_max=300,
    dS=1 grid, read at the node that coincides with the contract's spot. The new
    grid has different extent, spacing, and step count, so agreement is to
    discretization error rather than to the bit — and the band has to be wide
    enough to allow the new answer to be *better*, which for the three-month
    60%-volatility contract it is by 0.07.
    """
    style, label = key.split(".")
    expected = FD_SPOT_BASELINE[key]["price"]

    got = option(style, label, "FD").priceOption().price

    assert got == pytest.approx(expected, abs=tolerance(expected, 0.10, 0.01))


@pytest.mark.parametrize("label", sorted(CASES))
def test_the_derived_grid_is_no_less_accurate_than_the_one_it_replaced(label):
    """The claim worth making, where a closed form exists to make it against.

    "Nothing moved" was the wrong test to keep: MIN_FD_STEPS moved several of
    these deliberately. What has to hold is that none of them moved away from
    the truth. The slack absorbs cases already accurate to a rounding error,
    where the old value can win by a hair on noise alone.
    """
    from pricing import European_Option

    truth = option("european", label, "BSM").priceOption().price
    before = FD_SPOT_BASELINE[f"european.{label}"]["price"]
    after = option("european", label, "FD").priceOption().price

    assert abs(after - truth) <= abs(before - truth) + 0.005, (
        f"{label}: error against closed form grew from "
        f"{abs(before - truth):.4f} to {abs(after - truth):.4f}"
    )


# --------------------------------------------------------------------------
# Scenario 3: the derived grid is fine enough
# --------------------------------------------------------------------------

# Four of the sixteen combinations, not all of them. Doubling the nodes is an
# 8x dense-solve cost — the two-year case alone takes twenty seconds — and U13
# is what makes the solver cheap enough to widen this.
#
# The three labels are the widest grids in the table: highest volatility,
# longest dated, and furthest from the money. Both styles derive their grid
# through the same `fd_grid` call, so the extra American case is there to
# confirm the early-exercise path does not change the conclusion rather than to
# re-test the derivation.
RESOLUTION_CASES = [
    ("european", "high_vol_call_3m"),
    ("european", "otm_put_2y"),
    ("european", "otm_call_1y"),
    ("american", "otm_put_2y"),
]


@pytest.mark.parametrize("style,label", RESOLUTION_CASES)
def test_doubling_the_resolution_barely_moves_the_price(style, label):
    coarse = option(style, label, "FD").priceOption().price

    fine_opt = option(style, label, "FD")
    fine_opt.setFDResolution(nodes=800)
    fine = fine_opt.priceOption().price

    limit = tolerance(coarse, RESOLUTION_ABS, RESOLUTION_REL)
    assert abs(fine - coarse) < limit, (
        f"{style}.{label}: {coarse:.5f} -> {fine:.5f} on doubling, "
        f"moved {abs(fine - coarse):.5f} against a limit of {limit:.5f}"
    )


# --------------------------------------------------------------------------
# Scenario 4: gamma survives the grid following spot
# --------------------------------------------------------------------------


@pytest.mark.parametrize("label", sorted(CASES))
def test_finite_difference_gamma_is_real(label):
    """Non-zero, and agreeing with the closed form that has the same value.

    Non-zero alone would pass on noise. Every contract in the table is European
    with a strictly positive closed-form gamma, so the stronger claim is
    available and worth making.
    """
    fd = option("european", label, "FD").priceOption()
    closed_form = option("european", label, "BSM").priceOption()

    assert closed_form.gamma > 0, "case chosen badly — closed-form gamma is zero"
    assert fd.gamma > 0
    assert fd.gamma == pytest.approx(closed_form.gamma, rel=0.05)


@pytest.mark.parametrize("label", sorted(CASES))
def test_american_finite_difference_gamma_is_non_zero(label):
    """No closed form to check against, so only the weaker claim."""
    assert option("american", label, "FD").priceOption().gamma > 0


# --------------------------------------------------------------------------
# The grid rule itself, across an envelope wider than the case table
# --------------------------------------------------------------------------

# The case table tops out at sig=0.6, T=2, and the first version of the grid
# rule passed every test above while being unusable outside that box: an
# uncapped exp(5*sig*sqrt(T)) put the boundary at 1400x strike for sig=0.8,
# T=3 and mispriced by 1140. These two tests are what catch that class of
# error — they check the grid the contract produces, not just the contracts we
# happened to tabulate.
ENVELOPE = [(sig, T) for sig in (0.1, 0.3, 0.6, 0.8, 1.2) for T in (1 / 12, 1.0, 3.0)]


@pytest.mark.parametrize("sig,T", ENVELOPE)
def test_grid_both_contains_and_resolves_the_contract(sig, T):
    """Wide enough for the boundary to be harmless, fine enough to see spot."""
    spot = 100.0
    S, dS = fd_grid(spot, 100.0, sig, T)

    assert S[0] >= 2.0 * spot, "boundary too close to the contract"
    assert S[0] <= 15.0 * spot, "boundary so far out that spot is unresolved"
    assert dS <= spot / 25, f"spot spans fewer than 25 cells (dS={dS:.2f})"
    assert S[-1] == pytest.approx(dS), "grid must start one step above zero"
    assert np.allclose(np.diff(S), -dS), "grid must be uniform"


@pytest.mark.parametrize("sig,T", ENVELOPE)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_finite_differences_stay_accurate_across_the_envelope(kind, sig, T):
    """Against closed form, over the same envelope.

    A guard against a discretization that stops working, not a convergence
    tolerance — U5 sets those, on a tighter envelope. 2% is where the two known
    degradations sit: extent truncation at the top corner (sig=1.2, T=3 runs
    about 1.7%) and the Crank-Nicolson kink at the short end. Anything past this
    is a new failure, not a known one.
    """
    from pricing import European_Option

    fd = European_Option(kind, 100.0, 100.0, 0.05, sig, 0.02, T, "FD").priceOption()
    closed_form = European_Option(kind, 100.0, 100.0, 0.05, sig, 0.02, T, "BSM").priceOption()

    assert fd.price == pytest.approx(closed_form.price, rel=0.02, abs=0.05)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_short_dated_contracts_get_enough_time_steps(kind):
    """The regression that MIN_FD_STEPS exists for.

    A step per trading day gives a one-month contract 21 of them, which
    misprices by about 3%. Pinning the floor here rather than only the price
    keeps the reason visible: the earlier code passed every price assertion in
    this file while being wrong for short maturities, because nothing in the
    case table expires inside three months.
    """
    from pricing import European_Option

    T = 1 / 12
    fd = European_Option(kind, 100.0, 100.0, 0.05, 0.6, 0.02, T, "FD").priceOption()
    closed_form = European_Option(kind, 100.0, 100.0, 0.05, 0.6, 0.02, T, "BSM").priceOption()

    assert round(T * 252) < MIN_FD_STEPS, "case no longer exercises the floor"
    assert fd.price == pytest.approx(closed_form.price, rel=0.005)
