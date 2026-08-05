"""R4: finite-difference Greeks against their closed-form counterparts.

Per Greek, because the failures are per Greek. Delta and gamma already agree to
better than a percent — they come from spatial differences on the solver's own
grid (U4). Theta, vega and rho come from bumping and re-pricing, and all three
are wrong for separate reasons:

* **vega** is not a finite-difference failure at all. ``BSM`` mixes the two
  equivalent identities, computing ``S * exp(-rT) * n(d2)`` where the ``S`` and
  the discount factor belong to different forms. The result is the true vega
  scaled by ``S / K``, so the closed form — not the bumped estimate — is the
  side that is wrong. The ratio in the failure output makes that legible: at
  moneyness 0.8 the bumped vega is 1.25x the closed form, and 1/0.8 = 1.25.
* **theta** uses a one-sided difference with a 0.05-year bump, which is a fifth
  of a six-month contract's remaining life.
* **rho** reuses the volatility epsilon as the interest-rate bump, and is also
  one-sided.

Tolerances are what a correct implementation should meet, not what this one
does. Three of the five fail until U6 rebuilds the estimation on central
differences with per-input bump scales (KTD5).
"""

import pytest

from pricing import European_Option
from pricing.contracts import (
    BUMP_RATE,
    BUMP_RATE_FLOOR,
    BUMP_SPOT,
    BUMP_TIME,
    BUMP_TIME_CAP,
    BUMP_VOL,
    BUMP_VOL_FLOOR,
)

K, R, SIG, DIVIDEND = 100.0, 0.05, 0.25, 0.02
MONEYNESS = [0.8, 0.9, 1.0, 1.1, 1.2]
MATURITIES = [0.5, 1.0]

# (relative tolerance, absolute floor). The floor matters where a Greek passes
# through zero and a relative test would demand impossible precision — but it
# has to stay small enough not to swallow the defect. An earlier floor of 0.05
# on theta hid a 64% error, because the contract where the one-sided bump does
# most damage is also one where theta is only -0.08. These are what a central
# difference with a per-input bump scale should comfortably achieve; the units
# are the Greek's own, so theta's floor is 0.02 of value per year.
TOLERANCE = {
    "delta": (0.02, 0.005),
    "gamma": (0.03, 0.0005),
    "theta": (0.05, 0.02),
    "vega": (0.02, 0.05),
    "rho": (0.025, 0.02),
}

# Why rho is the loosest. These tolerances are floored by the *price* accuracy
# of the method being differenced, not by the differencing. The 80-strike-100
# six-month call is priced 0.95% out by the finite-difference scheme, and its
# rho comes back 1.03% out — the error is inherited, not introduced. Measured
# across bump sizes from 1e-4 to 1e-2 the rho error does not move from 1.03%,
# which is what rules out the bump as the cause.
#
# That makes this sweep a weak test of the estimation machinery, so
# test_central_differences_recover_exact_greeks below tests it separately
# against prices with no discretization error at all, where the tolerance can
# be three orders of magnitude tighter.

CASES = [
    (m, T, kind) for m in MONEYNESS for T in MATURITIES for kind in ("call", "put")
]


def both(m, T, kind):
    args = (kind, m * K, K, R, SIG, DIVIDEND, T)
    return (
        European_Option(*args, "FD").priceOption(),
        European_Option(*args, "BSM").priceOption(),
    )


@pytest.mark.parametrize("greek", sorted(TOLERANCE))
@pytest.mark.parametrize("m,T,kind", CASES)
def test_finite_difference_greek_matches_closed_form(greek, m, T, kind):
    fd, closed_form = both(m, T, kind)
    estimated, exact = getattr(fd, greek), getattr(closed_form, greek)

    rel, floor = TOLERANCE[greek]
    limit = max(floor, rel * abs(exact))
    ratio = estimated / exact if exact else float("nan")

    assert abs(estimated - exact) <= limit, (
        f"{greek} at moneyness {m}, T={T}, {kind}: "
        f"finite difference {estimated:.5f} vs closed form {exact:.5f} "
        f"(ratio {ratio:.4f}, allowed {limit:.5f})"
    )


@pytest.mark.parametrize("m,T,kind", CASES)
def test_central_differences_recover_exact_greeks(m, T, kind):
    """The estimation machinery on its own, with the discretization removed.

    Differencing closed-form prices instead of a solver's. Those prices carry
    no grid, no lattice, and no sampling error, so whatever comes back is
    attributable to the bump scheme alone — which is the thing KTD5 changed.

    Run against the old one-sided machinery on these same twenty contracts, it
    fails on all three of its Greeks: theta by 146x the tolerance (73% error on
    the six-month 80-strike put), vega and rho by 3x. Those vega and rho
    numbers are the bump error alone, with the S/K identity defect excluded —
    this differences prices, and the identity never entered.

    This is the test that pins the rebuild. The sweep above cannot: it is
    floored by the pricing error of the method it differences.
    """
    args = (kind, m * K, K, R, SIG, DIVIDEND, T)
    exact = European_Option(*args, "BSM").priceOption()

    def price_at(**overrides):
        opt = European_Option(*args, "BSM")
        for name, value in overrides.items():
            setattr(opt, name, value)
        return opt.BSM().price

    h_S = BUMP_SPOT * m * K
    h_sig = max(BUMP_VOL_FLOOR, BUMP_VOL * SIG)
    h_r = max(BUMP_RATE_FLOOR, BUMP_RATE * abs(R))
    h_T = min(BUMP_TIME * T, BUMP_TIME_CAP)

    up, down = price_at(S0=m * K + h_S), price_at(S0=m * K - h_S)
    estimated = {
        "delta": (up - down) / (2 * h_S),
        "gamma": (up - 2 * exact.price + down) / h_S**2,
        "theta": -(price_at(T=T + h_T) - price_at(T=T - h_T)) / (2 * h_T),
        "vega": (price_at(sig=SIG + h_sig) - price_at(sig=SIG - h_sig)) / (2 * h_sig),
        "rho": (price_at(r=R + h_r) - price_at(r=R - h_r)) / (2 * h_r),
    }

    for greek, value in estimated.items():
        target = getattr(exact, greek)
        assert value == pytest.approx(target, rel=0.005, abs=1e-4), (
            f"{greek} at moneyness {m}, T={T}, {kind}: "
            f"central difference {value:.6f} vs closed form {target:.6f}"
        )


@pytest.mark.parametrize("kind", ["call", "put"])
def test_price_option_actually_uses_central_differences(kind):
    """That the shipped code path is central, not merely that central is right.

    The test above computes its own differences, so it validates the bump
    constants while saying nothing about whether ``priceOption`` uses them. This
    one compares ``priceOption``'s output against differences taken by hand
    around the same contract, and a one-sided implementation misses by orders of
    magnitude more than the tolerance here.
    """
    args = (kind, 100.0, K, R, SIG, DIVIDEND, 1.0)
    steps = 200

    def price_at(**overrides):
        opt = European_Option(*args, "BT")
        opt.setTreeSteps(steps)
        for name, value in overrides.items():
            setattr(opt, name, value)
        return float(opt.BT())

    opt = European_Option(*args, "BT")
    opt.setTreeSteps(steps)
    got = opt.priceOption()

    h_S = BUMP_SPOT * 100.0
    h_sig = max(BUMP_VOL_FLOOR, BUMP_VOL * SIG)
    h_r = max(BUMP_RATE_FLOOR, BUMP_RATE * abs(R))
    h_T = min(BUMP_TIME * 1.0, BUMP_TIME_CAP)

    up, down = price_at(S0=100.0 + h_S), price_at(S0=100.0 - h_S)
    expected = {
        "delta": (up - down) / (2 * h_S),
        "gamma": (up - 2 * got.price + down) / h_S**2,
        "theta": -(price_at(T=1.0 + h_T) - price_at(T=1.0 - h_T)) / (2 * h_T),
        "vega": (price_at(sig=SIG + h_sig) - price_at(sig=SIG - h_sig)) / (2 * h_sig),
        "rho": (price_at(r=R + h_r) - price_at(r=R - h_r)) / (2 * h_r),
    }

    for greek, value in expected.items():
        assert getattr(got, greek) == pytest.approx(value, rel=1e-9, abs=1e-9), greek


def test_pricing_twice_gives_the_same_answer():
    """The bumps restore what they borrow.

    ``priceOption`` mutates spot, maturity, volatility and rate on the instance
    while estimating, and pins the finite-difference discretization. Every one
    of those is restored in a ``finally``, so an exception mid-estimate cannot
    leave a contract quietly holding a bumped input.
    """
    for method, configure in (
        ("BT", lambda o: o.setTreeSteps(200)),
        ("FD", lambda o: None),
    ):
        opt = European_Option("call", 100.0, K, R, SIG, DIVIDEND, 1.0, method)
        configure(opt)
        first, second = opt.priceOption(), opt.priceOption()
        assert first == second, method
        assert (opt.S0, opt.T, opt.sig, opt.r) == (100.0, 1.0, SIG, R)
        assert opt._fd_pinned is None


def test_vega_agrees_at_every_moneyness_not_just_at_the_money():
    """The shape of the defect, asserted as the shape of its absence.

    Before U6 this test held the opposite: the ratio of bumped to closed-form
    vega came out at exactly ``K / S`` — 1.2588 at moneyness 0.8, against
    1/0.8 = 1.25 — because the closed form mixed the two identities and
    returned the true vega scaled by ``S / K``. It was written that way so the
    fix could not be a loosened tolerance.

    Now it asserts the ratio is 1 everywhere. Kept as a separate test from the
    tolerance sweep because the failure mode is specifically a moneyness-shaped
    drift, and a single ratio at the money would not see it — that is the one
    point where the broken identity was right.
    """
    for m in MONEYNESS:
        fd, closed_form = both(m, 1.0, "call")
        ratio = fd.vega / closed_form.vega
        assert ratio == pytest.approx(1.0, rel=0.01), (
            f"at moneyness {m} the ratio is {ratio:.4f}; a ratio near "
            f"{1 / m:.4f} would mean the S/K scaling is back"
        )
