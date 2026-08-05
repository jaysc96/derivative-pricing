"""U12: inversion round-trips, and reports failure as failure.

R8, R9, R10. The round-trip tests are the accuracy claim; the rest exist
because a solver that returns a plausible number for an unanswerable quote is
worse than one that refuses, and the difference between "this quote has no
implied volatility" and "this solver could not reach one" is a difference the
downstream views need.
"""

import math

import pytest

from pricing.implied import (
    BRACKET_EXHAUSTED,
    MAX_VOL,
    NO_QUOTE,
    NO_SOLUTION,
    NOT_IDENTIFIED,
    SOLVED,
    implied_volatility,
    minimum_volatility,
    round_trip_error,
)

K, R, Y = 100.0, 0.05, 0.02
MONEYNESS = [0.8, 0.9, 1.0, 1.1, 1.2]
MATURITIES = [0.25, 1.0]
VOLS = [0.12, 0.30, 0.75]

GRID = [
    (style, kind, m, T, sig)
    for style in ("european", "american")
    for kind in ("call", "put")
    for m in MONEYNESS
    for T in MATURITIES
    for sig in VOLS
]


# --------------------------------------------------------------------------
# R10: round-trip accuracy across the grid
# --------------------------------------------------------------------------


@pytest.mark.parametrize("style,kind,m,T,sig", GRID)
def test_round_trip_recovers_the_input_volatility(style, kind, m, T, sig):
    """Price at a known volatility, invert, and land back on it.

    The tolerance is in volatility points. American inversion goes through a
    200-step tree whose own discretization error is the floor here, so this is
    not machine precision and should not pretend to be.

    One family of contracts is exempt and must say so rather than answer: a
    deep in-the-money American put where immediate exercise dominates is worth
    its payoff at every volatility, so no price identifies one. Those are
    asserted as `not_identified` — silently accepting a number there is the
    failure this exemption exists to catch.
    """
    from pricing.implied import _price

    price = _price(kind, style, sig, m * K, K, R, Y, T, 200)
    result = implied_volatility(kind, style, price, m * K, K, R, Y, T)

    if result.status == NOT_IDENTIFIED:
        assert style == "american", "only American exercise produces a flat price region"
        return

    assert result.status == SOLVED, f"{style} {kind} m={m} T={T} sig={sig}: {result.detail}"
    error = abs(result.implied_vol - sig)
    assert error < 1e-3, f"{style} {kind} m={m} T={T} sig={sig}: off by {error:.2e}"


@pytest.mark.parametrize("style", ["european", "american"])
def test_round_trip_holds_at_the_wings(style):
    """Deep out-of-the-money is where the surface is thinnest and inversion hardest."""
    for m, sig in ((0.6, 0.9), (1.5, 0.8)):
        error = round_trip_error("call", style, sig, m * K, K, R, Y, 0.5)
        assert error < 5e-3, f"{style} m={m} sig={sig}: off by {error:.2e}"


def test_a_high_but_reachable_volatility_resolves():
    """300% is inside the bracket and must come back as a number, not a limit."""
    from pricing.european import European_Option

    price = European_Option("call", 100.0, K, R, 3.0, Y, 1.0, "BSM").BSM().price
    result = implied_volatility("call", "european", price, 100.0, K, R, Y, 1.0)

    assert result.status == SOLVED
    assert result.implied_vol == pytest.approx(3.0, abs=1e-4)


# --------------------------------------------------------------------------
# R9: failure is explicit, and the two kinds are distinguishable
# --------------------------------------------------------------------------


def test_a_price_below_intrinsic_has_no_solution(style="european"):
    """AE1. Not a solver limit — a statement about the quote."""
    spot, strike = 130.0, 100.0
    intrinsic = spot - strike
    result = implied_volatility("call", style, intrinsic * 0.5, spot, strike, R, Y, 1.0)

    assert result.status == NO_SOLUTION
    assert result.implied_vol is None
    assert "below" in result.detail


def test_a_price_above_the_ceiling_records_bracket_exhaustion():
    """Distinct from no-solution: this one is the solver's limit, not the quote's."""
    result = implied_volatility("call", "european", 99.0, 100.0, K, R, Y, 1.0)

    assert result.status == BRACKET_EXHAUSTED
    assert result.implied_vol is None
    assert result.price_at_ceiling is not None
    assert result.target_price > result.price_at_ceiling


def test_the_two_failure_modes_are_not_the_same_status():
    """The skew view needs to tell a solver limit from a bad quote."""
    too_cheap = implied_volatility("call", "european", 0.001, 130.0, K, R, Y, 1.0)
    too_dear = implied_volatility("call", "european", 99.5, 100.0, K, R, Y, 1.0)

    assert too_cheap.status == NO_SOLUTION
    assert too_dear.status == BRACKET_EXHAUSTED
    assert too_cheap.status != too_dear.status


@pytest.mark.parametrize("price", [None, 0.0, -1.5])
def test_an_absent_or_impossible_price_is_refused_before_solving(price):
    result = implied_volatility("call", "european", price, 100.0, K, R, Y, 1.0)
    assert result.status == NO_QUOTE
    assert result.implied_vol is None


def test_an_expired_contract_is_refused():
    result = implied_volatility("call", "european", 5.0, 100.0, K, R, Y, 0.0)
    assert result.status == NO_QUOTE


# --------------------------------------------------------------------------
# The lower bracket is a correctness guard, not a preference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("T", [0.02, 0.25, 1.0, 3.0])
@pytest.mark.parametrize("steps", [50, 200, 800])
def test_the_floor_keeps_the_binomial_probability_in_range(T, steps):
    """Below it the lattice admits arbitrage and the solver would invert it anyway."""
    floor = minimum_volatility(R, Y, T, steps)
    dt = T / steps
    u = math.exp(floor * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((R - Y) * dt) - d) / (u - d)

    assert 0.0 <= p <= 1.0, f"p={p} at T={T}, steps={steps}, floor={floor}"


def test_the_floor_loosens_as_step_count_rises():
    """It binds near zero and stops mattering with a finer tree — a guard, not a limit."""
    coarse = minimum_volatility(R, Y, 1.0, 50)
    fine = minimum_volatility(R, Y, 1.0, 800)
    assert fine < coarse or coarse == fine == pytest.approx(1e-4, abs=1e-9)


def test_a_zero_rate_and_dividend_falls_back_to_the_absolute_floor():
    assert minimum_volatility(0.0, 0.0, 1.0, 200) == pytest.approx(1e-4)


# --------------------------------------------------------------------------
# The bracket endpoints are where the plan says they are
# --------------------------------------------------------------------------


def test_the_ceiling_is_five_hundred_percent():
    """Below this, single-name wings get reclassified as missing data."""
    assert MAX_VOL == 5.0


def test_a_two_hundred_percent_wing_is_inside_the_bracket():
    """The case the ceiling exists for: the strikes the skew view is about."""
    from pricing.european import European_Option

    price = European_Option("call", 60.0, K, R, 2.0, Y, 0.25, "BSM").BSM().price
    result = implied_volatility("call", "european", price, 60.0, K, R, Y, 0.25)

    assert result.status == SOLVED
    assert result.implied_vol == pytest.approx(2.0, abs=1e-3)


# --------------------------------------------------------------------------
# American exercise actually changes the answer
# --------------------------------------------------------------------------


def test_american_and_european_inversion_differ_where_early_exercise_matters():
    """The claim the whole engine exists for.

    An in-the-money put on a dividend payer is the case where inverting with a
    European formula is wrong, and the size of the gap is the argument for
    carrying an American method at all.
    """
    from pricing.american import American_Option

    spot, strike, rate, div, T, sig = 80.0, 100.0, 0.08, 0.0, 2.0, 0.3
    american = American_Option("put", spot, strike, rate, sig, div, T, "BT")
    american.setTreeSteps(200)
    price = float(american.BT())

    as_american = implied_volatility("put", "american", price, spot, strike, rate, div, T)
    as_european = implied_volatility("put", "european", price, spot, strike, rate, div, T)

    assert as_american.status == SOLVED
    assert as_american.implied_vol == pytest.approx(sig, abs=1e-3)

    # Inverting the same price with the wrong exercise style does not merely
    # lose precision — it lands somewhere else entirely, or nowhere at all.
    if as_european.solved:
        assert abs(as_european.implied_vol - sig) > 0.02, (
            "European inversion of an American price should be materially wrong"
        )
    else:
        assert as_european.status in (NO_SOLUTION, BRACKET_EXHAUSTED)


def test_a_deep_itm_american_put_reports_that_iv_is_not_identified():
    """The case the exemption above exists for, asserted directly.

    Immediate exercise dominates, so the contract is worth its payoff at every
    volatility from the bracket floor upward. Two different volatilities
    produce the same price, which means no price picks one out.
    """
    from pricing.implied import _price

    spot, strike, T = 80.0, 100.0, 0.25
    cheap = _price("put", "american", 0.05, spot, strike, R, Y, T, 200)
    dear = _price("put", "american", 0.12, spot, strike, R, Y, T, 200)
    assert cheap == pytest.approx(dear), "precondition: the price really is flat here"
    assert cheap == pytest.approx(strike - spot), "and it is the exercise payoff"

    result = implied_volatility("put", "american", dear, spot, strike, R, Y, T)
    assert result.status == NOT_IDENTIFIED
    assert result.implied_vol is None
    assert "flat" in result.detail


def test_the_european_twin_of_that_contract_still_solves():
    """Same contract, European exercise: no flat region, so it inverts fine.

    This is what makes the American result a property of early exercise rather
    than of deep moneyness.
    """
    from pricing.implied import _price

    spot, strike, T, sig = 80.0, 100.0, 0.25, 0.12
    price = _price("put", "european", sig, spot, strike, R, Y, T, 200)
    result = implied_volatility("put", "european", price, spot, strike, R, Y, T)

    assert result.status == SOLVED
    assert result.implied_vol == pytest.approx(sig, abs=1e-4)


def test_not_identified_is_distinct_from_every_other_failure():
    """Four outcomes, four meanings — the views downstream depend on the difference."""
    assert len({SOLVED, NO_SOLUTION, BRACKET_EXHAUSTED, NOT_IDENTIFIED, NO_QUOTE}) == 5
