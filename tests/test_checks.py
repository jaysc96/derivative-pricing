"""U16: three no-arbitrage bounds, evaluated on tradeable sides, never on mid.

Each pure check function is exercised in isolation first — matching the
plan's own scenario language ("a pair", "a butterfly", "calendar
monotonicity") — and ``find_violations`` is exercised separately against real
``ChainSnapshot``/``QuoteRecord`` objects, so the staleness gate and the
per-expiry/per-strike matching are proven against the actual domain shape,
not a shortcut fixture.
"""

import math
from datetime import date, datetime, timedelta, timezone

from marketdata import ChainSnapshot, QuoteRecord
from marketdata.checks import (
    BUTTERFLY,
    CALENDAR,
    DEFAULT_STALENESS_WINDOW,
    PUT_CALL_BAND,
    check_butterfly,
    check_calendar,
    check_put_call_band,
    find_violations,
    is_stale,
)

S, K, R, Y, T = 100.0, 100.0, 0.05, 0.02, 1.0
LOWER = S * math.exp(-Y * T) - K
UPPER = S - K * math.exp(-R * T)

EXPIRY = date(2026, 12, 18)
AS_OF = date(2026, 8, 6)
FRESH_TRADE = datetime(2026, 8, 6, 15, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# American put-call band
# --------------------------------------------------------------------------


def test_a_pair_outside_the_band_is_caught_on_the_lower_side():
    """AE2: buying the call at ask and selling the put at bid still profits."""
    violation = check_put_call_band(
        call_bid=0.9, call_ask=1.0, put_bid=5.0, put_ask=5.2, S=S, K=K, r=R, y=Y, T=T
    )
    assert violation is not None
    assert violation.kind == PUT_CALL_BAND
    assert "lower bound" in violation.detail


def test_a_pair_outside_the_band_is_caught_on_the_upper_side():
    """Selling the call at bid and buying the put at ask still profits."""
    violation = check_put_call_band(
        call_bid=10.0, call_ask=10.2, put_bid=0.5, put_ask=0.6, S=S, K=K, r=R, y=Y, T=T
    )
    assert violation is not None
    assert "upper bound" in violation.detail


def test_a_pair_inside_the_band_passes():
    assert (
        check_put_call_band(
            call_bid=2.9, call_ask=3.1, put_bid=2.0, put_ask=2.2, S=S, K=K, r=R, y=Y, T=T
        )
        is None
    )


def test_a_one_sided_leg_cannot_be_checked():
    """Nothing tradeable to check without both sides on both legs."""
    assert (
        check_put_call_band(
            call_bid=None, call_ask=1.0, put_bid=5.0, put_ask=5.2, S=S, K=K, r=R, y=Y, T=T
        )
        is None
    )


def test_bounds_evaluated_on_mid_prices_disagree_with_the_sides():
    """A gap that looks like a violation at the mid is not realizable at real prices.

    call_ask - put_bid = 0.0 and call_bid - put_ask = -4.0: neither breaches
    the band, so the sides-based check passes. But (call_mid - put_mid) =
    2.0 - 4.0 = -2.0, below LOWER (~-1.98) — a naive mid-price check would
    wrongly flag this as a violation nobody could actually trade into.
    """
    call_bid, call_ask = 1.0, 3.0
    put_bid, put_ask = 3.0, 5.0

    sides_result = check_put_call_band(
        call_bid=call_bid, call_ask=call_ask, put_bid=put_bid, put_ask=put_ask,
        S=S, K=K, r=R, y=Y, T=T,
    )
    call_mid = (call_bid + call_ask) / 2
    put_mid = (put_bid + put_ask) / 2
    mid_would_flag = (call_mid - put_mid) < LOWER or (call_mid - put_mid) > UPPER

    assert sides_result is None, "the sides-based check must not flag this pair"
    assert mid_would_flag, "the mid-based comparison must flag it, proving the two disagree"


# --------------------------------------------------------------------------
# Butterfly convexity
# --------------------------------------------------------------------------


def test_a_butterfly_violating_convexity_is_caught():
    violation = check_butterfly(
        low=(95.0, 5.8, 6.0), mid=(100.0, 4.5, 4.7), high=(105.0, 1.8, 2.0)
    )
    assert violation is not None
    assert violation.kind == BUTTERFLY


def test_a_convex_butterfly_passes():
    violation = check_butterfly(
        low=(95.0, 5.8, 6.0), mid=(100.0, 3.5, 3.7), high=(105.0, 1.8, 2.0)
    )
    assert violation is None


def test_butterfly_convexity_uses_weights_for_uneven_strikes():
    """80 is closer to 100 than 130 is; a violation exists at the weighted
    chord but would be invisible to the evenly-spaced low-2*mid+high formula.
    """
    # w_low = (130-100)/(130-80) = 0.6, w_high = 0.4
    # chord = 0.6*low_ask + 0.4*high_ask
    violation = check_butterfly(
        low=(80.0, 9.8, 10.0), mid=(100.0, 8.5, 8.7), high=(130.0, 1.8, 2.0)
    )
    assert violation is not None, "0.6*10.0 + 0.4*2.0 = 6.8 < mid_bid 8.5"


def test_butterfly_requires_strictly_increasing_strikes():
    import pytest

    with pytest.raises(ValueError):
        check_butterfly(low=(100.0, 1, 2), mid=(95.0, 1, 2), high=(105.0, 1, 2))


# --------------------------------------------------------------------------
# Calendar monotonicity
# --------------------------------------------------------------------------


def test_calendar_monotonicity_catches_a_near_dated_price_above_its_far_dated_counterpart():
    violation = check_calendar(near=(6.0, 6.2), far=(5.0, 5.2))
    assert violation is not None
    assert violation.kind == CALENDAR


def test_calendar_monotonicity_passes_when_far_dated_is_worth_more():
    assert check_calendar(near=(4.0, 4.2), far=(5.0, 5.2)) is None


# --------------------------------------------------------------------------
# Staleness gate
# --------------------------------------------------------------------------


def test_a_fresh_trade_is_not_stale():
    assert not is_stale(FRESH_TRADE, AS_OF)


def test_a_missing_trade_marker_is_treated_as_stale():
    assert is_stale(None, AS_OF)


def test_a_trade_outside_the_window_is_stale():
    old_trade = datetime.combine(AS_OF, datetime.min.time(), tzinfo=timezone.utc) - (
        DEFAULT_STALENESS_WINDOW + timedelta(days=1)
    )
    assert is_stale(old_trade, AS_OF)


# --------------------------------------------------------------------------
# find_violations: the real domain shapes, and the staleness gate wired in
# --------------------------------------------------------------------------


def quote(symbol, option_type, strike, bid, ask, *, traded=FRESH_TRADE):
    return QuoteRecord(
        contract_symbol=symbol, option_type=option_type, strike=strike,
        bid=bid, ask=ask, last=None, volume=10, open_interest=20,
        provider_iv=0.2, last_trade_at=traded,
    )


def chain(expiry, quotes, *, underlying_price=S, rate=R, dividend_yield=Y, as_of=AS_OF):
    return ChainSnapshot(
        provider="fake", symbol="TEST", expiry=expiry,
        captured_at=datetime.combine(as_of, datetime.min.time(), tzinfo=timezone.utc),
        as_of=as_of, origin="live", underlying_price=underlying_price,
        quotes=tuple(quotes), risk_free_rate=rate, dividend_yield=dividend_yield,
    )


def test_find_violations_catches_a_put_call_band_breach():
    quotes = [
        quote("C100", "call", 100.0, 0.9, 1.0),
        quote("P100", "put", 100.0, 5.0, 5.2),
    ]
    violations = find_violations([chain(EXPIRY, quotes)])
    assert any(v.kind == PUT_CALL_BAND for v in violations)
    matched = next(v for v in violations if v.kind == PUT_CALL_BAND)
    assert matched.legs == ("C100", "P100")


def test_find_violations_excludes_a_stale_leg_before_the_bound_runs():
    """The same pair as the band-breach test above, but the call's trade is stale.

    Without the staleness gate this pair would still fail the band; with it,
    the leg never reaches the check at all.
    """
    stale_trade = FRESH_TRADE - DEFAULT_STALENESS_WINDOW - timedelta(days=2)
    quotes = [
        quote("C100", "call", 100.0, 0.9, 1.0, traded=stale_trade),
        quote("P100", "put", 100.0, 5.0, 5.2),
    ]
    violations = find_violations([chain(EXPIRY, quotes)])
    assert not any(v.kind == PUT_CALL_BAND for v in violations)


def test_find_violations_catches_a_butterfly_across_real_quotes():
    quotes = [
        quote("C95", "call", 95.0, 5.8, 6.0),
        quote("C100", "call", 100.0, 4.5, 4.7),
        quote("C105", "call", 105.0, 1.8, 2.0),
    ]
    violations = find_violations([chain(EXPIRY, quotes)])
    matched = next(v for v in violations if v.kind == BUTTERFLY)
    assert matched.legs == ("C95", "C100", "C105")


def test_find_violations_catches_a_calendar_breach_across_expiries():
    near_expiry = EXPIRY
    far_expiry = EXPIRY + timedelta(days=30)
    near = chain(near_expiry, [quote("C100N", "call", 100.0, 6.0, 6.2)])
    far = chain(far_expiry, [quote("C100F", "call", 100.0, 5.0, 5.2)])

    violations = find_violations([near, far])
    matched = next(v for v in violations if v.kind == CALENDAR)
    assert matched.legs == ("C100N", "C100F")


def test_find_violations_skips_the_band_without_a_rate_or_yield():
    """A snapshot missing risk_free_rate/dividend_yield (U12b) can't evaluate the band."""
    quotes = [
        quote("C100", "call", 100.0, 0.9, 1.0),
        quote("P100", "put", 100.0, 5.0, 5.2),
    ]
    violations = find_violations([chain(EXPIRY, quotes, rate=None)])
    assert not any(v.kind == PUT_CALL_BAND for v in violations)


def test_find_violations_returns_nothing_for_a_clean_chain():
    quotes = [
        quote("C95", "call", 95.0, 5.8, 6.0),
        quote("C100", "call", 100.0, 3.5, 3.7),
        quote("C105", "call", 105.0, 1.8, 2.0),
        quote("P100", "put", 100.0, 2.0, 2.2),
    ]
    assert find_violations([chain(EXPIRY, quotes)]) == []
