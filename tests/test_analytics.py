"""U17: skew, term structure, and realized volatility, computed from the archive only.

Every skew/term-structure test goes through the real pipeline — write a
snapshot, run ``derive_batch``, read through ``build_skew``/
``build_term_structure`` — rather than constructing curve objects by hand, so
what is proven is that a real captured chain produces the right shape end to
end, including U16's exclusion (R21).
"""

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from analytics.realized import realized_volatility, realized_volatility_series
from analytics.surface import (
    INSUFFICIENT_DATA,
    OK,
    build_skew,
    build_term_structure,
)
from marketdata import ChainSnapshot, QuoteRecord, Store, UnderlyingBar
from marketdata.derive import derive_batch
from pricing.american import American_Option

R, Y = 0.05, 0.02
T0 = datetime(2026, 8, 6, 14, 0, tzinfo=timezone.utc)
NEAR = date(2026, 9, 18)
FAR = date(2026, 12, 18)
FAR2 = date(2027, 3, 19)


def price(strike, sig, expiry, *, S=100.0):
    T_years = (expiry - T0.date()).days / 365.0
    opt = American_Option("call", S, strike, R, sig, Y, T_years, "BT")
    opt.setTreeSteps(200)
    return float(opt.BT())


def quote(contract, strike, bid, ask, *, option_type="call", traded=T0):
    mid = (bid + ask) / 2 if bid and ask else None
    return QuoteRecord(
        contract_symbol=contract, option_type=option_type, strike=strike,
        bid=bid, ask=ask, last=mid, volume=10, open_interest=20,
        provider_iv=0.2, last_trade_at=traded,
    )


def snapshot(expiry, quotes, *, underlying=100.0):
    return ChainSnapshot(
        provider="fake", symbol="TEST", expiry=expiry, captured_at=T0,
        as_of=T0.date(), origin="live", underlying_price=underlying,
        quotes=tuple(quotes), risk_free_rate=R, dividend_yield=Y,
    )


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "analytics.db")


# --------------------------------------------------------------------------
# Skew
# --------------------------------------------------------------------------


def test_skew_is_computed_from_a_snapshot_with_known_volatilities(store):
    strikes_vols = [(90.0, 0.20), (100.0, 0.25), (110.0, 0.30)]
    quotes = [
        quote(f"C{strike:.0f}", strike, price(strike, sig, NEAR) - 0.01, price(strike, sig, NEAR) + 0.01)
        for strike, sig in strikes_vols
    ]
    store.write_snapshot(snapshot(NEAR, quotes))
    derive_batch(store)

    curves = build_skew(store, "TEST")
    assert len(curves) == 1
    curve = curves[0]
    assert curve.status == OK
    assert [pt.strike for pt in curve.points] == [90.0, 100.0, 110.0]
    for (strike, true_sig), pt in zip(strikes_vols, curve.points):
        assert pt.strike == strike
        assert pt.implied_vol == pytest.approx(true_sig, abs=5e-3)


def test_an_expiry_with_too_few_valid_quotes_reports_insufficient_data(store):
    quotes = [
        quote(f"C{strike:.0f}", strike, price(strike, sig, NEAR) - 0.01, price(strike, sig, NEAR) + 0.01)
        for strike, sig in [(95.0, 0.20), (105.0, 0.25)]
    ]
    store.write_snapshot(snapshot(NEAR, quotes))
    derive_batch(store)

    curves = build_skew(store, "TEST")
    assert len(curves) == 1
    assert curves[0].status == INSUFFICIENT_DATA


def test_skew_returns_nothing_before_any_capture(store):
    assert build_skew(store, "TEST") == []


# --------------------------------------------------------------------------
# Term structure
# --------------------------------------------------------------------------


def test_term_structure_is_computed_from_a_snapshot_with_known_volatilities(store):
    vols = {NEAR: 0.20, FAR: 0.25, FAR2: 0.30}
    for expiry, sig in vols.items():
        p = price(100.0, sig, expiry)
        store.write_snapshot(
            snapshot(expiry, [quote(f"C100_{expiry.isoformat()}", 100.0, p - 0.01, p + 0.01)])
        )
    derive_batch(store)

    curves = build_term_structure(store, "TEST")
    assert len(curves) == 1
    curve = curves[0]
    assert curve.status == OK
    assert [pt.expiry for pt in curve.points] == [NEAR, FAR, FAR2]
    for expiry, true_sig in vols.items():
        pt = next(p for p in curve.points if p.expiry == expiry)
        assert pt.implied_vol == pytest.approx(true_sig, abs=5e-3)


def test_a_strike_with_too_few_expiries_reports_insufficient_data_for_term_structure(store):
    for expiry in (NEAR, FAR):
        p = price(100.0, 0.25, expiry)
        store.write_snapshot(
            snapshot(expiry, [quote(f"C100_{expiry.isoformat()}", 100.0, p - 0.01, p + 0.01)])
        )
    derive_batch(store)

    curves = build_term_structure(store, "TEST")
    assert len(curves) == 1
    assert curves[0].status == INSUFFICIENT_DATA


# --------------------------------------------------------------------------
# R21: a U16 violation never reaches the surface
# --------------------------------------------------------------------------


def test_quotes_rejected_by_u16_are_absent_from_the_surface(store):
    good_strikes = [(90.0, 0.20), (100.0, 0.25), (110.0, 0.30)]
    quotes = [
        quote(f"C{strike:.0f}", strike, price(strike, sig, NEAR) - 0.01, price(strike, sig, NEAR) + 0.01)
        for strike, sig in good_strikes
    ]
    # A genuine put-call-band violation at strike 120: the call is priced far
    # too low relative to a rich put (call_ask - put_bid well below the lower
    # bound at these parameters).
    quotes.append(quote("C120", 120.0, 0.10, 0.20))
    quotes.append(quote("P120", 120.0, 30.0, 30.5, option_type="put"))

    store.write_snapshot(snapshot(NEAR, quotes))
    derive_batch(store)

    curves = build_skew(store, "TEST")
    strikes_in_surface = [pt.strike for pt in curves[0].points]
    assert strikes_in_surface == [90.0, 100.0, 110.0]
    assert 120.0 not in strikes_in_surface


# --------------------------------------------------------------------------
# Realized volatility
# --------------------------------------------------------------------------


def test_realized_volatility_matches_a_hand_computed_value():
    closes = [100.0, 102.0, 101.0, 105.0, 103.0]

    log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    expected = math.sqrt(variance) * math.sqrt(252)

    assert realized_volatility(closes) == pytest.approx(expected)


def test_realized_volatility_is_none_with_too_little_history():
    assert realized_volatility([100.0]) is None
    assert realized_volatility([100.0, 101.0]) is None


def test_realized_volatility_series_aligns_with_underlying_bars(store):
    bars = tuple(
        UnderlyingBar(
            symbol="TEST", bar_date=date(2026, 8, 1) + timedelta(days=i),
            open=100 + i, high=101 + i, low=99 + i, close=100 + i * 0.5,
            volume=1_000_000,
        )
        for i in range(25)
    )
    store.write_underlying(bars)
    rows = store.underlying_bars("TEST")

    series = realized_volatility_series(rows, window=21)

    assert len(series) == 25 - 21 + 1
    assert series[0][0] == date(2026, 8, 1) + timedelta(days=20)
    assert all(value > 0 for _date, value in series)
