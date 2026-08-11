"""U18: implied volatility against realized, aligned by day (R20).

Goes through the real pipeline like ``test_analytics.py`` does for the
surface — write snapshots across several days, derive, write underlying
bars, read through ``build_implied_vs_realized`` — rather than constructing
``ComparisonPoint`` objects by hand.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from analytics.comparison import build_implied_vs_realized
from marketdata import ChainSnapshot, QuoteRecord, Store, UnderlyingBar
from marketdata.derive import derive_batch
from pricing.american import American_Option

R, Y = 0.05, 0.02
NEAR = date(2026, 9, 18)
FAR = date(2026, 12, 18)


def price(strike, sig, expiry, as_of, *, S=100.0):
    T_years = (expiry - as_of).days / 365.0
    opt = American_Option("call", S, strike, R, sig, Y, T_years, "BT")
    opt.setTreeSteps(200)
    return float(opt.BT())


def quote(contract, strike, bid, ask, *, option_type="call", traded=None):
    mid = (bid + ask) / 2 if bid and ask else None
    return QuoteRecord(
        contract_symbol=contract, option_type=option_type, strike=strike,
        bid=bid, ask=ask, last=mid, volume=10, open_interest=20,
        provider_iv=0.2, last_trade_at=traded,
    )


def snapshot(as_of, quotes, *, expiry=NEAR, underlying=100.0):
    captured_at = datetime.combine(as_of, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=14)
    return ChainSnapshot(
        provider="fake", symbol="TEST", expiry=expiry, captured_at=captured_at,
        as_of=as_of, origin="live", underlying_price=underlying,
        quotes=tuple(quotes), risk_free_rate=R, dividend_yield=Y,
    )


def bar(day, close):
    return UnderlyingBar(
        symbol="TEST", bar_date=day, open=close, high=close + 1, low=close - 1,
        close=close, volume=1_000_000,
    )


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "comparison.db")


def test_returns_nothing_before_any_capture_or_bars(store):
    assert build_implied_vs_realized(store, "TEST") == []


def test_implied_side_is_the_at_the_money_call_iv_not_a_mean_across_strikes(store):
    """Spot is 100, so the 100 strike is the reading -- not the 0.283 average
    of the three. The wings are deliberately far from the body here: an
    unweighted mean is exactly what made this figure unusable on the real
    archive, where the front expiry's out-of-the-money implied vols run into
    the hundreds of percent."""
    as_of = date(2026, 8, 6)
    strikes_vols = [(95.0, 0.30), (100.0, 0.20), (105.0, 0.35)]
    quotes = [
        quote(f"C{k:.0f}", k, price(k, sig, NEAR, as_of) - 0.01, price(k, sig, NEAR, as_of) + 0.01)
        for k, sig in strikes_vols
    ]
    store.write_snapshot(snapshot(as_of, quotes))
    derive_batch(store)

    points = build_implied_vs_realized(store, "TEST")
    assert len(points) == 1
    assert points[0].as_of == as_of
    assert points[0].implied_vol == pytest.approx(0.20, abs=5e-3)
    assert points[0].realized_vol is None


def test_a_violating_leg_at_the_money_is_skipped_for_the_nearest_clean_strike(store):
    """R21 matters more here than on the surface: at the money the chosen leg
    *is* the answer, so a leg failing a no-arbitrage bound would set the day's
    number outright rather than nudging an average."""
    as_of = date(2026, 8, 6)
    fresh = datetime.combine(as_of, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=13)

    clean = [
        quote(f"C{k:.0f}", k, price(k, sig, NEAR, as_of) - 0.01, price(k, sig, NEAR, as_of) + 0.01, traded=fresh)
        for k, sig in [(100.5, 0.24), (97.0, 0.22)]
    ]
    # A call far too cheap against a rich put at the same strike: the put-call
    # band violation `find_violations` catches, sitting exactly at the money.
    violating = [
        quote("C100", 100.0, 0.10, 0.20, traded=fresh),
        quote("P100", 100.0, 30.0, 30.5, option_type="put", traded=fresh),
    ]
    store.write_snapshot(snapshot(as_of, clean + violating))
    derive_batch(store)

    points = build_implied_vs_realized(store, "TEST")
    assert len(points) == 1
    # 100.5 is the nearest surviving strike to a spot of 100.
    assert points[0].implied_vol == pytest.approx(0.24, abs=5e-3)


def test_an_expiry_with_nothing_left_to_invert_falls_through_to_the_next_one(store):
    """A zero-days-to-expiry chain routinely has no solved call left. Taking
    the nearest expiry unconditionally dropped those capture days from the
    series entirely instead of reading the expiry behind it."""
    as_of = date(2026, 8, 6)
    empty_front = [quote("P_NEAR", 100.0, 4.0, 4.2, option_type="put")]
    behind = [quote("C_FAR", 100.0, price(100.0, 0.28, FAR, as_of) - 0.01, price(100.0, 0.28, FAR, as_of) + 0.01)]

    store.write_snapshot(snapshot(as_of, empty_front, expiry=NEAR))
    store.write_snapshot(snapshot(as_of, behind, expiry=FAR))
    derive_batch(store)

    points = build_implied_vs_realized(store, "TEST")
    assert len(points) == 1
    assert points[0].implied_vol == pytest.approx(0.28, abs=5e-3)


def test_a_day_with_no_recorded_spot_reads_none_for_implied_rather_than_crashing(store):
    """underlying_price_at can itself return None -- no snapshot for this
    symbol has ever recorded a spot as of this moment. _atm_implied's early
    return on that must not raise, and the day should still surface via its
    realized side (proven here by giving it one) rather than the None
    implied reading silently deleting the whole day from the series."""
    as_of = date(2026, 8, 6)
    quotes = [quote("C0", 100.0, price(100.0, 0.2, NEAR, as_of) - 0.01, price(100.0, 0.2, NEAR, as_of) + 0.01)]
    store.write_snapshot(snapshot(as_of, quotes, underlying=None))
    derive_batch(store)

    closes = [100 + i * 0.3 for i in range(21)]
    bars = tuple(bar(as_of - timedelta(days=20 - i), close) for i, close in enumerate(closes))
    store.write_underlying(bars)

    points = build_implied_vs_realized(store, "TEST", window=21)
    by_date = {p.as_of: p for p in points}
    assert as_of in by_date
    assert by_date[as_of].implied_vol is None
    assert by_date[as_of].realized_vol is not None


def test_realized_side_appears_once_the_window_fills_even_with_no_implied_yet(store):
    closes = [100 + i * 0.3 for i in range(21)]
    bars = tuple(bar(date(2026, 7, 1) + timedelta(days=i), close) for i, close in enumerate(closes))
    store.write_underlying(bars)

    points = build_implied_vs_realized(store, "TEST", window=21)
    assert len(points) == 1
    assert points[0].as_of == date(2026, 7, 1) + timedelta(days=20)
    assert points[0].realized_vol is not None
    assert points[0].implied_vol is None


def test_implied_and_realized_align_on_a_shared_day(store):
    closes = [100 + i * 0.3 for i in range(21)]
    bars = tuple(bar(date(2026, 7, 1) + timedelta(days=i), close) for i, close in enumerate(closes))
    store.write_underlying(bars)
    shared_day = date(2026, 7, 1) + timedelta(days=20)

    quotes = [quote("C0", 100.0, price(100.0, 0.22, NEAR, shared_day) - 0.01, price(100.0, 0.22, NEAR, shared_day) + 0.01)]
    store.write_snapshot(snapshot(shared_day, quotes))
    derive_batch(store)

    points = build_implied_vs_realized(store, "TEST", window=21)
    assert len(points) == 1
    assert points[0].implied_vol is not None
    assert points[0].realized_vol is not None


def test_a_settled_expiry_is_not_used_for_a_later_capture_days_implied_side(store):
    """The front expiry is whichever is still live *as of that day's own capture* —
    a contract that later settles must not keep contributing to days after it expired."""
    early_day = date(2026, 8, 1)
    late_day = date(2026, 9, 20)  # after NEAR (2026-09-18) has settled

    early_quotes = [quote("C0", 100.0, price(100.0, 0.20, NEAR, early_day) - 0.01, price(100.0, 0.20, NEAR, early_day) + 0.01)]
    store.write_snapshot(snapshot(early_day, early_quotes, expiry=NEAR))

    far_quotes = [quote("C1", 100.0, price(100.0, 0.30, FAR, late_day) - 0.01, price(100.0, 0.30, FAR, late_day) + 0.01)]
    store.write_snapshot(snapshot(late_day, far_quotes, expiry=FAR))
    derive_batch(store)

    points = build_implied_vs_realized(store, "TEST")
    by_date = {p.as_of: p.implied_vol for p in points}
    assert by_date[early_day] == pytest.approx(0.20, abs=5e-3)
    assert by_date[late_day] == pytest.approx(0.30, abs=5e-3)
