"""U17: skew, term structure, and realized volatility, computed from the archive only.

Every skew/term-structure test goes through the real pipeline — write a
snapshot, run ``derive_batch``, read through ``build_skew``/
``build_term_structure`` — rather than constructing curve objects by hand, so
what is proven is that a real captured chain produces the right shape end to
end, including U16's exclusion (R21).
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from analytics.realized import realized_volatility, realized_volatility_series
from analytics.surface import (
    INSUFFICIENT_DATA,
    OK,
    _reconstruct_chain,
    build_skew,
    build_term_structure,
    build_violations,
)
from marketdata import ChainSnapshot, QuoteRecord, Store, UnderlyingBar
from marketdata.checks import PUT_CALL_BAND
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
# Expired expiries do not reach the surface or contaminate the violation scan
# --------------------------------------------------------------------------


def test_an_expiry_already_settled_as_of_the_latest_capture_is_absent_from_skew(store):
    """`symbols_and_expiries` returns every expiry ever captured, including
    ones that had already expired by the time this snapshot was taken.
    Rendering one as a current skew curve would show a contract nobody can
    trade any longer as though it were live."""
    settled = date(2026, 1, 15)  # before T0's date (2026-08-06)
    p = price(100.0, 0.25, NEAR)
    store.write_snapshot(
        snapshot(settled, [quote("C100_settled", 100.0, p - 0.01, p + 0.01)])
    )
    store.write_snapshot(
        snapshot(NEAR, [quote(f"C{s:.0f}", s, price(s, v, NEAR) - 0.01, price(s, v, NEAR) + 0.01)
                         for s, v in [(90.0, 0.20), (100.0, 0.25), (110.0, 0.30)]])
    )
    derive_batch(store)

    curves = build_skew(store, "TEST")
    assert [c.expiry for c in curves] == [NEAR]


def test_a_settled_expiry_does_not_contaminate_the_violation_scan_for_a_live_one(store):
    """A settled expiry's own contracts have real bid/ask that could pair with
    a live expiry's in a calendar check if both were fed to `find_violations`
    together -- excluding it before that scan runs, not just before display,
    is what `_live_expiries` guarantees."""
    settled = date(2026, 1, 15)
    good_strikes = [(90.0, 0.20), (100.0, 0.25), (110.0, 0.30)]
    live_quotes = [
        quote(f"C{s:.0f}", s, price(s, v, NEAR) - 0.01, price(s, v, NEAR) + 0.01)
        for s, v in good_strikes
    ]
    store.write_snapshot(
        # A deliberately rich quote on the settled expiry -- if it were fed
        # into the calendar check alongside NEAR's cheaper near-dated quotes,
        # `near_bid > far_ask` could misfire depending on ordering.
        snapshot(settled, [quote("C100_settled", 100.0, 500.0, 500.5)])
    )
    store.write_snapshot(snapshot(NEAR, live_quotes))
    derive_batch(store)

    curves = build_skew(store, "TEST")
    assert len(curves) == 1
    assert curves[0].status == OK
    assert [pt.strike for pt in curves[0].points] == [90.0, 100.0, 110.0]


# --------------------------------------------------------------------------
# Chain reconstruction uses the most recently observed row
# --------------------------------------------------------------------------


def test_chain_reconstruction_uses_the_most_recently_observed_snapshot_context(store):
    """Two captures of the same chain, at different times, carry different
    underlying prices. The reconstructed chain's spot must come from the
    later capture -- not from whichever row the query happens to return
    first, which the store gives no ordering guarantee on."""
    earlier = quote("C100", 100.0, 9.0, 9.2, traded=T0 - timedelta(hours=2))
    later_time = T0 + timedelta(hours=1)
    later = QuoteRecord(
        contract_symbol="C100", option_type="call", strike=100.0,
        bid=9.5, ask=9.7, last=9.6, volume=10, open_interest=20,
        provider_iv=0.2, last_trade_at=later_time,
    )
    store.write_snapshot(
        ChainSnapshot(
            provider="fake", symbol="TEST", expiry=NEAR, captured_at=T0,
            as_of=T0.date(), origin="live", underlying_price=100.0,
            quotes=(earlier,), risk_free_rate=0.05, dividend_yield=0.02,
        )
    )
    store.write_snapshot(
        ChainSnapshot(
            provider="fake", symbol="TEST", expiry=NEAR, captured_at=later_time,
            as_of=later_time.date(), origin="live", underlying_price=104.0,
            quotes=(later,), risk_free_rate=0.06, dividend_yield=0.03,
        )
    )

    chain = _reconstruct_chain(store, "TEST", NEAR, later_time + timedelta(minutes=1))
    assert chain.underlying_price == pytest.approx(104.0)
    assert chain.risk_free_rate == pytest.approx(0.06)
    assert chain.dividend_yield == pytest.approx(0.03)


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
# build_violations: what U16 excluded, shown rather than dropped (R21)
# --------------------------------------------------------------------------


def test_an_excluded_leg_is_visible_through_build_violations(store):
    good_strikes = [(90.0, 0.20), (100.0, 0.25), (110.0, 0.30)]
    quotes = [
        quote(f"C{strike:.0f}", strike, price(strike, sig, NEAR) - 0.01, price(strike, sig, NEAR) + 0.01)
        for strike, sig in good_strikes
    ]
    quotes.append(quote("C120", 120.0, 0.10, 0.20))
    quotes.append(quote("P120", 120.0, 30.0, 30.5, option_type="put"))

    store.write_snapshot(snapshot(NEAR, quotes))
    derive_batch(store)

    rows = build_violations(store, "TEST")
    assert {r.contract_symbol for r in rows} == {"C120", "P120"}
    assert all(r.kind == PUT_CALL_BAND for r in rows)
    assert all(r.expiry == NEAR for r in rows)
    call_row = next(r for r in rows if r.contract_symbol == "C120")
    assert call_row.strike == 120.0
    assert call_row.option_type == "call"
    assert "lower bound" in call_row.detail


def test_a_clean_surface_has_no_violations(store):
    good_strikes = [(90.0, 0.20), (100.0, 0.25), (110.0, 0.30)]
    quotes = [
        quote(f"C{strike:.0f}", strike, price(strike, sig, NEAR) - 0.01, price(strike, sig, NEAR) + 0.01)
        for strike, sig in good_strikes
    ]
    store.write_snapshot(snapshot(NEAR, quotes))
    derive_batch(store)

    assert build_violations(store, "TEST") == []


def test_build_violations_returns_nothing_before_any_capture(store):
    assert build_violations(store, "TEST") == []


# --------------------------------------------------------------------------
# Realized volatility
# --------------------------------------------------------------------------


def test_realized_volatility_matches_a_hand_computed_value():
    """The expected figure is a literal, not a formula run alongside
    ``analytics.realized``'s own — a hand-rolled re-derivation here would
    share any bug the production formula has (an off-by-one in the sample
    variance's denominator, for instance) and could never catch it. Computed
    once, independently, via three routes that agree to float precision:
    ``numpy.std(log_returns, ddof=1)``, ``statistics.stdev(log_returns)``,
    and the manual sum-of-squares formula — all times ``sqrt(252)``, on the
    fixed closes series below.
    """
    closes = [100.0, 102.0, 101.0, 105.0, 103.0]
    expected = 0.4248874656319516

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
