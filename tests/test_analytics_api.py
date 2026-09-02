"""U18: the analytics API (symbols, surface, comparison, violations), through Flask's real routing.

Every test drives the endpoints through Flask's test client, the same
discipline ``test_api.py`` uses for pricing — what is proven is the HTTP
contract a real caller sees, not just that the underlying ``analytics``
functions return the right shape. The archive is a real ``Store`` on a temp
path, injected through ``app.config["ANALYTICS_STORE"]``
(``api/analytics_routes.py``'s own test seam) — never the real
``data/quotes.db``.
"""

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from app import app as flask_app  # noqa: E402

from marketdata import ChainSnapshot, QuoteRecord, Store, TRACKED
from marketdata.derive import derive_batch
from pricing.american import American_Option

R, Y = 0.05, 0.02
NEAR = date(2026, 9, 18)


def price(strike, sig, expiry, as_of, *, S=100.0):
    T_years = (expiry - as_of).days / 365.0
    opt = American_Option("call", S, strike, R, sig, Y, T_years, "BT")
    opt.setTreeSteps(200)
    return float(opt.BT())


#: Close enough to every test's ``as_of`` (2026-08-06 unless noted) to read as
#: fresh under ``marketdata.checks.is_stale``'s one-day window -- a missing or
#: stale marker is excluded from the arbitrage scan before any bound runs,
#: which would make the deliberate violation below invisible.
TRADED = datetime(2026, 8, 6, 13, 0, tzinfo=timezone.utc)


def quote(contract, strike, bid, ask, *, option_type="call", traded=TRADED):
    mid = (bid + ask) / 2 if bid and ask else None
    return QuoteRecord(
        contract_symbol=contract, option_type=option_type, strike=strike,
        bid=bid, ask=ask, last=mid, volume=10, open_interest=20,
        provider_iv=0.2, last_trade_at=traded,
    )


def snapshot(symbol, as_of, quotes, *, captured_at=None, expiry=NEAR, underlying=100.0):
    moment = captured_at or (
        datetime.combine(as_of, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=14)
    )
    return ChainSnapshot(
        provider="fake", symbol=symbol, expiry=expiry, captured_at=moment,
        as_of=as_of, origin="live", underlying_price=underlying,
        quotes=tuple(quotes), risk_free_rate=R, dividend_yield=Y,
    )


@pytest.fixture
def store(tmp_path):
    test_store = Store(tmp_path / "analytics_api.db")
    flask_app.config["ANALYTICS_STORE"] = test_store
    yield test_store
    flask_app.config.pop("ANALYTICS_STORE", None)


@pytest.fixture
def client():
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as test_client:
        yield test_client


# --------------------------------------------------------------------------
# The tracked set (R37)
# --------------------------------------------------------------------------


def test_symbols_endpoint_reflects_the_tracked_set_not_a_hardcoded_list(store, client):
    response = client.get("/api/analytics/symbols")
    body = response.get_json()

    assert response.status_code == 200
    assert body["symbols"] == list(TRACKED)
    assert body["default"] == TRACKED[0]


@pytest.mark.parametrize("endpoint", ["surface", "comparison", "violations"])
def test_a_symbol_outside_the_tracked_set_is_rejected_before_reaching_the_query_layer(
    store, client, endpoint
):
    response = client.get(f"/api/analytics/{endpoint}", query_string={"symbol": "ZZZZ"})

    assert response.status_code == 400
    assert "must be one of" in response.get_json()["error"]


# --------------------------------------------------------------------------
# Surface (R19, R33, R36)
# --------------------------------------------------------------------------


def test_surface_returns_skew_and_the_snapshots_own_capture_time(store, client):
    as_of = date(2026, 8, 6)
    captured_at = datetime(2026, 8, 6, 14, 0, tzinfo=timezone.utc)
    strikes_vols = [(90.0, 0.20), (100.0, 0.25), (110.0, 0.30)]
    quotes = [
        quote(f"C{strike:.0f}", strike, price(strike, sig, NEAR, as_of) - 0.01, price(strike, sig, NEAR, as_of) + 0.01)
        for strike, sig in strikes_vols
    ]
    store.write_snapshot(snapshot("SPY", as_of, quotes, captured_at=captured_at))
    derive_batch(store)

    response = client.get("/api/analytics/surface", query_string={"symbol": "SPY"})
    body = response.get_json()

    assert response.status_code == 200
    assert body["symbol"] == "SPY"
    # The snapshot's own timestamp, not whatever time the test happens to run at.
    assert body["capture_time"] == captured_at.isoformat()
    assert len(body["skew"]) == 1
    assert body["skew"][0]["status"] == "ok"
    assert [p["strike"] for p in body["skew"][0]["points"]] == [90.0, 100.0, 110.0]
    # Only one captured expiry -- term structure per strike is short of the
    # three-expiry floor and must say so rather than draw a two-point line.
    assert all(curve["status"] == "insufficient_data" for curve in body["term_structure"])


def test_surface_before_any_capture_reports_no_data_rather_than_erroring(store, client):
    response = client.get("/api/analytics/surface", query_string={"symbol": "SPY"})
    body = response.get_json()

    assert response.status_code == 200
    assert body["capture_time"] is None
    assert body["skew"] == []
    assert body["term_structure"] == []


# --------------------------------------------------------------------------
# AE3: the last successful capture time survives a currently-unreachable provider
# --------------------------------------------------------------------------


def test_the_view_shows_the_last_successful_capture_time_when_the_provider_is_currently_unreachable(
    store, client
):
    """This endpoint never calls the provider (KTD9) -- it only ever reads
    what a past successful run persisted, so "the provider is unreachable
    right now" cannot change what it reports. Nothing here simulates a
    live provider outage; the point is that this route has no code path
    that could depend on one."""
    stale_capture = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    store.write_snapshot(
        snapshot("SPY", date(2026, 7, 1), [quote("C0", 100.0, 9.0, 9.2)], captured_at=stale_capture)
    )

    body = client.get("/api/analytics/surface", query_string={"symbol": "SPY"}).get_json()

    assert body["capture_time"] == stale_capture.isoformat()


# --------------------------------------------------------------------------
# Comparison (R20)
# --------------------------------------------------------------------------


def test_comparison_returns_one_point_for_a_single_captured_day(store, client):
    as_of = date(2026, 8, 6)
    quotes = [quote("C0", 100.0, price(100.0, 0.22, NEAR, as_of) - 0.01, price(100.0, 0.22, NEAR, as_of) + 0.01)]
    store.write_snapshot(snapshot("SPY", as_of, quotes))
    derive_batch(store)

    body = client.get("/api/analytics/comparison", query_string={"symbol": "SPY"}).get_json()

    assert body["symbol"] == "SPY"
    assert len(body["points"]) == 1
    assert body["points"][0]["as_of"] == as_of.isoformat()
    assert body["points"][0]["implied_vol"] == pytest.approx(0.22, abs=5e-3)
    assert body["points"][0]["realized_vol"] is None


# --------------------------------------------------------------------------
# Violations (R21)
# --------------------------------------------------------------------------


def test_violations_are_returned_with_a_count(store, client):
    as_of = date(2026, 8, 6)
    good = [
        quote(f"C{strike:.0f}", strike, price(strike, sig, NEAR, as_of) - 0.01, price(strike, sig, NEAR, as_of) + 0.01)
        for strike, sig in [(90.0, 0.20), (100.0, 0.25), (110.0, 0.30)]
    ]
    violating = [
        quote("C120", 120.0, 0.10, 0.20),
        quote("P120", 120.0, 30.0, 30.5, option_type="put"),
    ]
    store.write_snapshot(snapshot("SPY", as_of, good + violating))
    derive_batch(store)

    body = client.get("/api/analytics/violations", query_string={"symbol": "SPY"}).get_json()

    assert body["symbol"] == "SPY"
    assert body["count"] == 2
    assert {row["contract_symbol"] for row in body["violations"]} == {"C120", "P120"}
    assert all(row["kind"] == "put_call_band" for row in body["violations"])


def test_violations_before_any_capture_is_an_empty_table_not_an_error(store, client):
    body = client.get("/api/analytics/violations", query_string={"symbol": "SPY"}).get_json()

    assert body["count"] == 0
    assert body["violations"] == []
