"""U8: the archive accumulates, deduplicates by observation, and stays point-in-time."""

from datetime import date, datetime, timedelta, timezone

import pytest

from marketdata import ChainSnapshot, QuoteRecord, Store, UnderlyingBar
from marketdata.secrets import MissingCredential, get_secret, load_dotenv

EXPIRY = date(2026, 9, 18)
T0 = datetime(2026, 8, 5, 14, 0, tzinfo=timezone.utc)


def quote(symbol_suffix="C00600000", *, bid=12.5, ask=12.9, traded=None, strike=600.0):
    return QuoteRecord(
        contract_symbol=f"SPY260918{symbol_suffix}",
        option_type="call" if "C" in symbol_suffix else "put",
        strike=strike,
        bid=bid,
        ask=ask,
        last=12.7,
        volume=431,
        open_interest=2210,
        provider_iv=0.1842,
        last_trade_at=traded or datetime(2026, 8, 5, 13, 58, tzinfo=timezone.utc),
    )


def snapshot(*quotes, captured_at=T0, origin="live", as_of=None):
    return ChainSnapshot(
        provider="yfinance",
        symbol="SPY",
        expiry=EXPIRY,
        captured_at=captured_at,
        as_of=as_of or captured_at.date(),
        origin=origin,
        underlying_price=604.0,
        quotes=quotes,
    )


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "quotes.db")


def test_captures_accumulate_rather_than_overwrite(store):
    """Two pulls of the same symbol and expiry are two snapshots, not one."""
    store.write_snapshot(snapshot(quote()))
    store.write_snapshot(snapshot(quote(bid=13.0, ask=13.4), captured_at=T0 + timedelta(hours=2)))

    assert store.coverage()["snapshots"] == 2
    assert store.coverage()["quotes"] == 2


def test_an_unchanged_quote_is_not_stored_twice(store):
    """Polling a quiet hour adds a snapshot and no quote rows."""
    store.write_snapshot(snapshot(quote()))
    _, written = store.write_snapshot(snapshot(quote(), captured_at=T0 + timedelta(hours=1)))

    assert written == 0
    assert store.coverage()["snapshots"] == 2
    assert store.coverage()["quotes"] == 1


def test_a_moved_quote_is_stored(store):
    """Same contract, same trade stamp, different market — that is new information."""
    store.write_snapshot(snapshot(quote(bid=12.5, ask=12.9)))
    _, written = store.write_snapshot(
        snapshot(quote(bid=12.6, ask=13.0), captured_at=T0 + timedelta(hours=1))
    )
    assert written == 1
    assert store.coverage()["quotes"] == 2


def test_recency_marker_is_stored_apart_from_capture_time(store):
    traded = datetime(2026, 7, 28, 14, 2, tzinfo=timezone.utc)
    store.write_snapshot(snapshot(quote(traded=traded)))

    row = store.chain_as_of("SPY", EXPIRY, T0 + timedelta(minutes=1))[0]
    assert row["last_trade_at"] == traded.isoformat()
    assert row["captured_at"] == T0.isoformat()
    assert row["last_trade_at"] != row["captured_at"]


def test_origin_and_as_of_survive_the_round_trip(store):
    """A backfilled row must never be mistaken for one we observed."""
    store.write_snapshot(
        snapshot(
            quote(),
            captured_at=T0,
            origin="backfill",
            as_of=date(2026, 1, 5),
        )
    )
    row = store.chain_as_of("SPY", EXPIRY, T0 + timedelta(minutes=1))[0]
    assert row["origin"] == "backfill"
    assert row["as_of"] == "2026-01-05"
    assert row["captured_at"] == T0.isoformat()


def test_origin_is_constrained_to_two_values(store):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        store.write_snapshot(snapshot(quote(), origin="guessed"))


# --------------------------------------------------------------------------
# Point-in-time reads
# --------------------------------------------------------------------------


def test_chain_as_of_returns_what_was_knowable_then(store):
    """Not the current state of the contract — the state at that moment.

    This is the property a backtest depends on, and getting it wrong is how
    lookahead bias enters a dataset that looks correct.
    """
    store.write_snapshot(snapshot(quote(bid=12.5, ask=12.9), captured_at=T0))
    store.write_snapshot(
        snapshot(quote(bid=20.0, ask=20.4), captured_at=T0 + timedelta(hours=3))
    )

    early = store.chain_as_of("SPY", EXPIRY, T0 + timedelta(hours=1))
    late = store.chain_as_of("SPY", EXPIRY, T0 + timedelta(hours=4))

    assert len(early) == 1 and early[0]["bid"] == pytest.approx(12.5)
    assert len(late) == 1 and late[0]["bid"] == pytest.approx(20.0)


def test_chain_as_of_before_any_capture_is_empty(store):
    store.write_snapshot(snapshot(quote()))
    assert store.chain_as_of("SPY", EXPIRY, T0 - timedelta(days=1)) == []


def test_coverage_counts_two_sided_quotes(store):
    store.write_snapshot(
        snapshot(
            quote("C00600000", bid=12.5, ask=12.9),
            quote("C00900000", bid=0.0, ask=0.05, strike=900.0),
            quote("P00600000", bid=9.1, ask=9.4),
        )
    )
    coverage = store.coverage()
    assert coverage["quotes"] == 3
    assert coverage["two_sided"] == 2
    assert coverage["symbols"] == 1


# --------------------------------------------------------------------------
# The capture record
# --------------------------------------------------------------------------


def test_unproductive_runs_are_recorded(store):
    store.record_run(
        started_at=T0, provider="yfinance", symbol="SPY", productive=False,
        reason="empty_response",
    )
    store.record_run(
        started_at=T0, provider="yfinance", symbol="QQQ", productive=True, contracts=412,
    )
    record = store.run_record()
    assert len(record) == 2
    assert {r["reason"] for r in record} == {"empty_response", None}


def test_raw_error_text_cannot_reach_the_record(store):
    """U11 commits artifacts derived from this table into a public repository."""
    with pytest.raises(ValueError, match="fixed codes"):
        store.record_run(
            started_at=T0,
            provider="yfinance",
            symbol="SPY",
            productive=False,
            reason="HTTPError at https://query2.finance.yahoo.com/?crumb=SECRET",
        )


# --------------------------------------------------------------------------
# Underlying bars
# --------------------------------------------------------------------------


def bar(day, close=604.0):
    return UnderlyingBar(
        symbol="SPY", bar_date=day, open=600.0, high=606.0, low=598.0,
        close=close, volume=71_000_000,
    )


def test_underlying_backfill_is_idempotent(store):
    bars = (bar(date(2026, 8, 3)), bar(date(2026, 8, 4)))
    store.write_underlying(bars)
    store.write_underlying(bars)

    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM underlying_bars").fetchone()[0] == 2


def test_underlying_rewrite_updates_rather_than_duplicates(store):
    store.write_underlying((bar(date(2026, 8, 4), close=604.0),))
    store.write_underlying((bar(date(2026, 8, 4), close=607.5),))

    with store.connect() as conn:
        rows = conn.execute("SELECT close FROM underlying_bars").fetchall()
    assert len(rows) == 1 and rows[0][0] == pytest.approx(607.5)


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------


def test_missing_required_credential_fails_fast_and_by_name(monkeypatch):
    monkeypatch.delenv("MARKETDATA_API_TOKEN", raising=False)
    with pytest.raises(MissingCredential, match="MARKETDATA_API_TOKEN"):
        get_secret("MARKETDATA_API_TOKEN")


def test_optional_credential_returns_none(monkeypatch):
    monkeypatch.delenv("MARKETDATA_API_TOKEN", raising=False)
    assert get_secret("MARKETDATA_API_TOKEN", required=False) is None


def test_dotenv_never_shadows_the_real_environment(tmp_path, monkeypatch):
    """A host's configured secret must win over a stale local checkout."""
    env_file = tmp_path / ".env"
    env_file.write_text("MARKETDATA_API_TOKEN=from-file\n# a comment\n\n")
    monkeypatch.setenv("MARKETDATA_API_TOKEN", "from-environment")

    load_dotenv(env_file)
    assert get_secret("MARKETDATA_API_TOKEN") == "from-environment"


def test_dotenv_supplies_what_the_environment_lacks(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('MARKETDATA_API_TOKEN="quoted-value"\n')
    monkeypatch.delenv("MARKETDATA_API_TOKEN", raising=False)

    load_dotenv(env_file)
    assert get_secret("MARKETDATA_API_TOKEN") == "quoted-value"
