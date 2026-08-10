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


def snapshot(*quotes, captured_at=T0, origin="live", as_of=None, risk_free_rate=None, dividend_yield=None):
    return ChainSnapshot(
        provider="yfinance",
        symbol="SPY",
        expiry=EXPIRY,
        captured_at=captured_at,
        as_of=as_of or captured_at.date(),
        origin=origin,
        underlying_price=604.0,
        quotes=quotes,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
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


def test_rate_and_yield_survive_the_round_trip(store):
    """Captured now, alongside the quotes — not looked up later at derive time."""
    store.write_snapshot(snapshot(quote(), risk_free_rate=0.0373, dividend_yield=0.0074))
    row = store.chain_as_of("SPY", EXPIRY, T0 + timedelta(minutes=1))[0]
    assert row["risk_free_rate"] == pytest.approx(0.0373)
    assert row["dividend_yield"] == pytest.approx(0.0074)


def test_a_missing_rate_or_yield_survives_as_null_not_a_fabricated_default(store):
    store.write_snapshot(snapshot(quote()))
    row = store.chain_as_of("SPY", EXPIRY, T0 + timedelta(minutes=1))[0]
    assert row["risk_free_rate"] is None
    assert row["dividend_yield"] is None


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


# --------------------------------------------------------------------------
# Deduplication must not swallow information
#
# The three tests below pin defects an earlier value-keyed UNIQUE constraint
# had. Each one passed silently before, because the fixtures that covered this
# area used fully-populated quotes moving in one direction.
# --------------------------------------------------------------------------


def test_a_quote_with_no_bid_still_deduplicates(store):
    """SQLite treats NULLs as distinct, so a UNIQUE across them constrains nothing.

    Roughly a third of real contracts have no bid. Under a value-keyed
    constraint every one of them re-inserted on every capture, forever, and
    the count of stored quotes stopped meaning distinct observations.
    """
    for hour in range(3):
        store.write_snapshot(
            snapshot(quote(bid=None, ask=0.05), captured_at=T0 + timedelta(hours=hour))
        )

    assert store.coverage()["quotes"] == 1, "an unchanged bidless quote is one observation"


def test_a_quote_that_returns_to_an_earlier_price_is_a_new_observation(store):
    """Bid and ask oscillate; a return to a previous level is not a duplicate.

    Keying dedup on the values themselves collides with an observation hours
    old and drops the current one, which then leaves the point-in-time read
    asserting a price that had already been superseded.
    """
    for hour, (bid, ask) in enumerate([(1.00, 1.10), (1.05, 1.15), (1.00, 1.10)]):
        store.write_snapshot(
            snapshot(quote(bid=bid, ask=ask), captured_at=T0 + timedelta(hours=hour))
        )

    assert store.coverage()["quotes"] == 3
    latest = store.chain_as_of("SPY", EXPIRY, T0 + timedelta(hours=2))
    assert latest[0]["bid"] == pytest.approx(1.00), "the read returned a superseded price"


def test_an_untraded_contract_still_tracks_its_moving_quote(store):
    """`last_trade_at` is frozen all day for a contract nobody trades.

    That collapses a value key to bid and ask alone, which is exactly the pair
    that moves — so the contracts least likely to trade were the ones whose
    price history was most likely to be dropped.
    """
    frozen = datetime(2026, 7, 28, 14, 2, tzinfo=timezone.utc)
    for hour, bid in enumerate([1.00, 1.05, 1.00, 1.05]):
        store.write_snapshot(
            snapshot(
                quote(bid=bid, ask=bid + 0.10, traded=frozen),
                captured_at=T0 + timedelta(hours=hour),
            )
        )

    assert store.coverage()["quotes"] == 4


def test_replaying_one_capture_writes_no_second_copy(store):
    """A retried run must be idempotent, even though a revert is not."""
    for _ in range(2):
        store.write_snapshot(snapshot(quote(), captured_at=T0))
    assert store.coverage()["quotes"] == 1


# --------------------------------------------------------------------------
# Backfill has to be reachable, not merely storable
# --------------------------------------------------------------------------


def test_a_backfilled_row_is_readable_at_the_date_it_describes(store):
    """The `as_of` column exists so this query works.

    Filtering the point-in-time read on capture time instead made every
    backfilled row invisible: it is written today, so it never satisfies a
    cutoff in the past it was written to fill.
    """
    store.write_snapshot(
        snapshot(quote(), captured_at=T0, origin="backfill", as_of=date(2026, 1, 15))
    )

    rows = store.chain_as_of("SPY", EXPIRY, datetime(2026, 1, 16, tzinfo=timezone.utc))
    assert len(rows) == 1 and rows[0]["origin"] == "backfill"


def test_a_backfilled_row_is_not_knowable_before_its_day_closes(store):
    """A vendor's record of a day is not available partway through that day.

    Dating it to the start of its `as_of` day would make every backfilled
    quote readable hours before it existed, which is precisely the lookahead
    the archive is built to keep out. It is dated to the day's end instead.
    """
    store.write_snapshot(
        snapshot(quote(), captured_at=T0, origin="backfill", as_of=date(2026, 1, 15))
    )

    midday = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    assert store.chain_as_of("SPY", EXPIRY, midday) == []


def test_a_backfill_does_not_displace_the_live_history_it_precedes(store):
    """Written last, but it describes January — so it sorts before February."""
    live = datetime(2026, 2, 2, 15, 0, tzinfo=timezone.utc)
    store.write_snapshot(snapshot(quote(bid=9.0, ask=9.2), captured_at=live))
    store.write_snapshot(
        snapshot(
            quote(bid=3.0, ask=3.2),
            captured_at=T0,
            origin="backfill",
            as_of=date(2026, 1, 15),
        )
    )

    january = store.chain_as_of("SPY", EXPIRY, datetime(2026, 1, 20, tzinfo=timezone.utc))
    february = store.chain_as_of("SPY", EXPIRY, datetime(2026, 2, 3, tzinfo=timezone.utc))

    assert january[0]["bid"] == pytest.approx(3.0)
    assert february[0]["bid"] == pytest.approx(9.0), "the backfill overwrote later history"


def test_a_database_from_an_older_schema_is_refused_by_name(tmp_path):
    """There is no migration path (KTD7) for a non-additive delta, so silence
    would be the wrong answer. Version 1 predates observed_at and changed
    quotes' own UNIQUE constraint -- not a delta `_ADDITIVE_COLUMNS` covers."""
    import sqlite3

    from marketdata.store import SchemaMismatch

    path = tmp_path / "old.db"
    Store(path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 1")

    with pytest.raises(SchemaMismatch, match="schema version 1"):
        Store(path)


def test_a_database_from_a_newer_schema_is_refused_not_silently_downgraded(tmp_path):
    """A rollback to older code against an already-migrated database must
    hard-fail rather than guess at a downgrade path nothing here writes."""
    import sqlite3

    from marketdata.store import SchemaMismatch

    path = tmp_path / "future.db"
    Store(path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 99")

    with pytest.raises(SchemaMismatch, match="schema version 99"):
        Store(path)


_V2_SCHEMA = """
CREATE TABLE snapshots (
    id INTEGER PRIMARY KEY, provider TEXT NOT NULL, symbol TEXT NOT NULL,
    expiry TEXT NOT NULL, captured_at TEXT NOT NULL, as_of TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('live', 'backfill')), underlying_price REAL
);
CREATE TABLE quotes (
    id INTEGER PRIMARY KEY, snapshot_id INTEGER NOT NULL REFERENCES snapshots (id),
    contract_symbol TEXT NOT NULL, option_type TEXT NOT NULL CHECK (option_type IN ('call', 'put')),
    strike REAL NOT NULL, bid REAL, ask REAL, last REAL, volume INTEGER,
    open_interest INTEGER, provider_iv REAL, last_trade_at TEXT, observed_at TEXT NOT NULL,
    UNIQUE (contract_symbol, observed_at)
);
CREATE TABLE capture_runs (
    id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, provider TEXT NOT NULL, symbol TEXT NOT NULL,
    productive INTEGER NOT NULL, reason TEXT, contracts INTEGER NOT NULL DEFAULT 0,
    expiries_requested INTEGER NOT NULL DEFAULT 0, expiries_captured INTEGER NOT NULL DEFAULT 0,
    two_sided INTEGER NOT NULL DEFAULT 0
);
"""


def _seed_v2_database(path):
    """A real schema-version-2 shape (9d0d33d's, pre risk_free_rate /
    dividend_yield / implied_vols) with one snapshot and one quote in it."""
    import sqlite3

    conn = sqlite3.connect(path)
    conn.executescript(_V2_SCHEMA)
    conn.execute("PRAGMA user_version = 2")
    conn.execute(
        "INSERT INTO snapshots (provider, symbol, expiry, captured_at, as_of, origin, underlying_price) "
        "VALUES ('fake','SPY','2026-09-18','2026-08-01T00:00:00','2026-08-01','live', 604.0)"
    )
    conn.execute(
        "INSERT INTO quotes (snapshot_id, contract_symbol, option_type, strike, bid, ask, observed_at) "
        "VALUES (1, 'SPYC600', 'call', 600.0, 12.5, 12.9, '2026-08-01T00:00:00')"
    )
    conn.commit()
    conn.close()


def test_a_version_2_database_is_upgraded_in_place_rather_than_refused(tmp_path):
    """The 2->3 delta is purely additive (two nullable snapshot columns, one
    new table) -- opening it must add what is missing and keep every row,
    not force a delete-and-recapture of an archive that needed no rebuild."""
    import sqlite3

    path = tmp_path / "v2.db"
    _seed_v2_database(path)

    store = Store(path)

    assert sqlite3.connect(path).execute("PRAGMA user_version").fetchone()[0] == 3
    rows = list(store.connect().execute("SELECT * FROM snapshots"))
    assert len(rows) == 1
    assert rows[0]["symbol"] == "SPY"
    assert rows[0]["risk_free_rate"] is None, "new column exists, old row's value is honestly unknown"
    quotes = list(store.connect().execute("SELECT * FROM quotes"))
    assert len(quotes) == 1
    assert quotes[0]["contract_symbol"] == "SPYC600"
    tables = {r[0] for r in store.connect().execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "implied_vols" in tables


def test_a_version_2_upgrade_is_safe_to_retry_after_a_partial_run(tmp_path):
    """ALTER TABLE commits immediately in SQLite's autocommit mode, ahead of
    this method's own version-bump commit -- a crash in between would leave
    a column already added but user_version still at 2. The next open must
    not crash on "duplicate column"."""
    import sqlite3

    path = tmp_path / "v2-partial.db"
    _seed_v2_database(path)
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE snapshots ADD COLUMN risk_free_rate REAL")
        # dividend_yield deliberately left unadded and user_version left at 2,
        # simulating a crash partway through the migration.

    store = Store(path)

    assert sqlite3.connect(path).execute("PRAGMA user_version").fetchone()[0] == 3
    cols = {r[1] for r in store.connect().execute("PRAGMA table_info(snapshots)")}
    assert {"risk_free_rate", "dividend_yield"} <= cols


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


def test_raw_coverage_matches_the_global_totals(store):
    """Same counting rule as ``coverage()`` — one definition of two-sided, not two."""
    store.write_snapshot(
        snapshot(
            quote("C00600000", bid=12.5, ask=12.9),
            quote("C00900000", bid=0.0, ask=0.05, strike=900.0),
        )
    )
    assert store.raw_coverage() == {"quotes": 2, "two_sided": 1}


def test_raw_coverage_narrows_to_one_chain(store):
    other_expiry = EXPIRY + timedelta(days=30)
    store.write_snapshot(snapshot(quote(bid=12.5, ask=12.9)))
    store.write_snapshot(
        ChainSnapshot(
            provider="yfinance", symbol="SPY", expiry=other_expiry, captured_at=T0,
            as_of=T0.date(), origin="live", underlying_price=604.0,
            quotes=(quote("C00600000C2", bid=None, ask=5.0),),
        )
    )

    narrowed = store.raw_coverage(symbol="SPY", expiry=EXPIRY)
    everything = store.raw_coverage()
    assert narrowed == {"quotes": 1, "two_sided": 1}
    assert everything == {"quotes": 2, "two_sided": 1}


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
