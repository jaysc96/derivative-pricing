"""SQLite persistence for raw quotes, underlying bars, and the capture record.

One file (KTD7), because re-inverting accumulated history is a query over every
stored quote — cheap against an indexed table, a directory walk against dated
flat files. It also keeps the hosting requirement as small as it can be.

**Raw quotes carry no volatility column of our own** (KTD9). What we compute
lives in a separate derived table stamped with the engine version, so the raw
layer stays the single source of truth and can be re-inverted wholesale when
the engine changes. The provider's *own* implied volatility is stored here,
which is not a contradiction: it is part of the observation, not something we
derived, and the difference between it and ours is a product.

**Quotes are deduplicated by observation, not by capture.** The unique key is
the contract plus its recency marker plus both sides of the quote, so polling
twice in a day stores one row when nothing moved and two when something did.
A quote row therefore belongs to the capture that *first* saw it. That makes
the archive point-in-time by construction — ``chain_as_of`` reconstructs what
was knowable at a moment by taking the latest observation per contract at or
before it, which is what a backtest needs and what a per-capture snapshot table
would quietly get wrong.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date, datetime
from pathlib import Path

from .adapter import ChainSnapshot, QuoteRecord, UnderlyingBar

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY,
    provider        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    expiry          TEXT NOT NULL,
    captured_at     TEXT NOT NULL,
    as_of           TEXT NOT NULL,
    origin          TEXT NOT NULL CHECK (origin IN ('live', 'backfill')),
    underlying_price REAL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_expiry
    ON snapshots (symbol, expiry, as_of);

CREATE TABLE IF NOT EXISTS quotes (
    id              INTEGER PRIMARY KEY,
    snapshot_id     INTEGER NOT NULL REFERENCES snapshots (id),
    contract_symbol TEXT NOT NULL,
    option_type     TEXT NOT NULL CHECK (option_type IN ('call', 'put')),
    strike          REAL NOT NULL,
    bid             REAL,
    ask             REAL,
    last            REAL,
    volume          INTEGER,
    open_interest   INTEGER,
    provider_iv     REAL,
    last_trade_at   TEXT,
    UNIQUE (contract_symbol, last_trade_at, bid, ask)
);
CREATE INDEX IF NOT EXISTS idx_quotes_snapshot ON quotes (snapshot_id);
CREATE INDEX IF NOT EXISTS idx_quotes_contract ON quotes (contract_symbol);

CREATE TABLE IF NOT EXISTS underlying_bars (
    symbol   TEXT NOT NULL,
    bar_date TEXT NOT NULL,
    open     REAL NOT NULL,
    high     REAL NOT NULL,
    low      REAL NOT NULL,
    close    REAL NOT NULL,
    volume   INTEGER,
    PRIMARY KEY (symbol, bar_date)
);

CREATE TABLE IF NOT EXISTS capture_runs (
    id          INTEGER PRIMARY KEY,
    started_at  TEXT NOT NULL,
    provider    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    productive  INTEGER NOT NULL,
    reason      TEXT,
    contracts   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON capture_runs (started_at);
"""

#: The only reasons a run is allowed to record. Never raw exception text and
#: never a URL — U11 commits artifacts derived from this table into a public
#: repository, and raw error text is how a credential reaches git history.
FAILURE_REASONS = frozenset(
    {
        "rate_limited",
        "timeout",
        "http_error",
        "network_error",
        "empty_response",
        "malformed",
        "all_zero_sided",
    }
)


class Store:
    """Owns the SQLite file. Every write goes through here."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connect_and_migrate()

    def _connect_and_migrate(self) -> None:
        with closing(self.connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def write_snapshot(self, snapshot: ChainSnapshot) -> tuple[int, int]:
        """Persist a chain. Returns the snapshot id and how many quotes were new.

        Quotes already seen with the same recency marker and the same two sides
        are ignored rather than duplicated, so a second poll in a quiet hour
        adds a snapshot row and no quote rows.
        """
        with closing(self.connect()) as conn:
            cursor = conn.execute(
                """INSERT INTO snapshots
                   (provider, symbol, expiry, captured_at, as_of, origin, underlying_price)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot.provider,
                    snapshot.symbol,
                    snapshot.expiry.isoformat(),
                    snapshot.captured_at.isoformat(),
                    snapshot.as_of.isoformat(),
                    snapshot.origin,
                    snapshot.underlying_price,
                ),
            )
            snapshot_id = int(cursor.lastrowid)

            written = 0
            for quote in snapshot.quotes:
                result = conn.execute(
                    """INSERT OR IGNORE INTO quotes
                       (snapshot_id, contract_symbol, option_type, strike, bid, ask,
                        last, volume, open_interest, provider_iv, last_trade_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        snapshot_id,
                        quote.contract_symbol,
                        quote.option_type,
                        quote.strike,
                        quote.bid,
                        quote.ask,
                        quote.last,
                        quote.volume,
                        quote.open_interest,
                        quote.provider_iv,
                        quote.last_trade_at.isoformat() if quote.last_trade_at else None,
                    ),
                )
                written += result.rowcount
            conn.commit()
            return snapshot_id, written

    def write_underlying(self, bars: tuple[UnderlyingBar, ...]) -> int:
        """Upsert daily bars. Idempotent — running twice does not duplicate."""
        with closing(self.connect()) as conn:
            written = 0
            for bar in bars:
                result = conn.execute(
                    """INSERT INTO underlying_bars
                       (symbol, bar_date, open, high, low, close, volume)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT (symbol, bar_date) DO UPDATE SET
                         open=excluded.open, high=excluded.high, low=excluded.low,
                         close=excluded.close, volume=excluded.volume""",
                    (
                        bar.symbol,
                        bar.bar_date.isoformat(),
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume,
                    ),
                )
                written += result.rowcount
            conn.commit()
            return written

    def record_run(
        self,
        *,
        started_at: datetime,
        provider: str,
        symbol: str,
        productive: bool,
        reason: str | None = None,
        contracts: int = 0,
    ) -> None:
        """Record one symbol's outcome in one capture run.

        An unproductive run is the point of this table. The live provider
        degrades by returning empty frames rather than raising, so without a
        record of zero-contract runs a field rename on the provider's side
        would leave every run reporting success while history quietly stopped.
        """
        if reason is not None and reason not in FAILURE_REASONS:
            raise ValueError(
                f"reason {reason!r} is not one of the fixed codes; "
                "raw error text must never reach this table"
            )
        with closing(self.connect()) as conn:
            conn.execute(
                """INSERT INTO capture_runs
                   (started_at, provider, symbol, productive, reason, contracts)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (started_at.isoformat(), provider, symbol, int(productive), reason, contracts),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def chain_as_of(self, symbol: str, expiry: date, moment: datetime) -> list[sqlite3.Row]:
        """What was knowable about this chain at ``moment``.

        The latest observation per contract at or before that time — not the
        current state of the contract. This is the read a backtest needs, and
        keeping it as a query rather than a stored per-capture chain is what
        makes point-in-time correctness a property of the archive rather than
        of whoever writes the query.
        """
        with closing(self.connect()) as conn:
            return list(
                conn.execute(
                    """SELECT q.*, s.captured_at, s.as_of, s.origin, s.underlying_price
                       FROM quotes q
                       JOIN snapshots s ON s.id = q.snapshot_id
                       WHERE s.symbol = ? AND s.expiry = ? AND s.captured_at <= ?
                         AND q.id IN (
                           SELECT MAX(q2.id) FROM quotes q2
                           JOIN snapshots s2 ON s2.id = q2.snapshot_id
                           WHERE s2.symbol = s.symbol AND s2.expiry = s.expiry
                             AND s2.captured_at <= ?
                           GROUP BY q2.contract_symbol
                         )
                       ORDER BY q.option_type, q.strike""",
                    (symbol, expiry.isoformat(), moment.isoformat(), moment.isoformat()),
                )
            )

    def coverage(self) -> dict:
        """Counts the evidence artifacts and the quality gate both read."""
        with closing(self.connect()) as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS quotes,
                          SUM(CASE WHEN bid > 0 AND ask > 0 THEN 1 ELSE 0 END) AS two_sided,
                          COUNT(DISTINCT contract_symbol) AS contracts
                   FROM quotes"""
            ).fetchone()
            snapshots = conn.execute(
                """SELECT COUNT(*) AS n, COUNT(DISTINCT symbol) AS symbols,
                          MIN(captured_at) AS first, MAX(captured_at) AS last
                   FROM snapshots"""
            ).fetchone()
            return {
                "quotes": row["quotes"] or 0,
                "two_sided": row["two_sided"] or 0,
                "contracts": row["contracts"] or 0,
                "snapshots": snapshots["n"] or 0,
                "symbols": snapshots["symbols"] or 0,
                "first_capture": snapshots["first"],
                "last_capture": snapshots["last"],
            }

    def run_record(self) -> list[sqlite3.Row]:
        """Every recorded capture outcome, newest first."""
        with closing(self.connect()) as conn:
            return list(
                conn.execute("SELECT * FROM capture_runs ORDER BY started_at DESC")
            )
