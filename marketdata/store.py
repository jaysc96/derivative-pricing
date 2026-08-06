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

**The risk-free rate and dividend yield live on the snapshot, not the derived
row.** Inverting a price needs both, and they are market observations exactly
like the quotes — captured at the same moment, stored once, never re-fetched.
Re-fetching today's rate to re-invert a January snapshot would be a quieter
version of the lookahead bias the point-in-time archive exists to prevent.
Both are nullable: a rate-fetch hiccup should not cost the chain itself, and
the derived layer is what declines to invert a snapshot missing either, with
its own explicit reason rather than a silent gap.

**Quotes are deduplicated against the previous observation, not against all
history.** Polling a quiet hour stores no new quote row; a quote that moves
stores one. The comparison is against the *latest stored observation for that
contract*, in Python, which matters for two reasons an earlier value-keyed
``UNIQUE`` got wrong:

* A ``UNIQUE`` spanning nullable columns does not constrain anything in SQLite,
  which treats NULLs as distinct. Roughly a third of real contracts have no
  bid, so those rows deduplicated against nothing and re-inserted on every
  single capture.
* A quote that *returns* to an earlier value is new information, not a
  duplicate. Bid and ask oscillate between two levels all day, and
  ``last_trade_at`` is frozen for a contract nobody has traded — so a key built
  from those values collides with an observation hours old, drops the current
  one, and leaves the archive asserting a stale price.

**Every stored quote carries its own ``observed_at``**, and that is the column
the point-in-time read orders on. For a live pull it is the capture time. For a
backfilled row it is the end of the ``as_of`` day, because a vendor's record of
a past date describes that date rather than the moment we asked for it — and
filtering those rows by capture time, as an earlier version did, made them
invisible to every historical query.

``chain_as_of`` therefore reconstructs what was knowable at a moment by taking
the latest observation per contract at or before it, which is what a backtest
needs and what a per-capture snapshot table would quietly get wrong.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date, datetime, time, timezone
from pathlib import Path

from .adapter import ChainSnapshot, QuoteRecord, UnderlyingBar

#: Bumped whenever the shape below changes. There is no migration tooling here
#: (KTD7 keeps this to one file), so a database written by an older shape is
#: refused by name rather than read as if it matched.
SCHEMA_VERSION = 3

#: SQLite's parameter limit is 32,766 on modern builds and 999 on older ones.
#: Chains run to a few hundred contracts, so chunking the lookup keeps the
#: dedup query within the smaller bound without anyone having to know which
#: build they are on.
_PARAM_CHUNK = 400

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY,
    provider        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    expiry          TEXT NOT NULL,
    captured_at     TEXT NOT NULL,
    as_of           TEXT NOT NULL,
    origin          TEXT NOT NULL CHECK (origin IN ('live', 'backfill')),
    underlying_price REAL,
    risk_free_rate  REAL,
    dividend_yield  REAL
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
    observed_at     TEXT NOT NULL,
    -- Both columns are NOT NULL, so this constraint actually constrains. It
    -- makes re-running one capture idempotent; it is not the dedup rule, which
    -- compares against the previous observation in write_snapshot.
    UNIQUE (contract_symbol, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_quotes_snapshot ON quotes (snapshot_id);
CREATE INDEX IF NOT EXISTS idx_quotes_observed ON quotes (contract_symbol, observed_at);

-- KTD9's derived table. `status` and `implied_vol` mirror InversionResult, so
-- the same vocabulary describes why a contract has no volatility whether the
-- reason is the solver's or upstream of it (missing rate/yield, no quote).
-- UNIQUE per (quote_id, engine_version) lets a rebuild recompute every quote
-- under a new version without colliding with rows the old version left behind
-- — the rebuild deletes those explicitly rather than relying on this to.
CREATE TABLE IF NOT EXISTS implied_vols (
    id              INTEGER PRIMARY KEY,
    quote_id        INTEGER NOT NULL REFERENCES quotes (id),
    engine_version  INTEGER NOT NULL,
    status          TEXT NOT NULL,
    implied_vol     REAL,
    detail          TEXT,
    computed_at     TEXT NOT NULL,
    UNIQUE (quote_id, engine_version)
);
CREATE INDEX IF NOT EXISTS idx_implied_vols_quote ON implied_vols (quote_id);
CREATE INDEX IF NOT EXISTS idx_implied_vols_version ON implied_vols (engine_version);

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
    contracts   INTEGER NOT NULL DEFAULT 0,
    -- A run that captured one of the four expiries it asked for is not the
    -- same event as one that captured all four, and recording only
    -- `productive` cannot tell them apart.
    expiries_requested INTEGER NOT NULL DEFAULT 0,
    expiries_captured  INTEGER NOT NULL DEFAULT 0,
    two_sided   INTEGER NOT NULL DEFAULT 0
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
        "not_supported",
    }
)


class SchemaMismatch(RuntimeError):
    """The database on disk was written by a different schema version."""


def observed_at(snapshot: ChainSnapshot) -> str:
    """When the quotes in this snapshot describe the market.

    For a live pull that is the capture time. For a backfilled row it is the
    end of the ``as_of`` day: a vendor's record of a past date is a statement
    about that date, and ordering it by when we happened to ask would file a
    January quote after everything captured since.
    """
    if snapshot.origin == "backfill":
        return datetime.combine(snapshot.as_of, time.max, tzinfo=timezone.utc).isoformat()
    return snapshot.captured_at.isoformat()


class Store:
    """Owns the SQLite file. Every write goes through here."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connect_and_migrate()

    def _connect_and_migrate(self) -> None:
        with closing(self.connect()) as conn:
            populated = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'quotes'"
            ).fetchone()[0]
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if populated and version != SCHEMA_VERSION:
                raise SchemaMismatch(
                    f"{self.path} was written by schema version {version}, and this "
                    f"code expects {SCHEMA_VERSION}. There is no migration path "
                    "(KTD7); delete the file and re-capture, or keep it aside and "
                    "point at a new one."
                )
            conn.executescript(SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def _previous_observations(
        self, conn: sqlite3.Connection, contracts: list[str], before: str
    ) -> dict[str, tuple]:
        """The latest stored observation at or before ``before``, per contract.

        Scoped to the contracts in the snapshot being written rather than the
        whole table, and chunked to stay inside SQLite's parameter limit.
        """
        latest: dict[str, tuple] = {}
        for start in range(0, len(contracts), _PARAM_CHUNK):
            chunk = contracts[start : start + _PARAM_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"""SELECT q.contract_symbol, q.bid, q.ask, q.last_trade_at
                    FROM quotes q
                    JOIN (
                      SELECT contract_symbol, MAX(observed_at) AS mx
                      FROM quotes
                      WHERE contract_symbol IN ({placeholders}) AND observed_at <= ?
                      GROUP BY contract_symbol
                    ) prev
                      ON prev.contract_symbol = q.contract_symbol
                     AND prev.mx = q.observed_at""",
                (*chunk, before),
            ).fetchall()
            for row in rows:
                latest[row["contract_symbol"]] = (
                    row["bid"],
                    row["ask"],
                    row["last_trade_at"],
                )
        return latest

    def write_snapshot(self, snapshot: ChainSnapshot) -> tuple[int, int]:
        """Persist a chain. Returns the snapshot id and how many quotes were new.

        A quote whose two sides and recency marker match the *previous stored
        observation for that contract* is not written again, so a second poll
        in a quiet hour adds a snapshot row and no quote rows. Anything else —
        including a return to a price last seen hours ago — is a new
        observation and is stored as one.
        """
        stamp = observed_at(snapshot)
        with closing(self.connect()) as conn:
            cursor = conn.execute(
                """INSERT INTO snapshots
                   (provider, symbol, expiry, captured_at, as_of, origin, underlying_price,
                    risk_free_rate, dividend_yield)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot.provider,
                    snapshot.symbol,
                    snapshot.expiry.isoformat(),
                    snapshot.captured_at.isoformat(),
                    snapshot.as_of.isoformat(),
                    snapshot.origin,
                    snapshot.underlying_price,
                    snapshot.risk_free_rate,
                    snapshot.dividend_yield,
                ),
            )
            snapshot_id = int(cursor.lastrowid)

            previous = self._previous_observations(
                conn, [q.contract_symbol for q in snapshot.quotes], stamp
            )

            written = 0
            for quote in snapshot.quotes:
                traded = quote.last_trade_at.isoformat() if quote.last_trade_at else None
                if previous.get(quote.contract_symbol) == (quote.bid, quote.ask, traded):
                    continue
                result = conn.execute(
                    """INSERT OR IGNORE INTO quotes
                       (snapshot_id, contract_symbol, option_type, strike, bid, ask,
                        last, volume, open_interest, provider_iv, last_trade_at,
                        observed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                        traded,
                        stamp,
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
        expiries_requested: int = 0,
        expiries_captured: int = 0,
        two_sided: int = 0,
    ) -> None:
        """Record one symbol's outcome in one capture run.

        An unproductive run is the point of this table. The live provider
        degrades by returning empty frames rather than raising, so without a
        record of zero-contract runs a field rename on the provider's side
        would leave every run reporting success while history quietly stopped.

        The expiry counts exist for the partial case, which is the same failure
        wearing a success's clothes: a symbol that returned one of the four
        expiries it asked for advanced history and is honestly productive, but
        recording only that would hide three quarters of a missing surface.
        """
        if reason is not None and reason not in FAILURE_REASONS:
            raise ValueError(
                f"reason {reason!r} is not one of the fixed codes; "
                "raw error text must never reach this table"
            )
        with closing(self.connect()) as conn:
            conn.execute(
                """INSERT INTO capture_runs
                   (started_at, provider, symbol, productive, reason, contracts,
                    expiries_requested, expiries_captured, two_sided)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    started_at.isoformat(),
                    provider,
                    symbol,
                    int(productive),
                    reason,
                    contracts,
                    expiries_requested,
                    expiries_captured,
                    two_sided,
                ),
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

        Selection is on ``observed_at``, which is the market time the quote
        describes. Ordering on row id instead would be a proxy for it that
        holds only while rows arrive in market order, and breaks the moment a
        backfill lands after the live history it belongs before.
        """
        cutoff = moment.isoformat()
        with closing(self.connect()) as conn:
            return list(
                conn.execute(
                    """SELECT q.*, s.captured_at, s.as_of, s.origin, s.underlying_price,
                              s.risk_free_rate, s.dividend_yield
                       FROM quotes q
                       JOIN snapshots s ON s.id = q.snapshot_id
                       JOIN (
                         SELECT q2.contract_symbol AS cs, MAX(q2.observed_at) AS mx
                         FROM quotes q2
                         JOIN snapshots s2 ON s2.id = q2.snapshot_id
                         WHERE s2.symbol = ? AND s2.expiry = ? AND q2.observed_at <= ?
                         GROUP BY q2.contract_symbol
                       ) latest
                         ON latest.cs = q.contract_symbol AND latest.mx = q.observed_at
                       WHERE s.symbol = ? AND s.expiry = ?
                       ORDER BY q.option_type, q.strike""",
                    (symbol, expiry.isoformat(), cutoff, symbol, expiry.isoformat()),
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

    # ------------------------------------------------------------------
    # Derived layer (KTD9)
    # ------------------------------------------------------------------

    def pending_for_derivation(self, engine_version: int) -> list[sqlite3.Row]:
        """Raw quotes not yet derived at ``engine_version``, with what inverting them needs.

        Every pending quote is returned, two-sided or not — the solver's own
        no-quote path is what records a one-sided quote's exclusion, so this
        does not duplicate that check. ``time_to_expiry`` is computed from
        ``as_of``, not ``captured_at``: for a live row they agree, but a
        backfilled row's ``captured_at`` is today, and expiry minus today
        would be nonsense for a date the row actually describes months ago.
        """
        with closing(self.connect()) as conn:
            return list(
                conn.execute(
                    """SELECT q.id AS quote_id, q.option_type, q.strike, q.bid, q.ask,
                              s.symbol, s.expiry, s.as_of, s.underlying_price,
                              s.risk_free_rate, s.dividend_yield,
                              CAST(julianday(s.expiry) - julianday(s.as_of) AS REAL) / 365.0
                                AS time_to_expiry
                       FROM quotes q
                       JOIN snapshots s ON s.id = q.snapshot_id
                       LEFT JOIN implied_vols iv
                         ON iv.quote_id = q.id AND iv.engine_version = ?
                       WHERE iv.id IS NULL""",
                    (engine_version,),
                )
            )

    def write_derived(
        self,
        *,
        quote_id: int,
        engine_version: int,
        status: str,
        implied_vol: float | None,
        detail: str,
        computed_at: datetime,
    ) -> None:
        """Persist one derived row. Replaces rather than duplicates on retry."""
        with closing(self.connect()) as conn:
            conn.execute(
                """INSERT INTO implied_vols
                   (quote_id, engine_version, status, implied_vol, detail, computed_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT (quote_id, engine_version) DO UPDATE SET
                     status=excluded.status, implied_vol=excluded.implied_vol,
                     detail=excluded.detail, computed_at=excluded.computed_at""",
                (quote_id, engine_version, status, implied_vol, detail, computed_at.isoformat()),
            )
            conn.commit()

    def clear_derived(self) -> int:
        """Wipe the derived table. The wholesale half of KTD9's rebuild.

        Raw quotes are untouched — this only ever removes rows this layer
        computed, never anything captured.
        """
        with closing(self.connect()) as conn:
            result = conn.execute("DELETE FROM implied_vols")
            conn.commit()
            return result.rowcount

    def derived_coverage(
        self, engine_version: int, *, symbol: str | None = None, expiry: date | None = None
    ) -> dict:
        """Counts by status at one engine version, for the evidence surface.

        Global by default; ``symbol``/``expiry`` narrow to one chain, for a
        per-chain breakdown without a second, parallel query living outside
        this class.
        """
        clauses = ["iv.engine_version = ?"]
        params: list = [engine_version]
        joined = ""
        if symbol is not None or expiry is not None:
            joined = "JOIN quotes q ON q.id = iv.quote_id JOIN snapshots s ON s.id = q.snapshot_id"
            if symbol is not None:
                clauses.append("s.symbol = ?")
                params.append(symbol)
            if expiry is not None:
                clauses.append("s.expiry = ?")
                params.append(expiry.isoformat())
        with closing(self.connect()) as conn:
            rows = conn.execute(
                f"""SELECT iv.status, COUNT(*) AS n FROM implied_vols iv
                    {joined}
                    WHERE {' AND '.join(clauses)}
                    GROUP BY iv.status""",
                params,
            )
            return {row["status"]: row["n"] for row in rows}

    def symbols_and_expiries(self) -> list[tuple[str, date]]:
        """Every distinct chain the archive has ever captured, for a surface index."""
        with closing(self.connect()) as conn:
            rows = conn.execute("SELECT DISTINCT symbol, expiry FROM snapshots ORDER BY symbol, expiry")
            return [(r["symbol"], date.fromisoformat(r["expiry"])) for r in rows]

    def derived_as_of(
        self, symbol: str, expiry: date, moment: datetime, engine_version: int
    ) -> list[sqlite3.Row]:
        """Solved implied volatilities as of ``moment`` — the analytics read.

        Reads only ``implied_vols`` joined back to the raw layer for display
        fields (strike, option type). Never calls the solver: an analytics
        view built on this touches stored numbers, not live computation, which
        is what keeps a page load proportional to rows returned rather than to
        the pricer calls a fresh inversion would cost.
        """
        cutoff = moment.isoformat()
        with closing(self.connect()) as conn:
            return list(
                conn.execute(
                    """SELECT q.option_type, q.strike, iv.implied_vol, iv.status,
                              q.observed_at
                       FROM implied_vols iv
                       JOIN quotes q ON q.id = iv.quote_id
                       JOIN snapshots s ON s.id = q.snapshot_id
                       JOIN (
                         SELECT q2.contract_symbol AS cs, MAX(q2.observed_at) AS mx
                         FROM quotes q2
                         JOIN snapshots s2 ON s2.id = q2.snapshot_id
                         WHERE s2.symbol = ? AND s2.expiry = ? AND q2.observed_at <= ?
                         GROUP BY q2.contract_symbol
                       ) latest ON latest.cs = q.contract_symbol AND latest.mx = q.observed_at
                       WHERE s.symbol = ? AND s.expiry = ? AND iv.engine_version = ?
                         AND iv.status = 'solved'
                       ORDER BY q.option_type, q.strike""",
                    (symbol, expiry.isoformat(), cutoff, symbol, expiry.isoformat(), engine_version),
                )
            )

    def raw_coverage(self, *, symbol: str | None = None, expiry: date | None = None) -> dict:
        """Quote and two-sided counts for the raw layer (R35's coverage artifact).

        Global by default; ``symbol``/``expiry`` narrow to one chain, mirroring
        ``derived_coverage``'s shape so the two artifacts read the same way.
        Two-sidedness uses the same ``bid > 0 AND ask > 0`` test as
        ``coverage()`` and ``derive.py``'s own mid-price — one definition of
        "two-sided," not three that could quietly drift apart.
        """
        clauses = []
        params: list = []
        joined = ""
        if symbol is not None or expiry is not None:
            joined = "JOIN snapshots s ON s.id = q.snapshot_id"
            if symbol is not None:
                clauses.append("s.symbol = ?")
                params.append(symbol)
            if expiry is not None:
                clauses.append("s.expiry = ?")
                params.append(expiry.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(self.connect()) as conn:
            row = conn.execute(
                f"""SELECT COUNT(*) AS quotes,
                           SUM(CASE WHEN bid > 0 AND ask > 0 THEN 1 ELSE 0 END) AS two_sided
                    FROM quotes q {joined} {where}""",
                params,
            ).fetchone()
            return {"quotes": row["quotes"] or 0, "two_sided": row["two_sided"] or 0}
