"""U12b: the derived layer inverts stored quotes, and only stored quotes.

Every test here goes through the real pipeline — write a snapshot to the
store, run ``derive_batch``/``rebuild``, read back through the store's own
read methods — rather than calling ``pricing.implied`` directly, because the
plan's own bar for this unit is that the layer is *derived from the archive*,
not merely a wrapper around the solver. The three scenarios the plan names
explicitly are marked below; the rest is this suite's own coverage.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from marketdata import ChainSnapshot, QuoteRecord, Store
from marketdata.derive import NO_MARKET_CONTEXT, derive_batch, needs_rebuild, rebuild
from pricing import ENGINE_VERSION
from pricing.american import American_Option
from pricing.implied import NO_QUOTE

EXPIRY = date(2026, 12, 18)
T0 = datetime(2026, 8, 6, 14, 0, tzinfo=timezone.utc)
K, R, Y = 100.0, 0.05, 0.02
#: Matches T0 -> EXPIRY exactly, so a price generated at this T and re-derived
#: through the store's own julianday computation round-trips precisely rather
#: than approximately.
T_YEARS = (EXPIRY - T0.date()).days / 365.0


def price_american(kind: str, sig: float, *, S: float = 100.0, K: float = K) -> float:
    opt = American_Option(kind, S, K, R, sig, Y, T_YEARS, "BT")
    opt.setTreeSteps(200)
    return float(opt.BT())


def quote(*, contract="TESTC00100000", kind="call", strike=K, bid, ask):
    mid = (bid + ask) / 2 if bid and ask else None
    return QuoteRecord(
        contract_symbol=contract, option_type=kind, strike=strike,
        bid=bid, ask=ask, last=mid, volume=10, open_interest=20,
        provider_iv=0.2, last_trade_at=T0,
    )


def snapshot(quotes, *, rate=R, dividend_yield=Y, underlying_price=100.0, captured_at=T0):
    return ChainSnapshot(
        provider="fake", symbol="TEST", expiry=EXPIRY, captured_at=captured_at,
        as_of=captured_at.date(), origin="live", underlying_price=underlying_price,
        quotes=tuple(quotes), risk_free_rate=rate, dividend_yield=dividend_yield,
    )


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "derive.db")


# --------------------------------------------------------------------------
# Round trip through the full pipeline
# --------------------------------------------------------------------------


def test_a_solved_quote_recovers_its_input_volatility(store):
    true_sig = 0.30
    price = price_american("call", true_sig)
    store.write_snapshot(snapshot([quote(bid=price - 0.01, ask=price + 0.01)]))

    written = derive_batch(store)
    assert written == 1

    rows = store.derived_as_of("TEST", EXPIRY, T0 + timedelta(minutes=1), ENGINE_VERSION)
    assert len(rows) == 1
    assert rows[0]["implied_vol"] == pytest.approx(true_sig, abs=5e-3)


def test_derive_batch_is_incremental(store):
    """A second run with no new quotes has nothing pending."""
    store.write_snapshot(snapshot([quote(bid=9.0, ask=9.2)]))
    first = derive_batch(store)
    second = derive_batch(store)
    assert first == 1
    assert second == 0


def test_derive_batch_picks_up_only_what_a_later_capture_actually_added(store):
    """Three captures, three separate `derive_batch` calls: each processes
    only the quote its own capture wrote, never re-touching earlier ones —
    the high-water-mark claim, proven across more than one increment."""
    store.write_snapshot(snapshot([quote(contract="TESTC00100000", bid=9.0, ask=9.2)]))
    assert derive_batch(store) == 1

    store.write_snapshot(
        snapshot([quote(contract="TESTC00110000", strike=110.0, bid=4.0, ask=4.2)],
                 captured_at=T0 + timedelta(hours=1))
    )
    assert derive_batch(store) == 1

    store.write_snapshot(
        snapshot([quote(contract="TESTC00120000", strike=120.0, bid=1.5, ask=1.7)],
                 captured_at=T0 + timedelta(hours=2))
    )
    assert derive_batch(store) == 1
    assert derive_batch(store) == 0

    rows = store.derived_as_of("TEST", EXPIRY, T0 + timedelta(hours=3), ENGINE_VERSION)
    assert len(rows) == 3


# --------------------------------------------------------------------------
# R34 / plan scenario: a missing side is excluded, and recorded with its reason
# --------------------------------------------------------------------------


def test_a_zero_or_absent_bid_is_excluded_and_recorded(store):
    store.write_snapshot(
        snapshot([
            quote(contract="TESTC00090000", strike=90.0, bid=0.0, ask=11.5),
            quote(contract="TESTC00110000", strike=110.0, bid=None, ask=2.1),
        ])
    )

    derive_batch(store)

    with store.connect() as conn:
        rows = conn.execute("SELECT status, detail FROM implied_vols").fetchall()
    assert len(rows) == 2, "excluded, not dropped — a row exists for each"
    assert {r["status"] for r in rows} == {NO_QUOTE}
    assert all(r["detail"] for r in rows)
    # And the exclusion is real: an analytics read at the current version
    # returns nothing for a chain that never solved.
    assert store.derived_as_of("TEST", EXPIRY, T0 + timedelta(minutes=1), ENGINE_VERSION) == []


# --------------------------------------------------------------------------
# The derived layer's own gap: missing market context
# --------------------------------------------------------------------------


def test_a_snapshot_with_no_rate_is_not_sent_to_the_solver(store):
    """A rate-fetch hiccup at capture time should not look like a solver failure."""
    store.write_snapshot(snapshot([quote(bid=9.0, ask=9.2)], rate=None))
    derive_batch(store)

    with store.connect() as conn:
        row = conn.execute("SELECT status, implied_vol FROM implied_vols").fetchone()
    assert row["status"] == NO_MARKET_CONTEXT
    assert row["implied_vol"] is None


def test_a_snapshot_with_no_dividend_yield_is_also_withheld(store):
    store.write_snapshot(snapshot([quote(bid=9.0, ask=9.2)], dividend_yield=None))
    derive_batch(store)

    with store.connect() as conn:
        status = conn.execute("SELECT status FROM implied_vols").fetchone()["status"]
    assert status == NO_MARKET_CONTEXT


# --------------------------------------------------------------------------
# Plan scenario: changing the engine version rebuilds from the raw quotes
# --------------------------------------------------------------------------


def test_changing_the_engine_version_rebuilds_from_the_same_raw_quotes(store):
    price = price_american("call", 0.30)
    store.write_snapshot(snapshot([quote(bid=price - 0.01, ask=price + 0.01)]))

    n1 = rebuild(store, engine_version=1)
    n2 = rebuild(store, engine_version=2)
    assert n1 == n2 == 1, "the same one raw quote, recomputed under each version"

    with store.connect() as conn:
        versions = {r["engine_version"] for r in conn.execute("SELECT DISTINCT engine_version FROM implied_vols")}
    assert versions == {2}, "the rebuild replaced version 1's rows rather than accumulating alongside them"


def test_rebuild_actually_recomputes_rather_than_reusing_a_stale_row(store):
    """Change what the raw quote implies between two rebuilds; the second must reflect it.

    Proves the table is genuinely re-derived — not just re-stamped with a new
    version number over an old, cached implied volatility.
    """
    low = price_american("call", 0.15)
    store.write_snapshot(snapshot([quote(bid=low - 0.01, ask=low + 0.01)]))
    rebuild(store, engine_version=1)
    with store.connect() as conn:
        first = conn.execute("SELECT implied_vol FROM implied_vols").fetchone()["implied_vol"]

    # A distinct observed_at (same date, later moment), or this write dedups
    # against the first — a moved price minutes later is new information, not
    # a second write of the same one. Same contract: this is one contract's
    # price moving, not two independent ones.
    high = price_american("call", 0.45)
    store.write_snapshot(
        snapshot([quote(bid=high - 0.01, ask=high + 0.01)], captured_at=T0 + timedelta(hours=1))
    )
    rebuild(store, engine_version=2)
    with store.connect() as conn:
        rows = conn.execute("SELECT implied_vol FROM implied_vols ORDER BY implied_vol").fetchall()

    assert first == pytest.approx(0.15, abs=5e-3)
    assert [r["implied_vol"] for r in rows] == pytest.approx([0.15, 0.45], abs=5e-3)


def test_rebuild_derives_the_new_version_before_removing_the_old_one(store):
    """An interruption between deriving and cleanup must leave the *old*
    version intact and queryable, not the table empty at every version.

    Calls the two halves `rebuild` composes directly, stopping short of the
    cleanup step, to prove the ordering rather than trust the source read —
    a reordering regression here would only ever show up as a production
    outage during an engine-version bump, never in the happy-path test above.
    """
    price = price_american("call", 0.30)
    store.write_snapshot(snapshot([quote(bid=price - 0.01, ask=price + 0.01)]))
    rebuild(store, engine_version=1)

    derive_batch(store, engine_version=2)  # the half rebuild() runs first
    # Simulated crash right here, before clear_stale_derived ever runs.

    with store.connect() as conn:
        versions = {r["engine_version"] for r in conn.execute("SELECT DISTINCT engine_version FROM implied_vols")}
    assert versions == {1, 2}, "old version must still be present mid-rebuild, not wiped upfront"

    rows = store.derived_as_of("TEST", EXPIRY, T0 + timedelta(minutes=1), 1)
    assert len(rows) == 1, "version 1 must still be readable during the window before cleanup"


def test_needs_rebuild_is_false_on_a_fresh_archive(store):
    """Nothing derived yet is the ordinary starting state, not a stale one."""
    assert needs_rebuild(store) is False


def test_needs_rebuild_is_false_once_derived_at_the_current_version(store):
    store.write_snapshot(snapshot([quote(bid=9.0, ask=9.2)]))
    derive_batch(store, engine_version=ENGINE_VERSION)
    assert needs_rebuild(store, engine_version=ENGINE_VERSION) is False


def test_needs_rebuild_is_true_when_only_a_stale_version_is_present(store):
    store.write_snapshot(snapshot([quote(bid=9.0, ask=9.2)]))
    derive_batch(store, engine_version=1)
    assert needs_rebuild(store, engine_version=2) is True


# --------------------------------------------------------------------------
# Plan scenario: an analytics read touches only the derived table
# --------------------------------------------------------------------------


def test_derived_coverage_counts_by_status(store):
    store.write_snapshot(
        snapshot([
            quote(bid=9.0, ask=9.2),
            quote(contract="TESTC00090000", strike=90.0, bid=None, ask=11.5),
        ])
    )
    derive_batch(store)

    coverage = store.derived_coverage(ENGINE_VERSION)
    assert coverage.get("solved", 0) + coverage.get(NO_QUOTE, 0) == 2


def test_derived_coverage_narrows_to_one_chain(store):
    other_expiry = EXPIRY + timedelta(days=30)
    store.write_snapshot(snapshot([quote(bid=9.0, ask=9.2)]))
    store.write_snapshot(
        ChainSnapshot(
            provider="fake", symbol="TEST", expiry=other_expiry, captured_at=T0,
            as_of=T0.date(), origin="live", underlying_price=100.0,
            quotes=(quote(contract="TESTC00100000C2", bid=None, ask=5.0),),
            risk_free_rate=R, dividend_yield=Y,
        )
    )
    derive_batch(store)

    narrowed = store.derived_coverage(ENGINE_VERSION, symbol="TEST", expiry=EXPIRY)
    everything = store.derived_coverage(ENGINE_VERSION)
    assert sum(narrowed.values()) == 1
    assert sum(everything.values()) == 2


def test_symbols_and_expiries_lists_every_captured_chain(store):
    store.write_snapshot(snapshot([quote(bid=9.0, ask=9.2)]))
    assert store.symbols_and_expiries() == [("TEST", EXPIRY)]


def test_an_analytics_read_never_calls_the_solver(store, monkeypatch):
    price = price_american("call", 0.30)
    store.write_snapshot(snapshot([quote(bid=price - 0.01, ask=price + 0.01)]))
    derive_batch(store)  # populates the row with the real solver, unpatched

    def boom(*_args, **_kwargs):
        raise AssertionError("an analytics read must not call the solver")

    monkeypatch.setattr("pricing.implied.implied_volatility", boom)

    rows = store.derived_as_of("TEST", EXPIRY, T0 + timedelta(minutes=1), ENGINE_VERSION)
    assert len(rows) == 1
    assert rows[0]["implied_vol"] == pytest.approx(0.30, abs=5e-3)
