"""U10: the capture run survives rate limits, and never reports silent failure as success."""

from datetime import date, datetime, timedelta, timezone

import pytest

from marketdata import ChainSnapshot, NotSupported, QuoteRecord, RateLimited, Store
from marketdata.adapter import MalformedResponse, ProviderUnavailable
from marketdata.capture import capture_symbol, fallback_trigger_state, run_capture

NOW = datetime(2026, 8, 5, 14, 0, tzinfo=timezone.utc)
TODAY_EXPIRY = date(2026, 8, 5)

# Far enough ahead to stay in the future across every simulated run day below.
# A nearer date silently turns the multi-day trigger tests into expiry-skip
# tests — the capture correctly reports empty_response once the expiry passes,
# which is right behaviour and the wrong thing to be measuring there.
NEAR = date(2026, 12, 18)


def quote(strike=600.0, *, bid=12.5, ask=12.9, kind="call"):
    return QuoteRecord(
        contract_symbol=f"SPY{kind[0].upper()}{strike:.0f}",
        option_type=kind,
        strike=strike,
        bid=bid,
        ask=ask,
        last=12.7,
        volume=10,
        open_interest=20,
        provider_iv=0.18,
        last_trade_at=NOW - timedelta(hours=1),
    )


def chain(symbol, expiry, quotes):
    return ChainSnapshot(
        provider="fake", symbol=symbol, expiry=expiry, captured_at=NOW,
        as_of=NOW.date(), origin="live", underlying_price=604.0, quotes=tuple(quotes),
    )


class FakeAdapter:
    """Scriptable provider. Each call pops the next behaviour for that method."""

    name = "fake"

    def __init__(self, *, expiries=(NEAR,), chain_script=None, expiry_error=None):
        self._expiries = expiries
        self._expiry_error = expiry_error
        self._chain_script = list(chain_script or [])
        self.chain_calls = 0

    def expiries(self, _symbol):
        if self._expiry_error:
            raise self._expiry_error
        return tuple(self._expiries)

    def option_chain(self, symbol, expiry, as_of=None):
        self.chain_calls += 1
        if not self._chain_script:
            return chain(symbol, expiry, [quote()])
        step = self._chain_script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step if step is not None else chain(symbol, expiry, [quote()])


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "capture.db")


def no_sleep(_seconds):
    return None


# --------------------------------------------------------------------------
# Retry and backoff
# --------------------------------------------------------------------------


def test_rate_limit_triggers_backoff_then_succeeds(store):
    adapter = FakeAdapter(chain_script=[RateLimited("429"), None])
    delays = []

    outcome = capture_symbol(
        adapter, store, "SPY", sleep=delays.append, today=NOW.date()
    )

    assert outcome.productive
    assert adapter.chain_calls == 2
    assert delays == [2.0], "one backoff between the two attempts"


def test_backoff_grows_between_attempts(store):
    adapter = FakeAdapter(
        chain_script=[RateLimited("429"), RateLimited("429"), RateLimited("429")]
    )
    delays = []

    outcome = capture_symbol(
        adapter, store, "SPY", sleep=delays.append, today=NOW.date()
    )

    assert not outcome.productive
    assert outcome.reason == "rate_limited"
    assert delays == [2.0, 4.0], "exponential, and no sleep after the final attempt"


def test_exhausted_retries_leave_prior_history_intact(store):
    good = FakeAdapter()
    capture_symbol(good, store, "SPY", sleep=no_sleep, today=NOW.date())
    before = store.coverage()["quotes"]

    failing = FakeAdapter(chain_script=[RateLimited("429")] * 3)
    outcome = capture_symbol(failing, store, "SPY", sleep=no_sleep, today=NOW.date())

    assert not outcome.productive
    assert store.coverage()["quotes"] == before


def test_malformed_response_is_not_retried(store):
    """Asking a broken parser twice produces the same break."""
    adapter = FakeAdapter(chain_script=[MalformedResponse("bid column vanished")])
    outcome = capture_symbol(adapter, store, "SPY", sleep=no_sleep, today=NOW.date())

    assert outcome.reason == "malformed"
    assert adapter.chain_calls == 1


def test_timeout_and_network_failures_get_distinct_codes(store):
    timed_out = FakeAdapter(chain_script=[ProviderUnavailable("provider timed out")] * 3)
    assert capture_symbol(timed_out, store, "SPY", sleep=no_sleep, today=NOW.date()).reason == "timeout"

    unreachable = FakeAdapter(chain_script=[ProviderUnavailable("network failure")] * 3)
    assert capture_symbol(unreachable, store, "SPY", sleep=no_sleep, today=NOW.date()).reason == "network_error"


# --------------------------------------------------------------------------
# Silent failure is the failure mode that matters
# --------------------------------------------------------------------------


def test_zero_contracts_is_recorded_as_empty_response(store):
    adapter = FakeAdapter(chain_script=[chain("SPY", NEAR, [])])
    outcome = capture_symbol(adapter, store, "SPY", sleep=no_sleep, today=NOW.date())

    assert not outcome.productive
    assert outcome.reason == "empty_response"


def test_an_all_zero_sided_chain_is_unproductive_not_a_valid_snapshot(store):
    """Quotes arrived; none is usable. The surface did not advance."""
    dead = [quote(600.0, bid=0.0, ask=0.05), quote(610.0, bid=None, ask=0.02)]
    adapter = FakeAdapter(chain_script=[chain("SPY", NEAR, dead)])

    outcome = capture_symbol(adapter, store, "SPY", sleep=no_sleep, today=NOW.date())

    assert not outcome.productive
    assert outcome.reason == "all_zero_sided"
    assert outcome.contracts == 2
    assert store.coverage()["quotes"] == 2, "raw layer still keeps what it saw"


def test_an_empty_expiry_list_is_recorded(store):
    adapter = FakeAdapter(expiries=())
    outcome = capture_symbol(adapter, store, "SPY", sleep=no_sleep, today=NOW.date())
    assert outcome.reason == "empty_response"


# --------------------------------------------------------------------------
# Partial success is not success
# --------------------------------------------------------------------------

FOUR = (NEAR, date(2027, 1, 15), date(2027, 2, 19), date(2027, 3, 19))


def test_a_symbol_that_lost_three_of_four_expiries_is_marked_degraded(store):
    """It advanced history, so it is productive — and it is not healthy.

    Recording only `productive` cannot tell a full chain from a quarter of
    one, which is the same silent degradation the empty-response check exists
    to catch, arriving one level down.
    """
    class MostlyBlocked:
        name = "fake"

        def expiries(self, _symbol):
            return FOUR

        def option_chain(self, symbol, expiry, as_of=None):
            if expiry != FOUR[0]:
                raise RateLimited("429")
            return chain(symbol, expiry, [quote()])

    outcome = capture_symbol(MostlyBlocked(), store, "SPY", sleep=no_sleep, today=NOW.date())

    assert outcome.productive, "one good expiry is still history worth keeping"
    assert outcome.degraded
    assert (outcome.expiries, outcome.expiries_requested) == (1, 4)


def test_a_complete_capture_is_not_degraded(store):
    outcome = capture_symbol(FakeAdapter(), store, "SPY", sleep=no_sleep, today=NOW.date())
    assert outcome.productive and not outcome.degraded


def test_the_expiry_shortfall_reaches_the_record(store):
    """An outcome nobody persists cannot inform the fallback decision."""
    class MostlyBlocked:
        name = "fake"

        def expiries(self, _symbol):
            return FOUR

        def option_chain(self, symbol, expiry, as_of=None):
            if expiry != FOUR[0]:
                raise RateLimited("429")
            return chain(symbol, expiry, [quote()])

    run_capture(MostlyBlocked(), store, ("SPY",), sleep=no_sleep, now=NOW)

    row = store.run_record()[0]
    assert row["productive"] == 1 and row["reason"] is None
    assert row["expiries_requested"] == 4 and row["expiries_captured"] == 1
    assert row["two_sided"] == 1
    assert fallback_trigger_state(store)["degraded_in_window"] == 1


def test_a_capability_the_provider_lacks_is_not_filed_as_an_http_error(store):
    """U11 publishes these codes, so they have to mean what they say."""
    adapter = FakeAdapter(chain_script=[NotSupported("no dated chains")])
    outcome = capture_symbol(adapter, store, "SPY", sleep=no_sleep, today=NOW.date())

    assert outcome.reason == "not_supported"
    assert adapter.chain_calls == 1, "a missing capability does not improve on retry"


# --------------------------------------------------------------------------
# Partial failure keeps what worked
# --------------------------------------------------------------------------


def test_one_blocked_symbol_does_not_discard_the_others(store):
    class PerSymbol:
        name = "fake"

        def expiries(self, symbol):
            if symbol == "QQQ":
                raise RateLimited("429")
            return (NEAR,)

        def option_chain(self, symbol, expiry, as_of=None):
            return chain(symbol, expiry, [quote()])

    result = run_capture(
        PerSymbol(), store, ("SPY", "QQQ", "IWM"), sleep=no_sleep, now=NOW
    )

    by_symbol = {o.symbol: o for o in result.outcomes}
    assert by_symbol["SPY"].productive and by_symbol["IWM"].productive
    assert not by_symbol["QQQ"].productive
    assert by_symbol["QQQ"].reason == "rate_limited"
    assert result.productive, "the run advanced history even though one symbol failed"
    assert store.coverage()["snapshots"] == 2


def test_every_symbol_outcome_reaches_the_record(store):
    class PerSymbol:
        name = "fake"

        def expiries(self, symbol):
            return (NEAR,)

        def option_chain(self, symbol, expiry, as_of=None):
            if symbol == "IWM":
                raise RateLimited("429")
            return chain(symbol, expiry, [quote()])

    run_capture(PerSymbol(), store, ("SPY", "IWM"), sleep=no_sleep, now=NOW)

    record = {row["symbol"]: row for row in store.run_record()}
    assert record["SPY"]["productive"] == 1 and record["SPY"]["reason"] is None
    assert record["IWM"]["productive"] == 0 and record["IWM"]["reason"] == "rate_limited"


# --------------------------------------------------------------------------
# Expiry-day handling and deduplication
# --------------------------------------------------------------------------


def test_expiry_day_contracts_are_skipped_by_default(store):
    """42% two-sided against 76-83% further out — a different population."""
    adapter = FakeAdapter(expiries=(TODAY_EXPIRY, NEAR))
    capture_symbol(adapter, store, "SPY", sleep=no_sleep, today=NOW.date())

    assert adapter.chain_calls == 1, "only the later expiry was pulled"


def test_expiry_day_can_be_kept_when_asked(store):
    adapter = FakeAdapter(expiries=(TODAY_EXPIRY, NEAR))
    capture_symbol(
        adapter, store, "SPY", skip_expiry_day=False, sleep=no_sleep, today=NOW.date()
    )
    assert adapter.chain_calls == 2


def test_two_runs_in_a_day_do_not_duplicate_an_unchanged_quote(store):
    for _ in range(2):
        capture_symbol(FakeAdapter(), store, "SPY", sleep=no_sleep, today=NOW.date())

    assert store.coverage()["snapshots"] == 2
    assert store.coverage()["quotes"] == 1


# --------------------------------------------------------------------------
# The fallback trigger, evaluated against the record
# --------------------------------------------------------------------------


def test_trigger_is_quiet_while_runs_are_productive(store):
    for day in range(4):
        run_capture(
            FakeAdapter(), store, ("SPY",), sleep=no_sleep, now=NOW + timedelta(days=day)
        )
    assert fallback_trigger_state(store)["triggered"] is False


def test_three_consecutive_unproductive_runs_fire_the_trigger(store):
    run_capture(FakeAdapter(), store, ("SPY",), sleep=no_sleep, now=NOW)
    for day in range(1, 4):
        run_capture(
            FakeAdapter(chain_script=[RateLimited("429")] * 3),
            store, ("SPY",), sleep=no_sleep, now=NOW + timedelta(days=day),
        )

    state = fallback_trigger_state(store)
    assert state["consecutive_unproductive"] == 3
    assert state["triggered"] is True


def test_a_majority_of_unproductive_runs_fires_the_trigger(store):
    """Alternating failure never reaches three in a row but is still failing."""
    for day in range(6):
        script = [RateLimited("429")] * 3 if day % 2 == 0 else None
        run_capture(
            FakeAdapter(chain_script=script),
            store, ("SPY",), sleep=no_sleep, now=NOW + timedelta(days=day),
        )

    state = fallback_trigger_state(store)
    assert state["consecutive_unproductive"] < 3
    assert state["unproductive_in_window"] == 3
    assert state["triggered"] is False, "three of six is not a majority"

    run_capture(
        FakeAdapter(chain_script=[RateLimited("429")] * 3),
        store, ("SPY",), sleep=no_sleep, now=NOW + timedelta(days=6),
    )
    assert fallback_trigger_state(store)["triggered"] is True


def test_the_window_is_seven_days_rather_than_seven_runs(store):
    """At any cadence but one-a-day those differ, and the window silently shrinks.

    Six captures a day over two days is twelve runs. Counting the last *seven*
    of them looks back barely a day, so a rough morning outvotes a healthy
    yesterday and the trigger fires on a provider that is mostly working.
    Four unproductive of twelve is not a majority; four of seven is.
    """
    # Yesterday clean, today rough — but never three in a row, so this test
    # measures the window rule rather than the consecutive one.
    pattern = {0: [True] * 6, 1: [False, True, False, True, False, False]}
    for day, slots in pattern.items():
        for slot, healthy in enumerate(slots):
            run_capture(
                FakeAdapter(chain_script=None if healthy else [RateLimited("429")] * 3),
                store, ("SPY",), sleep=no_sleep, now=NOW + timedelta(days=day, hours=slot),
            )

    state = fallback_trigger_state(store)
    assert state["consecutive_unproductive"] < 3, "precondition: the other rule is quiet"
    assert state["window_size"] == 12, "a seven-day window covers both days of runs"
    assert state["unproductive_in_window"] == 4
    assert state["triggered"] is False, "four of twelve is not a majority"


def test_an_empty_record_reports_a_real_boolean(store):
    """`triggered` is read by callers; an empty list is not False."""
    state = fallback_trigger_state(store)
    assert state["triggered"] is False
    assert state["runs_recorded"] == 0
