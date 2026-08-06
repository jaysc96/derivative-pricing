"""The scheduled entry point reports its conclusion as an exit status.

Nothing reads stdout on a host, so the integer is the whole interface between
this job and whatever runs it. A capture that fails while exiting 0 is
indistinguishable from one that worked, which is the same silent-success
failure U10 exists to prevent, arriving one layer up.

Every test here injects a fake adapter. None reaches the network — the live
provider is an unofficial scraper and a suite that depends on Yahoo being
reachable fails for reasons that have nothing to do with the code.
"""

import importlib.util
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from marketdata import ChainSnapshot, QuoteRecord, RateLimited, Store

REPO = Path(__file__).parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location(
        "capture_script", REPO / "scripts" / "capture.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture_script = load_script()

NOW = datetime(2026, 8, 6, 14, 0, tzinfo=timezone.utc)
NEAR = date(2026, 12, 18)


def quote(strike=600.0, *, bid=12.5, ask=12.9):
    return QuoteRecord(
        contract_symbol=f"X{strike:.0f}", option_type="call", strike=strike,
        bid=bid, ask=ask, last=12.7, volume=10, open_interest=20,
        provider_iv=0.18, last_trade_at=NOW - timedelta(hours=1),
    )


class FakeAdapter:
    name = "fake"

    def __init__(self, *, error=None, quotes=None):
        self._error = error
        self._quotes = quotes if quotes is not None else [quote()]

    def expiries(self, _symbol):
        return (NEAR,)

    def option_chain(self, symbol, expiry, as_of=None):
        if self._error:
            raise self._error
        return ChainSnapshot(
            provider="fake", symbol=symbol, expiry=expiry, captured_at=NOW,
            as_of=NOW.date(), origin="live", underlying_price=604.0,
            quotes=tuple(self._quotes),
        )

    def risk_free_rate(self, as_of=None):
        return 0.05

    def dividend_yield(self, symbol, as_of=None):
        return 0.02

    def underlying_history(self, symbol, start, end):
        return ()


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "capture.db")


def no_sleep(_seconds):
    return None


def run(store, argv=(), **kwargs):
    kwargs.setdefault("sleep", no_sleep)
    return capture_script.main(list(argv), store=store, **kwargs)


# --------------------------------------------------------------------------
# Exit status is the interface
# --------------------------------------------------------------------------


def test_a_productive_capture_exits_zero(store):
    assert run(store, ["--symbols", "SPY"], adapter=FakeAdapter()) == 0


def test_an_unproductive_capture_exits_non_zero(store):
    """Exiting 0 here would make a dead provider look like a working one."""
    code = run(store, ["--symbols", "SPY"], adapter=FakeAdapter(error=RateLimited("429")))
    assert code == capture_script.EXIT_UNPRODUCTIVE


def test_a_chain_with_no_bids_exits_non_zero(store):
    """Quotes arrived and none is usable — the surface did not advance."""
    dead = [quote(600.0, bid=0.0, ask=0.05), quote(610.0, bid=None, ask=0.02)]
    code = run(store, ["--symbols", "SPY"], adapter=FakeAdapter(quotes=dead))
    assert code == capture_script.EXIT_UNPRODUCTIVE


def test_a_fired_trigger_outranks_a_single_bad_run(store):
    """One failure is noise; a fired trigger is a standing instruction.

    A scheduler that sees only 1 cannot tell "retry tomorrow" from "this
    provider is finished", so the escalation gets its own status.
    """
    for day in range(3):
        capture_script.main(
            ["--symbols", "SPY"], store=store, sleep=no_sleep,
            adapter=FakeAdapter(error=RateLimited("429")),
        )
    code = run(store, ["--symbols", "SPY"], adapter=FakeAdapter(error=RateLimited("429")))
    assert code == capture_script.EXIT_TRIGGERED
    assert capture_script.EXIT_TRIGGERED > capture_script.EXIT_UNPRODUCTIVE


def test_an_empty_symbol_list_is_refused_rather_than_reported_as_success(store):
    assert run(store, ["--symbols", ""], adapter=FakeAdapter()) != 0


# --------------------------------------------------------------------------
# Status is readable without writing code
# --------------------------------------------------------------------------


def test_status_reports_the_archive_without_capturing(store, capsys):
    adapter = FakeAdapter()
    run(store, ["--symbols", "SPY"], adapter=adapter)
    before = store.coverage()["quotes"]

    code = run(store, ["--status"])
    output = capsys.readouterr().out

    assert code == 0
    assert store.coverage()["quotes"] == before, "--status must not write"
    assert "archive" in output and "fallback trigger" in output
    assert "TRIGGERED        False" in output


def test_status_on_an_empty_archive_does_not_fail(store, capsys):
    assert run(store, ["--status"]) == 0
    assert "0" in capsys.readouterr().out


def test_status_surfaces_a_fired_trigger(store, capsys):
    for _ in range(3):
        capture_script.main(
            ["--symbols", "SPY"], store=store, sleep=no_sleep,
            adapter=FakeAdapter(error=RateLimited("429")),
        )
    code = run(store, ["--status"])
    assert code == capture_script.EXIT_TRIGGERED
    assert "Build the fallback adapter" in capsys.readouterr().out


# --------------------------------------------------------------------------
# What reaches the log
# --------------------------------------------------------------------------


def test_only_fixed_reason_codes_are_printed(store, capsys):
    """A host's job log is not as private as it looks.

    The store refuses raw error text on the same grounds; a script that prints
    what the table refuses to hold would defeat the point of both.
    """
    crumb = "https://query2.finance.yahoo.com/v7/finance/options/SPY?crumb=SECRET"
    run(store, ["--symbols", "SPY"], adapter=FakeAdapter(error=RateLimited(crumb)))

    output = capsys.readouterr().out
    assert "crumb" not in output and "SECRET" not in output
    assert "https://" not in output
    assert "rate_limited" in output


def test_a_degraded_symbol_says_so(store, capsys):
    """Productive and short of what it asked for is not the same as healthy."""
    class Partial:
        name = "fake"

        def expiries(self, _symbol):
            return (NEAR, date(2027, 1, 15), date(2027, 2, 19))

        def option_chain(self, symbol, expiry, as_of=None):
            if expiry != NEAR:
                raise RateLimited("429")
            return ChainSnapshot(
                provider="fake", symbol=symbol, expiry=expiry, captured_at=NOW,
                as_of=NOW.date(), origin="live", underlying_price=604.0,
                quotes=(quote(),),
            )

        def risk_free_rate(self, as_of=None):
            return 0.05

        def dividend_yield(self, symbol, as_of=None):
            return 0.02

        def underlying_history(self, symbol, start, end):
            return ()

    code = run(store, ["--symbols", "SPY"], adapter=Partial())
    output = capsys.readouterr().out

    assert code == 0, "one good expiry is still history worth keeping"
    assert "degraded" in output and "1/3 expiries" in output


# --------------------------------------------------------------------------
# The tracked set
# --------------------------------------------------------------------------


def test_the_tracked_set_matches_the_set_the_spike_measures():
    """A tracked set that drifts from the validated set invalidates the spike."""
    spike = (REPO / "scripts" / "provider_spike.py").read_text()
    for symbol in capture_script.TRACKED:
        assert f'"{symbol}"' in spike, f"{symbol} is captured but never validated"


def test_symbols_are_normalised(store, capsys):
    run(store, ["--symbols", " spy , qqq "], adapter=FakeAdapter())
    output = capsys.readouterr().out
    assert "SPY" in output and "QQQ" in output


def test_expiry_day_contracts_are_skipped_unless_asked_for(store):
    """42% two-sided against 76-83% further out — a different population."""
    seen = []

    class Watching(FakeAdapter):
        def expiries(self, _symbol):
            return (NOW.date(), NEAR)

        def option_chain(self, symbol, expiry, as_of=None):
            seen.append(expiry)
            return super().option_chain(symbol, expiry, as_of)

    run(store, ["--symbols", "SPY"], adapter=Watching())
    assert seen == [NEAR]

    seen.clear()
    run(store, ["--symbols", "SPY", "--include-expiry-day"], adapter=Watching())
    assert NOW.date() in seen
