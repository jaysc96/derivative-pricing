"""The unattended capture run.

Walks the tracked set, pulls each symbol's near expiries, writes raw quotes
through the store, and records what happened — including, especially, when
nothing happened.

**A run that returns nothing is a failure, not a success.** This is the failure
mode that matters most here. The provider is an unofficial scraper that
degrades by returning empty frames rather than raising, so a field rename on
Yahoo's side would leave every scheduled run reporting success while history
quietly stopped accruing, and the fallback trigger would never fire. Zero
contracts is recorded as ``empty_response``; a chain where nothing has a bid is
recorded as ``all_zero_sided``. Both count against the trigger.

Reasons are drawn from a fixed set of codes, never raw exception text and never
a request URL — the store enforces that, and U11 commits artifacts derived from
the run record into a public repository.

Partial failure keeps what worked. One blocked symbol does not discard the
other five; the batch is not a transaction, because a day of history for five
symbols is worth more than consistency across six. **Partial success is
recorded as partial**, though: a symbol that returned one of the four expiries
it asked for is genuinely productive and genuinely degraded, and the run record
carries both counts so the second half is not invisible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from .adapter import (
    MalformedResponse,
    NotSupported,
    ProviderUnavailable,
    RateLimited,
)
from .store import Store

DEFAULT_EXPIRIES_PER_SYMBOL = 4
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 2.0

#: Runs a window needs before the majority rule may fire. Without a floor,
#: "more than half the window" is satisfied by the very first unproductive run
#: on a fresh archive — one bad afternoon reading as a standing instruction to
#: go build a second provider. Matched to the consecutive rule's three, so
#: neither path escalates on less evidence than the other.
MIN_RUNS_FOR_MAJORITY = 3


@dataclass
class SymbolOutcome:
    """What happened for one symbol in one run."""

    symbol: str
    productive: bool
    contracts: int = 0
    two_sided: int = 0
    reason: str | None = None
    expiries: int = 0
    expiries_requested: int = 0

    @property
    def degraded(self) -> bool:
        """Productive, but short of what it set out to collect.

        Distinct from unproductive. A degraded run still advanced history, so
        suppressing it would be wrong; treating it as an unqualified success
        would hide a provider that has started refusing most of what we ask
        for while still answering enough to look alive.
        """
        return self.productive and self.expiries < self.expiries_requested


@dataclass
class CaptureResult:
    """What happened across the whole run."""

    started_at: datetime
    provider: str
    outcomes: list[SymbolOutcome] = field(default_factory=list)

    @property
    def productive(self) -> bool:
        """A run is productive when any symbol produced usable quotes."""
        return any(outcome.productive for outcome in self.outcomes)

    @property
    def contracts(self) -> int:
        return sum(outcome.contracts for outcome in self.outcomes)

    @property
    def two_sided(self) -> int:
        return sum(outcome.two_sided for outcome in self.outcomes)

    @property
    def degraded(self) -> bool:
        """Any symbol came back short of the expiries it asked for."""
        return any(outcome.degraded for outcome in self.outcomes)


def _with_backoff(call, *, max_attempts: int, backoff: float, sleep=time.sleep):
    """Retry on rate limits and transient failures; give up with a reason code.

    Returns ``(value, None)`` or ``(None, reason_code)``. Malformed responses
    are not retried — asking a broken parser twice produces the same break, and
    the code is the signal that the provider's shape changed.
    """
    delay = backoff
    last_reason = "http_error"
    for attempt in range(max_attempts):
        try:
            return call(), None
        except RateLimited:
            last_reason = "rate_limited"
        except ProviderUnavailable as exc:
            last_reason = "timeout" if "timed out" in str(exc) else "network_error"
        except MalformedResponse:
            return None, "malformed"
        except NotSupported:
            # Not a failure of the network — a capability this provider does
            # not have. U11 publishes these codes, so it should not be filed
            # under the one that means the provider answered with an error.
            return None, "not_supported"

        if attempt < max_attempts - 1:
            sleep(delay)
            delay *= 2
    return None, last_reason


def capture_symbol(
    adapter,
    store: Store,
    symbol: str,
    *,
    expiries_per_symbol: int = DEFAULT_EXPIRIES_PER_SYMBOL,
    skip_expiry_day: bool = True,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF_SECONDS,
    sleep=time.sleep,
    today: date | None = None,
) -> SymbolOutcome:
    """Capture one symbol's near expiries. Never raises; records instead."""
    expiries, reason = _with_backoff(
        lambda: adapter.expiries(symbol),
        max_attempts=max_attempts,
        backoff=backoff,
        sleep=sleep,
    )
    if reason is not None:
        return SymbolOutcome(symbol=symbol, productive=False, reason=reason)
    if not expiries:
        return SymbolOutcome(symbol=symbol, productive=False, reason="empty_response")

    # Expiry-day contracts are a different population — the spike measured 42%
    # two-sided against 76-83% further out, concentrated in puts nobody bids
    # for hours before they expire. Skipping them raises the usable fraction
    # and drops the quotes whose implied volatilities would be least reliable.
    reference = today or datetime.now(timezone.utc).date()
    if skip_expiry_day:
        expiries = tuple(e for e in expiries if e > reference)

    wanted = expiries[:expiries_per_symbol]
    contracts = two_sided = captured_expiries = 0
    last_reason: str | None = None

    for expiry in wanted:
        snapshot, reason = _with_backoff(
            lambda e=expiry: adapter.option_chain(symbol, e),
            max_attempts=max_attempts,
            backoff=backoff,
            sleep=sleep,
        )
        if reason is not None:
            last_reason = reason
            continue

        store.write_snapshot(snapshot)
        captured_expiries += 1
        contracts += len(snapshot.quotes)
        two_sided += sum(1 for quote in snapshot.quotes if quote.two_sided)

    if contracts == 0:
        return SymbolOutcome(
            symbol=symbol,
            productive=False,
            reason=last_reason or "empty_response",
            expiries=captured_expiries,
            expiries_requested=len(wanted),
        )
    if two_sided == 0:
        # Quotes arrived and none of them is usable. Stored, because the raw
        # layer keeps what it saw, but the run did not advance the surface.
        return SymbolOutcome(
            symbol=symbol,
            productive=False,
            contracts=contracts,
            two_sided=0,
            reason="all_zero_sided",
            expiries=captured_expiries,
            expiries_requested=len(wanted),
        )
    return SymbolOutcome(
        symbol=symbol,
        productive=True,
        contracts=contracts,
        two_sided=two_sided,
        expiries=captured_expiries,
        expiries_requested=len(wanted),
    )


def run_capture(
    adapter,
    store: Store,
    symbols: tuple[str, ...],
    *,
    expiries_per_symbol: int = DEFAULT_EXPIRIES_PER_SYMBOL,
    skip_expiry_day: bool = True,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF_SECONDS,
    sleep=time.sleep,
    now: datetime | None = None,
) -> CaptureResult:
    """Capture the tracked set once, recording every symbol's outcome."""
    started_at = now or datetime.now(timezone.utc)
    result = CaptureResult(started_at=started_at, provider=adapter.name)

    for symbol in symbols:
        outcome = capture_symbol(
            adapter,
            store,
            symbol,
            expiries_per_symbol=expiries_per_symbol,
            skip_expiry_day=skip_expiry_day,
            max_attempts=max_attempts,
            backoff=backoff,
            sleep=sleep,
            today=started_at.date(),
        )
        result.outcomes.append(outcome)
        store.record_run(
            started_at=started_at,
            provider=adapter.name,
            symbol=symbol,
            productive=outcome.productive,
            reason=outcome.reason,
            contracts=outcome.contracts,
            expiries_requested=outcome.expiries_requested,
            expiries_captured=outcome.expiries,
            two_sided=outcome.two_sided,
        )

    return result


def fallback_trigger_state(store: Store, *, window_days: int = 7) -> dict:
    """Evaluate the fallback trigger against the record rather than memory.

    The trigger fires on three consecutive unproductive runs, or on more than
    half of the runs in a seven-day window. Returning the counts rather than a
    boolean keeps the decision legible when it is taken.

    The window is measured in days, back from the most recent run — not in
    runs. Those coincide at one capture a day and diverge immediately at any
    other cadence, which would quietly shorten the window exactly when the
    schedule got busier. ``window_days <= 0`` means every run on record.

    Degraded runs are counted and reported but do not fire the trigger on
    their own. Deciding how much shortfall is worth building a second provider
    for is a judgement about a provider we have not yet watched degrade.

    The majority rule needs ``MIN_RUNS_FOR_MAJORITY`` runs on record before it
    can fire, because a majority of one is one.
    """
    record = store.run_record()
    by_run: dict[str, list] = {}
    for row in record:
        by_run.setdefault(row["started_at"], []).append(row)

    runs = sorted(by_run.items(), reverse=True)
    consecutive = 0
    for _stamp, rows in runs:
        if any(row["productive"] for row in rows):
            break
        consecutive += 1

    if runs and window_days > 0:
        cutoff = datetime.fromisoformat(runs[0][0]) - timedelta(days=window_days)
        recent = [(s, rows) for s, rows in runs if datetime.fromisoformat(s) > cutoff]
    else:
        recent = runs

    unproductive = sum(1 for _s, rows in recent if not any(r["productive"] for r in rows))
    degraded = sum(
        1
        for _s, rows in recent
        if any(r["expiries_captured"] < r["expiries_requested"] for r in rows)
    )

    majority = len(recent) >= MIN_RUNS_FOR_MAJORITY and unproductive * 2 > len(recent)

    return {
        "runs_recorded": len(runs),
        "consecutive_unproductive": consecutive,
        "unproductive_in_window": unproductive,
        "degraded_in_window": degraded,
        "window_size": len(recent),
        "triggered": bool(consecutive >= 3 or majority),
    }
