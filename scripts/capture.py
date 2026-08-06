"""Capture the tracked set once. The scheduled entry point.

    python scripts/capture.py                       # one capture
    python scripts/capture.py --status              # read the archive, ask nothing
    python scripts/capture.py --symbols AAPL,MSFT   # override the tracked set

U10 built the capture loop and U8 built the archive under it, and until this
file existed neither had a caller outside its own tests. A capture job that can
only be driven from a Python REPL cannot be put on a cron, run by an agent, or
scheduled by a host — which is the only way it accrues anything.

**The archive is the product and it can only be filled forwards.** ``yfinance``
serves the current chain and nothing else; that is why ``option_chain`` raises
``NotSupported`` for a dated request. A day this does not run is a day of
option-chain history that cannot be bought back from this provider later.

**Schedule this during or near US market hours.** The provider serves a chain
with bid and ask both zero outside them — measured at 05:33 UTC, 481 of 481
contracts zeroed on both sides, against 69% two-sided at 22:21 UTC. Contract
counts look normal either way, so a cron set to a convenient hour collects a
full-looking chain of nothing. The run reports ``all_zero_sided`` and exits
non-zero rather than pretending otherwise, but a schedule that earns that
result every night accrues no history at all. US equity options trade
13:30-20:00 UTC while the US observes daylight time (roughly March-November)
and 14:30-21:00 UTC the rest of the year — the shift is the *US* clock change,
not the scheduling host's own time zone or its own daylight-saving calendar,
which will not generally move on the same dates.

**Each run also derives.** After a productive capture, every raw quote not yet
inverted at the current engine version is (KTD9) — cheap, since only what
capture just wrote is pending, and it is what keeps the derived table current
without a second scheduled job to forget to set up.

**Exit status is the whole interface to a scheduler.** Nothing watches stdout on
a host, so the run's conclusion has to survive as an integer:

===  ====================================================================
  0  productive, and the fallback trigger is quiet
  1  the run was unproductive — no symbol produced a usable quote
  2  the fallback trigger has fired, and a human needs to look
===  ====================================================================

Status 2 outranks 1: a single bad run is noise, while a fired trigger is the
standing signal that this provider has stopped being viable and the second
adapter has to be built. Both are non-zero so a scheduler surfaces either.

Only the fixed reason codes are printed, never exception text and never a URL —
the same discipline U10 puts on the run record, for the same reason, since a
host's job log is not always as private as it looks.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from marketdata import Store, YFinanceAdapter  # noqa: E402
from marketdata.capture import fallback_trigger_state, run_capture  # noqa: E402
from marketdata.derive import derive_batch  # noqa: E402

#: Provisional, and deliberately the same six the spike measures — a tracked
#: set that drifts from the set being validated makes the spike's numbers
#: describe something other than what is being captured. U2 narrows this to the
#: plan's three-to-six once enough daily observations have accumulated; until
#: then, capturing all six costs one request per symbol more and keeps the
#: choice open, which is the cheaper mistake while history is unrecoverable.
TRACKED = ("SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA")

DEFAULT_DB = Path(__file__).parent.parent / "data" / "quotes.db"
DEFAULT_EXPIRIES = 4

EXIT_OK = 0
EXIT_UNPRODUCTIVE = 1
EXIT_TRIGGERED = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--symbols",
        default=",".join(TRACKED),
        help="comma-separated tracked set (default: %(default)s)",
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help="archive path (default: %(default)s)"
    )
    parser.add_argument(
        "--expiries",
        type=int,
        default=DEFAULT_EXPIRIES,
        help="near expiries per symbol (default: %(default)s)",
    )
    parser.add_argument(
        "--include-expiry-day",
        action="store_true",
        help="keep contracts expiring today; the spike measured these at 42%% "
        "two-sided against 76-83%% further out",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="report what the archive holds and exit without capturing",
    )
    return parser


def report_status(store: Store) -> int:
    """What the archive holds and whether the provider is still working.

    The fallback trigger decides whether someone has to go build a second
    provider, and until this printed it, evaluating it meant calling a function
    by hand — so the signal existed and nobody was positioned to receive it.
    """
    coverage = store.coverage()
    trigger = fallback_trigger_state(store)

    print("archive")
    print(f"  quotes           {coverage['quotes']:,} ({coverage['two_sided']:,} two-sided)")
    print(f"  contracts        {coverage['contracts']:,}")
    print(f"  snapshots        {coverage['snapshots']:,} across {coverage['symbols']} symbols")
    print(f"  first capture    {coverage['first_capture'] or '-'}")
    print(f"  last capture     {coverage['last_capture'] or '-'}")

    print("\nfallback trigger")
    print(f"  runs recorded    {trigger['runs_recorded']}")
    print(f"  unproductive     {trigger['unproductive_in_window']} of {trigger['window_size']} in window")
    print(f"  degraded         {trigger['degraded_in_window']} of {trigger['window_size']} in window")
    print(f"  consecutive      {trigger['consecutive_unproductive']}")
    print(f"  TRIGGERED        {trigger['triggered']}")

    if trigger["triggered"]:
        print("\nThe trigger has fired. Build the fallback adapter (marketdata.app).")
        return EXIT_TRIGGERED
    return EXIT_OK


def main(argv=None, *, adapter=None, store=None, sleep=time.sleep) -> int:
    """``adapter``, ``store`` and ``sleep`` are injected by tests.

    No test here reaches the network, and none of them should pay the real
    backoff between retries either — a suite that sleeps for its own retry
    policy stops being run.
    """
    args = build_parser().parse_args(argv)
    store = store or Store(args.db)

    if args.status:
        return report_status(store)

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    if not symbols:
        print("no symbols to capture")
        return EXIT_UNPRODUCTIVE

    adapter = adapter or YFinanceAdapter()
    result = run_capture(
        adapter,
        store,
        symbols,
        expiries_per_symbol=args.expiries,
        skip_expiry_day=not args.include_expiry_day,
        sleep=sleep,
    )

    for outcome in result.outcomes:
        if outcome.productive:
            state = "degraded " if outcome.degraded else "ok       "
            detail = f"{outcome.contracts:>5} contracts, {outcome.two_sided:>5} two-sided"
        else:
            state = "FAILED   "
            detail = outcome.reason or "unknown"
        print(
            f"  {outcome.symbol:<6} {state} "
            f"{outcome.expiries}/{outcome.expiries_requested} expiries  {detail}"
        )

    share = result.two_sided / result.contracts if result.contracts else 0.0
    print(
        f"\n{result.contracts:,} contracts, {result.two_sided:,} two-sided ({share:.1%})"
        f"{' — DEGRADED' if result.degraded else ''}"
    )

    derived = derive_batch(store)
    print(f"{derived:,} quote(s) derived")

    trigger = fallback_trigger_state(store)
    if trigger["triggered"]:
        print(
            f"\nFallback trigger FIRED: "
            f"{trigger['consecutive_unproductive']} consecutive unproductive, "
            f"{trigger['unproductive_in_window']} of {trigger['window_size']} in window. "
            "Build the fallback adapter (marketdata.app)."
        )
        return EXIT_TRIGGERED
    if not result.productive:
        print("\nRun was unproductive: no symbol produced a usable quote.")
        return EXIT_UNPRODUCTIVE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
