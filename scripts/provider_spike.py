"""Read-only spike against yfinance, to answer the questions U8 freezes.

    python scripts/provider_spike.py

Four questions, none of which can be answered by reading documentation:

1. Are bid and ask *populated*, or merely present as columns of zeros? The
   quality gate in R14 depends on real two-sided quotes, and a column full of
   zeros would look identical to a working feed until the surface came out wrong.
2. Which timestamp fields does the provider actually fill? R32's staleness rule
   needs a per-contract recency marker, and the schema freezes at U8 — after
   that, discovering there isn't one is expensive.
3. How many contracts are there across the candidate set? That is R11's N.
4. Does sustained daily polling draw a block? Only answerable by polling daily,
   which is why this unit is calendar-bound and starts early.

Throwaway by construction — the Definition of Done requires it to be deleted
once the tracked set, N and W are settled. It appends one record per run to
docs/evidence/provider-spike.jsonl so that several days of observations
accumulate into an answer for question 4.

Deliberately gentle: about thirty requests per run, spaced. The question is
whether *normal* polling gets blocked, and hammering a free endpoint to find
its ceiling would be both rude and a measurement of the wrong thing.

Failures record an error *class*, never the exception text or a URL — the
artifact is committed to a public repository, and that is the same discipline
U10 puts on the capture job.
"""

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

OUTPUT = Path(__file__).parent.parent / "docs" / "evidence" / "provider-spike.jsonl"

# Liquid and optionable, spanning an index ETF, a tech-heavy ETF, a small-cap
# ETF, and three single names of differing option volume. Six candidates to
# choose three to six from.
CANDIDATES = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA"]

EXPIRIES_PER_SYMBOL = 4
DELAY_SECONDS = 1.0

# Fields worth knowing the fill rate of, because a schema is about to freeze
# around whichever of them turn out to be real.
FIELDS = ["bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"]


def classify(exc):
    """An error class, never the message. The artifact is public."""
    name = type(exc).__name__
    text = str(exc).lower()
    if "rate" in text or "429" in text or "too many" in text:
        return "rate_limited"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "connection" in text or "network" in text or "resolve" in text:
        return "network_error"
    return f"other:{name}"


def summarize(frame, kind, symbol, expiry, now):
    both = (frame["bid"] > 0) & (frame["ask"] > 0)
    record = {
        "symbol": symbol,
        "expiry": expiry,
        "type": kind,
        "contracts": int(len(frame)),
        "both_sided": int(both.sum()),
        "both_sided_fraction": round(float(both.mean()), 4) if len(frame) else None,
    }

    for field in FIELDS:
        if field in frame.columns:
            filled = (frame[field].notna() & (frame[field] > 0)).mean()
            record[f"{field}_populated"] = round(float(filled), 4)

    # The recency marker question. lastTradeDate is the only timestamp the
    # provider returns per contract, and it is a *trade* time, not a quote
    # time: a contract quoted continuously but untraded since Friday carries
    # Friday's stamp. Recording the age distribution is what shows whether it
    # can stand in for a quote timestamp at all.
    if "lastTradeDate" in frame.columns and len(frame):
        stamps = frame["lastTradeDate"].dropna()
        record["lastTradeDate_populated"] = round(float(len(stamps) / len(frame)), 4)
        if len(stamps):
            ages = [(now - s.to_pydatetime()).total_seconds() / 3600 for s in stamps]
            record["trade_age_hours"] = {
                "median": round(statistics.median(ages), 2),
                "p90": round(sorted(ages)[int(0.9 * len(ages)) - 1], 2),
                "max": round(max(ages), 2),
            }
            # Two-sided *and* traded recently is the population a quality gate
            # would actually keep.
            fresh = sum(1 for a in ages if a < 24)
            record["traded_within_24h_fraction"] = round(fresh / len(frame), 4)
    return record


def main():
    now = datetime.now(timezone.utc)
    run = {
        "captured_at": now.isoformat(),
        "provider": "yfinance",
        "provider_version": yf.__version__,
        "requests": 0,
        "failures": {},
        "chains": [],
    }
    started = time.perf_counter()

    for symbol in CANDIDATES:
        ticker = yf.Ticker(symbol)
        try:
            run["requests"] += 1
            expiries = list(ticker.options)[:EXPIRIES_PER_SYMBOL]
        except Exception as exc:  # noqa: BLE001 - classifying, not handling
            reason = classify(exc)
            run["failures"][reason] = run["failures"].get(reason, 0) + 1
            print(f"{symbol}: expiry list failed ({reason})")
            continue

        print(f"{symbol}: {len(expiries)} expiries")
        for expiry in expiries:
            time.sleep(DELAY_SECONDS)
            try:
                run["requests"] += 1
                chain = ticker.option_chain(expiry)
            except Exception as exc:  # noqa: BLE001
                reason = classify(exc)
                run["failures"][reason] = run["failures"].get(reason, 0) + 1
                print(f"  {expiry}: failed ({reason})")
                continue

            for kind, frame in (("call", chain.calls), ("put", chain.puts)):
                summary = summarize(frame, kind, symbol, expiry, now)
                run["chains"].append(summary)
            call, put = run["chains"][-2], run["chains"][-1]
            print(
                f"  {expiry}: {call['contracts'] + put['contracts']:>4} contracts, "
                f"two-sided {call['both_sided'] + put['both_sided']:>4} "
                f"({(call['both_sided'] + put['both_sided']) / max(call['contracts'] + put['contracts'], 1):.0%})"
            )

    run["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    run["total_contracts"] = sum(c["contracts"] for c in run["chains"])
    run["total_two_sided"] = sum(c["both_sided"] for c in run["chains"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a") as handle:
        handle.write(json.dumps(run, sort_keys=True) + "\n")

    print(
        f"\n{run['requests']} requests in {run['elapsed_seconds']}s, "
        f"{run['total_contracts']} contracts, "
        f"{run['total_two_sided']} two-sided "
        f"({run['total_two_sided'] / max(run['total_contracts'], 1):.1%}), "
        f"failures {run['failures'] or 'none'}"
    )
    print(f"appended to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
