# Provider spike findings

Running record for U2. Raw observations accumulate in `provider-spike.jsonl`,
one JSON object per run; this file is the reading of them.

**Status: incomplete.** One observation so far. The question U2 exists to answer
that cannot be rushed — whether sustained daily polling draws a block — needs
several more days. Everything below is provisional, and the number most likely
to move is the two-sided fraction, because the first run landed after the close.

## Observation 1 — 2026-08-05 22:21 UTC

Six candidates, four near expiries each. 30 requests, 31.5s, **no failures**,
3,681 contracts, **69.1% with both a non-zero bid and a non-zero ask**.

### Bid is the entire constraint

`ask` is populated on **100% of contracts, without exception**. In all 48
symbol-expiry-type groups the two-sided fraction equals the bid fill rate
exactly. Nothing is ever missing an ask; the gate is only ever about whether
anyone is bidding.

This matters more than the headline number. A quality rule written as "require
both sides" is, against this provider, a rule about bids alone — and a
mid-price built from a zero bid is not a wide quote, it is half a quote at half
its value. R14's gate must reject on `bid > 0`, not on presence of the field.

### Expiry-day contracts are a different population

| Expiry | Days out | Contracts | Two-sided |
|---|---|---:|---:|
| 2026-08-05 | 0 | 866 | **42%** |
| 2026-08-06 | 1 | 515 | 71% |
| 2026-08-07 | 2 | 1254 | 76% |
| 2026-08-10 | 5 | 777 | **83%** |
| 2026-08-12 | 7 | 269 | 82% |

And within expiry day the split is by option type, not by symbol: SPY puts 7%,
QQQ puts 11%, IWM puts 6%, MSFT puts 8% — against calls at 61-88% on the same
chains. Out-of-the-money puts hours from expiry are worth nothing and nobody
bids for them.

**Excluding zero-day-to-expiry contracts raises the overall two-sided fraction
from 69% to about 78%** and removes a population whose implied volatilities
would be the least reliable points on the surface anyway. Worth deciding before
U8 freezes the schema, because it changes N.

### There is a recency marker, but it is not the one R32 assumes

`lastTradeDate` is populated on **100% of contracts**. That answers the
existence question, and the schema can carry it.

It is a **trade** timestamp, not a quote timestamp, and the provider returns no
quote timestamp at all. A contract quoted continuously all day but last traded
on Friday carries Friday's stamp while its bid and ask are current. So the field
cannot answer "is this quote stale" — only "has anyone traded this recently".

Observed ages, measured against a 22:21 UTC capture (18:21 ET, after the 16:00
close): median 2-8 hours across most chains, **74% of contracts traded within
24 hours**. The outliers are informative — IWM expiry-day puts show a median
age of 30 hours, which is the same illiquid population the bid fill rate
flagged.

R32's staleness rule has to be written against what exists. Two usable readings:
treat `lastTradeDate` as a liquidity signal rather than a freshness one, or take
the capture timestamp as the quote time and accept that the provider gives no
per-contract way to detect a frozen quote. The first is more honest and is what
the field actually measures.

### Other fields

`impliedVolatility` is populated on 100% of contracts — Yahoo's own number.
Not a substitute for computing it (the whole point is the inversion and its
round-trip evidence), but a free cross-check: a systematic gap between this
library's surface and the provider's would be worth investigating before
publishing either. `openInterest` runs 77-100%. `volume` and `lastPrice` are
present throughout.

## Observation 2 — 2026-08-06 05:33 UTC — the first finding is time-bound

Not a spike run. The first live capture through `scripts/capture.py`, SPY only,
two expiries, 481 contracts. **Every one came back with bid and ask both zero.**

Verified against `yfinance` directly, bypassing the adapter, so this is the
provider and not our parsing: 96 of 96 calls on the front expiry with
`bid == 0.0` and `ask == 0.0`, `lastPrice` populated throughout, and
`impliedVolatility` pinned at `0.00001` — a placeholder, not a computation.

**This contradicts the headline finding above.** "Ask is populated on 100% of
contracts, without exception" is true of 22:21 UTC and false of 05:33 UTC. The
first observation landed at 18:21 ET, inside the extended session, where quotes
persist. This one landed at 01:33 ET, roughly nine hours after the close, and
Yahoo serves a zeroed chain then rather than the last known quote or a null.

The correction matters in three places:

* **The quality gate is unchanged but its motivation widens.** R14 rejecting on
  `bid > 0` still holds. What is new is that a whole capture can be zeros on
  both sides, which is not a thin market — it is the provider declining to
  answer, in a shape indistinguishable from one.
* **The capture schedule is load-bearing, not an operational detail.** A cron
  set to a convenient hour rather than a market hour collects nothing usable
  while reporting a full contract count. U10's `all_zero_sided` check caught
  exactly this on its first real run, which is the reassuring half; the other
  half is that the check was needed on run one.
* **N is unaffected** — 481 contracts arrived. Contract counts and quote
  quality vary independently, so sizing on N remains safe.

Time of day is now a variable the remaining runs have to span, alongside the
consecutive-days question. One observation per day at a fixed hour cannot
distinguish "the provider degraded" from "we asked at a worse time".

## Provisional answers

| Question | Provisional answer | Confidence |
|---|---|---|
| Bid and ask populated? | **Depends on the hour.** 22:21 UTC: ask always, bid 69% (78% ex-0DTE). 05:33 UTC: neither, on any contract | Medium — two observations, two very different answers |
| Per-contract recency marker? | `lastTradeDate`, but it times trades not quotes | High |
| N (contracts per capture) | ~3,700 across 6 symbols x 4 expiries | Low — depends on the tracked set and the 0DTE decision |
| W (capture window) | Not yet — derives from the schedule, which is undecided | — |
| Rate-limit headroom | 30 requests, no failures, no throttling | **Very low — one run** |
| Sustained polling blocked? | Unknown | Needs several days |

## What the next runs need to settle

1. **Several consecutive days**, which is the only way to answer the blocking
   question. This is why U2 starts early.
2. **At least one run during market hours.** Observation 1 was after the close
   and observation 2 was in the middle of the night; between them they bracket
   the bid fill rate at 69% and 0% without establishing what it is while the
   market is open. That number decides whether the surface is worth building.
3. **Runs at several hours of the day**, which observation 2 turned from a
   refinement into a prerequisite. Until the time-of-day curve is known, a
   fixed-hour schedule cannot tell a degrading provider from a badly chosen
   alarm clock — and the fallback trigger reads exactly that signal.
4. **The tracked set and expiry depth**, which together fix N.

The note on U2 already stands: this runs from a development machine, and the
provider throttles by IP reputation, so no ceiling measured here transfers to a
shared host. U9 re-runs it from the deployment target before anything is sized
on the number.
