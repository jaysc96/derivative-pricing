---
name: Derivative Pricing
last_updated: 2026-08-05
---

# Derivative Pricing Strategy

## Target problem

Anyone trying to do something quantitative with options has to start from an
implied volatility they can't verify. Every platform publishes an IV column
computed with undisclosed assumptions — which rate, which dividend, mid or
last, and frequently a European formula applied to American contracts — so the
same contract carries different numbers on different screens with no way to
tell which is wrong or by how much. Recomputing it yourself is not a way out:
that needs a pricer you trust, correct American exercise, and the raw quote the
number came from, and the raw quote is exactly what the platforms don't publish.

## Our approach

Correct where others are wrong, and checkable so that's not just a claim.
Single-stock contracts are priced American, with the rate and dividend actually
used recorded beside the result; the raw quote is stored verbatim and never
overwritten, the pricer is version-stamped, and the entire surface can be
re-derived from the archive at any time. Accuracy sets the bar and auditability
is the guiding policy beneath it — when they conflict, most obviously where a
faster approximate method would buy more coverage, auditability wins.

## Who it's for

**Primary:** The individual systematic trader or quant-minded hobbyist writing
their own screens, signals, and backtests over options — they're hiring this to
get an implied-vol series they can build on without staking the result on a
number they can't reproduce.

**Later:** The independent trader or small shop who has to justify a mark to a
client or a partner. Named here because it says which future work is
on-strategy, not because it drives decisions today.

## Key metrics

- **Reproduction rate** — share of published IVs that re-derive to the same
  value from the archived raw quote at the recorded engine version. Target
  100%; anything less means the provenance chain is broken, which is the
  approach failing. Measured by batch re-derivation against the archive.
- **Gated quote coverage** — share of tracked contracts passing the quality
  gate. 69% at first measurement, 78% excluding expiry-day contracts. Falls
  when the provider degrades or a tracked name goes illiquid. Measured per
  capture run.
- **Capture continuity** — share of scheduled captures that succeeded in the
  trailing 30 days, and the longest unbroken run. The history is the moat and
  this is the only thing protecting it. Measured from the capture job log.
- **Round-trip inversion error** — worst |IV → price → IV| error across the
  surface. The accuracy standard stated as a number rather than a claim.
  Measured by the test suite and the batch job.

_No lagging outcome metric yet._ For a signal builder that would be repeat
pulls of the same series, and it is unmeasurable until a series is published
and someone is pulling it. Revisit then rather than carry a placeholder now.

## Tracks

### Engine correctness

The pricing and inversion math, and the evidence that it is right: six-method
convergence, American exercise, round-trip accuracy, generated evidence
artifacts.

_Why it serves the approach:_ it is the accuracy standard everything else is
measured against.

### Archive and provenance

Capture, verbatim raw storage that is never overwritten, engine version
stamping, re-derivability, and the quality gate. History is built from both
directions — live capture accruing forward, and backfilled raw quotes
recomputed through the same engine — with every row labelled by which it is,
because "the quote we observed" and "the quote a vendor records" are not the
same claim.

_Why it serves the approach:_ this track is auditability.

### Derived views and access

The surface, rich-versus-cheap, IV rank, and the API or exports a signal
builder actually consumes.

_Why it serves the approach:_ without it the other two are correct, checkable,
and used by nobody.

## Not working on

- **Coverage breadth** — three symbols someone can audit over three thousand
  they must take on faith.
- **The premium-seller signal product** — a real market, but it wants a number
  rather than provenance, which puts it at odds with the approach.
- **Being a chain viewer** — no competing on UI polish or display speed.
