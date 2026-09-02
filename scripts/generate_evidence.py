"""Regenerate the committed evidence artifacts.

    python scripts/generate_evidence.py           # write all three artifacts
    python scripts/generate_evidence.py --check   # fail if convergence.md would change

Three artifacts, two different reproducibility rules.

`docs/evidence/convergence.md` is pure function of the code: the stochastic
methods are seeded and every value is rounded before being written, so a clean
checkout reproduces it byte for byte. `--check` is what makes committing it
safe, and is the only artifact `--check` covers.

`docs/evidence/coverage.md` and `docs/evidence/surface.md` are not that kind
of artifact — both read the accumulated market-data store, which is not in
the repository and changes on every capture. Coverage is available from raw
capture alone (U11); the surface needs inversion and so arrives with U12b's
derived layer. The byte-for-byte rule cannot apply to a table that is
expected to differ from one regeneration to the next; their rule instead is
the one R35's note states: regenerated, never hand-edited, with the
generating query and the source snapshot's own timestamp recorded inside the
artifact, so a reader can tell what moment it describes without trusting the
commit date. `--check` does not touch either — checking a store-derived table
for staleness would fail immediately after any real capture, which is not
staleness, it is the table doing its job.

Deliberately excluded: timings. They are the one thing that cannot reproduce,
and they live in docs/evidence/benchmarks.md with their own hardware caveat.
"""

import argparse
import itertools
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from pricing import AMERICAN_METHODS, ENGINE_VERSION, EUROPEAN_METHODS, American_Option, European_Option

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from reference_values import (  # noqa: E402
    HULL_AMERICAN_CONVERGED,
    HULL_AMERICAN_FIVE_STEP,
    HULL_EUROPEAN,
)

from marketdata import Store  # noqa: E402
from marketdata.capture import fallback_trigger_state  # noqa: E402

OUTPUT = Path(__file__).parent.parent / "docs" / "evidence" / "convergence.md"
SURFACE_OUTPUT = Path(__file__).parent.parent / "docs" / "evidence" / "surface.md"
COVERAGE_OUTPUT = Path(__file__).parent.parent / "docs" / "evidence" / "coverage.md"
STORE_PATH = Path(__file__).parent.parent / "data" / "quotes.db"

TREE_STEPS = 200
MC_PATHS, MC_STEPS = 50_000, 50
LSMC_PATHS = 20_000
SEED = 42

K, R, SIG, DIVIDEND = 100.0, 0.05, 0.25, 0.03
MONEYNESS = [0.8, 0.9, 1.0, 1.1, 1.2]
MATURITIES = [0.5, 1.0]

GREEKS = ["delta", "gamma", "theta", "vega", "rho"]


def price(style, kind, S, y, T, method, K=K, r=R, sig=SIG):
    cls = European_Option if style == "european" else American_Option
    opt = cls(kind, S, K, r, sig, y, T, method)
    if method in ("BT", "TT"):
        opt.setTreeSteps(TREE_STEPS)
    elif method == "MC":
        opt.setSeedVariables(SEED, MC_PATHS, T / MC_STEPS)
    elif method == "LSMC":
        opt.setSeedVariables(SEED, LSMC_PATHS)
    if method == "BSM":
        return opt.BSM().price
    return float(opt.method())


def convergence_table(style, methods):
    lines = [
        "| Moneyness | T | " + " | ".join(methods) + " | Max spread |",
        "|---|---|" + "---|" * (len(methods) + 1),
    ]
    worst = 0.0
    for m, T in itertools.product(MONEYNESS, MATURITIES):
        values = [price(style, "call", m * K, DIVIDEND, T, meth) for meth in methods]
        spread = max(values) - min(values)
        worst = max(worst, spread)
        cells = " | ".join(f"{v:.4f}" for v in values)
        lines.append(f"| {m:.1f} | {T:.2f} | {cells} | {spread:.4f} |")
    return "\n".join(lines), worst


def greek_table():
    lines = [
        "| Moneyness | T | " + " | ".join(g.capitalize() for g in GREEKS) + " |",
        "|---|---|" + "---|" * len(GREEKS),
    ]
    worst = 0.0
    for m, T in itertools.product(MONEYNESS, MATURITIES):
        args = ("call", m * K, K, R, SIG, DIVIDEND, T)
        fd = European_Option(*args, "FD").priceOption()
        exact = European_Option(*args, "BSM").priceOption()
        cells = []
        for greek in GREEKS:
            a, b = getattr(fd, greek), getattr(exact, greek)
            rel = abs(a - b) / max(abs(b), 1e-12)
            worst = max(worst, rel)
            cells.append(f"{rel * 100:.2f}%")
        lines.append(f"| {m:.1f} | {T:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines), worst


def anchor_table():
    rows = []
    p = HULL_EUROPEAN["params"]
    for kind in ("call", "put"):
        printed = HULL_EUROPEAN[kind]
        for method in EUROPEAN_METHODS:
            got = price("european", kind, p["S"], p["y"], p["T"], method, p["K"], p["r"], p["sig"])
            rows.append(
                f"| European {kind} | {method} | {printed:.2f} | {got:.4f} | {got - printed:+.4f} |"
            )

    a = HULL_AMERICAN_FIVE_STEP
    q = a["params"]
    opt = American_Option("put", q["S"], q["K"], q["r"], q["sig"], q["y"], q["T"], "BT")
    opt.setTreeSteps(a["steps"])
    got = float(opt.BT())
    rows.append(
        f"| American put, {a['steps']}-step tree | BT | {a['put']:.2f} | {got:.4f} | {got - a['put']:+.4f} |"
    )

    c = HULL_AMERICAN_CONVERGED
    q = c["params"]
    for method in ("BT", "TT", "FD"):
        got = price("american", "put", q["S"], q["y"], q["T"], method, q["K"], q["r"], q["sig"])
        rows.append(
            f"| American put, converged | {method} | {c['put']:.4f} | {got:.4f} | {got - c['put']:+.4f} |"
        )

    header = [
        "| Contract | Method | Reference | This library | Difference |",
        "|---|---|---|---|---|",
    ]
    return "\n".join(header + rows)


def render():
    european, eu_worst = convergence_table("european", list(EUROPEAN_METHODS))
    american, am_worst = convergence_table("american", list(AMERICAN_METHODS))
    greeks, greek_worst = greek_table()

    return f"""# Convergence evidence

Generated by `scripts/generate_evidence.py`. Do not edit by hand — the
generator is run with `--check` to confirm this file reproduces byte for byte,
so an edit here becomes a failure rather than a correction.

Six methods, none sharing an implementation path: a closed form, two lattices,
two Monte Carlo estimators, and a PDE solver. Agreement between them is the
claim, and the point of having six is that a mistake would have to be made
identically in several to survive.

Contracts below are calls at strike {K:.0f}, rate {R:.2f}, volatility {SIG:.2f},
dividend yield {DIVIDEND:.2f}. Trees use {TREE_STEPS} steps, Monte Carlo
{MC_PATHS:,} paths, Longstaff-Schwartz {LSMC_PATHS:,}, all seeded.

## European methods

Worst spread across the grid: **{eu_worst:.4f}**.

{european}

## American methods

No closed form exists, so these agree only with each other. Longstaff-Schwartz
is biased low by construction, which is most of the spread.

Worst spread across the grid: **{am_worst:.4f}**.

{american}

## Greeks against closed form

Finite-difference Greeks as a relative error against Black-Scholes. Delta and
gamma come from spatial differences on the solver's own grid; theta, vega and
rho from central differences with a bump scaled to each input.

Worst relative error: **{greek_worst * 100:.2f}%**. Rho is the loosest, and the
reason is that these errors are inherited rather than introduced — a method
whose price is 0.95% out cannot produce a rho that is better.

{greeks}

## Against an outside reference

**Read the caveat before citing this section.** The parameter sets and printed
values are Hull's, but the edition and page could not be verified, so this is
not yet an external anchor in the sense R31 asks for. What it is: agreement
with an independent implementation in `tests/reference_values.py` that shares
no code with `pricing/`, reproducing the printed figures to their stated
precision.

One trap the table encodes. Hull's 4.49 for the American put is the output of a
**five-step** tree, not a converged price — the converged value is 4.2842.
Checking a 200-step method against 4.49 would be wrong by 0.2 while looking
correct, so the five-step row pins the five-step tree and the converged rows
are labelled as this library's own number.

{anchor_table()}
"""


def _status_row(counts: dict) -> str:
    order = ("solved", "no_quote", "no_solution", "bracket_exhausted", "not_identified", "no_market_context")
    return " | ".join(str(counts.get(status, 0)) for status in order)


#: A real chain runs to a hundred-plus strikes; convergence.md's own grid is
#: ten rows. An evenly-strided sample stays a compact illustration of the
#: shape rather than a raw dump, while still spanning the whole chain.
MAX_STRIKE_ROWS = 24


def _stride_sample(strikes: list[float]) -> tuple[list[float], int]:
    """Evenly-strided subset of ``strikes``, always keeping the last one."""
    stride = max(1, len(strikes) // MAX_STRIKE_ROWS)
    sampled = strikes[::stride]
    if sampled[-1] != strikes[-1]:
        sampled.append(strikes[-1])
    return sampled, stride


def _strike_table(store: Store, symbol: str, expiry: date, moment: datetime) -> str:
    rows = store.derived_as_of(symbol, expiry, moment, ENGINE_VERSION)
    if not rows:
        return "*No solved contract in this chain yet.*"
    by_strike: dict[float, dict[str, float]] = {}
    for row in rows:
        by_strike.setdefault(row["strike"], {})[row["option_type"]] = row["implied_vol"]
    strikes = sorted(by_strike)
    sampled, stride = _stride_sample(strikes)

    lines = ["| Strike | Call IV | Put IV |", "|---|---|---|"]
    for strike in sampled:
        sides = by_strike[strike]
        call = f"{sides['call']:.4f}" if "call" in sides else "-"
        put = f"{sides['put']:.4f}" if "put" in sides else "-"
        lines.append(f"| {strike:.1f} | {call} | {put} |")
    note = (
        f"\n\n{len(strikes)} solved strikes in this chain; every {stride} shown."
        if stride > 1 else ""
    )
    return "\n".join(lines) + note


def _two_sided_strike_table(store: Store, symbol: str, expiry: date, moment: datetime) -> str:
    """Per-strike bid/ask presence for one chain — the raw layer's own shape.

    Unlike ``_strike_table``, this reads ``chain_as_of`` rather than the
    derived table: coverage is a property of what was quoted, not of what
    solved, so a strike belongs here whether or not U12b's solver ever
    accepted it.
    """
    rows = store.chain_as_of(symbol, expiry, moment)
    if not rows:
        return "*No quote captured for this chain yet.*"
    by_strike: dict[float, dict[str, bool]] = {}
    for row in rows:
        two_sided = bool(row["bid"] and row["bid"] > 0 and row["ask"] and row["ask"] > 0)
        by_strike.setdefault(row["strike"], {})[row["option_type"]] = two_sided
    strikes = sorted(by_strike)
    sampled, stride = _stride_sample(strikes)

    lines = ["| Strike | Call two-sided | Put two-sided |", "|---|---|---|"]
    for strike in sampled:
        sides = by_strike[strike]
        call = "yes" if sides.get("call") else ("no" if "call" in sides else "-")
        put = "yes" if sides.get("put") else ("no" if "put" in sides else "-")
        lines.append(f"| {strike:.1f} | {call} | {put} |")
    note = (
        f"\n\n{len(strikes)} strikes in this chain; every {stride} shown."
        if stride > 1 else ""
    )
    return "\n".join(lines) + note


def render_surface() -> str:
    """The IV surface artifact — reads the store, never the solver (KTD9).

    Unlike ``render()`` above, this is not byte-for-byte reproducible and does
    not try to be: the store it reads changes on every capture, so the rule is
    "regenerated, never hand-edited," with the generating query and the source
    snapshot's own timestamp recorded here rather than trusted from the commit
    date. See the module docstring.
    """
    generated_at = datetime.now(timezone.utc).isoformat()

    if not STORE_PATH.exists():
        return f"""# Implied volatility surface

Generated by `scripts/generate_evidence.py` at {generated_at}. Not checked
for staleness — see the module docstring.

No archive exists yet at `{STORE_PATH.relative_to(STORE_PATH.parent.parent)}`.
This section fills in once `scripts/capture.py` has run at least once.
"""

    store = Store(STORE_PATH)
    chains = store.symbols_and_expiries()
    if not chains:
        return f"""# Implied volatility surface

Generated by `scripts/generate_evidence.py` at {generated_at}. Not checked
for staleness — see the module docstring.

The archive exists but has captured nothing yet.
"""

    summary = ["| Symbol | Expiry | Solved | No quote | No solution | Bracket exhausted | Not identified | No market context |",
               "|---|---|---|---|---|---|---|---|"]
    best = None
    best_solved = -1
    for symbol, expiry in chains:
        counts = store.derived_coverage(ENGINE_VERSION, symbol=symbol, expiry=expiry)
        summary.append(f"| {symbol} | {expiry.isoformat()} | {_status_row(counts)} |")
        solved = counts.get("solved", 0)
        if solved > best_solved:
            best, best_solved = (symbol, expiry), solved

    coverage = store.coverage()
    source_timestamp = coverage["last_capture"] or "unknown"

    detail = ""
    if best is not None and best_solved > 0:
        symbol, expiry = best
        moment = datetime.fromisoformat(coverage["last_capture"])
        detail = f"""
## {symbol}, expiry {expiry.isoformat()}

The chain with the most solved contracts, as an example of the shape rather
than a claim about this symbol specifically.

{_strike_table(store, symbol, expiry, moment)}
"""

    return f"""# Implied volatility surface

Generated by `scripts/generate_evidence.py` at {generated_at}, reading
`{STORE_PATH.relative_to(STORE_PATH.parent.parent)}` at engine version {ENGINE_VERSION}.
Not checked for staleness: this table is expected to differ from one
regeneration to the next, and a `--check` failure would describe the table
doing its job, not a defect. See the module docstring.

**Source snapshot timestamp: {source_timestamp}.** This is the moment the
underlying data describes, distinct from the generation time above — the
figure that matters for deciding whether this table is current.

Counts are by inversion status per chain (KTD9). `No market context` means the
snapshot itself is missing a risk-free rate or dividend yield, a capture-time
gap rather than a solver outcome; every other non-solved status is the
solver's own conclusion about the quote, detailed in `pricing/implied.py`.

{chr(10).join(summary)}
{detail}"""


def _run_record_table(store: Store, *, max_rows: int = 15) -> str:
    runs = store.run_record()[:max_rows]
    if not runs:
        return "*No capture run recorded yet.*"
    lines = [
        "| Started | Symbol | Outcome | Reason | Contracts | Two-sided | Expiries |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in runs:
        outcome = "productive" if row["productive"] else "FAILED"
        reason = row["reason"] or "-"
        lines.append(
            f"| {row['started_at']} | {row['symbol']} | {outcome} | {reason} | "
            f"{row['contracts']} | {row['two_sided']} | "
            f"{row['expiries_captured']}/{row['expiries_requested']} |"
        )
    return "\n".join(lines)


def render_coverage() -> str:
    """Raw-capture legibility: history stats, strike/expiry coverage, run record (R35, U11).

    Available from raw capture alone — unlike ``render_surface``, nothing here
    needs inversion. Same reproducibility rule as ``render_surface``: reads a
    store that changes on every capture, so this is regenerated and self-
    documented rather than checked byte for byte. See the module docstring.
    """
    generated_at = datetime.now(timezone.utc).isoformat()
    header = (
        f"Generated by `scripts/generate_evidence.py` at {generated_at}. Not "
        "checked for staleness — see the module docstring."
    )

    if not STORE_PATH.exists():
        return f"""# Market data coverage

{header}

No archive exists yet at `{STORE_PATH.relative_to(STORE_PATH.parent.parent)}`.
This section fills in once `scripts/capture.py` has run at least once.
"""

    store = Store(STORE_PATH)
    coverage = store.coverage()
    if not coverage["snapshots"]:
        return f"""# Market data coverage

{header}

The archive exists but has captured nothing yet.
"""

    two_sided_share = coverage["two_sided"] / coverage["quotes"] if coverage["quotes"] else 0.0

    chains = store.symbols_and_expiries()
    chain_summary = [
        "| Symbol | Expiry | Contracts | Two-sided | Share |",
        "|---|---|---|---|---|",
    ]
    best = None
    best_quotes = -1
    for symbol, expiry in chains:
        chain_coverage = store.raw_coverage(symbol=symbol, expiry=expiry)
        share = (
            chain_coverage["two_sided"] / chain_coverage["quotes"]
            if chain_coverage["quotes"]
            else 0.0
        )
        chain_summary.append(
            f"| {symbol} | {expiry.isoformat()} | {chain_coverage['quotes']} | "
            f"{chain_coverage['two_sided']} | {share:.1%} |"
        )
        if chain_coverage["quotes"] > best_quotes:
            best, best_quotes = (symbol, expiry), chain_coverage["quotes"]

    detail = ""
    if best is not None and best_quotes > 0:
        symbol, expiry = best
        moment = datetime.fromisoformat(coverage["last_capture"])
        detail = f"""
## {symbol}, expiry {expiry.isoformat()}

The chain with the most captured contracts, as an example of the shape rather
than a claim about this symbol specifically.

{_two_sided_strike_table(store, symbol, expiry, moment)}
"""

    trigger = fallback_trigger_state(store)

    return f"""# Market data coverage

{header}

**Source snapshot timestamp: {coverage['last_capture'] or 'unknown'}.** This is
the moment the underlying data describes, distinct from the generation time
above.

## Accumulated history

- {coverage['snapshots']:,} snapshot(s) across {coverage['symbols']} symbol(s)
- {coverage['quotes']:,} quote(s), {coverage['two_sided']:,} two-sided ({two_sided_share:.1%})
- {coverage['contracts']:,} distinct contract(s)
- First capture: {coverage['first_capture'] or '-'}
- Last capture: {coverage['last_capture'] or '-'}

## Quote coverage by chain

{chr(10).join(chain_summary)}
{detail}
## Unproductive-run record

The fallback trigger (`marketdata.capture.fallback_trigger_state`) watches this
same record for the provider going unreliable.
{trigger['runs_recorded']} run(s) recorded so far;
{trigger['unproductive_in_window']} of {trigger['window_size']} unproductive in the trailing window;
{trigger['consecutive_unproductive']} consecutive unproductive right now.
Triggered: **{trigger['triggered']}**.

{_run_record_table(store)}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the artifact would change")
    args = parser.parse_args()

    generated = render()

    if args.check:
        if not OUTPUT.exists():
            print(f"{OUTPUT} does not exist; run without --check")
            return 1
        if OUTPUT.read_text() != generated:
            print(f"{OUTPUT} is stale — regenerate with scripts/generate_evidence.py")
            return 1
        print(f"{OUTPUT} reproduces byte for byte")
        return 0

    OUTPUT.write_text(generated)
    print(f"wrote {OUTPUT}")

    SURFACE_OUTPUT.write_text(render_surface())
    print(f"wrote {SURFACE_OUTPUT}")

    COVERAGE_OUTPUT.write_text(render_coverage())
    print(f"wrote {COVERAGE_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
