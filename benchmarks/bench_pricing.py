"""Timing for the six pricing methods, against a recorded baseline.

    python benchmarks/bench_pricing.py            # compare against the baseline
    python benchmarks/bench_pricing.py --record   # overwrite the baseline

The baseline in ``benchmarks/baseline.json`` is the post-U6, pre-vectorization
code. R12 wants a recorded before-and-after rather than an assertion that
something got faster, so the ratio this prints is the deliverable, not the
absolute numbers — those depend on the machine.

One thing the ratio deliberately excludes: the finite-difference matrix
factorization, which moved out of the time loop during U4 and is therefore
already in the baseline. Claiming it here would be counting it twice.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from pricing import American_Option, European_Option

BASELINE_PATH = Path(__file__).parent / "baseline.json"

# One contract, priced every way it can be. At the money and one year, so no
# method is doing unusually little work.
CONTRACT = dict(S=100.0, K=100.0, r=0.05, sig=0.25, y=0.02, T=1.0)

TREE_STEPS = 200
MC_PATHS, MC_DT = 50_000, 1 / 252
LSMC_PATHS = 20_000
SEED = 42

CASES = [
    ("european", "BSM"),
    ("european", "BT"),
    ("european", "TT"),
    ("european", "MC"),
    ("european", "FD"),
    ("american", "BT"),
    ("american", "TT"),
    ("american", "LSMC"),
    ("american", "FD"),
]

# Slow methods get fewer repeats. Timing noise matters less than wall-clock
# here — a benchmark nobody runs because it takes ten minutes is not a gate.
REPEATS = {"BSM": 200, "BT": 5, "TT": 3, "MC": 3, "LSMC": 3, "FD": 5}


def build(style, method):
    cls = European_Option if style == "european" else American_Option
    c = CONTRACT
    opt = cls("call", c["S"], c["K"], c["r"], c["sig"], c["y"], c["T"], method)
    if method in ("BT", "TT"):
        opt.setTreeSteps(TREE_STEPS)
    elif method == "MC":
        opt.setSeedVariables(SEED, MC_PATHS, MC_DT)
    elif method == "LSMC":
        opt.setSeedVariables(SEED, LSMC_PATHS)
    return opt


def measure(style, method):
    """Median of N full priceOption calls, in milliseconds.

    priceOption rather than the bare method, because that is what every caller
    uses and it is where the bump count multiplies the cost.
    """
    build(style, method).priceOption()  # warm any import-time cost
    timings = []
    for _ in range(REPEATS[method]):
        start = time.perf_counter()
        build(style, method).priceOption()
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="overwrite the baseline")
    args = parser.parse_args()

    results = {f"{style}.{method}": measure(style, method) for style, method in CASES}

    if args.record:
        BASELINE_PATH.write_text(
            json.dumps({k: round(v, 4) for k, v in results.items()}, indent=2, sort_keys=True) + "\n"
        )
        print(f"baseline written to {BASELINE_PATH}")

    baseline = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}

    print(f"{'method':<18}{'baseline ms':>13}{'current ms':>12}{'speedup':>10}")
    print("-" * 53)
    total_before = total_after = 0.0
    for key, current in results.items():
        before = baseline.get(key)
        if before is None:
            print(f"{key:<18}{'—':>13}{current:>12.2f}{'—':>10}")
            continue
        total_before += before
        total_after += current
        print(f"{key:<18}{before:>13.2f}{current:>12.2f}{before / current:>9.1f}x")

    if total_before:
        print("-" * 53)
        print(f"{'total':<18}{total_before:>13.2f}{total_after:>12.2f}{total_before / total_after:>9.1f}x")


if __name__ == "__main__":
    main()
