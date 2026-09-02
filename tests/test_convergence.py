"""R2 and R31: the methods agree with each other, and with something outside.

Written before the defects are fixed, so several of these fail on purpose. What
fails, and why, is recorded in ``docs/evidence/pre-fix-failures.md``. A green
run at this point would mean the tolerances are too loose to be worth having.

Tolerances are not guessed. Each method carries an error budget against truth —
an absolute floor plus a fraction of the price — and a pair is allowed the sum
of its two budgets. That is the right composition because the budgets bound
each method's distance from the same true value, so the triangle inequality
bounds their distance from each other. It also means a tolerance can be
justified one method at a time instead of once per pair.
"""

import itertools

import pytest
from cases import STYLES

from pricing import AMERICAN_METHODS, EUROPEAN_METHODS
from reference_values import (
    HULL_AMERICAN_CONVERGED,
    HULL_AMERICAN_FIVE_STEP,
    HULL_EUROPEAN,
)

K, R, SIG = 100.0, 0.05, 0.25
MONEYNESS = [0.8, 0.9, 1.0, 1.1, 1.2]
MATURITIES = [0.5, 1.0]

TREE_STEPS = 200
MC_PATHS, MC_STEPS = 50_000, 50
LSMC_PATHS = 20_000
SEED = 42

# (absolute floor, fraction of price). Discretization budgets for the
# deterministic methods, sampling error for the stochastic ones.
BUDGET = {
    "BSM": (0.000, 0.0000),   # closed form, the reference the others are measured against
    "BT": (0.030, 0.0040),    # O(1/n) tree bias at 200 steps
    "TT": (0.030, 0.0040),
    "FD": (0.060, 0.0080),    # grid truncation plus interpolation at spot
    "MC": (0.100, 0.0060),    # ~3 standard errors, antithetic, 50k paths
    "LSMC": (0.200, 0.0250),  # sampling plus regression bias; biased low by construction
}

GRID = [(m, T) for m in MONEYNESS for T in MATURITIES]


def pair_tolerance(a, b, reference):
    floor = BUDGET[a][0] + BUDGET[b][0]
    rel = BUDGET[a][1] + BUDGET[b][1]
    return max(floor, rel * abs(reference))


def price(style, kind, S, y, T, method, K=K, r=R, sig=SIG):
    opt = STYLES[style](kind, S, K, r, sig, y, T, method)
    if method in ("BT", "TT"):
        opt.setTreeSteps(TREE_STEPS)
    elif method == "MC":
        opt.setSeedVariables(SEED, MC_PATHS, T / MC_STEPS)
    elif method == "LSMC":
        opt.setSeedVariables(SEED, LSMC_PATHS)
    if method == "BSM":
        return opt.BSM().price
    return float(opt.method())


def _convergence_case(style, methods):
    return [
        (a, b, m, T, kind)
        for a, b in itertools.combinations(methods, 2)
        for m, T in GRID
        for kind in ("call", "put")
    ]


# --------------------------------------------------------------------------
# Pairwise agreement, without dividends
# --------------------------------------------------------------------------
#
# Splitting the grid by dividend is what localizes the Monte Carlo defect. Its
# drift omits the dividend yield, so at y=0 it is correct and these pass; the
# same comparisons on the paying grid below fail by two to three currency
# units. Two identical test bodies over different y is the cheapest way to say
# "the sampling is fine, the drift is not".


@pytest.mark.parametrize("a,b,m,T,kind", _convergence_case("european", EUROPEAN_METHODS))
def test_european_methods_agree_without_dividends(a, b, m, T, kind):
    va = price("european", kind, m * K, 0.0, T, a)
    vb = price("european", kind, m * K, 0.0, T, b)
    assert abs(va - vb) <= pair_tolerance(a, b, max(abs(va), abs(vb)))


@pytest.mark.parametrize("a,b,m,T,kind", _convergence_case("american", AMERICAN_METHODS))
def test_american_methods_agree_without_dividends(a, b, m, T, kind):
    va = price("american", kind, m * K, 0.0, T, a)
    vb = price("american", kind, m * K, 0.0, T, b)
    assert abs(va - vb) <= pair_tolerance(a, b, max(abs(va), abs(vb)))


# --------------------------------------------------------------------------
# Pairwise agreement, with a dividend
# --------------------------------------------------------------------------

DIVIDEND = 0.03


@pytest.mark.parametrize("a,b,m,T,kind", _convergence_case("european", EUROPEAN_METHODS))
def test_european_methods_agree_with_dividends(a, b, m, T, kind):
    va = price("european", kind, m * K, DIVIDEND, T, a)
    vb = price("european", kind, m * K, DIVIDEND, T, b)
    assert abs(va - vb) <= pair_tolerance(a, b, max(abs(va), abs(vb)))


@pytest.mark.parametrize("a,b,m,T,kind", _convergence_case("american", AMERICAN_METHODS))
def test_american_methods_agree_with_dividends(a, b, m, T, kind):
    va = price("american", kind, m * K, DIVIDEND, T, a)
    vb = price("american", kind, m * K, DIVIDEND, T, b)
    assert abs(va - vb) <= pair_tolerance(a, b, max(abs(va), abs(vb)))


# --------------------------------------------------------------------------
# R31: anchored outside the repository
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("method", EUROPEAN_METHODS)
def test_european_anchor_matches_the_published_value(method, kind):
    """Every European method, against Hull's printed two decimals."""
    p = HULL_EUROPEAN["params"]
    got = price("european", kind, p["S"], p["y"], p["T"], method, p["K"], p["r"], p["sig"])
    expected = HULL_EUROPEAN[kind]
    assert got == pytest.approx(expected, abs=pair_tolerance(method, "BSM", expected) + 0.005)


def test_american_anchor_matches_the_published_five_step_value():
    """Hull's illustration is a five-step tree, so the anchor is a five-step tree.

    Pinning the step count is the point. The converged value for these
    parameters is 4.2842, so a 200-step method checked against the printed 4.49
    would be wrong by 0.2 while appearing to agree with the book.
    """
    p = HULL_AMERICAN_FIVE_STEP["params"]
    opt = STYLES["american"]("put", p["S"], p["K"], p["r"], p["sig"], p["y"], p["T"], "BT")
    opt.setTreeSteps(HULL_AMERICAN_FIVE_STEP["steps"])
    assert float(opt.BT()) == pytest.approx(HULL_AMERICAN_FIVE_STEP["put"], abs=0.005)


@pytest.mark.parametrize("method", ["BT", "TT", "FD"])
def test_american_converging_methods_reach_the_converged_anchor(method):
    """The same contract, valued properly rather than illustratively."""
    p = HULL_AMERICAN_CONVERGED["params"]
    got = price("american", "put", p["S"], p["y"], p["T"], method, p["K"], p["r"], p["sig"])
    expected = HULL_AMERICAN_CONVERGED["put"]
    assert got == pytest.approx(expected, abs=pair_tolerance(method, "BSM", expected) + 0.02)
