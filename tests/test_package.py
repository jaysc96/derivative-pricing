"""Proof that U3's extraction changed packaging and nothing else.

Three claims, one per test group:

1. ``pricing`` imports by name from an installed distribution, with no
   ``sys.path`` manipulation and no dependence on the working directory.
2. Every method named in the public API is reachable through it.
3. Every price and Greek matches the value the code produced *before* the move,
   recorded in ``tests/baseline_prices.json``.

The third is the one that matters. The pre-move code had known defects, and a
refactor that quietly fixed or worsened one would be indistinguishable from a
clean move without a recorded baseline. Two of the recorded entries are
crashes rather than numbers — deep out-of-the-money LSMC raises
``LinAlgError`` — and those are asserted just as strictly. U6 changes them
deliberately; until then, preserving the crash is what proves nothing moved.

U4 superseded the finite-difference entries on purpose: those results were
vectors across a caller-supplied grid, and there is no longer such a grid to
supply. ``tests/test_contract.py`` carries FD's proof forward by checking the
new price at spot against what the old grid produced at the same point.

Regenerate the baseline only when a change to the numbers is intended, and say
so in the commit that does it.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from cases import CASES, STYLES, option

from pricing import AMERICAN_METHODS, EUROPEAN_METHODS

BASELINE_PATH = Path(__file__).parent / "baseline_prices.json"
BASELINE = json.loads(BASELINE_PATH.read_text())
META = BASELINE["_meta"]

METHODS = {"european": EUROPEAN_METHODS, "american": AMERICAN_METHODS}

# The library stopped rounding in U4 — rounding is a display decision, and the
# convergence suite and implied-volatility inversion both need the precision
# the old three-decimal round threw away. The recorded values still carry three
# decimals, so half of the last place is exactly the right tolerance: the
# methods this file covers are otherwise untouched, and any real drift in them
# would be orders of magnitude larger.
TOL = 5e-4

ALL_KEYS = sorted(key for key in BASELINE if key != "_meta")
LIVE_KEYS = [key for key in ALL_KEYS if not key.endswith(".FD")]
SUPERSEDED_KEYS = [key for key in ALL_KEYS if key.endswith(".FD")]


def build(key):
    """Construct and discretize the option a baseline key names."""
    style, label, method = key.split(".")
    opt = option(style, label, method)
    if method in ("BT", "TT"):
        opt.setTreeSteps(META["tree_steps"])
    elif method in ("MC", "LSMC"):
        opt.setSeedVariables(META["seed"], META["mc_paths"], META["mc_dt"])
    return opt


def test_imports_from_an_installed_distribution(tmp_path):
    """Importable by name from anywhere, not just from the repo root.

    Run out of a temp directory so the repo is not on ``sys.path`` implicitly.
    An import that only resolves because the caller happens to be standing in
    the project root is not a package.
    """
    proc = subprocess.run(
        [sys.executable, "-c", "import pricing; print(pricing.__file__)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("pricing/__init__.py")


def test_no_path_manipulation_in_the_package():
    """No module reaches into ``sys.path`` to make itself importable."""
    for module in Path(__file__).parent.parent.glob("pricing/*.py"):
        assert "sys.path" not in module.read_text(), module


@pytest.mark.parametrize(
    "style,method",
    [(style, method) for style, methods in METHODS.items() for method in methods],
)
def test_every_advertised_method_is_reachable(style, method):
    """Each method the public API names resolves and binds on construction."""
    opt = STYLES[style]("call", 100.0, 100.0, 0.05, 0.20, 0.02, 1.0, method)
    assert opt.method_name == method
    assert callable(opt.method)


def test_the_six_methods_are_covered_between_the_two_styles():
    assert set(EUROPEAN_METHODS) | set(AMERICAN_METHODS) == {
        "BSM",
        "BT",
        "TT",
        "MC",
        "LSMC",
        "FD",
    }


@pytest.mark.parametrize("key", LIVE_KEYS)
def test_matches_pre_move_baseline(key):
    expected = BASELINE[key]
    opt = build(key)

    if "raises" in expected:
        with pytest.raises(getattr(np.linalg, expected["raises"])):
            opt.priceOption()
        return

    result = opt.priceOption()

    for name, want in expected.items():
        got = getattr(result, name)
        assert got == pytest.approx(want, abs=TOL), name


def test_baseline_covers_every_style_and_method():
    """Guard against a baseline that silently loses entries."""
    want = {
        f"{style}.{label}.{method}"
        for style, methods in METHODS.items()
        for label in CASES
        for method in methods
    }
    assert set(ALL_KEYS) == want


def test_only_the_finite_difference_entries_are_superseded():
    """Pin what U4 was allowed to change, so a later unit cannot widen it.

    The finite-difference entries recorded a vector across a grid the caller
    supplied. U4 removed that grid, so there is nothing left to compare against
    and ``tests/test_contract.py`` takes over. Every other method is untouched
    and stays under the assertion above.
    """
    assert len(SUPERSEDED_KEYS) == 16
    assert all(key.endswith(".FD") for key in SUPERSEDED_KEYS)
    assert len(LIVE_KEYS) == 56
