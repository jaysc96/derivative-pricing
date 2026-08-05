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

Regenerate the baseline only when a change to the numbers is intended, and say
so in the commit that does it.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pricing import (
    AMERICAN_METHODS,
    EUROPEAN_METHODS,
    American_Option,
    European_Option,
)

BASELINE_PATH = Path(__file__).parent / "baseline_prices.json"
BASELINE = json.loads(BASELINE_PATH.read_text())
META = BASELINE["_meta"]

# Spread across moneyness, maturity, exercise style, and dividend, so a move
# that only happens to preserve the at-the-money case is caught. Must stay in
# step with the labels recorded in the baseline.
CASES = {
    #  label:            (option_type,  S0,    K,     r,    sig,  y,    T)
    "atm_call_1y": ("call", 100.0, 100.0, 0.05, 0.20, 0.02, 1.0),
    "atm_put_1y": ("put", 100.0, 100.0, 0.05, 0.20, 0.02, 1.0),
    "itm_call_1y": ("call", 120.0, 100.0, 0.05, 0.20, 0.02, 1.0),
    "otm_call_1y": ("call", 80.0, 100.0, 0.05, 0.20, 0.02, 1.0),
    "itm_put_6m": ("put", 80.0, 100.0, 0.03, 0.35, 0.00, 0.5),
    "otm_put_2y": ("put", 130.0, 100.0, 0.05, 0.25, 0.04, 2.0),
    "high_vol_call_3m": ("call", 100.0, 100.0, 0.01, 0.60, 0.00, 0.25),
    "low_vol_call_1y": ("call", 100.0, 95.0, 0.05, 0.10, 0.02, 1.0),
}

STYLES = {
    "european": (European_Option, EUROPEAN_METHODS),
    "american": (American_Option, AMERICAN_METHODS),
}

# The move is a copy, so agreement should be exact. The tolerance guards against
# last-bit differences between platforms, not against real drift — the recorded
# values carry three decimals, so anything meaningful is orders of magnitude
# larger than this.
TOL = 1e-9

BASELINE_KEYS = sorted(key for key in BASELINE if key != "_meta")


def build(key):
    """Construct and discretize the option a baseline key names."""
    style, label, method = key.split(".")
    cls, _ = STYLES[style]
    option_type, S0, K, r, sig, y, T = CASES[label]

    opt = cls(option_type, S0, K, r, sig, y, T, method)
    if method in ("BT", "TT"):
        opt.setTreeSteps(META["tree_steps"])
    elif method in ("MC", "LSMC"):
        opt.setSeedVariables(META["seed"], META["mc_paths"], META["mc_dt"])
    elif method == "FD":
        opt.setFDVariables(*META["fd_grid"])
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
    [(style, method) for style, (_, methods) in STYLES.items() for method in methods],
)
def test_every_advertised_method_is_reachable(style, method):
    """Each method the public API names resolves and binds on construction."""
    cls, _ = STYLES[style]
    opt = cls("call", 100.0, 100.0, 0.05, 0.20, 0.02, 1.0, method)
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


@pytest.mark.parametrize("key", BASELINE_KEYS)
def test_matches_pre_move_baseline(key):
    expected = BASELINE[key]
    opt = build(key)

    if "raises" in expected:
        with pytest.raises(getattr(np.linalg, expected["raises"])):
            opt.priceOption()
        return

    result = opt.priceOption()

    for name, want in expected.items():
        got = result[name]
        if isinstance(want, dict):
            # FD returns a vector over its own grid; the baseline records the
            # shape and three sampled nodes rather than 300 numbers per case.
            assert got.shape == (want["shape"],), name
            assert got[0] == pytest.approx(want["first"], abs=TOL), f"{name} first"
            assert got[want["shape"] // 2] == pytest.approx(
                want["mid"], abs=TOL
            ), f"{name} mid"
            assert got[-1] == pytest.approx(want["last"], abs=TOL), f"{name} last"
        else:
            assert float(got) == pytest.approx(want, abs=TOL), name


def test_baseline_covers_every_style_and_method():
    """Guard against a baseline that silently loses entries."""
    want = {
        f"{style}.{label}.{method}"
        for style, (_, methods) in STYLES.items()
        for label in CASES
        for method in methods
    }
    assert set(BASELINE_KEYS) == want
