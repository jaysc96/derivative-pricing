"""The committed evidence artifacts still reproduce.

A generated file that has drifted from its generator is worse than no file:
it reads as evidence while being whatever someone last edited by hand. This
turns that into a test failure.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent


def test_convergence_artifact_is_not_stale():
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "generate_evidence.py"), "--check"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_readme_points_at_artifacts_that_exist():
    """Every evidence link on the landing page resolves.

    R26 is about a reviewer evaluating the evidence without cloning, which a
    broken relative link defeats entirely.
    """
    import re

    readme = (REPO / "README.md").read_text()
    targets = re.findall(r"\]\((?!https?://)([^)#]+)", readme)
    assert targets, "no relative links found — has the README lost its evidence section?"

    missing = [t for t in targets if not (REPO / t).exists()]
    assert not missing, f"README links to missing files: {missing}"
