"""Present so that ``tests/`` is on the import path for ``cases``.

pytest inserts a test file's own directory into ``sys.path`` under the default
import mode, but that behaviour depends on rootdir discovery. A conftest here
pins it, so ``from cases import CASES`` resolves the same way whether pytest is
invoked from the repo root, from ``tests/``, or with an explicit file argument.
"""
