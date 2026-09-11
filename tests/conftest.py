"""pytest configuration for pygstat.

Several files under ``tests/`` are converted notebooks that run a full analysis
at import time. They are kept as runnable scripts, but excluded from pytest
collection so ``pytest`` stays a unit-test runner.

Run a notebook-style script directly, from the repository root::

    python tests/test_sgsim_meuse.py
"""

from pathlib import Path

collect_ignore = [
    "test_STPoisson_kriging.py",
    "test_krigeST.py",
    "test_multivariate_poisson_cokriging.py",
    "test_poisson_Kriging_apt_atp.py",
    "test_sgsim_meuse.py",
    "test_sisim_meuse.py",
    "test_utility_gslib_helpers.py",
    "test_variogram_krging.py",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
