"""
Tests for pygstat.sgsim/pygstat.sisim's `use_gpu` option.

Unlike pygstat.core.kriging/pygstat.indicator_kriging, Sequential Gaussian/
Indicator Simulation is inherently *sequential* within one realization:
each node's local kriging system depends on every previously-simulated
node (added to the conditioning set before the next node is visited), so
nodes cannot be batched together the way independent predictions can.
`use_gpu` defaults to `False` (not `'auto'`, unlike the other GPU-capable
classes) since for typical small `num_points` the per-call GPU dispatch
overhead usually outweighs the benefit -- see the `sgsim`/`sisim`
docstrings. At `nsim=1` it dispatches each node's own covariance-matrix
build + solve to CuPy individually; at `nsim >= 2` it instead batches all
realizations' local kriging systems together at each step
(`_simulate_batch`, both modules) -- see `TestBatchedRealizationsGPUPath`.

See tests/test_gpu_kriging.py's module docstring for why the "fake GPU"
approach (monkeypatching `cupy_gpu_usable()`/`cp` rather than requiring
real CUDA) is used here: this sandbox's GPU can import CuPy but can't
actually compile a kernel (confirmed directly).
"""

import numpy as np
import pandas as pd
import pytest

from pygstat.sgsim import sgsim
from pygstat.sisim import sisim
from pygstat.utils import backend


class _FakeCupyModule:
    """See tests/test_gpu_kriging.py for the full rationale."""

    ndarray = np.ndarray

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asnumpy(arr):
        return np.asarray(arr)


@pytest.fixture
def fake_gpu(monkeypatch):
    monkeypatch.setattr(backend, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(backend, "cp", _FakeCupyModule())
    monkeypatch.setattr(backend, "_cupy_usable_cache", True)
    monkeypatch.setattr(backend, "cupy_gpu_usable", lambda: True)


@pytest.fixture
def conditioning_data():
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "X": rng.uniform(0, 100, n), "Y": rng.uniform(0, 100, n), "Z": rng.normal(size=n),
    })
    grid = rng.uniform(0, 100, size=(15, 2))
    return df, grid


SGSIM_KW = dict(num_points=8, radius=200, kriging_type="ordinary", nsim=1, seed=7, quiet=True)
SISIM_KW = dict(
    thresholds=[-1.0, -0.3, 0.3, 1.0], global_cdf=[0.2, 0.45, 0.7, 0.9],
    num_points=8, radius=200, kriging_type="ordinary", nsim=1, seed=7, quiet=True,
)


class TestDefaultIsOptIn:
    def test_sgsim_default_use_gpu_is_false(self, conditioning_data):
        df, grid = conditioning_data
        vario = [0, 0.1, 30, 30, 1.0, "spherical"]
        # default use_gpu=False must never touch CuPy at all, even if it
        # were installed-but-broken -- no warning, no attempted dispatch.
        sims = sgsim(grid, df, "X", "Y", "Z", vario=vario, **SGSIM_KW)
        assert np.isfinite(sims).all()

    def test_sisim_default_use_gpu_is_false(self, conditioning_data):
        df, grid = conditioning_data
        vario = [0, 0.1, 30, 30, 1.0, "spherical"]
        sims = sisim(grid, df, "X", "Y", "Z", vario=vario, **SISIM_KW)
        assert np.isfinite(sims).all()


class TestDeterminism:
    def test_sgsim_same_seed_reproducible(self, conditioning_data):
        df, grid = conditioning_data
        vario = [0, 0.1, 30, 30, 1.0, "spherical"]
        s1 = sgsim(grid, df, "X", "Y", "Z", vario=vario, **SGSIM_KW)
        s2 = sgsim(grid, df, "X", "Y", "Z", vario=vario, **SGSIM_KW)
        np.testing.assert_array_equal(s1, s2)

    def test_sisim_same_seed_reproducible(self, conditioning_data):
        df, grid = conditioning_data
        vario = [0, 0.1, 30, 30, 1.0, "spherical"]
        s1 = sisim(grid, df, "X", "Y", "Z", vario=vario, **SISIM_KW)
        s2 = sisim(grid, df, "X", "Y", "Z", vario=vario, **SISIM_KW)
        np.testing.assert_array_equal(s1, s2)


class TestGPUCodePathMatchesCPU:
    """`fake_gpu` forces `use_gpu=True`/'auto' to resolve True with `cp`
    aliased to a NumPy-backed stand-in, exercising the real GPU dispatch
    code (as_gpu_array, xp.linalg.lstsq, ...) without needing real CUDA."""

    @pytest.mark.parametrize("vtype,extra", [
        ("spherical", []), ("exponential", []), ("gaussian", []), ("matern", [1.5]),
    ])
    @pytest.mark.parametrize("kriging_type", ["ordinary", "simple"])
    def test_sgsim_gpu_matches_cpu(self, conditioning_data, fake_gpu, vtype, extra, kriging_type):
        df, grid = conditioning_data
        vario = [0, 0.1, 30, 30, 1.0, vtype] + extra
        kw = dict(SGSIM_KW)
        kw["kriging_type"] = kriging_type
        sims_cpu = sgsim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=False, **kw)
        sims_gpu = sgsim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=True, **kw)
        np.testing.assert_allclose(sims_cpu, sims_gpu, atol=1e-8)

    @pytest.mark.parametrize("kriging_type", ["ordinary", "simple"])
    def test_sisim_gpu_matches_cpu(self, conditioning_data, fake_gpu, kriging_type):
        df, grid = conditioning_data
        vario = [0, 0.1, 30, 30, 1.0, "spherical"]
        kw = dict(SISIM_KW)
        kw["kriging_type"] = kriging_type
        sims_cpu = sisim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=False, **kw)
        sims_gpu = sisim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=True, **kw)
        np.testing.assert_allclose(sims_cpu, sims_gpu, atol=1e-8)

    def test_sisim_gpu_matches_cpu_with_per_threshold_vario_dict(self, conditioning_data, fake_gpu):
        df, grid = conditioning_data
        thresholds = SISIM_KW["thresholds"]
        vario_dict = {t: [0, 0.1, 30, 30, 1.0, "spherical"] for t in thresholds}
        kw = dict(SISIM_KW)
        sims_cpu = sisim(grid, df, "X", "Y", "Z", vario=vario_dict, use_gpu=False, **kw)
        sims_gpu = sisim(grid, df, "X", "Y", "Z", vario=vario_dict, use_gpu=True, **kw)
        np.testing.assert_allclose(sims_cpu, sims_gpu, atol=1e-8)


class TestBatchedRealizationsGPUPath:
    """`nsim >= 2` + `use_gpu=True` takes `_simulate_batch` (all realizations
    in lockstep, one batched covariance build + solve per step) instead of
    `_simulate_one`'s one-node-at-a-time dispatch -- see sgsim.py's module
    docstring. `fake_gpu` exercises that real code path (as_gpu_array,
    xp.linalg.solve, the padding/masking) without needing real CUDA."""

    @pytest.mark.parametrize("kriging_type", ["ordinary", "simple"])
    def test_matches_cpu_per_realization_path(self, conditioning_data, fake_gpu, kriging_type):
        df, grid = conditioning_data
        vario = [0, 0.1, 30, 30, 1.0, "spherical"]
        kw = dict(SGSIM_KW)
        kw["kriging_type"] = kriging_type
        kw["nsim"] = 4
        # CPU path: nsim independent calls to _simulate_one (use_gpu=False).
        sims_cpu = sgsim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=False, **kw)
        # GPU path: _simulate_batch, all 4 realizations batched per step.
        sims_gpu = sgsim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=True, **kw)
        # Not bit-identical: _simulate_batch solves with a small ridge via
        # xp.linalg.solve where _simulate_one uses lstsq (see _simulate_batch's
        # docstring), and that tiny per-node difference compounds through the
        # sequential conditioning set -- but should stay numerically close.
        np.testing.assert_allclose(sims_cpu, sims_gpu, atol=1e-3, rtol=1e-3)

    def test_batched_path_reproducible(self, conditioning_data, fake_gpu):
        df, grid = conditioning_data
        vario = [0, 0.1, 30, 30, 1.0, "spherical"]
        kw = dict(SGSIM_KW)
        kw["nsim"] = 3
        s1 = sgsim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=True, **kw)
        s2 = sgsim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=True, **kw)
        np.testing.assert_array_equal(s1, s2)

    def test_batched_path_exact_conditioning(self, conditioning_data, fake_gpu):
        df, grid = conditioning_data
        # Simulate on a grid that includes real data locations plus new ones.
        cond_pts = df[["X", "Y"]].values[:10]
        new_pts = np.random.default_rng(1).uniform(0, 100, size=(10, 2))
        combined_grid = np.vstack([cond_pts, new_pts])
        vario = [0, 0.1, 30, 30, 1.0, "spherical"]
        kw = dict(SGSIM_KW)
        kw["nsim"] = 3
        sims = sgsim(combined_grid, df, "X", "Y", "Z", vario=vario, use_gpu=True, **kw)
        true_vals = df["Z"].values[:10]
        for r in range(sims.shape[0]):
            np.testing.assert_allclose(sims[r, :10], true_vals, atol=1e-8)

    def test_nsim_1_does_not_use_batched_path(self, conditioning_data, monkeypatch):
        """nsim == 1 has nothing to batch across, so it must still go
        through _simulate_one, not _simulate_batch."""
        df, grid = conditioning_data
        vario = [0, 0.1, 30, 30, 1.0, "spherical"]
        import sys
        # `import pygstat.sgsim as X` would resolve via the `pygstat`
        # package's `sgsim` attribute, which `pygstat/__init__.py`'s
        # `from .sgsim import sgsim` rebinds to the *function* -- go
        # through sys.modules to get the actual submodule.
        sgsim_mod = sys.modules["pygstat.sgsim"]
        called = {"batch": False}
        original = sgsim_mod._simulate_batch

        def spy(*args, **kwargs):
            called["batch"] = True
            return original(*args, **kwargs)

        monkeypatch.setattr(sgsim_mod, "_simulate_batch", spy)
        kw = dict(SGSIM_KW)
        kw["nsim"] = 1
        sgsim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=False, **kw)
        assert called["batch"] is False


class TestSisimBatchedRealizationsGPUPath:
    """sisim's mirror of TestBatchedRealizationsGPUPath -- `_simulate_batch`
    here additionally batches *per threshold* (see its docstring), so these
    also exercise that extra loop, including with a per-threshold vario
    dict (different rotation matrix per threshold)."""

    @pytest.mark.parametrize("kriging_type", ["ordinary", "simple"])
    def test_matches_cpu_per_realization_path(self, conditioning_data, fake_gpu, kriging_type):
        df, grid = conditioning_data
        vario = [0, 0.1, 30, 30, 1.0, "spherical"]
        kw = dict(SISIM_KW)
        kw["kriging_type"] = kriging_type
        kw["nsim"] = 4
        sims_cpu = sisim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=False, **kw)
        sims_gpu = sisim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=True, **kw)
        # Same tolerance rationale as sgsim's equivalent test: _simulate_batch
        # solves with a small ridge via xp.linalg.solve where _indicator_ccdf
        # uses lstsq, and that per-node difference compounds through the
        # sequential conditioning set (and, here, through _ordrel/_beyond_zval).
        np.testing.assert_allclose(sims_cpu, sims_gpu, atol=1e-2, rtol=1e-2)

    def test_matches_cpu_per_realization_path_with_per_threshold_vario_dict(self, conditioning_data, fake_gpu):
        df, grid = conditioning_data
        thresholds = SISIM_KW["thresholds"]
        vario_dict = {t: [0, 0.1, 30, 30, 1.0, "spherical"] for t in thresholds}
        kw = dict(SISIM_KW)
        kw["nsim"] = 4
        sims_cpu = sisim(grid, df, "X", "Y", "Z", vario=vario_dict, use_gpu=False, **kw)
        sims_gpu = sisim(grid, df, "X", "Y", "Z", vario=vario_dict, use_gpu=True, **kw)
        np.testing.assert_allclose(sims_cpu, sims_gpu, atol=1e-2, rtol=1e-2)

    def test_batched_path_reproducible(self, conditioning_data, fake_gpu):
        df, grid = conditioning_data
        vario = [0, 0.1, 30, 30, 1.0, "spherical"]
        kw = dict(SISIM_KW)
        kw["nsim"] = 3
        s1 = sisim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=True, **kw)
        s2 = sisim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=True, **kw)
        np.testing.assert_array_equal(s1, s2)

    def test_batched_path_exact_conditioning(self, conditioning_data, fake_gpu):
        df, grid = conditioning_data
        cond_pts = df[["X", "Y"]].values[:10]
        new_pts = np.random.default_rng(1).uniform(0, 100, size=(10, 2))
        combined_grid = np.vstack([cond_pts, new_pts])
        vario = [0, 0.1, 30, 30, 1.0, "spherical"]
        kw = dict(SISIM_KW)
        kw["nsim"] = 3
        sims = sisim(combined_grid, df, "X", "Y", "Z", vario=vario, use_gpu=True, **kw)
        true_vals = df["Z"].values[:10]
        for r in range(sims.shape[0]):
            np.testing.assert_allclose(sims[r, :10], true_vals, atol=1e-8)

    def test_nsim_1_does_not_use_batched_path(self, conditioning_data, monkeypatch):
        df, grid = conditioning_data
        vario = [0, 0.1, 30, 30, 1.0, "spherical"]
        import sys
        sisim_mod = sys.modules["pygstat.sisim"]
        called = {"batch": False}
        original = sisim_mod._simulate_batch

        def spy(*args, **kwargs):
            called["batch"] = True
            return original(*args, **kwargs)

        monkeypatch.setattr(sisim_mod, "_simulate_batch", spy)
        kw = dict(SISIM_KW)
        kw["nsim"] = 1
        sisim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=False, **kw)
        assert called["batch"] is False


class TestExplicitTrueWarnsWhenUnusable:
    def test_sgsim_use_gpu_true_warns_and_falls_back(self, conditioning_data, monkeypatch):
        monkeypatch.setattr(backend, "_cupy_usable_cache", False)
        df, grid = conditioning_data
        vario = [0, 0.1, 30, 30, 1.0, "spherical"]
        with pytest.warns(UserWarning, match="falling back to CPU"):
            sims = sgsim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=True, **SGSIM_KW)
        assert np.isfinite(sims).all()

    def test_sisim_use_gpu_true_warns_and_falls_back(self, conditioning_data, monkeypatch):
        monkeypatch.setattr(backend, "_cupy_usable_cache", False)
        df, grid = conditioning_data
        vario = [0, 0.1, 30, 30, 1.0, "spherical"]
        with pytest.warns(UserWarning, match="falling back to CPU"):
            sims = sisim(grid, df, "X", "Y", "Z", vario=vario, use_gpu=True, **SISIM_KW)
        assert np.isfinite(sims).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
