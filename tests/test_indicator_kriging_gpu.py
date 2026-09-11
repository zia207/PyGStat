"""
Tests for pygstat.indicator_kriging's batched/GPU-capable prediction path.

Every prediction point is independent (unlike pygstat.sgsim/sisim's
sequential simulators), so the default k-nearest-neighbor mode batches all
`n_pred` points' kriging systems into one stacked `xp.linalg.solve` call
instead of looping in Python -- a real speedup on CPU (no Python-level
loop) and what makes `use_gpu` genuinely useful here. `search_radius` mode
(variable neighbor counts) isn't batchable this way and keeps the original
per-point loop.

See tests/test_gpu_kriging.py's module docstring for why the "fake GPU"
approach (monkeypatching `cupy_gpu_usable()`/`cp` rather than requiring
real CUDA) is used here: this sandbox's GPU can import CuPy but can't
actually compile a kernel (confirmed directly), so the GPU *code path* is
exercised via a NumPy-backed CuPy stand-in instead.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pygstat.indicator_kriging import (
    IndicatorKriging,
    _indicator_transform,
    _solve_ok_batch,
    _solve_ordinary_kriging_system,
)
from pygstat.utils import backend


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(60, 2))
    values = coords[:, 0] * 0.2 + coords[:, 1] * 0.1 + rng.normal(scale=3, size=60)
    coords_pred = rng.uniform(0, 100, size=(15, 2))
    return coords, values, coords_pred


class _FakeCupyModule:
    """See tests/test_gpu_kriging.py for the full rationale. A NumPy-backed
    stand-in for `cupy`: `ndarray` is `numpy.ndarray`, `asnumpy` (a real
    CuPy name with no NumPy equivalent) is a no-op, everything else
    delegates to `numpy`."""

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


class TestBatchedMatchesOriginalPerPointLoop:
    """The batched k-NN path must reproduce the original per-point-loop
    semantics exactly (same k, same neighbors, same regularization)."""

    def test_batched_result_matches_manual_per_point_loop(self, data):
        coords, values, coords_pred = data
        ik = IndicatorKriging(max_neighbors=8, use_gpu=False)
        ik.fit(coords, values)
        threshold = float(np.median(values))
        probs_batched = ik.predict(coords_pred, [threshold])[threshold]

        indicator_train = _indicator_transform(values, threshold)
        max_dist = np.percentile(
            np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)), 50
        )
        nugget, sill, rng_ = 0.01, 0.25, max_dist * 0.6
        k = min(8, len(coords))
        tree = cKDTree(coords)
        probs_manual = np.full(len(coords_pred), np.nan)
        for i, target in enumerate(coords_pred):
            _, idx = tree.query(target, k=k)
            w = _solve_ordinary_kriging_system(coords[idx], target, ik.cov_model, sill, rng_, nugget, 1e-8)
            probs_manual[i] = np.clip(np.dot(w, indicator_train[idx]), 0, 1)

        np.testing.assert_allclose(probs_batched, probs_manual, atol=1e-10)

    def test_multiple_thresholds_and_probability_bounds(self, data):
        coords, values, coords_pred = data
        ik = IndicatorKriging(max_neighbors=10, use_gpu=False)
        ik.fit(coords, values)
        thresholds = [float(np.percentile(values, p)) for p in (25, 50, 75)]
        probs = ik.predict(coords_pred, thresholds)
        assert set(probs.keys()) == set(thresholds)
        for t in thresholds:
            assert probs[t].shape == (len(coords_pred),)
            assert np.all((probs[t] >= 0) & (probs[t] <= 1))

    def test_search_radius_mode_still_works(self, data):
        """The non-batchable fallback path (variable neighbor counts)."""
        coords, values, coords_pred = data
        ik = IndicatorKriging(max_neighbors=8, search_radius=30.0, use_gpu=False)
        ik.fit(coords, values)
        threshold = float(np.median(values))
        probs = ik.predict(coords_pred, [threshold])[threshold]
        assert probs.shape == (len(coords_pred),)


class TestGPUDispatch:
    def test_use_gpu_auto_never_crashes(self, data):
        coords, values, coords_pred = data
        ik = IndicatorKriging(max_neighbors=8, use_gpu="auto")
        ik.fit(coords, values)
        probs = ik.predict(coords_pred, [float(np.median(values))])
        assert np.isfinite(next(iter(probs.values()))).all()

    def test_explicit_true_warns_and_falls_back_when_unusable(self, monkeypatch):
        monkeypatch.setattr(backend, "_cupy_usable_cache", False)
        with pytest.warns(UserWarning, match="falling back to CPU"):
            ik = IndicatorKriging(use_gpu=True)
        assert ik.use_gpu is False

    def test_solve_ok_batch_gpu_path_matches_cpu(self, data, fake_gpu):
        coords, values, coords_pred = data
        from pygstat.indicator_kriging import _spherical_covariance

        k = 8
        tree = cKDTree(coords)
        _, idx = tree.query(coords_pred, k=k)
        neighbor_coords = coords[idx]

        weights_cpu = _solve_ok_batch(
            neighbor_coords, coords_pred, _spherical_covariance,
            sill=0.25, rng=40.0, nugget=0.01, regularization=1e-8, use_gpu=False,
        )
        weights_gpu = _solve_ok_batch(
            neighbor_coords, coords_pred, _spherical_covariance,
            sill=0.25, rng=40.0, nugget=0.01, regularization=1e-8, use_gpu=True,
        )
        assert isinstance(weights_gpu, np.ndarray)
        np.testing.assert_allclose(weights_cpu, weights_gpu, rtol=1e-8)

    def test_full_predict_gpu_path_matches_cpu(self, data, fake_gpu):
        coords, values, coords_pred = data
        threshold = float(np.median(values))

        ik_cpu = IndicatorKriging(max_neighbors=8, use_gpu=False)
        ik_cpu.fit(coords, values)
        probs_cpu = ik_cpu.predict(coords_pred, [threshold])[threshold]

        ik_gpu = IndicatorKriging(max_neighbors=8, use_gpu=True)
        assert ik_gpu.use_gpu is True
        ik_gpu.fit(coords, values)
        probs_gpu = ik_gpu.predict(coords_pred, [threshold])[threshold]

        np.testing.assert_allclose(probs_cpu, probs_gpu, rtol=1e-8)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
