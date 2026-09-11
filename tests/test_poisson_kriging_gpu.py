"""
Tests for pygstat.poisson_kriging's batched/GPU-capable centroid prediction.

Every prediction point is independent, so PoissonKriging.predict batches all
n_pred local (k+1)×(k+1) systems into one stacked xp.linalg.solve — a real
speedup on CPU (no Python-level loop) and what makes use_gpu useful here.
predict_id (leave-one-out of a single area) and AreaPoissonKriging keep the
original per-target loop.

See tests/test_gpu_kriging.py's module docstring for why the "fake GPU"
approach (monkeypatching cupy_gpu_usable()/cp rather than requiring real
CUDA) is used here.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pygstat import PoissonKriging, Variogram
from pygstat.utils import backend


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(60, 2))
    values = 8.0 + 0.02 * coords[:, 0] + rng.normal(scale=0.4, size=60)
    pops = rng.uniform(5_000, 80_000, size=60)
    coords_pred = rng.uniform(0, 100, size=(15, 2))
    vg = Variogram(coords, values, model="exponential", n_lags=8, use_gpu=False)
    vg.fit()
    return coords, values, pops, coords_pred, vg


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


class TestBatchedMatchesOriginalPerPointLoop:
    def test_batched_result_matches_manual_per_point_loop(self, data):
        coords, values, pops, coords_pred, vg = data
        pk = PoissonKriging(vg, rate_base=100, use_gpu=False)
        pk.fit(coords, values, pops)
        zhat_b, sig_b = pk.predict(coords_pred, number_of_neighbors=8)

        k = 8
        tree = cKDTree(coords)
        _, idx = tree.query(coords_pred, k=k)
        zhat_m = np.empty(len(coords_pred))
        sig_m = np.empty(len(coords_pred))
        for i in range(len(coords_pred)):
            zhat_m[i], sig_m[i] = pk._solve(
                idx[i], coords_pred[i], False, False
            )

        np.testing.assert_allclose(zhat_b, zhat_m, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(sig_b, sig_m, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_predict_shapes_and_finite(self, data):
        coords, values, pops, coords_pred, vg = data
        pk = PoissonKriging(vg, rate_base=100, use_gpu=False)
        pk.fit(coords, values, pops)
        zhat, sig = pk.predict(coords_pred, number_of_neighbors=10)
        assert zhat.shape == (len(coords_pred),)
        assert sig.shape == (len(coords_pred),)
        assert np.isfinite(zhat).all()

    def test_too_many_neighbors_raises(self, data):
        coords, values, pops, coords_pred, vg = data
        pk = PoissonKriging(vg, rate_base=100, use_gpu=False)
        pk.fit(coords, values, pops)
        with pytest.raises(ValueError, match="Not enough known areas"):
            pk.predict(coords_pred, number_of_neighbors=len(coords) + 1)


class TestGPUDispatch:
    def test_use_gpu_auto_never_crashes(self, data):
        coords, values, pops, coords_pred, vg = data
        pk = PoissonKriging(vg, rate_base=100, use_gpu="auto")
        pk.fit(coords, values, pops)
        zhat, sig = pk.predict(coords_pred, number_of_neighbors=8)
        assert np.isfinite(zhat).all()

    def test_explicit_true_warns_and_falls_back_when_unusable(self, data, monkeypatch):
        monkeypatch.setattr(backend, "_cupy_usable_cache", False)
        _, _, _, _, vg = data
        with pytest.warns(UserWarning, match="falling back to CPU"):
            pk = PoissonKriging(vg, use_gpu=True)
        assert pk.use_gpu is False

    def test_full_predict_gpu_path_matches_cpu(self, data, fake_gpu):
        coords, values, pops, coords_pred, vg = data

        pk_cpu = PoissonKriging(vg, rate_base=100, use_gpu=False)
        pk_cpu.fit(coords, values, pops)
        z_cpu, s_cpu = pk_cpu.predict(coords_pred, number_of_neighbors=8)

        pk_gpu = PoissonKriging(vg, rate_base=100, use_gpu=True)
        assert pk_gpu.use_gpu is True
        pk_gpu.fit(coords, values, pops)
        z_gpu, s_gpu = pk_gpu.predict(coords_pred, number_of_neighbors=8)

        np.testing.assert_allclose(z_cpu, z_gpu, rtol=1e-8)
        np.testing.assert_allclose(s_cpu, s_gpu, rtol=1e-8, equal_nan=True)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
