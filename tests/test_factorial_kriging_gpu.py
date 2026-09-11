"""
Tests for pygstat.factorial_kriging's batched/GPU-capable extract path.

Default k-nearest-neighbor mode batches all `n_pred` points' kriging
systems into one stacked `xp.linalg.solve` call. `search_radius` mode
keeps the original per-point loop.

See tests/test_gpu_kriging.py for why the "fake GPU" approach
(monkeypatching `cupy_gpu_usable()`/`cp`) is used here.
"""

import numpy as np
import pytest

from pygstat.factorial_kriging import FactorialKriging
from pygstat.utils import backend


NUGGET = 0.5
STRUCTURES = [
    {"model": "spherical", "sill": 1.0, "range": 20.0},
    {"model": "spherical", "sill": 0.8, "range": 60.0},
]


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(60, 2))
    values = coords[:, 0] * 0.2 + coords[:, 1] * 0.1 + rng.normal(scale=3, size=60)
    coords_pred = rng.uniform(0, 100, size=(15, 2))
    return coords, values, coords_pred


class _FakeCupyModule:
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


class TestBatchedMatchesPerPointLoop:
    def test_total_matches_search_radius_when_all_neighbors_used(self, data):
        coords, values, coords_pred = data
        fk_knn = FactorialKriging(
            nugget=NUGGET, structures=STRUCTURES, max_neighbors=8, use_gpu=False
        )
        fk_knn.fit(coords, values)
        z_knn = fk_knn.extract(coords_pred, factor="total")

        fk_rad = FactorialKriging(
            nugget=NUGGET, structures=STRUCTURES, max_neighbors=8,
            search_radius=1e9, use_gpu=False,
        )
        fk_rad.fit(coords, values)
        z_rad = fk_rad.extract(coords_pred, factor="total")
        np.testing.assert_allclose(z_knn, z_rad, atol=1e-8)

    def test_search_radius_mode_still_works(self, data):
        coords, values, coords_pred = data
        fk = FactorialKriging(
            nugget=NUGGET, structures=STRUCTURES, max_neighbors=8,
            search_radius=30.0, use_gpu=False,
        )
        fk.fit(coords, values)
        z = fk.extract(coords_pred, factor=1)
        assert z.shape == (len(coords_pred),)


class TestGPUDispatch:
    def test_use_gpu_auto_never_crashes(self, data):
        coords, values, coords_pred = data
        fk = FactorialKriging(
            nugget=NUGGET, structures=STRUCTURES, max_neighbors=8, use_gpu="auto"
        )
        fk.fit(coords, values)
        z = fk.extract(coords_pred, factor="total")
        assert np.isfinite(z).all()

    def test_explicit_true_warns_and_falls_back_when_unusable(self, monkeypatch):
        monkeypatch.setattr(backend, "_cupy_usable_cache", False)
        with pytest.warns(UserWarning, match="falling back to CPU"):
            fk = FactorialKriging(
                nugget=NUGGET, structures=STRUCTURES, use_gpu=True
            )
        assert fk.use_gpu is False

    def test_full_extract_gpu_path_matches_cpu(self, data, fake_gpu):
        coords, values, coords_pred = data

        fk_cpu = FactorialKriging(
            nugget=NUGGET, structures=STRUCTURES, max_neighbors=8, use_gpu=False
        )
        fk_cpu.fit(coords, values)

        fk_gpu = FactorialKriging(
            nugget=NUGGET, structures=STRUCTURES, max_neighbors=8, use_gpu=True
        )
        assert fk_gpu.use_gpu is True
        fk_gpu.fit(coords, values)

        for factor in ("total", 1, 2, "denoised"):
            z_cpu = fk_cpu.extract(coords_pred, factor=factor)
            z_gpu = fk_gpu.extract(coords_pred, factor=factor)
            np.testing.assert_allclose(z_cpu, z_gpu, rtol=1e-8)

        zc, se_c = fk_cpu.extract(coords_pred, factor="total", return_variance=True)
        zg, se_g = fk_gpu.extract(coords_pred, factor="total", return_variance=True)
        np.testing.assert_allclose(zc, zg, rtol=1e-8)
        np.testing.assert_allclose(se_c, se_g, rtol=1e-8)
