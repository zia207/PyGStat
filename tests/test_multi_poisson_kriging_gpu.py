"""
Tests for pygstat.multi_poisson_kriging's batched/GPU k-NN prediction path.

The default global poisson_cokrige invert is unchanged. When
number_of_neighbors=k is set, every prediction point uses the same k
co-located neighbors and all n_pred systems are stacked into one
xp.linalg.solve — NumPy on CPU, CuPy on GPU.

See tests/test_gpu_kriging.py for why the fake-GPU monkeypatch is used.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pygstat.multi_poisson_kriging import (
    LMC,
    _cokrige_solve,
    poisson_cokrige,
)
from pygstat.utils import backend


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 40
    coords = rng.uniform(0, 100, size=(n, 2))
    pop = rng.uniform(5_000, 40_000, size=n)
    y0 = rng.poisson(pop * 0.08 / 100.0).astype(float)
    y1 = rng.poisson(pop * 0.25 / 100.0).astype(float)
    y2 = rng.poisson(pop * 0.30 / 100.0).astype(float)
    coords_pred = rng.uniform(0, 100, size=(12, 2))
    datasets = [
        dict(x=coords[:, 0], y=coords[:, 1], cases=y0, pop=pop),
        dict(x=coords[:, 0], y=coords[:, 1], cases=y1, pop=pop),
        dict(x=coords[:, 0], y=coords[:, 1], cases=y2, pop=pop),
    ]
    B = np.array([
        [1.2, 0.4, 0.3],
        [0.4, 0.9, 0.5],
        [0.3, 0.5, 1.1],
    ])
    lmc = LMC(B, rng=40.0, model="Exp")
    shared = {(0, 1): 0.02, (0, 2): 0.015, (1, 2): 0.04}
    return datasets, lmc, shared, coords_pred


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


class TestBatchedMatchesLocalLoop:
    def test_knn_batch_matches_global_solver_on_neighbors(self, data):
        datasets, lmc, shared, coords_pred = data
        k = 8
        pop_rate = 100.0
        res = poisson_cokrige(
            datasets, 0, lmc, pop_rate, shared, coords_pred=coords_pred,
            number_of_neighbors=k, use_gpu=False,
        )
        coords = np.column_stack([datasets[0]["x"], datasets[0]["y"]])
        tree = cKDTree(coords)
        _, idx = tree.query(coords_pred, k=k)
        pred_m = np.empty(len(coords_pred))
        var_m = np.empty(len(coords_pred))
        for i in range(len(coords_pred)):
            ii = idx[i]
            ds = [
                dict(
                    x=coords[ii, 0], y=coords[ii, 1],
                    cases=d["cases"][ii], pop=d["pop"][ii],
                )
                for d in datasets
            ]
            r = _cokrige_solve(ds, 0, lmc, pop_rate, shared, coords_pred[i:i + 1])
            pred_m[i], var_m[i] = r[0, 2], r[0, 3]
        np.testing.assert_allclose(res["pred"], pred_m, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(res["var"], var_m, rtol=1e-8, atol=1e-8)

    def test_global_path_still_works(self, data):
        datasets, lmc, shared, coords_pred = data
        res = poisson_cokrige(
            datasets, 0, lmc, 100.0, shared, coords_pred=coords_pred, use_gpu=False,
        )
        assert res["pred"].shape == (len(coords_pred),)
        assert np.isfinite(res["pred"]).all()

    def test_too_many_neighbors_raises(self, data):
        datasets, lmc, shared, coords_pred = data
        n = len(datasets[0]["x"])
        with pytest.raises(ValueError, match="Not enough known areas"):
            poisson_cokrige(
                datasets, 0, lmc, 100.0, shared, coords_pred=coords_pred,
                number_of_neighbors=n + 1, use_gpu=False,
            )


class TestGPUDispatch:
    def test_use_gpu_auto_never_crashes(self, data):
        datasets, lmc, shared, coords_pred = data
        res = poisson_cokrige(
            datasets, 0, lmc, 100.0, shared, coords_pred=coords_pred,
            number_of_neighbors=8, use_gpu="auto",
        )
        assert np.isfinite(res["pred"]).all()

    def test_explicit_true_warns_and_falls_back_when_unusable(self, data, monkeypatch):
        monkeypatch.setattr(backend, "_cupy_usable_cache", False)
        datasets, lmc, shared, coords_pred = data
        with pytest.warns(UserWarning, match="falling back to CPU"):
            poisson_cokrige(
                datasets, 0, lmc, 100.0, shared, coords_pred=coords_pred,
                number_of_neighbors=8, use_gpu=True,
            )

    def test_full_predict_gpu_path_matches_cpu(self, data, fake_gpu):
        datasets, lmc, shared, coords_pred = data
        cpu = poisson_cokrige(
            datasets, 0, lmc, 100.0, shared, coords_pred=coords_pred,
            number_of_neighbors=8, use_gpu=False,
        )
        gpu = poisson_cokrige(
            datasets, 0, lmc, 100.0, shared, coords_pred=coords_pred,
            number_of_neighbors=8, use_gpu=True,
        )
        np.testing.assert_allclose(cpu["pred"], gpu["pred"], rtol=1e-8)
        np.testing.assert_allclose(cpu["var"], gpu["var"], rtol=1e-8)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
