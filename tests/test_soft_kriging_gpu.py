"""
Tests for pygstat.soft_kriging's batched/GPU-capable prediction path.

k-nearest-neighbor Markov-Bayes systems share a neighbor count, so they
batch into one stacked ``xp.linalg.solve``. ``search_radius`` mode keeps
the original per-point loop.

See tests/test_gpu_kriging.py for the fake-GPU (NumPy-backed CuPy stand-in)
rationale.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pygstat.soft_kriging import (
    SoftKriging,
    _solve_markov_bayes_batch,
    _solve_markov_bayes_system,
    _exponential_covariance,
)
from pygstat.utils import backend


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    hard_coords = rng.uniform(0, 10, size=(40, 2))
    hard_ind = (hard_coords[:, 0] + rng.normal(scale=1, size=40) > 5).astype(float)
    soft_coords = rng.uniform(0, 10, size=(80, 2))
    soft_prob = np.clip(0.3 + 0.05 * soft_coords[:, 0] + rng.normal(scale=0.1, size=80), 0, 1)
    coords_pred = rng.uniform(0, 10, size=(20, 2))
    return hard_coords, hard_ind, soft_coords, soft_prob, coords_pred


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
    def test_batched_result_matches_manual_loop(self, data):
        hard_coords, hard_ind, soft_coords, soft_prob, coords_pred = data
        B, sill, rng_, nugget = 0.6, 0.20, 3.0, 0.05
        sk = SoftKriging(
            cov_model="exponential",
            max_neighbors_hard=8,
            max_neighbors_soft=10,
            search_radius=None,
            use_gpu=False,
        )
        sk.fit(hard_coords, hard_ind, soft_coords, soft_prob, B=B)
        prob_b, var_b = sk.predict(
            coords_pred, sill=sill, range_=rng_, nugget=nugget, return_variance=True,
        )

        k_h, k_s = 8, 10
        htree, stree = cKDTree(hard_coords), cKDTree(soft_coords)
        prob_m = np.empty(len(coords_pred))
        var_m = np.empty(len(coords_pred))
        c0 = sill + nugget
        for i, target in enumerate(coords_pred):
            _, hi = htree.query(target, k=k_h)
            _, si = stree.query(target, k=k_s)
            w, c_vec, mu = _solve_markov_bayes_system(
                hard_coords[hi], soft_coords[si], target,
                sk.cov_model, sill, rng_, nugget, B, sk.regularization,
            )
            vals = np.concatenate([hard_ind[hi], soft_prob[si]])
            prob_m[i] = np.clip(np.dot(w, vals), 0, 1)
            var_m[i] = max(c0 - np.dot(w, c_vec) - mu, 0.0)

        np.testing.assert_allclose(prob_b, prob_m, atol=1e-10)
        np.testing.assert_allclose(var_b, var_m, atol=1e-10)

    def test_hard_only_still_works(self, data):
        hard_coords, hard_ind, soft_coords, soft_prob, coords_pred = data
        sk = SoftKriging(
            max_neighbors_hard=10, max_neighbors_soft=0, use_gpu=False,
        )
        sk.fit(hard_coords, hard_ind, soft_coords, soft_prob, B=0.0)
        prob = sk.predict(coords_pred, sill=0.2, range_=3.0, nugget=0.05)
        assert prob.shape == (len(coords_pred),)
        assert np.all((prob >= 0) & (prob <= 1))

    def test_search_radius_mode_still_works(self, data):
        hard_coords, hard_ind, soft_coords, soft_prob, coords_pred = data
        sk = SoftKriging(
            max_neighbors_hard=8, max_neighbors_soft=8,
            search_radius=4.0, use_gpu=False,
        )
        sk.fit(hard_coords, hard_ind, soft_coords, soft_prob, B=0.5)
        prob = sk.predict(coords_pred, sill=0.2, range_=3.0, nugget=0.05)
        assert prob.shape == (len(coords_pred),)
        assert np.isfinite(prob).any()


class TestGPUDispatch:
    def test_use_gpu_auto_never_crashes(self, data):
        hard_coords, hard_ind, soft_coords, soft_prob, coords_pred = data
        sk = SoftKriging(max_neighbors_hard=8, max_neighbors_soft=8, use_gpu="auto")
        sk.fit(hard_coords, hard_ind, soft_coords, soft_prob, B=0.5)
        prob = sk.predict(coords_pred, sill=0.2, range_=3.0, nugget=0.05)
        assert np.isfinite(prob).all()

    def test_explicit_true_warns_and_falls_back_when_unusable(self, monkeypatch):
        monkeypatch.setattr(backend, "_cupy_usable_cache", False)
        with pytest.warns(UserWarning, match="falling back to CPU"):
            sk = SoftKriging(use_gpu=True)
        assert sk.use_gpu is False

    def test_batch_gpu_path_matches_cpu(self, data, fake_gpu):
        hard_coords, hard_ind, soft_coords, soft_prob, coords_pred = data
        k_h, k_s = 8, 10
        htree, stree = cKDTree(hard_coords), cKDTree(soft_coords)
        _, hi = htree.query(coords_pred, k=k_h)
        _, si = stree.query(coords_pred, k=k_s)
        kwargs = dict(
            cov_model=_exponential_covariance, sill=0.2, rng=3.0, nugget=0.05,
            B=0.6, regularization=1e-5,
        )
        w_cpu, c_cpu, mu_cpu = _solve_markov_bayes_batch(
            hard_coords[hi], soft_coords[si], coords_pred, use_gpu=False, **kwargs,
        )
        w_gpu, c_gpu, mu_gpu = _solve_markov_bayes_batch(
            hard_coords[hi], soft_coords[si], coords_pred, use_gpu=True, **kwargs,
        )
        np.testing.assert_allclose(w_cpu, w_gpu, rtol=1e-8)
        np.testing.assert_allclose(c_cpu, c_gpu, rtol=1e-8)
        np.testing.assert_allclose(mu_cpu, mu_gpu, rtol=1e-8)

    def test_full_predict_gpu_path_matches_cpu(self, data, fake_gpu):
        hard_coords, hard_ind, soft_coords, soft_prob, coords_pred = data
        kw = dict(max_neighbors_hard=8, max_neighbors_soft=10, search_radius=None)
        sk_cpu = SoftKriging(use_gpu=False, **kw)
        sk_cpu.fit(hard_coords, hard_ind, soft_coords, soft_prob, B=0.55)
        p_cpu, v_cpu = sk_cpu.predict(
            coords_pred, sill=0.2, range_=3.0, nugget=0.05, return_variance=True,
        )

        sk_gpu = SoftKriging(use_gpu=True, **kw)
        assert sk_gpu.use_gpu is True
        sk_gpu.fit(hard_coords, hard_ind, soft_coords, soft_prob, B=0.55)
        p_gpu, v_gpu = sk_gpu.predict(
            coords_pred, sill=0.2, range_=3.0, nugget=0.05, return_variance=True,
        )
        np.testing.assert_allclose(p_cpu, p_gpu, rtol=1e-8)
        np.testing.assert_allclose(v_cpu, v_gpu, rtol=1e-8)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
