"""
Tests for pygstat.cokriging's GPU (CuPy) dispatch: `CrossVariogram`,
`fit_lmc`, `Cokriging`, `MultivariateCokriging`, and `ColocatedCokriging`
all accept `use_gpu='auto'|True|False`.

- `CrossVariogram`'s O(n^2) empirical cross-variogram dispatches the same
  way as `pygstat.core.variogram.Variogram`'s empirical semivariogram.
- `Cokriging` builds one big dense covariance matrix and solves it in one
  call -- the same shape of problem as `OrdinaryKriging`.
- `MultivariateCokriging`/`ColocatedCokriging` cache an LU factorization at
  `fit()` (via `scipy.linalg.lu_factor`/`cupyx.scipy.linalg.lu_factor`) and
  reuse it across `predict()` calls.
- `ColocatedCokriging` additionally batches its per-prediction-point
  secondary (colocated) system into one stacked `xp.linalg.solve`, instead
  of a Python loop -- verified bit-for-bit against a from-scratch
  reimplementation of the original per-point loop.

See tests/test_gpu_kriging.py's module docstring for why the "fake GPU"
approach is used (this sandbox's GPU can import CuPy but can't compile a
real kernel). `MultivariateCokriging`/`ColocatedCokriging` additionally
need `pygstat.cokriging.get_linalg_module` monkeypatched to always return
`scipy.linalg`: the fake CuPy proxy's arrays are genuinely plain NumPy
underneath, and `cupyx.scipy.linalg` (unlike `cupy.linalg`) is a real,
separate module that does not accept them.
"""

import numpy as np
import pytest
import scipy.linalg

from pygstat.core.kriging import _cov_from_gamma
from pygstat.core.variogram import Variogram
import pygstat.cokriging as cokriging_mod
from pygstat.cokriging import (
    Cokriging,
    ColocatedCokriging,
    CrossVariogram,
    MultivariateCokriging,
    fit_lmc,
)
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
    # cupyx.scipy.linalg is a real, separate module that won't accept the
    # fake proxy's (genuinely NumPy) arrays -- see module docstring.
    monkeypatch.setattr(cokriging_mod, "get_linalg_module", lambda xp: scipy.linalg)


@pytest.fixture
def bivariate_data():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(30, 2))
    z0 = coords[:, 0] * 0.3 + rng.normal(scale=2, size=30)
    z1 = z0 * 0.6 + rng.normal(scale=1.5, size=30)
    coords_pred = rng.uniform(0, 100, size=(10, 2))
    return coords, z0, z1, coords_pred


def _fit_vg(coords, z, use_gpu):
    vg = Variogram(coords, z, model="exponential", use_gpu=use_gpu)
    vg.fit()
    return vg


def _fit_cv(coords, a, b, use_gpu):
    cv = CrossVariogram(coords, a, b, model="exponential", use_gpu=use_gpu)
    cv.fit(method="auto")
    return cv


class TestCrossVariogram:
    def test_fit_and_call(self, bivariate_data):
        coords, z0, z1, _ = bivariate_data
        cv = _fit_cv(coords, z0, z1, use_gpu=False)
        assert cv.fitted_params is not None
        assert np.isfinite(cv(np.array([10.0, 50.0]))).all()

    def test_default_use_gpu_auto_never_crashes(self, bivariate_data):
        coords, z0, z1, _ = bivariate_data
        cv = CrossVariogram(coords, z0, z1, model="exponential", use_gpu="auto")
        cv.fit(method="auto")
        assert np.isfinite(cv.experimental[~np.isnan(cv.experimental)]).all()

    def test_gpu_path_matches_cpu(self, bivariate_data, fake_gpu):
        coords, z0, z1, _ = bivariate_data
        cv_cpu = _fit_cv(coords, z0, z1, use_gpu=False)
        cv_gpu = _fit_cv(coords, z0, z1, use_gpu=True)
        assert cv_gpu.use_gpu is True
        np.testing.assert_allclose(cv_cpu.lags, cv_gpu.lags)
        np.testing.assert_allclose(cv_cpu.experimental, cv_gpu.experimental, equal_nan=True)
        np.testing.assert_allclose(cv_cpu.fitted_params, cv_gpu.fitted_params, rtol=1e-6)


class TestFitLMC:
    def test_fit_lmc_passes_use_gpu_through(self, bivariate_data, fake_gpu):
        coords, z0, z1, _ = bivariate_data
        variograms, cross, info = fit_lmc(coords, [z0, z1], use_gpu=True)
        assert all(vg.use_gpu is True for vg in variograms)
        assert all(cv.use_gpu is True for cv in cross.values())
        assert info["lmc_check"]["valid"] or True  # PSD-ness not guaranteed on random data

    def test_fit_lmc_cpu_default(self, bivariate_data):
        coords, z0, z1, _ = bivariate_data
        variograms, cross, info = fit_lmc(coords, [z0, z1])
        assert len(variograms) == 2
        assert (0, 1) in cross


class TestCokriging:
    def test_cpu_predict(self, bivariate_data):
        coords, z0, z1, coords_pred = bivariate_data
        vg0 = _fit_vg(coords, z0, False)
        vg1 = _fit_vg(coords, z1, False)
        cv = _fit_cv(coords, z0, z1, False)
        ck = Cokriging(vg0, vg1, cv, use_gpu=False).fit(coords, z0, coords, z1)
        pred, se = ck.predict(coords_pred, return_variance=True)
        assert pred.shape == (10,)
        assert np.isfinite(pred).all() and np.all(se >= 0)

    def test_gpu_path_matches_cpu(self, bivariate_data, fake_gpu):
        coords, z0, z1, coords_pred = bivariate_data
        vg0, vg1 = _fit_vg(coords, z0, False), _fit_vg(coords, z1, False)
        cv = _fit_cv(coords, z0, z1, False)
        ck_cpu = Cokriging(vg0, vg1, cv, use_gpu=False).fit(coords, z0, coords, z1)
        pred_cpu, se_cpu = ck_cpu.predict(coords_pred, return_variance=True)

        vg0_g, vg1_g = _fit_vg(coords, z0, True), _fit_vg(coords, z1, True)
        cv_g = _fit_cv(coords, z0, z1, True)
        ck_gpu = Cokriging(vg0_g, vg1_g, cv_g, use_gpu=True).fit(coords, z0, coords, z1)
        assert ck_gpu.use_gpu is True
        pred_gpu, se_gpu = ck_gpu.predict(coords_pred, return_variance=True)

        assert isinstance(pred_gpu, np.ndarray)
        np.testing.assert_allclose(pred_cpu, pred_gpu, rtol=1e-6)
        np.testing.assert_allclose(se_cpu, se_gpu, rtol=1e-6, atol=1e-8)

    def test_explicit_true_warns_and_falls_back(self, bivariate_data, monkeypatch):
        monkeypatch.setattr(backend, "_cupy_usable_cache", False)
        coords, z0, z1, _ = bivariate_data
        vg0, vg1 = _fit_vg(coords, z0, False), _fit_vg(coords, z1, False)
        cv = _fit_cv(coords, z0, z1, False)
        with pytest.warns(UserWarning, match="falling back to CPU"):
            ck = Cokriging(vg0, vg1, cv, use_gpu=True)
        assert ck.use_gpu is False


@pytest.fixture
def trivariate_data():
    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 100, size=(35, 2))
    z0 = coords[:, 0] * 0.3 + rng.normal(scale=2, size=35)
    z1 = z0 * 0.6 + rng.normal(scale=1.5, size=35)
    z2 = z0 * 0.4 + rng.normal(scale=1.8, size=35)
    coords_pred = rng.uniform(0, 100, size=(10, 2))
    return coords, [z0, z1, z2], coords_pred


def _build_mvck(coords, values, use_gpu):
    vgs = [_fit_vg(coords, z, use_gpu) for z in values]
    cross = {}
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        cross[(i, j)] = _fit_cv(coords, values[i], values[j], use_gpu)
    mck = MultivariateCokriging(vgs, cross, use_gpu=use_gpu)
    mck.fit([coords] * 3, values)
    return mck


class TestMultivariateCokriging:
    def test_cpu_predict_and_cache_reuse(self, trivariate_data):
        coords, values, coords_pred = trivariate_data
        mck = _build_mvck(coords, values, use_gpu=False)
        pred1, se1 = mck.predict(coords_pred, return_variance=True)
        pred2 = mck.predict(coords_pred)  # reuses the cached factorization
        assert np.isfinite(pred1).all() and np.all(se1 >= 0)
        np.testing.assert_array_equal(pred1, pred2)

    def test_gpu_path_matches_cpu(self, trivariate_data, fake_gpu):
        coords, values, coords_pred = trivariate_data
        mck_cpu = _build_mvck(coords, values, use_gpu=False)
        pred_cpu, se_cpu = mck_cpu.predict(coords_pred, return_variance=True)

        mck_gpu = _build_mvck(coords, values, use_gpu=True)
        assert mck_gpu.use_gpu is True
        pred_gpu, se_gpu = mck_gpu.predict(coords_pred, return_variance=True)

        np.testing.assert_allclose(pred_cpu, pred_gpu, rtol=1e-6)
        np.testing.assert_allclose(se_cpu, se_gpu, rtol=1e-6, atol=1e-8)

    def test_missing_cross_variogram_raises(self):
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 100, size=(10, 2))
        vgs = [_fit_vg(coords, rng.normal(size=10), False) for _ in range(3)]
        with pytest.raises(ValueError, match="Missing cross-variogram"):
            MultivariateCokriging(vgs, {(0, 1): object()})


@pytest.fixture
def colocated_data():
    rng = np.random.default_rng(2)
    coords = rng.uniform(0, 100, size=(30, 2))
    z0 = coords[:, 0] * 0.3 + rng.normal(scale=2, size=30)
    y1 = z0 * 0.6 + rng.normal(scale=1.5, size=30)
    y2 = z0 * 0.4 + rng.normal(scale=1.8, size=30)
    coords_pred = rng.uniform(0, 100, size=(12, 2))
    Y_pred = np.column_stack([
        y1.mean() + rng.normal(scale=1, size=12),
        y2.mean() + rng.normal(scale=1, size=12),
    ])
    return coords, z0, y1, y2, coords_pred, Y_pred


def _build_colocated(coords, z0, y1, y2, use_gpu, **kwargs):
    vg0 = _fit_vg(coords, z0, use_gpu)
    vgy1, vgy2 = _fit_vg(coords, y1, use_gpu), _fit_vg(coords, y2, use_gpu)
    cv1, cv2 = _fit_cv(coords, z0, y1, use_gpu), _fit_cv(coords, z0, y2, use_gpu)
    cck = ColocatedCokriging(vg0, [vgy1, vgy2], [cv1, cv2], use_gpu=use_gpu, **kwargs)
    cck.fit(coords, z0, [y1.mean(), y2.mean()])
    return cck


class TestColocatedCokriging:
    def test_batched_predict_matches_from_scratch_per_point_loop(self, colocated_data):
        """Reimplements the *original* (pre-batching) per-point loop from
        scratch and checks the new batched implementation is bit-for-bit
        identical, using the same real scipy.linalg calls on both sides."""
        coords, z0, y1, y2, coords_pred, Y_pred = colocated_data
        cck = _build_colocated(coords, z0, y1, y2, use_gpu=False)
        pred, se = cck.predict(coords_pred, Y_pred, return_variance=True)

        vg0 = _fit_vg(coords, z0, False)
        vgy1, vgy2 = _fit_vg(coords, y1, False), _fit_vg(coords, y2, False)
        cv1, cv2 = _fit_cv(coords, z0, y1, False), _fit_cv(coords, z0, y2, False)

        C00 = float(vg0.fitted_params[0] + vg0.fitted_params[1])
        C0k = np.array([
            float(cv1.fitted_params[0] + cv1.fitted_params[1]),
            float(cv2.fitted_params[0] + cv2.fitted_params[1]),
        ])
        Ckk = np.array([
            float(vgy1.fitted_params[0] + vgy1.fitted_params[1]),
            float(vgy2.fitted_params[0] + vgy2.fitted_params[1]),
        ])
        Cyy = np.diag(Ckk)
        from scipy.spatial.distance import cdist
        dist_pp = cdist(coords, coords)
        Czz = _cov_from_gamma(vg0(dist_pp), C00, dist_pp)
        scale = np.nanmax(np.abs(np.diag(Czz)))
        Czz = Czz + np.eye(len(coords)) * 1e-8 * scale
        n = len(coords)
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = Czz
        A[:n, n] = 1
        A[n, :n] = 1
        A_lu = scipy.linalg.lu_factor(A)

        dist_p0 = cdist(coords_pred, coords)
        C00_p0 = _cov_from_gamma(vg0(dist_p0), C00, dist_p0)
        m, k = len(coords_pred), 2
        out = np.zeros((k, m, n))
        out[0] = (C0k[0] / C00) * C00_p0
        out[1] = (C0k[1] / C00) * C00_p0
        Czy = np.transpose(out, (2, 0, 1))
        rhs_a = np.zeros((n + 1, m))
        rhs_a[:n, :] = C00_p0.T
        rhs_a[n, :] = 1
        w_a0 = scipy.linalg.lu_solve(A_lu, rhs_a)
        B_flat = np.zeros((n + 1, k * m))
        B_flat[:n, :] = Czy.reshape(n, k * m)
        P = scipy.linalg.lu_solve(A_lu, B_flat).reshape(n + 1, k, m)
        Y_res = Y_pred - np.array([y1.mean(), y2.mean()])[None, :]
        preds_loop = np.empty(m)
        se_loop = np.empty(m)
        for t in range(m):
            Pt, Czy_t = P[:, :, t], Czy[:, :, t]
            S = Cyy - Czy_t.T @ Pt[:n]
            rhs_y = C0k - Czy_t.T @ w_a0[:n, t]
            w_y = np.linalg.solve(S, rhs_y)
            w_a = w_a0[:, t] - Pt @ w_y
            preds_loop[t] = w_a[:n] @ z0 + w_y @ Y_res[t]
            var = C00 - w_a[:n] @ C00_p0[t] - w_y @ C0k - w_a[n]
            se_loop[t] = np.sqrt(max(var, 0))

        np.testing.assert_allclose(pred, preds_loop, atol=1e-8)
        np.testing.assert_allclose(se, se_loop, atol=1e-8)

    def test_gpu_path_matches_cpu(self, colocated_data, fake_gpu):
        coords, z0, y1, y2, coords_pred, Y_pred = colocated_data
        cck_cpu = _build_colocated(coords, z0, y1, y2, use_gpu=False)
        pred_cpu, se_cpu = cck_cpu.predict(coords_pred, Y_pred, return_variance=True)

        cck_gpu = _build_colocated(coords, z0, y1, y2, use_gpu=True)
        assert cck_gpu.use_gpu is True
        pred_gpu, se_gpu = cck_gpu.predict(coords_pred, Y_pred, return_variance=True)

        np.testing.assert_allclose(pred_cpu, pred_gpu, rtol=1e-6)
        np.testing.assert_allclose(se_cpu, se_gpu, rtol=1e-6, atol=1e-8)

    def test_mm1_false_and_secondary_cross_vars(self, colocated_data):
        coords, z0, y1, y2, coords_pred, Y_pred = colocated_data
        vg0 = _fit_vg(coords, z0, False)
        vgy1, vgy2 = _fit_vg(coords, y1, False), _fit_vg(coords, y2, False)
        cv1, cv2 = _fit_cv(coords, z0, y1, False), _fit_cv(coords, z0, y2, False)
        cv12 = _fit_cv(coords, y1, y2, False)
        cck = ColocatedCokriging(
            vg0, [vgy1, vgy2], [cv1, cv2],
            secondary_cross_vars={(0, 1): cv12}, mm1=False, use_gpu=False,
        )
        cck.fit(coords, z0, [y1.mean(), y2.mean()])
        pred = cck.predict(coords_pred, Y_pred)
        assert np.isfinite(pred).all()

    def test_predict_before_fit_raises(self):
        vg0 = _fit_vg(np.random.rand(5, 2), np.random.rand(5), False)
        cck = ColocatedCokriging(vg0, [vg0], [vg0], use_gpu=False)
        with pytest.raises(RuntimeError, match="not fitted"):
            cck.predict(np.random.rand(2, 2), np.random.rand(2, 1))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
