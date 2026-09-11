"""
Tests for pygstat.krigeST / krige_st's batched/GPU-capable local path.

Default k-nearest-neighbor ST kriging (finite ``nmax``) batches all
``n_pred`` points' ordinary-kriging systems into one stacked
``xp.linalg.solve`` call. Global kriging (``nmax=inf``) is unchanged
and always runs on CPU.

See tests/test_gpu_kriging.py for why the "fake GPU" approach
(monkeypatching ``cupy_gpu_usable()`` / ``cp``) is used here.
"""

import numpy as np
import pytest

from pygstat.krigeST import STKriging
from pygstat.krige_st import krige_st_local
from pygstat.st_variogram_models import vgm, vgm_st
from pygstat.utils import backend


def _metric_model(psill=1.0, range_=50.0, nugget=0.1, st_ani=10.0):
    return vgm_st(
        "metric",
        joint=vgm(psill=psill, model="exponential", range_=range_, nugget=nugget),
        st_ani=st_ani,
    )


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n, m = 40, 8
    coords = rng.uniform(0, 100, size=(n, 2))
    times = rng.uniform(0, 12, size=n)
    values = coords[:, 0] * 0.05 + times * 0.2 + rng.normal(scale=1.0, size=n)
    coords_pred = rng.uniform(0, 100, size=(m, 2))
    times_pred = rng.uniform(0, 12, size=m)
    model = _metric_model()
    return coords, times, values, coords_pred, times_pred, model


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
    def test_batched_result_matches_manual_per_point_loop(self, data, monkeypatch):
        coords, times, values, coords_pred, times_pred, model = data
        nmax = 8

        stk = STKriging(model, use_gpu=False).fit(coords, times, values)
        z_batch, sig_batch = stk.predict(coords_pred, times_pred, nmax=nmax)

        import importlib

        krige_st_mod = importlib.import_module("pygstat.krige_st")

        def _boom(*_a, **_k):
            raise RuntimeError("force per-point loop")

        monkeypatch.setattr(krige_st_mod, "_solve_st_ok_batch", _boom)
        z_loop, sig_loop = stk.predict(coords_pred, times_pred, nmax=nmax)

        np.testing.assert_allclose(z_batch, z_loop, atol=1e-8)
        np.testing.assert_allclose(sig_batch, sig_loop, atol=1e-8)

    def test_krige_st_local_returns_finite(self, data):
        coords, times, values, coords_pred, times_pred, model = data
        out = krige_st_local(
            coords, times, values, coords_pred, times_pred, model,
            nmax=8, compute_var=True, progress=False, use_gpu=False,
        )
        assert np.isfinite(out["var1.pred"]).all()
        assert np.isfinite(out["var1.var"]).all()
        assert out["var1.pred"].shape == (len(coords_pred),)


class TestGPUDispatch:
    def test_use_gpu_auto_never_crashes(self, data):
        coords, times, values, coords_pred, times_pred, model = data
        stk = STKriging(model, use_gpu="auto").fit(coords, times, values)
        zhat, sig = stk.predict(coords_pred, times_pred, nmax=8)
        assert np.isfinite(zhat).all()
        assert np.isfinite(sig).all()

    def test_explicit_true_warns_and_falls_back_when_unusable(self, monkeypatch):
        monkeypatch.setattr(backend, "_cupy_usable_cache", False)
        model = _metric_model()
        with pytest.warns(UserWarning, match="falling back to CPU"):
            stk = STKriging(model, use_gpu=True)
        assert stk.use_gpu is False

    def test_full_predict_gpu_path_matches_cpu(self, data, fake_gpu):
        coords, times, values, coords_pred, times_pred, model = data

        stk_cpu = STKriging(model, use_gpu=False).fit(coords, times, values)
        stk_gpu = STKriging(model, use_gpu=True).fit(coords, times, values)
        assert stk_gpu.use_gpu is True

        z_cpu, sig_cpu = stk_cpu.predict(coords_pred, times_pred, nmax=8)
        z_gpu, sig_gpu = stk_gpu.predict(coords_pred, times_pred, nmax=8)
        np.testing.assert_allclose(z_cpu, z_gpu, rtol=1e-8)
        np.testing.assert_allclose(sig_cpu, sig_gpu, rtol=1e-8)

        zc, _ = stk_cpu.predict(coords_pred, times_pred, nmax=8, compute_var=False)
        zg, _ = stk_gpu.predict(coords_pred, times_pred, nmax=8, compute_var=False)
        np.testing.assert_allclose(zc, zg, rtol=1e-8)
