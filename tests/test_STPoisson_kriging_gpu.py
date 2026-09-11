"""
Tests for pygstat.STPoissonKriging's batched/GPU k-NN prediction path.

:meth:`STPoissonKriging.predict` stacks every local ``(k+1)×(k+1)`` system
into one ``xp.linalg.solve``. :meth:`STPoissonKriging.predict_id` keeps
the original per-target loop.

See tests/test_gpu_kriging.py for why the fake-GPU monkeypatch is used.
"""

import numpy as np
import pytest

from pygstat.STPoisson_kriging import STPoissonKriging
from pygstat.st_variogram_models import vgm, vgm_st
from pygstat.utils import backend


def _metric_model(psill=80.0, range_=50.0, nugget=20.0, st_ani=10.0):
    return vgm_st(
        "metric",
        joint=vgm(psill=psill, model="exponential", range_=range_, nugget=nugget),
        st_ani=st_ani,
    )


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n_space, n_time, m = 20, 6, 8
    coords_u = rng.uniform(0, 100, size=(n_space, 2))
    times_u = np.arange(n_time, dtype=float)
    coords, times, values, pops, ids = [], [], [], [], []
    for i in range(n_space):
        pop = float(rng.uniform(5_000, 40_000))
        for t in times_u:
            coords.append(coords_u[i])
            times.append(t)
            pops.append(pop)
            ids.append(i)
            rate = 50.0 + 0.1 * coords_u[i, 0] + 0.8 * t + rng.normal(scale=3.0)
            values.append(max(rate, 1.0))
    coords = np.asarray(coords)
    times = np.asarray(times)
    values = np.asarray(values)
    pops = np.asarray(pops)
    ids = np.asarray(ids)
    coords_pred = rng.uniform(0, 100, size=(m, 2))
    times_pred = rng.uniform(0, n_time - 1, size=m)
    model = _metric_model()
    return coords, times, values, pops, ids, coords_pred, times_pred, model


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
    def test_batched_result_matches_manual_per_point_loop(self, data):
        coords, times, values, pops, ids, coords_pred, times_pred, model = data
        k = 8
        stpk = STPoissonKriging(model, rate_base=100000.0, use_gpu=False).fit(
            coords, times, values, pops, ids=ids,
        )
        z_batch, sig_batch = stpk.predict(coords_pred, times_pred, number_of_neighbors=k)

        z_loop = np.empty(len(coords_pred))
        sig_loop = np.empty(len(coords_pred))
        for i in range(len(coords_pred)):
            idx = stpk._neighbors(coords_pred[i], times_pred[i], None, k)
            z_loop[i], sig_loop[i] = stpk._solve(
                idx, coords_pred[i], times_pred[i], False, False,
            )
        np.testing.assert_allclose(z_batch, z_loop, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(sig_batch, sig_loop, rtol=1e-8, atol=1e-8)

    def test_predict_id_still_uses_loop(self, data):
        coords, times, values, pops, ids, *_rest, model = data
        stpk = STPoissonKriging(model, rate_base=100000.0, use_gpu=False).fit(
            coords, times, values, pops, ids=ids,
        )
        out = stpk.predict_id(ids[0], times[0], number_of_neighbors=8)
        assert np.isfinite(out["zhat"])
        assert np.isfinite(out["sig"]) or np.isnan(out["sig"])

    def test_too_few_neighbors_raises(self, data):
        coords, times, values, pops, ids, coords_pred, times_pred, model = data
        stpk = STPoissonKriging(model, rate_base=100000.0, use_gpu=False).fit(
            coords, times, values, pops, ids=ids,
        )
        with pytest.raises(ValueError, match="Not enough space-time neighbors"):
            stpk.predict(coords_pred[:1], times_pred[:1], number_of_neighbors=1)

    def test_too_many_neighbors_raises(self, data):
        coords, times, values, pops, ids, coords_pred, times_pred, model = data
        stpk = STPoissonKriging(model, rate_base=100000.0, use_gpu=False).fit(
            coords, times, values, pops, ids=ids,
        )
        with pytest.raises(ValueError, match="Not enough known area-years"):
            stpk.predict(
                coords_pred[:1], times_pred[:1],
                number_of_neighbors=len(values) + 1,
            )


class TestGPUDispatch:
    def test_use_gpu_auto_never_crashes(self, data):
        coords, times, values, pops, ids, coords_pred, times_pred, model = data
        stpk = STPoissonKriging(model, rate_base=100000.0, use_gpu="auto").fit(
            coords, times, values, pops, ids=ids,
        )
        zhat, sig = stpk.predict(coords_pred, times_pred, number_of_neighbors=8)
        assert np.isfinite(zhat).all()
        assert zhat.shape == (len(coords_pred),)

    def test_explicit_true_warns_and_falls_back_when_unusable(self, monkeypatch):
        monkeypatch.setattr(backend, "_cupy_usable_cache", False)
        model = _metric_model()
        with pytest.warns(UserWarning, match="falling back to CPU"):
            stpk = STPoissonKriging(model, use_gpu=True)
        assert stpk.use_gpu is False

    def test_full_predict_gpu_path_matches_cpu(self, data, fake_gpu):
        coords, times, values, pops, ids, coords_pred, times_pred, model = data
        stpk_cpu = STPoissonKriging(model, rate_base=100000.0, use_gpu=False).fit(
            coords, times, values, pops, ids=ids,
        )
        stpk_gpu = STPoissonKriging(model, rate_base=100000.0, use_gpu=True).fit(
            coords, times, values, pops, ids=ids,
        )
        assert stpk_gpu.use_gpu is True

        z_cpu, sig_cpu = stpk_cpu.predict(coords_pred, times_pred, number_of_neighbors=8)
        z_gpu, sig_gpu = stpk_gpu.predict(coords_pred, times_pred, number_of_neighbors=8)
        np.testing.assert_allclose(z_cpu, z_gpu, rtol=1e-8)
        np.testing.assert_allclose(sig_cpu, sig_gpu, rtol=1e-8, equal_nan=True)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
