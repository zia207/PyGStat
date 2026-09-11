"""
Tests for pygstat.ebk (Empirical Bayesian Kriging).

Covers: the fit/predict/predict_proba API on synthetic data, the central
claim of EBK versus plain kriging on real Meuse data (point estimates
agree closely, but EBK's total predictive standard error is larger
because it also captures semivariogram-parameter uncertainty), and
GPU-dispatch smoke tests using the same "fake CuPy" pattern as
tests/test_indicator_kriging_gpu.py / tests/test_disjunctive_kriging.py.

`n_realizations` is kept small in most tests (fit cost scales as
O(n_realizations * n_train) kriging solves via leave-one-out CV) --
correctness doesn't need a large ensemble, just >= 2.
"""

import numpy as np
import pandas as pd
import pytest

from pygstat.ebk import EmpiricalBayesianKriging
from pygstat import OrdinaryKriging, Variogram
from pygstat.utils import backend

DATA_DIR = __file__.rsplit("/tests/", 1)[0] + "/data"


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(7)
    coords = rng.uniform(0, 100, size=(50, 2))
    values = (
        coords[:, 0] * 0.15 + coords[:, 1] * 0.08 + rng.normal(scale=1.5, size=50)
    )
    coords_pred = rng.uniform(0, 100, size=(15, 2))
    return coords, values, coords_pred


class TestEBKSynthetic:
    def test_fit_sets_expected_attributes(self, synthetic_data):
        coords, values, _ = synthetic_data
        ebk = EmpiricalBayesianKriging(n_realizations=10, use_gpu=False, random_state=0)
        ebk.fit(coords, values)
        assert ebk.is_fitted_
        assert len(ebk.variograms_) == 10
        assert len(ebk.krige_models_) == 10
        assert ebk.weights_.shape == (10,)
        assert ebk.weights_.sum() == pytest.approx(1.0)
        assert np.all(ebk.weights_ >= 0)
        assert ebk.loo_rmse_.shape == (10,)

    def test_predict_shape_and_finite(self, synthetic_data):
        coords, values, coords_pred = synthetic_data
        ebk = EmpiricalBayesianKriging(n_realizations=10, use_gpu=False, random_state=0)
        ebk.fit(coords, values)
        pred = ebk.predict(coords_pred)
        assert pred.shape == (len(coords_pred),)
        assert np.isfinite(pred).all()

    def test_predict_return_variance(self, synthetic_data):
        coords, values, coords_pred = synthetic_data
        ebk = EmpiricalBayesianKriging(n_realizations=10, use_gpu=False, random_state=0)
        ebk.fit(coords, values)
        pred, se = ebk.predict(coords_pred, return_variance=True)
        assert pred.shape == se.shape == (len(coords_pred),)
        assert np.all(se >= 0)
        assert np.isfinite(se).all()

    def test_simple_kriging_type(self, synthetic_data):
        coords, values, coords_pred = synthetic_data
        ebk = EmpiricalBayesianKriging(
            n_realizations=8, kriging_type="simple", use_gpu=False, random_state=0
        )
        ebk.fit(coords, values)
        pred = ebk.predict(coords_pred)
        assert np.isfinite(pred).all()

    def test_predict_proba_bounds_and_shape(self, synthetic_data):
        coords, values, coords_pred = synthetic_data
        ebk = EmpiricalBayesianKriging(n_realizations=10, use_gpu=False, random_state=0)
        ebk.fit(coords, values)
        cutoffs = [float(np.percentile(values, p)) for p in (25, 50, 75)]
        probs = ebk.predict_proba(coords_pred, cutoffs)
        assert set(probs.keys()) == set(cutoffs)
        for zc in cutoffs:
            assert probs[zc].shape == (len(coords_pred),)
            assert np.all((probs[zc] >= 0) & (probs[zc] <= 1))

    def test_predict_proba_decreases_with_cutoff_on_average(self, synthetic_data):
        coords, values, coords_pred = synthetic_data
        ebk = EmpiricalBayesianKriging(n_realizations=10, use_gpu=False, random_state=0)
        ebk.fit(coords, values)
        low, mid, high = [float(np.percentile(values, p)) for p in (25, 50, 75)]
        probs = ebk.predict_proba(coords_pred, [low, mid, high])
        assert probs[low].mean() > probs[mid].mean() > probs[high].mean()

    def test_predict_proba_less_than(self, synthetic_data):
        coords, values, coords_pred = synthetic_data
        ebk = EmpiricalBayesianKriging(n_realizations=10, use_gpu=False, random_state=0)
        ebk.fit(coords, values)
        zc = float(np.median(values))
        p_gt = ebk.predict_proba(coords_pred, [zc], greater_than=True)[zc]
        p_lt = ebk.predict_proba(coords_pred, [zc], greater_than=False)[zc]
        np.testing.assert_allclose(p_gt + p_lt, 1.0, atol=1e-8)

    def test_invalid_kriging_type_raises(self):
        with pytest.raises(ValueError):
            EmpiricalBayesianKriging(kriging_type="bogus")

    def test_invalid_n_realizations_raises(self):
        with pytest.raises(ValueError):
            EmpiricalBayesianKriging(n_realizations=1)

    def test_predict_before_fit_raises(self, synthetic_data):
        _, _, coords_pred = synthetic_data
        ebk = EmpiricalBayesianKriging()
        with pytest.raises(RuntimeError):
            ebk.predict(coords_pred)
        with pytest.raises(RuntimeError):
            ebk.predict_proba(coords_pred, [1.0])

    def test_reproducible_with_random_state(self, synthetic_data):
        coords, values, coords_pred = synthetic_data
        ebk1 = EmpiricalBayesianKriging(n_realizations=8, use_gpu=False, random_state=42)
        ebk1.fit(coords, values)
        ebk2 = EmpiricalBayesianKriging(n_realizations=8, use_gpu=False, random_state=42)
        ebk2.fit(coords, values)
        np.testing.assert_allclose(ebk1.predict(coords_pred), ebk2.predict(coords_pred))


class TestEBKMeuse:
    """The central EBK claim, checked on real data: point estimates agree
    closely with plain OrdinaryKriging, but EBK's total predictive
    standard error is larger everywhere, because it also captures
    semivariogram-parameter uncertainty that plain kriging ignores."""

    @pytest.fixture(scope="class")
    def meuse(self):
        df = pd.read_csv(f"{DATA_DIR}/meuse.csv")
        grid = pd.read_csv(f"{DATA_DIR}/meuse_grid.csv")
        coords = df[["x", "y"]].values
        log_zinc = np.log(df["zinc"].values)
        gcoords = grid[["x", "y"]].values[:200]  # subset -- keep the test fast
        return coords, log_zinc, gcoords

    @pytest.fixture(scope="class")
    def fitted_ebk(self, meuse):
        coords, log_zinc, _ = meuse
        ebk = EmpiricalBayesianKriging(
            model="spherical", n_realizations=20, use_gpu=False, random_state=0
        )
        ebk.fit(coords, log_zinc)
        return ebk

    def test_agrees_with_ordinary_kriging_point_estimate(self, meuse, fitted_ebk):
        coords, log_zinc, gcoords = meuse
        ebk_pred = fitted_ebk.predict(gcoords)

        vg = Variogram(coords, log_zinc, model="spherical")
        vg.fit()
        ok_pred = OrdinaryKriging(vg).fit(coords, log_zinc).predict(gcoords)

        corr = np.corrcoef(ebk_pred, ok_pred)[0, 1]
        assert corr > 0.99

    def test_standard_error_exceeds_plain_kriging(self, meuse, fitted_ebk):
        coords, log_zinc, gcoords = meuse
        _, ebk_se = fitted_ebk.predict(gcoords, return_variance=True)

        vg = Variogram(coords, log_zinc, model="spherical")
        vg.fit()
        _, ok_se = OrdinaryKriging(vg).fit(coords, log_zinc).predict(gcoords, return_variance=True)

        # EBK's total SE (within + between-model) must dominate plain OK's
        # SE (within-model only) virtually everywhere -- that extra margin
        # *is* the semivariogram-parameter uncertainty plain kriging ignores.
        assert np.mean(ebk_se >= ok_se) > 0.9
        assert ebk_se.mean() > ok_se.mean()

    def test_semivariogram_ensemble_has_real_spread(self, fitted_ebk):
        """The refit ensemble should show genuine parameter uncertainty,
        not collapse onto a single value."""
        ranges = np.array([vg.fitted_params[2] for vg in fitted_ebk.variograms_])
        assert ranges.std() > 0

    def test_probability_maps_on_meuse_grid(self, meuse, fitted_ebk):
        _, _, gcoords = meuse
        cutoffs = [np.log(500.0), np.log(1000.0)]
        probs = fitted_ebk.predict_proba(gcoords, cutoffs)
        for zc in cutoffs:
            p = probs[zc]
            assert p.shape == (len(gcoords),)
            assert np.all((p >= 0) & (p <= 1))
        assert probs[cutoffs[0]].mean() > probs[cutoffs[1]].mean()


# ----------------------------------------------------------------------
# GPU dispatch smoke tests (fake CuPy, same pattern as
# tests/test_indicator_kriging_gpu.py / tests/test_disjunctive_kriging.py)
# ----------------------------------------------------------------------
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


class TestGPUDispatch:
    def test_use_gpu_true_matches_cpu(self, synthetic_data, fake_gpu):
        coords, values, coords_pred = synthetic_data
        ebk_cpu = EmpiricalBayesianKriging(n_realizations=6, use_gpu=False, random_state=1)
        ebk_cpu.fit(coords, values)
        ebk_gpu = EmpiricalBayesianKriging(n_realizations=6, use_gpu=True, random_state=1)
        ebk_gpu.fit(coords, values)

        pred_cpu = ebk_cpu.predict(coords_pred)
        pred_gpu = ebk_gpu.predict(coords_pred)
        np.testing.assert_allclose(pred_cpu, pred_gpu, atol=1e-6)

    def test_use_gpu_auto_never_crashes(self, synthetic_data, fake_gpu):
        coords, values, coords_pred = synthetic_data
        ebk = EmpiricalBayesianKriging(n_realizations=6, use_gpu="auto", random_state=1)
        ebk.fit(coords, values)
        pred = ebk.predict(coords_pred)
        assert np.isfinite(pred).all()
