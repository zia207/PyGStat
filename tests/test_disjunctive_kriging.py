"""
Tests for pygstat.disjunctive_kriging.

Covers, in order: the standalone Hermite-polynomial math (orthonormality,
indicator-coefficient reconstruction of a known step function), the
`DisjunctiveKriging` fit/predict/predict_proba API on synthetic data, a
real-data sanity check against `OrdinaryKriging` on Meuse (they estimate
different things -- linear vs. nonlinear -- but should broadly agree), and
GPU-dispatch smoke tests using the same "fake CuPy" pattern as
tests/test_indicator_kriging_gpu.py (this sandbox's GPU can import CuPy but
can't compile a kernel, so the GPU code *path* is exercised with a
NumPy-backed CuPy stand-in rather than requiring real CUDA).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from pygstat.disjunctive_kriging import (
    DisjunctiveKriging,
    hermite_polynomials,
    hermite_indicator_coefficients,
)
from pygstat import OrdinaryKriging, Variogram
from pygstat.utils import backend

DATA_DIR = __file__.rsplit("/tests/", 1)[0] + "/data"


# ----------------------------------------------------------------------
# Hermite polynomial math
# ----------------------------------------------------------------------
class TestHermitePolynomials:
    def test_orthonormal_under_standard_gaussian(self):
        """E[H_n(Y)H_m(Y)] = delta_nm for Y~N(0,1), checked by Gauss-Hermite
        quadrature (deterministic, not Monte Carlo -- high-order Hermite
        polynomials have heavy-tailed integrands that converge far too
        slowly under plain sampling to give a tight tolerance)."""
        n_max = 6
        pdf = norm.pdf
        for n in range(n_max + 1):
            for m in range(n, n_max + 1):
                val, _ = quad(
                    lambda y, n=n, m=m: hermite_polynomials(np.array([y]), n_max)[n, 0]
                    * hermite_polynomials(np.array([y]), n_max)[m, 0]
                    * pdf(y),
                    -10,
                    10,
                )
                expected = 1.0 if n == m else 0.0
                assert val == pytest.approx(expected, abs=1e-6)

    def test_known_values(self):
        y = np.array([0.0, 1.0, 2.0])
        H = hermite_polynomials(y, 3)
        np.testing.assert_allclose(H[0], 1.0)
        np.testing.assert_allclose(H[1], y)
        # H2(y) = (y^2 - 1) / sqrt(2)
        np.testing.assert_allclose(H[2], (y ** 2 - 1) / np.sqrt(2))

    def test_indicator_coefficients_correct_side_of_cutoff(self):
        """Truncated sum_n chi_n(yc) H_n(y) should land on the correct side
        of 0.5 well away from the discontinuity, even though a truncated
        Hermite series overshoots/undershoots right around a jump (Gibbs
        phenomenon -- see the aggregate convergence test below)."""
        yc = 0.3
        n_max = 40
        chi = hermite_indicator_coefficients(yc, n_max)

        y_test = np.array([yc - 2.5, yc - 1.5, yc + 1.5, yc + 2.5])
        H = hermite_polynomials(y_test, n_max)
        recon = (chi[:, None] * H).sum(axis=0)
        true_indicator = (y_test > yc).astype(float)
        assert np.all((recon > 0.5) == (true_indicator > 0.5))

    def test_indicator_reconstruction_improves_with_truncation_order(self):
        """A truncated Hermite series has pointwise Gibbs-phenomenon
        ringing that need not shrink monotonically at any single point,
        but the density-weighted L2 error against the true indicator over
        the whole real line must shrink as more Hermite orders are added."""
        yc = 0.3
        y = np.linspace(-4, 4, 2001)
        true_indicator = (y > yc).astype(float)
        weight = norm.pdf(y)

        errors = []
        for n_max in (5, 20, 80):
            chi = hermite_indicator_coefficients(yc, n_max)
            H = hermite_polynomials(y, n_max)
            recon = (chi[:, None] * H).sum(axis=0)
            l2 = np.sqrt(np.sum(weight * (recon - true_indicator) ** 2) / np.sum(weight))
            errors.append(l2)
        assert errors[0] > errors[1] > errors[2]

    def test_indicator_coefficient_zero_matches_normal_sf(self):
        for yc in (-1.0, 0.0, 1.5):
            chi0 = hermite_indicator_coefficients(yc, 5)[0]
            assert chi0 == pytest.approx(1.0 - norm.cdf(yc))


# ----------------------------------------------------------------------
# DisjunctiveKriging on synthetic data
# ----------------------------------------------------------------------
@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 100, size=(120, 2))
    latent = (
        np.sin(coords[:, 0] / 20) + np.cos(coords[:, 1] / 25) + rng.normal(scale=0.3, size=120)
    )
    values = np.exp(latent)  # positively skewed, like typical concentration data
    coords_pred = rng.uniform(0, 100, size=(30, 2))
    return coords, values, coords_pred


class TestDisjunctiveKrigingSynthetic:
    def test_fit_sets_expected_attributes(self, synthetic_data):
        coords, values, _ = synthetic_data
        dk = DisjunctiveKriging(model="spherical", n_hermite=10, use_gpu=False)
        dk.fit(coords, values)
        assert dk.is_fitted_
        assert dk.phi_.shape == (11,)
        assert dk.phi_[0] == pytest.approx(np.mean(values))
        assert len(dk.krige_by_order_) == 10
        assert 0.5 < dk.explained_variance_ratio_ < 1.5

    def test_predict_shape_and_finite(self, synthetic_data):
        coords, values, coords_pred = synthetic_data
        dk = DisjunctiveKriging(n_hermite=10, use_gpu=False).fit(coords, values)
        pred = dk.predict(coords_pred)
        assert pred.shape == (len(coords_pred),)
        assert np.isfinite(pred).all()

    def test_predict_return_variance(self, synthetic_data):
        coords, values, coords_pred = synthetic_data
        dk = DisjunctiveKriging(n_hermite=10, use_gpu=False).fit(coords, values)
        pred, se = dk.predict(coords_pred, return_variance=True)
        assert pred.shape == se.shape == (len(coords_pred),)
        assert np.all(se >= 0)
        assert np.isfinite(se).all()

    def test_predict_near_training_points_recovers_values(self, synthetic_data):
        """DK should reproduce the data reasonably closely (not necessarily
        exactly, since the fitted Y-variogram has a small nugget) at the
        exact training locations."""
        coords, values, _ = synthetic_data
        dk = DisjunctiveKriging(n_hermite=15, use_gpu=False).fit(coords, values)
        pred_train = dk.predict(coords)
        rmse = np.sqrt(np.mean((pred_train - values) ** 2))
        assert rmse < 0.5 * np.std(values)

    def test_predict_proba_bounds_and_shape(self, synthetic_data):
        coords, values, coords_pred = synthetic_data
        dk = DisjunctiveKriging(n_hermite=15, use_gpu=False).fit(coords, values)
        cutoffs = [float(np.percentile(values, p)) for p in (25, 50, 75)]
        probs = dk.predict_proba(coords_pred, cutoffs)
        assert set(probs.keys()) == set(cutoffs)
        for zc in cutoffs:
            assert probs[zc].shape == (len(coords_pred),)
            assert np.all((probs[zc] >= 0) & (probs[zc] <= 1))

    def test_predict_proba_decreases_with_cutoff_on_average(self, synthetic_data):
        """Mean exceedance probability across the grid must decrease as the
        cutoff increases (individual points can show small order-relation
        violations near a cutoff -- the classic DK/IK Gibbs-phenomenon
        artifact -- but the aggregate trend must be monotonic)."""
        coords, values, coords_pred = synthetic_data
        dk = DisjunctiveKriging(n_hermite=15, use_gpu=False).fit(coords, values)
        low, mid, high = [float(np.percentile(values, p)) for p in (25, 50, 75)]
        probs = dk.predict_proba(coords_pred, [low, mid, high])
        assert probs[low].mean() > probs[mid].mean() > probs[high].mean()

    def test_invalid_n_hermite_raises(self):
        with pytest.raises(ValueError):
            DisjunctiveKriging(n_hermite=0)

    def test_invalid_tails_raises(self):
        with pytest.raises(ValueError):
            DisjunctiveKriging(tails="bogus")

    def test_predict_before_fit_raises(self, synthetic_data):
        _, _, coords_pred = synthetic_data
        dk = DisjunctiveKriging()
        with pytest.raises(RuntimeError):
            dk.predict(coords_pred)
        with pytest.raises(RuntimeError):
            dk.predict_proba(coords_pred, [1.0])


# ----------------------------------------------------------------------
# Real-data end-to-end test: Meuse fit + meuse_grid prediction
# ----------------------------------------------------------------------
class TestDisjunctiveKrigingMeuse:
    @pytest.fixture(scope="class")
    def meuse(self):
        df = pd.read_csv(f"{DATA_DIR}/meuse.csv")
        grid = pd.read_csv(f"{DATA_DIR}/meuse_grid.csv")
        coords = df[["x", "y"]].values
        zinc = df["zinc"].values
        gcoords = grid[["x", "y"]].values
        return coords, zinc, gcoords

    def test_fit_and_predict_on_meuse_grid(self, meuse):
        coords, zinc, gcoords = meuse
        dk = DisjunctiveKriging(model="spherical", n_hermite=20, use_gpu=False)
        dk.fit(coords, zinc)
        pred, se = dk.predict(gcoords, return_variance=True)

        assert pred.shape == se.shape == (len(gcoords),)
        assert np.isfinite(pred).all() and np.isfinite(se).all()
        assert np.all(se >= 0)
        # Predicted zinc concentrations should stay in a physically
        # reasonable range around the observed data.
        assert pred.min() > 0
        assert pred.max() < 5 * zinc.max()

    def test_agrees_with_ordinary_kriging_on_log_zinc(self, meuse):
        """DK (nonlinear, via Gaussian anamorphosis) and OK on log(zinc)
        (linear, on a variance-stabilizing transform) estimate the same
        underlying field two different ways; they should correlate
        strongly on the same prediction grid even though they need not
        match exactly."""
        coords, zinc, gcoords = meuse
        dk = DisjunctiveKriging(model="spherical", n_hermite=20, use_gpu=False)
        dk.fit(coords, zinc)
        dk_pred = dk.predict(gcoords)

        vg = Variogram(coords, np.log(zinc), model="spherical")
        vg.fit()
        ok = OrdinaryKriging(vg).fit(coords, np.log(zinc))
        ok_pred = np.exp(ok.predict(gcoords))

        corr = np.corrcoef(dk_pred, ok_pred)[0, 1]
        assert corr > 0.9

    def test_probability_maps_on_meuse_grid(self, meuse):
        coords, zinc, gcoords = meuse
        dk = DisjunctiveKriging(model="spherical", n_hermite=20, use_gpu=False)
        dk.fit(coords, zinc)
        cutoffs = [200.0, 500.0, 1000.0]
        probs = dk.predict_proba(gcoords, cutoffs)

        assert set(probs.keys()) == set(cutoffs)
        for zc in cutoffs:
            p = probs[zc]
            assert p.shape == (len(gcoords),)
            assert np.all((p >= 0) & (p <= 1))
        # Higher cutoffs must be exceeded less often, on average, across
        # the whole grid.
        assert probs[200.0].mean() > probs[500.0].mean() > probs[1000.0].mean()


# ----------------------------------------------------------------------
# GPU dispatch smoke tests (fake CuPy, same pattern as
# tests/test_indicator_kriging_gpu.py)
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
        dk_cpu = DisjunctiveKriging(n_hermite=8, use_gpu=False).fit(coords, values)
        dk_gpu = DisjunctiveKriging(n_hermite=8, use_gpu=True).fit(coords, values)

        pred_cpu = dk_cpu.predict(coords_pred)
        pred_gpu = dk_gpu.predict(coords_pred)
        np.testing.assert_allclose(pred_cpu, pred_gpu, atol=1e-8)

        probs_cpu = dk_cpu.predict_proba(coords_pred, [float(np.median(values))])
        probs_gpu = dk_gpu.predict_proba(coords_pred, [float(np.median(values))])
        for zc in probs_cpu:
            np.testing.assert_allclose(probs_cpu[zc], probs_gpu[zc], atol=1e-8)

    def test_use_gpu_auto_never_crashes(self, synthetic_data, fake_gpu):
        coords, values, coords_pred = synthetic_data
        dk = DisjunctiveKriging(n_hermite=8, use_gpu="auto").fit(coords, values)
        pred = dk.predict(coords_pred)
        assert np.isfinite(pred).all()
