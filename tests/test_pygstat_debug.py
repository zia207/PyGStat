"""Regression tests for bugs fixed in the pygstat debug pass."""

import numpy as np
import pandas as pd
import pytest

from pygstat.core.variogram_models import cubic, matern, covariance_from_variogram
from pygstat.core.variogram import Variogram
from pygstat.core.kriging import OrdinaryKriging, SimpleKriging
from pygstat.cokriging import (
    Cokriging,
    CrossVariogram,
    MultivariateCokriging,
    ColocatedCokriging,
    fit_lmc,
)
from pygstat.universal_kriging import UniversalKriging
from pygstat.trans import transform_boxcox, back_transform_boxcox, transform_log
from pygstat.nscore import nscore_forward, nscore_back
from pygstat.sgsim import sgsim, _validate_vario, _octant_search
from pygstat.etype_estimates import compute_etype_from_probabilities
from pygstat.st_variogram_models import vgm, vgm_st, extract_par, insert_par
from pygstat.utils.anisotropy import transform_aniso
from pygstat.common import write_gslib, read_gslib
from pygstat.declus import declus_cell
from pygstat.validation import loo_cv


def _grid_data(n=20, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(n, 2))
    vals = rng.normal(10, 2, size=n)
    return coords, vals


def test_cubic_variogram_is_monotonic_and_gamma0_is_nugget():
    h = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    g = cubic(h, nugget=0.1, sill=1.0, range_=1.0)
    assert g[0] == pytest.approx(0.1)
    assert g[-1] == pytest.approx(1.1)
    assert np.all(np.diff(g[:5]) >= -1e-12)


def test_matern_accepts_scalar_h():
    g = matern(0.0, 0.1, 1.0, 2.0, nu=1.5)
    assert float(g) == pytest.approx(0.1)
    g2 = matern(np.array([0.0, 1.0]), 0.1, 1.0, 2.0, nu=1.5)
    assert g2.shape == (2,)
    assert g2[0] == pytest.approx(0.1)


def test_covariance_from_variogram_c0_includes_nugget():
    h = np.array([0.0, 1.0])
    c = covariance_from_variogram(h, nugget=0.2, sill=1.0, range_=5.0, model="spherical")
    assert c[0] == pytest.approx(1.2)
    assert c[1] < c[0]


def test_set_params_keyword_order_is_stable():
    coords, vals = _grid_data()
    vg = Variogram(coords, vals, model="spherical")
    vg.set_params(range_=10, sill=2, nugget=0.1)
    assert vg.fitted_params[0] == pytest.approx(0.1)
    assert vg.fitted_params[1] == pytest.approx(2)
    assert vg.fitted_params[2] == pytest.approx(10)


def test_unfitted_variogram_raises():
    coords, vals = _grid_data()
    vg = Variogram(coords, vals)
    with pytest.raises(RuntimeError, match="not fitted"):
        vg(1.0)


def test_ordinary_kriging_nugget_on_diagonal():
    coords, vals = _grid_data(n=15)
    vg = Variogram(coords, vals, model="spherical")
    vg.set_params(nugget=0.5, sill=1.0, range_=40)
    ok = OrdinaryKriging(vg).fit(coords, vals)
    pred, se = ok.predict(coords[:4], return_variance=True)
    assert pred.shape == (4,)
    assert se.shape == (4,)
    assert np.all(np.isfinite(pred))
    assert np.all(se >= 0)


def test_simple_kriging_runs():
    coords, vals = _grid_data(n=15)
    vg = Variogram(coords, vals, model="exponential")
    vg.set_params(nugget=0.1, sill=1.0, range_=30)
    sk = SimpleKriging(vg, mean=float(np.mean(vals))).fit(coords, vals)
    pred = sk.predict(coords[:3])
    assert pred.shape == (3,)


def test_universal_kriging_weights_include_intercept():
    coords, vals = _grid_data(n=20)
    vg = Variogram(coords, vals, model="spherical")
    vg.set_params(nugget=0.1, sill=1.0, range_=50)
    uk = UniversalKriging(vg, degree=1).fit(coords, vals)
    assert uk.F.shape[1] == 3  # 1, x, y
    pred, se = uk.predict(coords[:2], return_variance=True)
    assert pred.shape == (2,)
    assert np.all(np.isfinite(pred))


def test_cokriging_two_unbiasedness_constraints():
    coords, z1 = _grid_data(n=12, seed=1)
    z2 = z1 + np.random.default_rng(1).normal(0, 0.3, size=len(z1))
    vg1 = Variogram(coords, z1, model="exponential")
    vg1.set_params(nugget=0.05, sill=1.0, range_=40)
    vg2 = Variogram(coords, z2, model="exponential")
    vg2.set_params(nugget=0.05, sill=0.8, range_=40)
    cv = CrossVariogram(coords, z1, z2, model="exponential")
    cv.fit()
    cv.fitted_params = [0.02, 0.5, 40.0]
    ck = Cokriging(vg1, vg2, cv).fit(coords, z1, coords, z2)
    pred, se = ck.predict(coords[:3], return_variance=True)
    assert pred.shape == (3,)
    assert np.all(np.isfinite(pred))
    assert np.all(se >= 0)


def test_cross_variogram_fit_model_and_set_params():
    coords, z1 = _grid_data(n=20, seed=2)
    z2 = z1 + np.random.default_rng(2).normal(0, 0.4, size=len(z1))
    cv = CrossVariogram(coords, z1, z2, model="spherical", n_lags=8)
    cv.fit(method="auto")
    assert cv.fitted_params is not None
    assert len(cv.fitted_params) == 3
    assert np.all(np.isfinite(cv(cv.lags)))
    cv.set_params(nugget=0.01, c12=-0.2, range_=25.0)
    assert cv.fitted_params[1] == pytest.approx(-0.2)


def test_multivariate_cokriging_three_variables():
    coords, z1 = _grid_data(n=18, seed=3)
    rng = np.random.default_rng(3)
    z2 = 0.7 * z1 + rng.normal(0, 0.4, size=len(z1))
    z3 = 0.5 * z1 + rng.normal(0, 0.5, size=len(z1))
    vgs, cross, info = fit_lmc(coords, [z1, z2, z3], model="exponential", n_lags=8)
    assert info["lmc_check"]["valid"]
    ck = MultivariateCokriging(vgs, cross).fit(
        [coords, coords, coords], [z1, z2, z3]
    )
    pred, se = ck.predict(coords[:4], return_variance=True)
    assert pred.shape == (4,)
    assert np.all(np.isfinite(pred))
    assert np.all(se >= 0)


def test_colocated_cokriging_mm1():
    coords, z1 = _grid_data(n=16, seed=4)
    rng = np.random.default_rng(4)
    z2 = 0.8 * z1 + rng.normal(0, 0.3, size=len(z1))
    z3 = 0.6 * z1 + rng.normal(0, 0.4, size=len(z1))
    vgs, cross, _ = fit_lmc(coords, [z1, z2, z3], model="spherical", n_lags=8)
    cck = ColocatedCokriging(
        vgs[0],
        [vgs[1], vgs[2]],
        [cross[(0, 1)], cross[(0, 2)]],
        secondary_cross_vars={(0, 1): cross[(1, 2)]},
        mm1=True,
    ).fit(coords, z1, secondary_means=[float(z2.mean()), float(z3.mean())])
    Y = np.column_stack([z2[:5], z3[:5]])
    pred, se = cck.predict(coords[:5], Y, return_variance=True)
    assert pred.shape == (5,)
    assert np.all(np.isfinite(pred))
    assert np.all(se >= 0)


def test_boxcox_preserves_shape_and_min_val_zero():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = transform_boxcox(x, 0.5, min_val=0.0)
    assert t.shape == (2, 2)
    back = back_transform_boxcox(t, 0.5, min_val=0.0)
    assert back.shape == (2, 2)
    np.testing.assert_allclose(back, x, rtol=1e-6)


def test_log_transform_rejects_nonpositive():
    with pytest.raises(ValueError):
        transform_log(np.array([1.0, 0.0, 2.0]))


def test_nscore_uses_low_high_and_roundtrip():
    v = np.array([1.0, 2.0, 2.0, 3.0, 10.0])
    ns, vs, nss = nscore_forward(v, tails="linear", low=0.5, high=8.0)
    assert ns.shape == v.shape
    assert np.all(np.isfinite(ns))
    back = nscore_back(ns, vs, nss, tails="linear")
    assert back[0] == pytest.approx(1.0, rel=1e-5)
    with pytest.raises(ValueError):
        nscore_forward(v, tails="quadratic")


def test_octant_search_works_with_few_neighbors():
    df = pd.DataFrame({"X": [0, 1, 2, 3], "Y": [0, 0, 1, 1], "Z": [1.0, 2.0, 3.0, 4.0]})
    near = _octant_search(radius=10.0, num_points=4, loc=np.array([0.5, 0.5]), data=df)
    assert near.shape[1] == 3
    assert near.shape[0] > 0


def test_sgsim_runs_and_rejects_partial_sill():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.uniform(0, 10, 8),
        "y": rng.uniform(0, 10, 8),
        "z": rng.normal(0, 1, 8),
    })
    grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    vario = [0.0, 0.1, 5.0, 5.0, 1.1, "spherical"]
    sims = sgsim(grid, df, "x", "y", "z", num_points=4, vario=vario, radius=20.0, nsim=1, seed=0, quiet=True)
    assert sims.shape == (1, 3)
    assert np.all(np.isfinite(sims))
    with pytest.raises(ValueError, match="sill is total C\\(0\\)"):
        _validate_vario([0.0, 0.5, 5.0, 5.0, 0.2, "spherical"])


def test_etype_includes_lower_tail():
    thresholds = np.array([2.0, 4.0, 6.0])
    probs = [np.array([0.8]), np.array([0.4]), np.array([0.1])]
    et = compute_etype_from_probabilities(thresholds, probs, z_min=0.0)
    assert et.shape == (1,)
    assert et[0] > 0


def test_insert_par_prod_sum_space_nugget_range():
    space = vgm(1.0, "Sph", 10.0, nugget=0.2)
    time = vgm(0.5, "Sph", 5.0, nugget=0.1)
    model = vgm_st("productSum", space=space, time=time, k=0.3)
    par = extract_par(model)
    restored = insert_par(par, model)
    np.testing.assert_allclose(extract_par(restored), par)


def test_transform_aniso_validates_shape():
    with pytest.raises(ValueError):
        transform_aniso(np.array([1.0, 2.0, 3.0]))
    out = transform_aniso(np.array([[1.0, 2.0], [3.0, 4.0]]), angle=0, ratio=0.5)
    assert out.shape == (2, 2)


def test_gslib_roundtrip(tmp_path):
    path = tmp_path / "demo.dat"
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    write_gslib(path, "demo", ["a", "b"], data)
    title, names, out = read_gslib(path)
    assert title == "demo"
    assert names == ["a", "b"]
    np.testing.assert_allclose(out, data)
    with pytest.raises(ValueError):
        write_gslib(path, "demo", ["a"], data)


def test_declus_length_check():
    with pytest.raises(ValueError):
        declus_cell(np.array([1.0, 2.0]), np.array([1.0]))
    w = declus_cell(np.array([0.0, 1.0, 1.1, 5.0]), np.array([0.0, 1.0, 1.1, 5.0]), nx=2)
    assert len(w) == 4
    assert w.sum() == pytest.approx(4.0)


def test_loo_cv_ordinary_kriging():
    coords, vals = _grid_data(n=12)
    vg = Variogram(coords, vals, model="spherical")
    vg.fit()
    ok = OrdinaryKriging(vg)
    scores = loo_cv(ok, coords, vals)
    assert "rmse" in scores
    assert scores["predictions"].shape == (12,)
