"""Smoke tests: import every pygstat module and exercise the public API."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pygstat

SRC_ROOT = Path(pygstat.__file__).resolve().parent

OPTIONAL_MODULES = {
    "pygstat.regression_kriging_h2o",
    "pygstat.spatial_cv_h2o_rk",
    "pygstat.spatial_cv_automl",
    "pygstat.regression_kriging_pytorch_gpu",
    "pygstat.regression_kriging_gnn",
    "pygstat.kriging_STGNN",
    "pygstat.kriging_STGTN",
    "pygstat.kriging_CNNLSTM",
    "pygstat.regression_kriging_tf",
    "pygstat.spatial_cv_pytorch_rk",
    "pygstat.spatial_cv_tf",
}


def _synthetic(n=24, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 50, size=(n, 2))
    values = 10 + 0.05 * coords[:, 0] + rng.normal(0, 1.0, size=n)
    return coords, values, rng


def test_version_and_public_api():
    assert pygstat.__version__ == "0.1.0"
    for name in pygstat.__all__:
        assert hasattr(pygstat, name), f"public name {name!r} is missing"


def test_import_every_submodule():
    failed = []
    for mod in pkgutil.walk_packages([str(SRC_ROOT)], prefix="pygstat."):
        name = mod.name
        try:
            importlib.import_module(name)
        except Exception as exc:
            if name in OPTIONAL_MODULES:
                continue
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failed, "required modules failed to import:\n" + "\n".join(failed)


def test_core_variogram_and_kriging():
    coords, values, _ = _synthetic()
    vg = pygstat.Variogram(coords, values, model="spherical", n_lags=8)
    vg.set_params(nugget=0.1, sill=1.0, range_=20.0)
    ok = pygstat.OrdinaryKriging(vg).fit(coords, values)
    pred, se = ok.predict(coords[:4], return_variance=True)
    assert pred.shape == (4,)
    assert np.all(np.isfinite(pred))
    assert np.all(se >= 0)

    sk = pygstat.SimpleKriging(vg, mean=float(np.mean(values))).fit(coords, values)
    assert sk.predict(coords[:3]).shape == (3,)

    uk = pygstat.UniversalKriging(vg, degree=1).fit(coords, values)
    uk_pred = uk.predict(coords[:2])
    assert np.all(np.isfinite(uk_pred))

    dist, gamma, left, right = pygstat.compute_variogram_cloud(coords, values, maxlag=30)
    assert dist.size == gamma.size == left.size == right.size
    assert dist.size > 0


def test_regression_indicator_poisson_helpers():
    coords, values, rng = _synthetic(n=30)
    X = np.column_stack([coords, rng.normal(size=len(values))])
    rk = pygstat.RegressionKriging(regressor="linear")
    rk.fit(X, values, coords)
    pred = rk.predict(X[:5], coords[:5])
    assert pred.shape == (5,)

    ik = pygstat.IndicatorKriging(max_neighbors=8)
    ik.fit(coords, values)
    thresh = float(np.median(values))
    lags, gamma = pygstat.fit_indicator_variogram(coords, values, thresh, n_lags=6)
    assert lags.shape == gamma.shape
    probs = ik.predict(coords[:4], [thresh])
    assert thresh in probs
    etype = pygstat.compute_etype_from_probabilities(
        np.array([thresh]), [probs[thresh]], z_min=float(values.min())
    )
    assert etype.shape == (4,)

    pops = rng.integers(200, 2000, size=len(values)).astype(float)
    vg = pygstat.Variogram(coords, values, model="exponential", n_lags=8)
    vg.set_params(nugget=0.2, sill=1.5, range_=25.0)
    pk = pygstat.PoissonKriging(vg, rate_base=100.0).fit(coords, np.abs(values), pops)
    zhat, sig = pk.predict(coords[:3], number_of_neighbors=6)
    assert zhat.shape == (3,)
    assert np.all(np.isfinite(zhat))


def test_cokriging_soft_factorial_and_cv():
    coords, z1, rng = _synthetic(n=20, seed=1)
    z2 = 0.7 * z1 + rng.normal(0, 0.3, size=len(z1))
    vg1 = pygstat.Variogram(coords, z1, model="exponential", n_lags=6)
    vg1.set_params(nugget=0.05, sill=1.0, range_=30.0)
    vg2 = pygstat.Variogram(coords, z2, model="exponential", n_lags=6)
    vg2.set_params(nugget=0.05, sill=0.8, range_=30.0)
    cv = pygstat.CrossVariogram(coords, z1, z2, model="exponential", n_lags=6)
    cv.fitted_params = [0.02, 0.4, 30.0]
    ck = pygstat.Cokriging(vg1, vg2, cv).fit(coords, z1, coords, z2)
    assert np.all(np.isfinite(ck.predict(coords[:3])))

    hard_i = (z1 >= np.median(z1)).astype(float)
    soft = np.clip(0.5 + 0.2 * rng.normal(size=len(z1)), 0.05, 0.95)
    sk = pygstat.SoftKriging(max_neighbors_hard=8, max_neighbors_soft=8)
    sk.fit(coords, hard_i, coords + rng.normal(0, 0.4, coords.shape), soft, B=0.6)
    soft_pred = sk.predict(coords[:4], sill=0.2, range_=25.0, nugget=0.02)
    assert soft_pred.shape == (4,)
    assert np.all((soft_pred >= 0) & (soft_pred <= 1))

    fk = pygstat.FactorialKriging(
        nugget=0.1,
        structures=[
            {"model": "spherical", "sill": 0.8, "range": 15.0},
            {"model": "spherical", "sill": 0.4, "range": 40.0},
        ],
        max_neighbors=10,
    ).fit(coords, z1)
    extracted = fk.extract(coords[:4], factor="total")
    assert extracted.shape == (4,)

    scores = pygstat.loo_cv(pygstat.OrdinaryKriging(vg1), coords, z1)
    assert "rmse" in scores
    kfold = pygstat.kfold_cv(pygstat.OrdinaryKriging(vg1), coords, z1, n_splits=3)
    assert kfold["predictions"].shape == (len(z1),)


def test_gslib_helpers_simulation_and_kt3d(tmp_path):
    coords, values, rng = _synthetic(n=16, seed=2)
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "z": values})
    path = tmp_path / "pts.dat"
    pygstat.write_gslib(path, "synthetic", ["x", "y", "z"], df.values)
    title, names, data = pygstat.read_gslib(path)
    assert title == "synthetic"
    assert names == ["x", "y", "z"]
    np.testing.assert_allclose(data, df.values)

    ns, vs, nss = pygstat.nscore_forward(values)
    back = pygstat.nscore_back(ns, vs, nss)
    assert back.shape == values.shape
    tlog = pygstat.transform_log(np.abs(values) + 0.1)
    assert tlog.shape == values.shape
    w = pygstat.declus_cell(coords[:, 0], coords[:, 1], nx=3)
    assert len(w) == len(values)
    xx, yy = pygstat.add_coordinates_to_grid(4, 3, xorig=0, yorig=0, xsize=1, ysize=1)
    assert xx.shape == yy.shape == (12,)

    grid = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    vario = [0.0, 0.1, 25.0, 25.0, 1.1, "spherical"]
    sims = pygstat.sgsim(grid, df, "x", "y", "z", num_points=4, vario=vario, radius=80.0, nsim=1, seed=0, quiet=True)
    assert sims.shape == (1, 3)

    thresholds = np.percentile(values, [30, 70]).tolist()
    sis = pygstat.sisim(
        grid, df, "x", "y", "z",
        thresholds=thresholds, global_cdf=[0.3, 0.7], vario=vario,
        num_points=4, radius=80.0, nsim=1, seed=0, quiet=True,
    )
    assert sis.shape == (1, 3)

    vg_spec = {"nugget": 0.1, "structures": [{"type": "spherical", "sill": 0.9, "range": 30.0}]}
    result = pygstat.kt3d_grid(
        df, x="x", y="y", value="z", variogram=vg_spec,
        nx=4, xmn=5.0, xsiz=10.0, ny=4, ymn=5.0, ysiz=10.0,
        ktype="ordinary", search_radius=80.0, ndmin=1, ndmax=8,
    )
    assert "estimate" in result.columns
    assert result["estimate"].notna().any()


def test_spatiotemporal_and_multivariate_poisson():
    n_sites, n_times = 8, 4
    rng = np.random.default_rng(3)
    site_xy = rng.uniform(0, 50, size=(n_sites, 2))
    times = np.arange(n_times, dtype=float)
    panel = 10 + 0.04 * site_xy[:, 0:1] + rng.normal(0, 0.8, size=(n_sites, n_times))
    long_coords = np.repeat(site_xy, n_times, axis=0)
    long_times = np.tile(times, n_sites)
    long_values = panel.ravel()

    joint = pygstat.vgm(1.0, "Sph", 25.0, nugget=0.1)
    st_model = pygstat.vgm_st("metric", joint=joint, st_ani=5.0)
    emp = pygstat.empirical_st_variogram(site_xy, times, panel, n_space_bins=4)
    assert "gamma" in emp

    stk = pygstat.STKriging(st_model).fit(long_coords, long_times, long_values)
    zhat, sig = stk.predict(long_coords[:3], long_times[:3], nmax=8)
    assert zhat.shape == (3,)
    assert np.all(np.isfinite(zhat))

    pops = rng.integers(300, 1500, size=len(long_values)).astype(float)
    rates = np.abs(long_values) * 5
    stpk = pygstat.STPoissonKriging(st_model, rate_base=100.0).fit(
        long_coords, long_times, rates, pops
    )
    zhat2, _ = stpk.predict(long_coords[:3], long_times[:3], number_of_neighbors=8)
    assert zhat2.shape == (3,)

    mpk = pygstat.multi_poisson_kriging
    cases = rng.poisson(np.clip(rates, 1, None)).astype(float)
    data = {
        "x": long_coords[:, 0],
        "y": long_coords[:, 1],
        "cases": cases,
        "pop": pops,
    }
    pk_out = mpk.poisson_krige(
        data, sill=2.0, rng=20.0, nugget=0.2, model="Exp",
        pop_rate=100.0, coords_pred=long_coords[:4], smooth=False,
    )
    assert pk_out["pred"].shape == (4,)
    assert np.all(np.isfinite(pk_out["pred"]))


def test_io_and_backend():
    from pygstat.io.geopandas_io import from_geodataframe
    from pygstat.utils.backend import CUPY_AVAILABLE, asarray, to_numpy
    from pygstat.utils.anisotropy import transform_aniso

    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    coords, values, _ = _synthetic(n=8, seed=4)
    gdf = gpd.GeoDataFrame(
        {"z": values},
        geometry=[Point(xy) for xy in coords],
        crs="EPSG:3857",
    )
    xy, zz = from_geodataframe(gdf, "z")
    assert xy.shape == (8, 2)
    assert zz.shape == (8,)

    arr = asarray(coords, use_gpu=False)
    np.testing.assert_allclose(to_numpy(arr), coords)
    aniso = transform_aniso(coords, angle=30, ratio=0.5)
    assert aniso.shape == coords.shape
    assert isinstance(CUPY_AVAILABLE, bool)
