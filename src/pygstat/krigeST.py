#!/usr/bin/env python3
"""
krigeST.py
==========

Spatio-Temporal (Ordinary) Kriging convenience layer.

This module wraps :mod:`pygstat.krige_st` (a Python port of gstat's
``krigeST.R``, see ``krigeST.R`` at the repo root) and
:mod:`pygstat.st_variogram_models` (a port of ``stVariogramModels.R``) into
an ergonomic, scikit-learn-flavored class, :class:`STKriging`, plus a data
loader and a runnable demo/test against a real monitoring-network dataset.

Unlike :mod:`pygstat.STPoisson_kriging` (this repo's *Poisson* kriging
generalization, for area-level count/rate panels needing a
population-reliability correction), :class:`STKriging` is plain
spatio-temporal **ordinary kriging** of a continuous variable observed at
a set of point locations repeatedly over time -- the typical shape of an
environmental monitoring network (e.g. air-quality stations), where sites
do not all necessarily share the exact same set of time stamps.

Theory
------
Ordinary kriging estimates a continuous variable at an unobserved
space-time location $(\\mathbf u_0, t_0)$ as a weighted linear combination of
$N$ nearby observations,

.. math::

    \\hat Z(\\mathbf u_0, t_0) = \\sum_{i=1}^N \\lambda_i\\, Z(\\mathbf u_i, t_i),
    \\qquad \\sum_{i=1}^N \\lambda_i = 1,

with weights solving the (space-time) kriging system

.. math::

    \\begin{bmatrix} \\mathbf V & \\mathbf 1 \\\\ \\mathbf 1^\\top & 0 \\end{bmatrix}
    \\begin{bmatrix} \\boldsymbol\\lambda \\\\ \\mu \\end{bmatrix}
    = \\begin{bmatrix} \\mathbf v_0 \\\\ 1 \\end{bmatrix},

where $V_{ij} = C\\big((\\mathbf u_i,t_i) - (\\mathbf u_j,t_j)\\big)$ and
$v_{0,i} = C\\big((\\mathbf u_i,t_i) - (\\mathbf u_0,t_0)\\big)$ for a
space-time covariance $C$. The needed ingredient beyond ordinary
(spatial-only) kriging is $C$ as a function of *both* a spatial lag and a
temporal lag -- see the notebook (``tests/test_krigeST.ipynb``) for the
full derivation, the empirical semivariogram estimator, and the "metric"
space-time model used here.

Data used to test this module (see ``main()`` below)
-------------------------------------------------------
``data/CA_pm25_2025.csv`` -- daily PM2.5 concentration (micrograms/m3) at
163 California air-quality monitoring stations through 2025, aggregated
here to monthly station means (12 monthly time stamps). 156 of the 163
stations report every month; the other 7 have partial coverage -- fine
for :class:`STKriging` (which only ever needs local neighbors, found by a
KD-tree, and tolerates an irregular panel), but the empirical variogram
estimator (:func:`pygstat.st_variogram_models.empirical_st_variogram`)
needs a *complete* site x month matrix, so variogram estimation uses only
the 156 complete stations.

Usage
-----
Run directly to fit and validate the model against the CA PM2.5 data:

    python -m pygstat.krigeST

Or import and use programmatically::

    from pygstat import STKriging, load_ca_pm25_panel
    from pygstat.st_variogram_models import empirical_st_variogram, fit_metric_st_variogram

    panel = load_ca_pm25_panel("data/CA_pm25_2025.csv")
    emp = empirical_st_variogram(coords, months, pm25_matrix)
    fitted = fit_metric_st_variogram(emp)

    stk = STKriging(fitted, use_gpu="auto").fit(coords, months, pm25, ids=site_ids)
    zhat, sig = stk.predict(new_coords, new_months, nmax=20)

Local kriging (finite ``nmax``) batches every prediction point's
``k × k`` system into one ``xp.linalg.solve`` — NumPy on CPU, CuPy on
GPU. Global kriging (``nmax=inf``) stays on CPU.

References
----------
Pebesma, E. (2012). spacetime: Spatio-Temporal Data in R. Journal of
Statistical Software, 51(7).

Authors
-------
Zia Ahmed, PhD (zia207@gmail.com)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .krige_st import krige_st
from .st_variogram_models import (
    vgm,
    vgm_st,
    fit_st_variogram,
    empirical_st_variogram,
    fit_metric_st_variogram,
    joint_nugget_sill_range,
)

__all__ = [
    "STKriging",
    "load_ca_pm25_panel",
    "make_prediction_grid",
    "empirical_st_variogram",
    "fit_metric_st_variogram",
]


# ---------------------------------------------------------------------------
# Spatio-Temporal Ordinary Kriging
# ---------------------------------------------------------------------------
class STKriging:
    """
    Ergonomic wrapper around :func:`pygstat.krige_st.krige_st` for
    irregular space-time monitoring-network data (unequal numbers of
    observations per site, missing time stamps, etc. -- no complete grid
    required, unlike variogram *estimation*, which does need one).

    Parameters
    ----------
    st_model : dict
        A fitted spatio-temporal variogram model (e.g. from
        :func:`pygstat.st_variogram_models.fit_metric_st_variogram` /
        ``fit_st_variogram``), used as ``model_list`` in
        :func:`pygstat.krige_st.krige_st`.
    use_gpu : bool or 'auto', default='auto'
        For local kriging (finite ``nmax``) stack every prediction point's
        ``k × k`` system into one batched ``xp.linalg.solve`` — NumPy on
        CPU, CuPy on GPU. Resolved the same way as
        :class:`pygstat.indicator_kriging.IndicatorKriging` (a real kernel
        probe, not just ``import cupy``). Global kriging (``nmax=inf``)
        always runs on CPU.

    GPU (CuPy) support
    -----------------
    Local neighbourhood kriging (the default ``nmax`` path) is independent
    across prediction points, so pygstat batches into one stacked solve
    of shape ``(n_pred, nmax, k)``. ``use_gpu='auto'|True|False`` runs
    that solve on the GPU via CuPy when usable.

    Examples
    --------
    >>> stk = STKriging(fitted_st_model).fit(coords, months, pm25, ids=site_ids)
    >>> zhat, sig = stk.predict(new_coords, new_months, nmax=20)
    >>> stk.predict_id(60590007, 5, nmax=20)
    {'id': 60590007, 'time': 5, 'zhat': 8.4, 'sig': 1.9}
    """

    def __init__(self, st_model, use_gpu: Union[bool, str] = "auto"):
        from .utils.backend import resolve_cupy_use_gpu

        self.st_model = st_model
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)
        self.coords = None
        self.times = None
        self.values = None
        self.ids = None

    def fit(self, coords, times, values, ids=None):
        """
        Store the known space-time observations used as kriging neighbors.

        Parameters
        ----------
        coords : array-like, shape (n, 2)
        times : array-like, shape (n,)
        values : array-like, shape (n,)
        ids : array-like, shape (n,), optional
            Site identifier of each row, for :meth:`predict_id`.
        """
        coords = np.asarray(coords, dtype=float)
        times = np.asarray(times, dtype=float).ravel()
        values = np.asarray(values, dtype=float).ravel()
        n = len(values)
        if not (len(coords) == len(times) == n):
            raise ValueError("coords, times, and values must have the same length")
        if n < 2:
            raise ValueError("at least 2 observations are required to fit STKriging")

        self.coords = coords
        self.times = times
        self.values = values
        self.ids = np.asarray(ids) if ids is not None else np.arange(n)
        return self

    def predict(self, X_pred, T_pred, nmax=30, compute_var=True, progress=False):
        """
        Predict at new (unobserved) space-time locations.

        Parameters
        ----------
        X_pred : array-like, shape (m, 2)
        T_pred : array-like, shape (m,)
        nmax : int, default=30
            Number of nearest space-time neighbors per local kriging system
            (delegates to :func:`pygstat.krige_st.krige_st_local`). Use
            ``np.inf`` for global kriging (all training points every time
            -- only practical for a modest number of observations).
        compute_var : bool, default=True
            Also return the kriging standard error.

        Returns
        -------
        zhat : numpy array, shape (m,)
        sig : numpy array, shape (m,) or None
            Kriging standard errors (``None`` if ``compute_var=False``).
        """
        X_pred = np.atleast_2d(np.asarray(X_pred, dtype=float))
        T_pred = np.atleast_1d(np.asarray(T_pred, dtype=float))
        if len(T_pred) != len(X_pred):
            raise ValueError("X_pred and T_pred must have the same length")

        out = krige_st(
            self.coords, self.times, self.values,
            X_pred, T_pred, self.st_model,
            nmax=nmax, compute_var=compute_var, progress=progress,
            use_gpu=self.use_gpu,
        )
        zhat = np.asarray(out["var1.pred"]).ravel()
        if compute_var:
            sig = np.sqrt(np.maximum(np.asarray(out["var1.var"]).ravel(), 0.0))
            return zhat, sig
        return zhat, None

    def predict_id(self, site_id, time, nmax=30, compute_var=True):
        """
        Predict the value of one of the *known* observations, excluding it
        from its own neighbor search (leave-one-out). Useful for
        validating the model against observed values.

        Returns
        -------
        dict
            ``{"id": site_id, "time": time, "zhat": prediction, "sig": std_error}``
        """
        matches = np.where((self.ids == site_id) & np.isclose(self.times, time))[0]
        if len(matches) == 0:
            raise ValueError(f"(id={site_id!r}, time={time!r}) not found in the fitted data")
        target_idx = int(matches[0])
        keep = np.ones(len(self.values), dtype=bool)
        keep[target_idx] = False

        out = krige_st(
            self.coords[keep], self.times[keep], self.values[keep],
            self.coords[target_idx:target_idx + 1], self.times[target_idx:target_idx + 1],
            self.st_model, nmax=min(nmax, keep.sum()), compute_var=compute_var,
            progress=False, use_gpu=self.use_gpu,
        )
        zhat = float(np.asarray(out["var1.pred"]).ravel()[0])
        sig = (
            float(np.sqrt(max(np.asarray(out["var1.var"]).ravel()[0], 0.0)))
            if compute_var else np.nan
        )
        return {"id": site_id, "time": time, "zhat": zhat, "sig": sig}


# ---------------------------------------------------------------------------
# Data loading (California PM2.5 monitoring network test data)
# ---------------------------------------------------------------------------
def load_ca_pm25_panel(csv_path):
    """
    Build a monthly (site_id, month, x, y, pm25) panel from the daily
    ``CA_pm25_2025.csv`` monitoring-network CSV: aggregates daily
    concentrations to monthly station means and projects station
    lat/long to California Albers Equal Area (EPSG:3310, meters) so
    spatial distances are in a metric unit consistent with
    ``COUNTY_ATLANTIC.prj`` elsewhere in this repo.

    Returns
    -------
    pandas.DataFrame
        Columns ``site_id, month (0=Jan..11=Dec), x, y, lat, lon, pm25``,
        sorted by ``(site_id, month)``.
    """
    import geopandas as gpd

    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["month"] = (df["Date"].dt.month - 1).astype(int)  # 0=Jan .. 11=Dec

    monthly = (
        df.groupby(["Site ID", "month"])
        .agg(pm25=("Daily_PM2.5_ug_m3", "mean"), lat=("Lat", "first"), lon=("Long", "first"))
        .reset_index()
        .rename(columns={"Site ID": "site_id"})
    )

    pts = gpd.GeoDataFrame(
        monthly, geometry=gpd.points_from_xy(monthly["lon"], monthly["lat"]), crs="EPSG:4326"
    ).to_crs("EPSG:3310")
    monthly["x"] = pts.geometry.x.to_numpy()
    monthly["y"] = pts.geometry.y.to_numpy()

    return monthly.sort_values(["site_id", "month"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prediction grid
# ---------------------------------------------------------------------------
def make_prediction_grid(boundary, cell_size=5000.0):
    """
    Build a regular ``cell_size`` x ``cell_size`` grid of point centers
    covering ``boundary``, clipped to the cells whose center falls inside
    the boundary polygon(s).

    Parameters
    ----------
    boundary : geopandas.GeoDataFrame or geopandas.GeoSeries
        Polygon(s) defining the area to grid, **in a projected (metric)
        CRS matching the data you intend to krige** -- e.g. reproject to
        ``EPSG:3310`` first to align with :func:`load_ca_pm25_panel`'s
        station coordinates, since a shapefile's own CRS (here
        ``US_STATE.shp``'s CONUS Albers) need not match.

    cell_size : float, default=5000.0
        Grid spacing in the boundary's CRS units (meters, so 5000.0 = 5 km).

    Returns
    -------
    geopandas.GeoDataFrame
        One row per grid cell center inside the boundary, columns
        ``x, y, geometry`` (``geometry`` in ``boundary``'s CRS).
    """
    import geopandas as gpd

    geom = boundary.geometry.union_all() if hasattr(boundary, "geometry") else boundary.union_all()
    minx, miny, maxx, maxy = geom.bounds
    xs = np.arange(minx + cell_size / 2, maxx, cell_size)
    ys = np.arange(miny + cell_size / 2, maxy, cell_size)
    xx, yy = np.meshgrid(xs, ys)

    pts = gpd.GeoDataFrame(
        {"x": xx.ravel(), "y": yy.ravel()},
        geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()),
        crs=boundary.crs,
    )
    inside = pts.geometry.within(geom)
    return pts.loc[inside].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fit and validate Spatio-Temporal Ordinary Kriging on the "
                     "California monthly PM2.5 monitoring-network data."
    )
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument("--csv", default=str(repo_root / "data" / "CA_pm25_2025.csv"))
    parser.add_argument("--n-test-sites", type=int, default=25,
                         help="Number of stations held out entirely for validation.")
    parser.add_argument("--nmax", type=int, default=20,
                         help="Number of space-time neighbors per kriging system.")
    parser.add_argument("--n-space-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="krigeST_predictions.csv")
    args = parser.parse_args()

    panel = load_ca_pm25_panel(args.csv)
    n_sites = panel["site_id"].nunique()
    print(f"Loaded {len(panel)} site-month rows ({n_sites} stations x up to 12 months)")

    # Empirical-variogram estimation needs a *complete* site x month matrix;
    # a handful of stations have partial-year coverage.
    counts = panel.groupby("site_id").size()
    complete_sites = counts[counts == 12].index.to_numpy()
    print(f"{len(complete_sites)}/{n_sites} stations report all 12 months "
          f"(used for variogram estimation); all {n_sites} are used for kriging.")

    # --- Hold out whole stations for validation
    rng = np.random.default_rng(args.seed)
    test_sites = rng.choice(complete_sites, size=args.n_test_sites, replace=False)
    is_test = panel["site_id"].isin(test_sites)
    train = panel.loc[~is_test].reset_index(drop=True)
    test = panel.loc[is_test].reset_index(drop=True)
    print(f"Train: {train['site_id'].nunique()} stations ({len(train)} rows)  |  "
          f"Test (held out stations): {test['site_id'].nunique()} stations ({len(test)} rows)")

    # --- Empirical + fitted space-time variogram (complete training stations only)
    train_complete = train[train["site_id"].isin(complete_sites)]
    train_wide = train_complete.pivot(index="site_id", columns="month", values="pm25").sort_index()
    train_coords = (
        train_complete[["site_id", "x", "y"]].drop_duplicates("site_id").set_index("site_id")
        .loc[train_wide.index][["x", "y"]].to_numpy()
    )
    months = np.sort(train_complete["month"].unique()).astype(float)

    print("Estimating empirical space-time semivariogram ...")
    emp = empirical_st_variogram(train_coords, months, train_wide.to_numpy(),
                                  n_space_bins=args.n_space_bins)
    print(f"  {len(emp['dist'])} (space-bin, time-lag) cells from {emp['np'].sum():.0f} pairs")

    print("Fitting 'metric' space-time variogram model ...")
    fitted_model = fit_metric_st_variogram(emp)
    nugget, sill, prange = joint_nugget_sill_range(fitted_model["joint"])
    print(f"  nugget={nugget:.3f}  sill={sill:.3f}  range={prange:.0f} m  "
          f"stAni={fitted_model['stAni']:.0f} m/month  MSE={fitted_model['MSE']:.4f}")

    # --- Fit Spatio-Temporal Kriging on ALL training stations (complete + partial)
    stk = STKriging(fitted_model).fit(
        coords=train[["x", "y"]].to_numpy(),
        times=train["month"].to_numpy(float),
        values=train["pm25"].to_numpy(),
        ids=train["site_id"].to_numpy(),
    )

    # --- Validate: predict every held-out station-month from training neighbors
    print(f"Predicting {len(test)} held-out station-months (nmax={args.nmax}) ...")
    zhat, sig = stk.predict(test[["x", "y"]].to_numpy(), test["month"].to_numpy(float),
                             nmax=args.nmax)
    test = test.assign(zhat=zhat, sig=sig)

    resid = test["pm25"] - test["zhat"]
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    r2 = 1.0 - float(np.sum(resid ** 2)) / float(np.sum((test["pm25"] - test["pm25"].mean()) ** 2))
    baseline_rmse = float(np.sqrt(np.mean((test["pm25"] - train["pm25"].mean()) ** 2)))

    print("\n--- Held-out station validation ---")
    print(f"  RMSE            : {rmse:.3f} ug/m3")
    print(f"  MAE             : {mae:.3f}")
    print(f"  R^2             : {r2:.3f}")
    print(f"  Naive-mean RMSE : {baseline_rmse:.3f}  (baseline for comparison)")
    print(f"  Mean kriging sd : {test['sig'].mean():.3f}")

    # --- Leave-one-out sanity check on training stations
    print("\n--- Leave-one-out check on 10 random training station-months ---")
    sample = train.sample(10, random_state=args.seed)
    loo = pd.DataFrame([
        stk.predict_id(row["site_id"], row["month"], nmax=args.nmax)
        for _, row in sample.iterrows()
    ])
    loo["observed"] = sample["pm25"].to_numpy()
    print(loo[["id", "time", "observed", "zhat", "sig"]].to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))

    # --- Forecast one month beyond the observed range
    print("\n--- Forecast: month index 12 (one month beyond Dec 2025) ---")
    demo_sites = train["site_id"].drop_duplicates().sample(3, random_state=args.seed).to_numpy()
    demo_coords = (
        train[["site_id", "x", "y"]].drop_duplicates("site_id").set_index("site_id")
        .loc[demo_sites][["x", "y"]].to_numpy()
    )
    fzhat, fsig = stk.predict(demo_coords, np.full(len(demo_sites), 12.0), nmax=args.nmax)
    for site, zh, sg in zip(demo_sites, fzhat, fsig):
        last_obs = train.loc[train["site_id"] == site].sort_values("month").iloc[-1]["pm25"]
        print(f"  Site {site}: Dec 2025 observed={last_obs:.2f}  ->  "
              f"forecast={zh:.2f} +/- {sg:.2f}")

    test[["site_id", "month", "pm25", "zhat", "sig"]].to_csv(args.output, index=False)
    print(f"\nSaved held-out predictions to {args.output}")


if __name__ == "__main__":
    main()
