#!/usr/bin/env python3
"""
STPoisson_kriging.py
=====================

Spatio-Temporal Poisson Kriging for areal disease/rate panel data.

Poisson Kriging (Goovaerts, 2006) smooths a rate that has been computed as
``count / population`` for an area (e.g. a county). A naive semivariogram/
kriging of the raw rate is misleading because a rate estimated from a small
population is far less reliable than the same rate estimated from a large
one; Poisson Kriging corrects for this by adding a population-dependent term
to the diagonal of the kriging covariance matrix, down-weighting areas with
a small population support.

``pygstat.poisson_kriging.PoissonKriging`` already implements this, but only
in space: each area is one ``(x, y, rate, population)`` row. This script
generalizes it to **space-time** panel data (the same areas observed
repeatedly over time, e.g. a yearly county rate), by combining two pieces
already in this repository:

* the population-reliability diagonal correction of
  :mod:`pygstat.poisson_kriging` (Goovaerts, 2006, Eq. 3/7), and
* the space-time covariance engine of :mod:`pygstat.krige_st` /
  :mod:`pygstat.st_variogram_models` (a Python port of gstat's
  ``krigeST.R`` / ``stVariogramModels.R`` -- see ``krigeST.R`` and
  ``stVariogramModels.R`` at the repo root).

The result, :class:`STPoissonKriging`, is an *area-year* kriging model:
ordinary kriging in a joint (space, time) covariance, with each
neighbor's own population added as a Goovaerts-style reliability term on
the diagonal. It reduces to ``PoissonKriging`` when every area has a single
time stamp, and to plain space-time ordinary kriging when every population
is equal (the reliability term becomes a constant nugget).

GPU (CuPy) support
------------------
:meth:`STPoissonKriging.predict` batches every local ``(k+1)×(k+1)``
system into one stacked ``xp.linalg.solve`` of shape
``(n_pred, k+1, k+1)`` — NumPy on CPU, CuPy on GPU — using the same
space-time covariance batcher as :func:`pygstat.krige_st.krige_st_local`.
Pass ``use_gpu='auto'|True|False``. :meth:`STPoissonKriging.predict_id`
(leave-one-out of a single area-year) keeps the original per-target loop.

Data used to test this script (see ``main()`` below)
------------------------------------------------------
North-Atlantic-seaboard US county lung/bronchus cancer mortality rate panel:

* ``data/lbc_atlantic.csv`` -- wide panel, one row per county (``FIPS``),
  one column per year 1998-2012, age-adjusted rate per 100,000.
* ``data/lbc_rate.csv`` -- one row per county with its projected centroid
  (``x``, ``y``, Albers Equal Area meters, matching
  ``data/COUNTY_ATLANTIC.prj``) and its population (``pop``).
* ``data/COUNTY_ATLANTIC.shp`` (+ ``.dbf``/``.shx``/``.prj``) -- the county
  polygons the two CSVs above were derived from; not required to run the
  model (which only needs point centroids), used here only to double check
  that every ``FIPS`` lines up.

**Important limitation.** The attached CSVs give one population figure per
county (its approximate population over the study period), not a
year-by-year population. True Poisson Kriging needs population *at the time
of each count*; lacking that, this script assumes population is constant
across 1998-2012 for each county. That is the same assumption implicitly
made by ``tests/test_poisson_kriging.py`` (the existing spatial-only
Poisson Kriging test), which draws its population from this same
``lbc_rate.csv``. Supply a true year-by-year population table via
``--population-cols`` (see ``--help``) if you have one.

Usage
-----
Run directly to fit and validate the model against the Atlantic county data:

    python -m pygstat.STPoisson_kriging

Or import and use programmatically::

    from pygstat import STPoissonKriging, empirical_st_variogram
    from pygstat.st_variogram_models import vgm, vgm_st, fit_st_variogram

    emp = empirical_st_variogram(coords, times, rates)
    joint0 = vgm(psill=..., model="exponential", range_=..., nugget=...)
    model0 = vgm_st("metric", joint=joint0, st_ani=...)
    fitted = fit_st_variogram(emp, model0)

    stpk = STPoissonKriging(fitted).fit(coords, times, rates, populations, ids=fips)
    zhat, sig = stpk.predict(new_coords, new_times, number_of_neighbors=30)

References
----------
Goovaerts, P. (2006). Geostatistical analysis of disease data: accounting
for spatial support and population density in the isopleth mapping of
cancer mortality risk. International Journal of Health Geographics, 5(1).

Pebesma, E. (2012). spacetime: Spatio-Temporal Data in R. Journal of
Statistical Software, 51(7). (source of the ``krigeST``/space-time
variogram algorithms this script's ``pygstat`` dependencies are ported
from, and of the Atlantic county lung/bronchus cancer dataset used to test
this script.)

Authors
-------
Zia Ahmed, PhD (zia207@gmail.com)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .krige_st import cov_fn_st, _cov_st_batch
from .st_variogram_models import (
    vgm,
    vgm_st,
    fit_st_variogram,
    empirical_st_variogram,
    fit_metric_st_variogram,
    joint_nugget_sill_range,
)
from .poisson_kriging import _pk_diagonal_weight, _detect_rate_base
from .utils.backend import (
    as_gpu_array,
    get_array_module,
    resolve_cupy_use_gpu,
    to_numpy,
)

# Re-exported here for backwards compatibility: earlier versions of this
# module defined `empirical_st_variogram`/`fit_metric_st_variogram` directly.
# They are generic (no Poisson/population dependency) and now live in
# `pygstat.st_variogram_models` so `pygstat.krigeST` can share them too.
__all__ = [
    "empirical_st_variogram",
    "fit_metric_st_variogram",
    "STPoissonKriging",
    "load_atlantic_panel",
]


# ---------------------------------------------------------------------------
# Spatio-Temporal Poisson Kriging
# ---------------------------------------------------------------------------
class STPoissonKriging:
    """
    Spatio-temporal Poisson Kriging of area-year rate/count data.

    Ordinary kriging in a joint space-time covariance (any model type
    supported by :func:`pygstat.krige_st.cov_fn_st`: ``separable``,
    ``productSum``, ``sumMetric``, ``simpleSumMetric``, or ``metric``),
    with a Goovaerts (2006) population-reliability term
    ``rate_base * m* / n(u_i)`` added to each neighbor's own diagonal entry
    -- the direct space-time generalization of
    :class:`pygstat.poisson_kriging.PoissonKriging`.

    Parameters
    ----------
    st_model : dict
        A fitted spatio-temporal variogram model
        (e.g. from :func:`fit_metric_st_variogram` /
        :func:`pygstat.st_variogram_models.fit_st_variogram`).

    rate_base : float, optional
        The ``B`` in ``rate = B * count / population`` (e.g. 100000 for a
        per-100,000 rate). Auto-detected from the data if not given (see
        :func:`pygstat.poisson_kriging._detect_rate_base`).
    use_gpu : bool or {'auto'}, default='auto'
        Solve the batched k-nearest-neighbor systems in :meth:`predict`
        on the GPU via CuPy when usable. Has no effect on :meth:`predict_id`.

    Examples
    --------
    >>> stpk = STPoissonKriging(fitted_st_model).fit(coords, years, rates, pops, ids=fips)
    >>> zhat, sig = stpk.predict(new_coords, new_years, number_of_neighbors=30)
    >>> stpk.predict_id(34017, 2005, number_of_neighbors=30)
    {'id': 34017, 'year': 2005, 'zhat': 61.2, 'sig': 4.8}
    """

    def __init__(self, st_model, rate_base=None, use_gpu="auto"):
        self.st_model = st_model
        self.rate_base = rate_base
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)
        self.coords = None
        self.times = None
        self.values = None
        self.populations = None
        self.ids = None
        self.st_ani = float(st_model.get("stAni", 1.0)) if st_model.get("stAni") else 1.0
        self._tree = None
        self._c00_cache = None

    def fit(self, coords, times, values, populations, ids=None):
        """
        Store the known area-years used as Poisson Kriging neighbors.

        Parameters
        ----------
        coords : array-like, shape (n, 2)
        times : array-like, shape (n,)
        values : array-like, shape (n,)
            Areal rate of each area-year.
        populations : array-like, shape (n,)
            Population of each area-year. Must be strictly positive.
        ids : array-like, shape (n,), optional
            Area identifier (e.g. FIPS) of each row, for :meth:`predict_id`.
        """
        coords = np.asarray(coords, dtype=float)
        times = np.asarray(times, dtype=float).ravel()
        values = np.asarray(values, dtype=float).ravel()
        populations = np.asarray(populations, dtype=float).ravel()
        n = len(values)
        if not (len(coords) == len(times) == n == len(populations)):
            raise ValueError(
                "coords, times, values, and populations must have the same length"
            )
        if np.any(populations <= 0):
            raise ValueError("populations must be strictly positive")

        if self.rate_base is None:
            mstar = _pk_diagonal_weight(values, populations)
            self.rate_base = _detect_rate_base(mstar)

        self.coords = coords
        self.times = times
        self.values = values
        self.populations = populations
        self.ids = np.asarray(ids) if ids is not None else np.arange(n)

        scaled = np.column_stack([coords[:, 0], coords[:, 1], times * self.st_ani])
        self._tree = cKDTree(scaled)
        self._c00_cache = float(
            np.asarray(
                cov_fn_st(coords[:1], times[:1], coords[:1], times[:1], self.st_model)
            ).reshape(-1)[0]
        )
        return self

    def _neighbors(self, x0, t0, exclude_idx, k):
        q = np.array([x0[0], x0[1], t0 * self.st_ani])
        extra = 1 if exclude_idx is not None else 0
        k_query = min(k + extra, len(self.values))
        dist, idx = self._tree.query(q, k=k_query)
        idx = np.atleast_1d(idx)
        if exclude_idx is not None:
            idx = idx[idx != exclude_idx]
        return idx[:k]

    def _solve(self, idx, x0, t0, raise_when_negative_prediction, raise_when_negative_error):
        if len(idx) < 2:
            raise ValueError(
                "Not enough space-time neighbors to build a kriging system "
                f"(got {len(idx)}); increase number_of_neighbors or check the data."
            )
        coords_i = self.coords[idx]
        times_i = self.times[idx]
        values_i = self.values[idx]
        pops_i = self.populations[idx]
        n = len(idx)

        V = np.asarray(cov_fn_st(coords_i, times_i, coords_i, times_i, self.st_model))
        # Poisson (population) reliability correction on the diagonal
        # (Goovaerts 2006, Eq. 3/7): neighbor i gets `rate_base * m* / n(u_i)`
        # added to its own variance, m* the population-weighted mean rate of
        # the neighborhood -- the space-time analogue of
        # pygstat.poisson_kriging.PoissonKriging._solve.
        mstar = _pk_diagonal_weight(values_i, pops_i)
        V = V + np.diag(self.rate_base * mstar / pops_i)

        target_coord = np.asarray(x0, dtype=float).reshape(1, 2)
        target_time = np.asarray([t0], dtype=float)
        v0 = np.asarray(
            cov_fn_st(coords_i, times_i, target_coord, target_time, self.st_model)
        ).ravel()

        K = np.zeros((n + 1, n + 1))
        K[:n, :n] = V
        K[:n, n] = 1.0
        K[n, :n] = 1.0
        rhs = np.append(v0, 1.0)

        try:
            w = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                "Singular space-time Poisson Kriging matrix -- check for "
                "duplicate area/time coordinates or too few distinct neighbors."
            ) from e

        zhat = float(values_i.dot(w[:-1]))
        if zhat < 0 and raise_when_negative_prediction:
            raise ValueError(
                f"Predicted value is {zhat} and should not be lower than 0. "
                f"Check the variogram model or number_of_neighbors."
            )

        sigmasq = self._c00_cache - float(w.dot(rhs))
        if sigmasq < 0:
            if raise_when_negative_error:
                raise ValueError(
                    f"Predicted error variance is {sigmasq} and should not be "
                    f"lower than 0. Check the variogram model or number_of_neighbors."
                )
            sigma = np.nan
        else:
            sigma = np.sqrt(sigmasq)
        return zhat, sigma

    def _solve_batch(self, idx, X_pred, T_pred, use_gpu,
                     raise_when_negative_prediction, raise_when_negative_error):
        """
        Solve ``n_pred`` independent ST Poisson-kriging systems in one
        batched ``xp.linalg.solve`` of shape ``(n_pred, k+1, k+1)``.
        """
        n_pred, k = idx.shape
        neighbor_coords = self.coords[idx]
        neighbor_times = self.times[idx]
        values = self.values[idx]
        pops = self.populations[idx]
        X_pred = np.asarray(X_pred, dtype=float)
        T_pred = np.asarray(T_pred, dtype=float).ravel()

        if use_gpu:
            nc = as_gpu_array(neighbor_coords)
            nt = as_gpu_array(neighbor_times)
            tg = as_gpu_array(X_pred)
            tt = as_gpu_array(T_pred)
            values_xp = as_gpu_array(values)
            pops_xp = as_gpu_array(pops)
        else:
            nc, nt = neighbor_coords, neighbor_times
            tg, tt = X_pred, T_pred
            values_xp, pops_xp = values, pops
        xp = get_array_module(nc)

        V = _cov_st_batch(nc, nt, nc, nt, self.st_model)
        v0 = _cov_st_batch(
            nc, nt, tg[:, None, :], tt[:, None], self.st_model,
        )[:, :, 0]

        mstar = xp.sum(values_xp * pops_xp, axis=1) / xp.sum(pops_xp, axis=1)
        penalty = self.rate_base * mstar[:, None] / pops_xp
        diag = xp.arange(k)
        V[:, diag, diag] = V[:, diag, diag] + penalty

        K = xp.zeros((n_pred, k + 1, k + 1))
        K[:, :k, :k] = V
        K[:, :k, k] = 1.0
        K[:, k, :k] = 1.0
        rhs = xp.ones((n_pred, k + 1, 1))
        rhs[:, :k, 0] = v0

        try:
            w = xp.linalg.solve(K, rhs)[:, :, 0]
        except xp.linalg.LinAlgError:
            w = xp.linalg.lstsq(K, rhs, rcond=None)[0][:, :, 0]

        zhat = xp.sum(values_xp * w[:, :k], axis=1)
        sigmasq = float(self._c00_cache) - xp.sum(w * rhs[:, :, 0], axis=1)
        zhat = to_numpy(zhat)
        sigmasq = to_numpy(sigmasq)

        if raise_when_negative_prediction and np.any(zhat < 0):
            raise ValueError(
                f"Predicted value is {zhat.min()} and should not be lower than 0. "
                f"Check the variogram model or number_of_neighbors."
            )
        sig = np.sqrt(np.where(sigmasq >= 0, sigmasq, np.nan))
        if raise_when_negative_error and np.any(sigmasq < 0):
            raise ValueError(
                f"Predicted error variance is {np.nanmin(sigmasq)} and should not be "
                f"lower than 0. Check the variogram model or number_of_neighbors."
            )
        return zhat, sig

    def predict(self, X_pred, T_pred, number_of_neighbors=30,
                raise_when_negative_prediction=False, raise_when_negative_error=False):
        """
        Predict the areal rate at new (unobserved) area-time combinations.

        Each target uses its ``number_of_neighbors`` nearest known area-years
        in scaled ``(x, y, a·t)`` space. All ``n_pred`` systems are stacked
        and solved in one batched ``xp.linalg.solve`` — NumPy on CPU, CuPy
        on GPU when ``use_gpu`` is true.

        Parameters
        ----------
        X_pred : array-like, shape (m, 2)
            Coordinates of the locations to predict.
        T_pred : array-like, shape (m,)
            Time stamp of each prediction (same units as ``times`` in :meth:`fit`).
        number_of_neighbors : int, default=30
            Number of known area-years used to build each local kriging system.

        Returns
        -------
        zhat, sig : numpy arrays, shape (m,)
            Predicted rates and kriging standard errors.
        """
        if self.coords is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_pred = np.atleast_2d(np.asarray(X_pred, dtype=float))
        T_pred = np.atleast_1d(np.asarray(T_pred, dtype=float))
        m = len(X_pred)
        if len(T_pred) != m:
            raise ValueError("X_pred and T_pred must have the same length")
        k = int(number_of_neighbors)
        n_train = len(self.values)
        if k < 2:
            raise ValueError(
                "Not enough space-time neighbors to build a kriging system "
                f"(got {k}); increase number_of_neighbors or check the data."
            )
        if k > n_train:
            raise ValueError(
                "Not enough known area-years to satisfy number_of_neighbors "
                f"({k}); only {n_train} available."
            )
        query = np.column_stack([X_pred[:, 0], X_pred[:, 1], T_pred * self.st_ani])
        _, idx = self._tree.query(query, k=k)
        idx = np.asarray(idx, dtype=int)
        if k == 1:
            idx = idx.reshape(-1, 1)
        else:
            idx = np.atleast_2d(idx)
            if idx.shape[1] != k:
                idx = idx.reshape(m, k)
        return self._solve_batch(
            idx, X_pred, T_pred, self.use_gpu,
            raise_when_negative_prediction, raise_when_negative_error,
        )

    def predict_id(self, area_id, time, number_of_neighbors=30,
                    raise_when_negative_prediction=False, raise_when_negative_error=False):
        """
        Predict the value of one of the *known* area-years, excluding it from
        its own neighbor search (leave-one-out). Useful for validating the
        model against observed rates.

        Returns
        -------
        dict
            ``{"id": area_id, "year": time, "zhat": prediction, "sig": std_error}``
        """
        matches = np.where((self.ids == area_id) & np.isclose(self.times, time))[0]
        if len(matches) == 0:
            raise ValueError(f"(id={area_id!r}, time={time!r}) not found in the fitted data")
        target_idx = int(matches[0])
        x0 = self.coords[target_idx]
        t0 = self.times[target_idx]
        idx = self._neighbors(x0, t0, target_idx, number_of_neighbors)
        zhat, sig = self._solve(
            idx, x0, t0, raise_when_negative_prediction, raise_when_negative_error
        )
        return {"id": area_id, "year": time, "zhat": zhat, "sig": sig}


# ---------------------------------------------------------------------------
# Data loading (Atlantic county lung/bronchus cancer test data)
# ---------------------------------------------------------------------------
def load_atlantic_panel(data_dir):
    """
    Build a long (FIPS, year, rate, x, y, pop) panel from
    ``lbc_atlantic.csv`` (wide rate panel) + ``lbc_rate.csv`` (centroid +
    population per county). See the module docstring for the population
    caveat.
    """
    data_dir = Path(data_dir)
    atlantic = pd.read_csv(data_dir / "lbc_atlantic.csv")
    rate_meta = pd.read_csv(data_dir / "lbc_rate.csv")

    year_cols = [c for c in atlantic.columns if c != "FIPS"]
    long = atlantic.melt(id_vars="FIPS", value_vars=year_cols,
                          var_name="year", value_name="rate")
    long["year"] = long["year"].astype(int)

    long = long.merge(rate_meta[["FIPS", "x", "y", "pop"]], on="FIPS", how="left")
    missing = long[long["pop"].isna()]["FIPS"].unique()
    if len(missing):
        raise ValueError(
            f"{len(missing)} FIPS in lbc_atlantic.csv have no matching row in "
            f"lbc_rate.csv (no population/coordinates): {missing[:10]}"
        )
    return long.sort_values(["FIPS", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fit and validate Spatio-Temporal Poisson Kriging on the "
                     "Atlantic county lung/bronchus cancer rate panel."
    )
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument("--data-dir", default=str(repo_root / "data"))
    parser.add_argument("--n-test-counties", type=int, default=60,
                         help="Number of counties held out entirely for validation.")
    parser.add_argument("--n-neighbors", type=int, default=30,
                         help="Number of space-time neighbors per kriging system.")
    parser.add_argument("--n-space-bins", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="STPoisson_kriging_predictions.csv")
    parser.add_argument("--plot", default="STPoisson_kriging_diagnostics.png")
    args = parser.parse_args()

    panel = load_atlantic_panel(args.data_dir)
    n_counties = panel["FIPS"].nunique()
    n_years = panel["year"].nunique()
    print(f"Loaded {len(panel)} county-year rows "
          f"({n_counties} counties x {n_years} years, "
          f"{panel['year'].min()}-{panel['year'].max()})")

    # --- Hold out whole counties for validation (spatial-temporal
    # extrapolation: none of a held-out county's years are seen in training).
    rng = np.random.default_rng(args.seed)
    fips_all = panel["FIPS"].unique()
    test_fips = rng.choice(fips_all, size=args.n_test_counties, replace=False)
    is_test = panel["FIPS"].isin(test_fips)
    train = panel.loc[~is_test].reset_index(drop=True)
    test = panel.loc[is_test].reset_index(drop=True)
    print(f"Train: {train['FIPS'].nunique()} counties ({len(train)} rows)  |  "
          f"Test (held out counties): {test['FIPS'].nunique()} counties ({len(test)} rows)")

    # --- Empirical + fitted space-time variogram (training data only)
    train_wide = train.pivot(index="FIPS", columns="year", values="rate").sort_index()
    train_coords = (
        train[["FIPS", "x", "y"]].drop_duplicates("FIPS").set_index("FIPS").loc[train_wide.index]
        [["x", "y"]].to_numpy()
    )
    years = np.sort(train["year"].unique()).astype(float)

    print("Estimating empirical space-time semivariogram ...")
    emp = empirical_st_variogram(train_coords, years, train_wide.to_numpy(),
                                  n_space_bins=args.n_space_bins)
    print(f"  {len(emp['dist'])} (space-bin, time-lag) cells from {emp['np'].sum():.0f} pairs")

    print("Fitting 'metric' space-time variogram model ...")
    fitted_model = fit_metric_st_variogram(emp)
    nugget, sill, prange = joint_nugget_sill_range(fitted_model["joint"])
    print(f"  nugget={nugget:.3f}  sill={sill:.3f}  range={prange:.0f} m  "
          f"stAni={fitted_model['stAni']:.0f} m/yr  MSE={fitted_model['MSE']:.4f}")

    # --- Fit Spatio-Temporal Poisson Kriging on the training set
    stpk = STPoissonKriging(fitted_model).fit(
        coords=train[["x", "y"]].to_numpy(),
        times=train["year"].to_numpy(float),
        values=train["rate"].to_numpy(),
        populations=train["pop"].to_numpy(),
        ids=train["FIPS"].to_numpy(),
    )
    print(f"  auto-detected rate_base = {stpk.rate_base:.0f}")

    # --- Validate: predict every held-out county-year from training neighbors
    print(f"Predicting {len(test)} held-out county-years "
          f"(number_of_neighbors={args.n_neighbors}) ...")
    zhat, sig = stpk.predict(
        test[["x", "y"]].to_numpy(), test["year"].to_numpy(float),
        number_of_neighbors=args.n_neighbors,
    )
    test = test.assign(zhat=zhat, sig=sig)

    resid = test["rate"] - test["zhat"]
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    r2 = 1.0 - float(np.sum(resid ** 2)) / float(np.sum((test["rate"] - test["rate"].mean()) ** 2))

    # Naive baseline for comparison: each held-out county-year predicted by
    # the overall training mean rate (no spatial or temporal information).
    baseline_pred = train["rate"].mean()
    base_resid = test["rate"] - baseline_pred
    base_rmse = float(np.sqrt(np.mean(base_resid ** 2)))

    print("\n--- Held-out county validation ---")
    print(f"  RMSE            : {rmse:.3f}  (rate units, per 100,000)")
    print(f"  MAE             : {mae:.3f}")
    print(f"  R^2             : {r2:.3f}")
    print(f"  Naive-mean RMSE : {base_rmse:.3f}  (baseline for comparison)")
    print(f"  Mean kriging sd : {test['sig'].mean():.3f}")

    # --- Leave-one-out sanity check on a handful of TRAINING points
    print("\n--- Leave-one-out check on 10 random training area-years ---")
    sample = train.sample(10, random_state=args.seed)
    loo_rows = []
    for _, row in sample.iterrows():
        out = stpk.predict_id(row["FIPS"], row["year"], number_of_neighbors=args.n_neighbors)
        loo_rows.append({"FIPS": row["FIPS"], "year": row["year"],
                          "observed": row["rate"], "zhat": out["zhat"], "sig": out["sig"]})
    loo = pd.DataFrame(loo_rows)
    print(loo.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # --- Forecast one year beyond the observed range (2013) for 3 counties
    print("\n--- Forecast: 2013 (one year beyond the observed 1998-2012 range) ---")
    demo_fips = train["FIPS"].drop_duplicates().sample(3, random_state=args.seed).to_numpy()
    demo_coords = (
        train[["FIPS", "x", "y"]].drop_duplicates("FIPS").set_index("FIPS").loc[demo_fips]
        [["x", "y"]].to_numpy()
    )
    fzhat, fsig = stpk.predict(demo_coords, np.full(len(demo_fips), 2013.0),
                                number_of_neighbors=args.n_neighbors)
    for fips, zh, sg in zip(demo_fips, fzhat, fsig):
        last_obs = train.loc[train["FIPS"] == fips].sort_values("year").iloc[-1]["rate"]
        print(f"  FIPS {fips}: 2012 observed={last_obs:.2f}  ->  "
              f"2013 forecast={zh:.2f} +/- {sg:.2f}")

    test[["FIPS", "year", "rate", "zhat", "sig"]].to_csv(args.output, index=False)
    print(f"\nSaved held-out predictions to {args.output}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(test["rate"], test["zhat"], s=10, alpha=0.5)
        lims = [min(test["rate"].min(), test["zhat"].min()),
                max(test["rate"].max(), test["zhat"].max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Observed rate (per 100,000)")
        ax.set_ylabel("ST Poisson Kriging prediction")
        ax.set_title(f"Held-out counties (RMSE={rmse:.2f})")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Saved diagnostic plot to {args.plot}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
