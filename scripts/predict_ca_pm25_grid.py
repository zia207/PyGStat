#!/usr/bin/env python3
"""
predict_ca_pm25_grid.py
========================

Predict monthly PM2.5 (2025) on a 5 km x 5 km grid covering California.

Extracts the California boundary from ``data/US_STATE.shp``, builds a
regular 5 km grid clipped to it (:func:`pygstat.krigeST.make_prediction_grid`),
fits Spatio-Temporal Ordinary Kriging (:class:`pygstat.krigeST.STKriging`)
on the full ``data/CA_pm25_2025.csv`` monitoring-network panel (163
stations, all 12 months), and krige-predicts PM2.5 at every grid cell for
each of the 12 months.

Outputs
-------
* ``data/CA_pm25_grid_predictions.csv`` -- one row per (grid cell, month):
  ``x, y, month, month_name, pm25_pred, pm25_sd``.
* a 12-panel PNG map of the predicted monthly PM2.5 surface (path printed
  at the end; pass ``--plot`` to choose it).

Usage
-----
    python scripts/predict_ca_pm25_grid.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pygstat import STKriging, load_ca_pm25_panel
from pygstat.krigeST import make_prediction_grid
from pygstat.st_variogram_models import (
    empirical_st_variogram,
    fit_metric_st_variogram,
    joint_nugget_sill_range,
)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main():
    parser = argparse.ArgumentParser(
        description="Krige monthly PM2.5 (2025) on a 5km grid over California."
    )
    parser.add_argument("--states-shp", default=str(REPO_ROOT / "data" / "US_STATE.shp"))
    parser.add_argument("--csv", default=str(REPO_ROOT / "data" / "CA_pm25_2025.csv"))
    parser.add_argument("--cell-size", type=float, default=5000.0, help="Grid spacing in meters.")
    parser.add_argument("--nmax", type=int, default=20)
    parser.add_argument("--n-space-bins", type=int, default=10)
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "CA_pm25_grid_predictions.csv"))
    parser.add_argument("--plot", default=str(REPO_ROOT / "CA_pm25_grid_2025.png"))
    args = parser.parse_args()

    # --- 1. California boundary, reprojected to match the station CRS (EPSG:3310)
    states = gpd.read_file(args.states_shp)
    ca = states[states["STATE"] == "California"]
    if ca.empty:
        raise ValueError("California not found in US_STATE.shp's STATE column")
    ca = ca.to_crs("EPSG:3310")
    print(f"California boundary: {len(ca)} polygon feature(s), reprojected to EPSG:3310")

    # --- 2. 5km x 5km prediction grid, clipped to California
    grid = make_prediction_grid(ca, cell_size=args.cell_size)
    print(f"Prediction grid: {len(grid)} cells at "
          f"{args.cell_size / 1000:.0f}km x {args.cell_size / 1000:.0f}km inside California")

    # --- 3. Station panel + empirical/fitted space-time variogram
    panel = load_ca_pm25_panel(args.csv)
    counts = panel.groupby("site_id").size()
    complete_sites = counts[counts == 12].index.to_numpy()
    comp = panel[panel["site_id"].isin(complete_sites)]
    wide = comp.pivot(index="site_id", columns="month", values="pm25").sort_index()
    coords = (
        comp[["site_id", "x", "y"]].drop_duplicates("site_id").set_index("site_id")
        .loc[wide.index][["x", "y"]].to_numpy()
    )
    months = np.sort(comp["month"].unique()).astype(float)

    print(f"Loaded {len(panel)} station-month rows ({panel['site_id'].nunique()} stations); "
          f"{len(complete_sites)} report all 12 months (used for variogram estimation).")
    print("Estimating empirical space-time semivariogram ...")
    emp = empirical_st_variogram(coords, months, wide.to_numpy(), n_space_bins=args.n_space_bins)
    fitted_model = fit_metric_st_variogram(emp)
    nugget, sill, prange = joint_nugget_sill_range(fitted_model["joint"])
    print(f"  nugget={nugget:.3f}  sill={sill:.2f}  range={prange:.0f} m  "
          f"stAni={fitted_model['stAni']:.0f} m/month  MSE={fitted_model['MSE']:.3f}")

    # --- 4. Fit STKriging on ALL stations (complete + partial-year)
    stk = STKriging(fitted_model).fit(
        coords=panel[["x", "y"]].to_numpy(),
        times=panel["month"].to_numpy(float),
        values=panel["pm25"].to_numpy(),
        ids=panel["site_id"].to_numpy(),
    )

    # --- 5. Predict at every grid cell, for each of the 12 months
    grid_xy = grid[["x", "y"]].to_numpy()
    rows = []
    for m in range(12):
        zhat, sig = stk.predict(grid_xy, np.full(len(grid_xy), float(m)), nmax=args.nmax)
        # Ordinary kriging is a linear interpolator with no positivity
        # constraint; a handful of cells far from any station (~0.1% here)
        # can come out very slightly negative. PM2.5 concentration cannot
        # be negative, so clip -- this only ever nudges values that were
        # already within a fraction of a ug/m3 of zero.
        n_clipped = int(np.sum(zhat < 0))
        zhat = np.clip(zhat, 0.0, None)
        if n_clipped:
            print(f"  ({MONTH_NAMES[m]}: clipped {n_clipped} slightly-negative cells to 0)")
        rows.append(pd.DataFrame({
            "x": grid_xy[:, 0], "y": grid_xy[:, 1],
            "month": m, "month_name": MONTH_NAMES[m],
            "pm25_pred": zhat, "pm25_sd": sig,
        }))
        print(f"  {MONTH_NAMES[m]}: mean={zhat.mean():.2f}  "
              f"range=[{zhat.min():.2f}, {zhat.max():.2f}] ug/m3")
    out = pd.concat(rows, ignore_index=True)

    out.to_csv(args.output, index=False)
    print(f"\nSaved {len(out)} grid-month predictions to {args.output}")

    # --- 6. 12-panel map
    vmin, vmax = out["pm25_pred"].quantile(0.02), out["pm25_pred"].quantile(0.98)
    fig, axes = plt.subplots(3, 4, figsize=(16, 13), sharex=True, sharey=True)
    for m, ax in enumerate(axes.ravel()):
        sub = out[out["month"] == m]
        sc = ax.scatter(sub["x"], sub["y"], c=sub["pm25_pred"], cmap="RdYlGn_r",
                         vmin=vmin, vmax=vmax, s=5, marker="s")
        ca.boundary.plot(ax=ax, color="black", linewidth=0.6)
        ax.set_title(MONTH_NAMES[m], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(sc, ax=axes, shrink=0.6, label="Predicted PM2.5 (ug/m3)")
    fig.suptitle(f"California PM2.5 -- Spatio-Temporal Kriging, "
                 f"{args.cell_size / 1000:.0f}km grid, 2025", fontsize=14)
    fig.savefig(args.plot, dpi=120, bbox_inches="tight")
    print(f"Saved 12-panel map to {args.plot}")


if __name__ == "__main__":
    main()
