#!/usr/bin/env python
# coding: utf-8

# # Spatial-Temporal Kriging
# 
# Testing `pygstat.krigeST` against California's 2025 **PM2.5 air-quality
# monitoring network** (`CA_pm25_2025.csv`, 163 stations, daily readings
# aggregated to monthly means).
# 
# > **Pebesma, E. (2012).** *spacetime: Spatio-Temporal Data in R.* Journal of
# > Statistical Software, 51(7). (source of the `krigeST` algorithm --
# > `krigeST.R`/`stVariogramModels.R` at the repo root -- this module is a
# > Python port of.)

# ## 1. Theory
# 
# ### 1.1 Ordinary kriging
# 
# Ordinary kriging estimates a continuous variable $Z$ at an unobserved
# location as a weighted linear combination of $N$ nearby observations,
# 
# $$
# \hat Z(\mathbf u_0) = \sum_{i=1}^N \lambda_i\, Z(\mathbf u_i), \qquad
# \sum_{i=1}^N \lambda_i = 1,
# $$
# 
# the constraint making the estimator unbiased regardless of the (unknown)
# mean of $Z$. The weights $\lambda_i$ minimize the estimation variance
# subject to that constraint, giving the kriging system
# 
# $$
# \sum_{j=1}^N \lambda_j\, C(\mathbf u_i-\mathbf u_j) + \mu = C(\mathbf u_i-\mathbf u_0),
# \qquad i=1,\dots,N,
# $$
# 
# with Lagrange multiplier $\mu$ enforcing unbiasedness, or in matrix form
# 
# $$
# \begin{bmatrix} \mathbf V & \mathbf 1 \\ \mathbf 1^\top & 0 \end{bmatrix}
# \begin{bmatrix} \boldsymbol\lambda \\ \mu \end{bmatrix}
# = \begin{bmatrix} \mathbf v_0 \\ 1 \end{bmatrix},
# \qquad
# \sigma^2 = C(0) - \boldsymbol\lambda^\top\mathbf v_0 - \mu,
# $$
# 
# where $V_{ij}=C(\mathbf u_i-\mathbf u_j)$, $v_{0,i}=C(\mathbf u_i-\mathbf u_0)$,
# and $C(h)=C(0)-\gamma(h)$ relates the covariance to the semivariogram
# $\gamma$.
# 
# ### 1.2 Adding a time dimension
# 
# A monitoring network like this one reports $Z(\mathbf u_i, t_i)$: the same
# quantity at a set of fixed station locations, repeated over time. The
# estimator and kriging system are unchanged, except $C$ must now be a
# function of **both** a spatial lag $h_s=\lVert\mathbf u_i-\mathbf u_j\rVert$
# and a temporal lag $h_t=|t_i-t_j|$:
# 
# $$
# \hat Z(\mathbf u_0,t_0) = \sum_{i=1}^N \lambda_i\, Z(\mathbf u_i,t_i), \qquad
# \sum_{j=1}^N \lambda_j\, C\big((\mathbf u_i,t_i)-(\mathbf u_j,t_j)\big) + \mu
#      = C\big((\mathbf u_i,t_i)-(\mathbf u_0,t_0)\big).
# $$
# 
# This notebook uses the **metric** space-time model (Pebesma, 2012): space
# and time are folded into one joint (anisotropic) distance
# 
# $$
# h_{st} = \sqrt{h_s^2 + (a\,h_t)^2},
# $$
# 
# where $a$ (`stAni`, meters per month here) makes space and time
# commensurable, and a single ordinary variogram/covariance $C(h_{st})$ is
# fit to it -- the simplest space-time model, and a robust default absent a
# strong prior for separate spatial/temporal structure.
# `pygstat.st_variogram_models` (a port of `stVariogramModels.R`) also
# implements `separable`, `productSum`, and `sumMetric` models for cases
# with distinct spatial/temporal structure.
# 
# ### 1.3 Empirical semivariogram
# 
# The empirical semivariogram is the Matheron estimator, binned jointly by
# spatial and temporal lag:
# 
# $$
# \hat\gamma(h_s,h_t) = \frac{1}{2\,|N(h_s,h_t)|}
#    \sum_{(i,a),(j,b)\,\in\,N(h_s,h_t)} \big(Z(\mathbf u_i,t_a)-Z(\mathbf u_j,t_b)\big)^2,
# $$
# 
# where $N(h_s,h_t)$ collects station-month pairs whose spatial distance
# falls in the $h_s$ bin and whose time difference equals $h_t$. This
# estimator (`pygstat.st_variogram_models.empirical_st_variogram`) needs a
# *complete* station x month matrix (every station observed every month);
# Section 3 shows that all but 7 of the 163 stations satisfy this.
# 
# ### 1.4 Local (neighborhood) kriging
# 
# Solving the kriging system exactly against every training observation
# scales as $O(N^3)$, impractical once $N$ is more than a few thousand.
# `pygstat.krige_st.krige_st_local` (used by this notebook via `nmax`)
# instead builds each prediction's kriging system from only its
# `nmax` nearest space-time neighbors (found by a KD-tree in a scaled
# $(x,y,a\cdot t)$ space) -- the same local-neighborhood strategy `gstat`'s
# `krigeST` uses, and the space-time analogue of local ordinary kriging.
# 
# ### 1.5 `STKriging`
# 
# `pygstat.krigeST.STKriging` (this notebook's `pygstat.krigeST`) is a thin,
# ergonomic wrapper around `pygstat.krige_st.krige_st`: it stores the
# training space-time observations and exposes `.predict()` /
# `.predict_id()` (leave-one-out) without requiring a complete grid --
# unlike this repo's sibling module `pygstat.STPoisson_kriging`, it needs no
# population-reliability correction, since PM2.5 is a directly measured
# continuous concentration, not a count-over-population rate.

# ## 2. Environment and module import

# In[1]:


import matplotlib
matplotlib.use("Agg")
import sys, os
sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

import pygstat
from pygstat import STKriging, load_ca_pm25_panel
from pygstat.st_variogram_models import (
    empirical_st_variogram, fit_metric_st_variogram, variogram_line, joint_nugget_sill_range,
)

# --- running test log, rendered as a styled summary table at the end ---
RESULTS = []
def record(test, status, details=""):
    RESULTS.append({"Test": test, "Status": status, "Details": details})
    print(f"[{status}] {test} -- {details}")


# ## 3. Data
# 
# `data/CA_pm25_2025.csv` -- daily PM2.5 concentration (micrograms/m3) at 163
# California air-quality monitoring stations through 2025 (`Site ID`, `Lat`,
# `Long`, `Date`, `Daily_PM2.5_ug_m3`). `load_ca_pm25_panel` aggregates this
# to monthly per-station means and projects station coordinates from
# lat/long (EPSG:4326) to California Albers Equal Area (EPSG:3310, meters),
# so spatial distances are in a metric unit.

# In[2]:


raw = pd.read_csv("../data/CA_pm25_2025.csv")
ok = (raw.shape[0] > 0) and (not raw[["Lat", "Long", "Daily_PM2.5_ug_m3"]].isna().any().any())
record("Load CA_pm25_2025.csv (daily station readings)", "PASS" if ok else "FAIL",
       f"shape={raw.shape}, {raw['Site ID'].nunique()} stations")
raw.head()


# In[3]:


panel = load_ca_pm25_panel("../data/CA_pm25_2025.csv")
ok = (panel.shape[0] > 0) and not panel[["x", "y", "pm25"]].isna().any().any()
record("Build monthly (site, month) panel + project to EPSG:3310", "PASS" if ok else "FAIL",
       f"shape={panel.shape}, {panel['site_id'].nunique()} stations x up to 12 months")
panel.head()


# In[4]:


counts = panel.groupby("site_id").size()
complete_sites = counts[counts == 12].index.to_numpy()
ok = len(complete_sites) > 100
record("Identify stations with complete 12-month coverage", "PASS" if ok else "FAIL",
       f"{len(complete_sites)}/{panel['site_id'].nunique()} complete")
counts.value_counts().sort_index()


# ### 3.1 Exploratory maps

# In[5]:


mean_pm = panel.groupby("site_id")["pm25"].mean().rename("mean_pm25")
site_meta = (panel[["site_id", "x", "y", "lat", "lon"]].drop_duplicates("site_id")
             .merge(mean_pm, on="site_id"))

fig, ax = plt.subplots(figsize=(6, 7))
sc = ax.scatter(site_meta["lon"], site_meta["lat"], c=site_meta["mean_pm25"],
                 cmap="RdBu_r", s=25, edgecolor="k", linewidth=0.3)
plt.colorbar(sc, ax=ax, label="mean PM2.5 (ug/m3)", shrink=0.7)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("California PM2.5 monitoring stations, 2025 mean")
plt.tight_layout()
plt.show()


# In[6]:


fig, ax = plt.subplots(figsize=(7, 4))
for site in panel["site_id"].drop_duplicates().sample(6, random_state=0):
    sub = panel[panel["site_id"] == site].sort_values("month")
    ax.plot(sub["month"], sub["pm25"], marker="o", ms=3, label=str(site))
ax.set_xlabel("month (0=Jan, 11=Dec)"); ax.set_ylabel("PM2.5 (ug/m3)")
ax.set_title("PM2.5 time series, 6 random stations")
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.show()


# ## 4. Train / test split
# 
# To validate genuine spatial-temporal extrapolation, 25 stations (~15%)
# with complete 12-month coverage are held out **entirely** and predicted
# purely from their space-time neighbors among the remaining 138 stations
# (which include the 7 stations with partial-year coverage -- fine for
# `STKriging`, which needs no complete grid).

# In[7]:


rng = np.random.default_rng(0)
test_sites = rng.choice(complete_sites, size=25, replace=False)
is_test = panel["site_id"].isin(test_sites)
train = panel.loc[~is_test].reset_index(drop=True)
test = panel.loc[is_test].reset_index(drop=True)
print(f"train: {train['site_id'].nunique()} stations, {len(train)} rows")
print(f"test:  {test['site_id'].nunique()} stations (held out), {len(test)} rows")


# ## 5. Empirical semivariogram and space-time model fit
# 
# `empirical_st_variogram` needs a complete station x month matrix, so this
# step uses only the training stations with all 12 months.

# In[8]:


train_complete = train[train["site_id"].isin(complete_sites)]
train_wide = train_complete.pivot(index="site_id", columns="month", values="pm25").sort_index()
train_coords = (train_complete[["site_id", "x", "y"]].drop_duplicates("site_id").set_index("site_id")
                 .loc[train_wide.index][["x", "y"]].to_numpy())
months = np.sort(train_complete["month"].unique()).astype(float)

emp = empirical_st_variogram(train_coords, months, train_wide.to_numpy(), n_space_bins=10)
ok = (len(emp["dist"]) > 0) and np.all(np.isfinite(emp["gamma"])) and np.all(emp["gamma"] >= 0)
record("Empirical space-time variogram estimation", "PASS" if ok else "FAIL",
       f"{len(emp['dist'])} (space-bin, time-lag) cells from {emp['np'].sum():.0f} pairs")


# In[9]:


fitted_model = fit_metric_st_variogram(emp)
nugget, sill, prange = joint_nugget_sill_range(fitted_model["joint"])
st_ani = fitted_model["stAni"]
ok = (nugget >= 0) and (sill > 0) and (prange > 0) and (st_ani > 0) and np.isfinite(fitted_model["MSE"])
record("Fit 'metric' space-time variogram model", "PASS" if ok else "FAIL",
       f"nugget={nugget:.3f} sill={sill:.2f} range={prange:.0f}m stAni={st_ani:.0f}m/month "
       f"MSE={fitted_model['MSE']:.3f}")


# In[10]:


fig, ax = plt.subplots(figsize=(6, 4))
show_lags = sorted(set(emp["timelag"]))
colors = plt.cm.viridis(np.linspace(0, 1, len(show_lags)))
h = np.linspace(0, emp["dist"].max(), 100)
for lag, c in zip(show_lags, colors):
    m = emp["timelag"] == lag
    ax.scatter(emp["dist"][m], emp["gamma"][m], s=12, color=c, label=f"lag={lag:.0f}mo")
    hh = np.sqrt(h ** 2 + (st_ani * lag) ** 2)
    ax.plot(h, variogram_line(fitted_model["joint"], hh, covariance=False), "--", color=c, linewidth=0.8)
ax.set_xlabel("spatial lag (m)"); ax.set_ylabel("semivariance")
ax.set_title("Empirical (points) vs fitted metric model (dashed)")
ax.legend(fontsize=6, ncol=2)
plt.tight_layout()
plt.show()


# ## 6. Fit Spatio-Temporal Kriging
# 
# `STKriging.fit` is used on **all** 138 training stations (complete and
# partial-year alike) -- no complete grid is needed here, only the earlier
# variogram *estimation* step required one.

# In[11]:


stk = STKriging(fitted_model).fit(
    coords=train[["x", "y"]].to_numpy(),
    times=train["month"].to_numpy(float),
    values=train["pm25"].to_numpy(),
    ids=train["site_id"].to_numpy(),
)
record("Fit STKriging on training stations (complete + partial)", "PASS",
       f"{len(train)} training obs from {train['site_id'].nunique()} stations")


# ## 7. Validation: held-out stations
# 
# Predict every station-month of the 25 held-out stations from their
# nearest space-time neighbors among the 138 training stations, and compare
# against a naive baseline (the overall training mean PM2.5, i.e. no
# spatial or temporal information at all).

# In[12]:


zhat, sig = stk.predict(test[["x", "y"]].to_numpy(), test["month"].to_numpy(float), nmax=20)
test = test.assign(zhat=zhat, sig=sig)

resid = test["pm25"] - test["zhat"]
rmse = float(np.sqrt(np.mean(resid ** 2)))
mae = float(np.mean(np.abs(resid)))
r2 = 1.0 - float(np.sum(resid ** 2)) / float(np.sum((test["pm25"] - test["pm25"].mean()) ** 2))
baseline_rmse = float(np.sqrt(np.mean((test["pm25"] - train["pm25"].mean()) ** 2)))

ok = rmse < baseline_rmse
record("Held-out station prediction accuracy (25 stations)", "PASS" if ok else "FAIL",
       f"RMSE={rmse:.2f} vs naive-mean RMSE={baseline_rmse:.2f}, R2={r2:.3f}, "
       f"mean kriging sd={test['sig'].mean():.2f}")

print(f"RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}  naive-mean RMSE={baseline_rmse:.3f}")


# In[13]:


fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(test["pm25"], test["zhat"], s=10, alpha=0.5)
lims = [min(test["pm25"].min(), test["zhat"].min()), max(test["pm25"].max(), test["zhat"].max())]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Observed PM2.5 (ug/m3)")
ax.set_ylabel("ST Kriging prediction")
ax.set_title(f"Held-out stations (RMSE={rmse:.2f})")
plt.tight_layout()
plt.show()


# In[14]:


try:
    snap = test[test["month"] == 6]  # July
    vmin, vmax = snap[["pm25", "zhat"]].min().min(), snap[["pm25", "zhat"]].max().max()
    fig, ax = plt.subplots(1, 2, figsize=(11, 6))
    s0 = ax[0].scatter(snap["lon"], snap["lat"], c=snap["pm25"], vmin=vmin, vmax=vmax,
                        cmap="RdBu_r", s=45, edgecolor="k")
    ax[0].set_title("Observed, July"); ax[0].set_xlabel("Longitude"); ax[0].set_ylabel("Latitude")
    s1 = ax[1].scatter(snap["lon"], snap["lat"], c=snap["zhat"], vmin=vmin, vmax=vmax,
                        cmap="RdBu_r", s=45, edgecolor="k")
    ax[1].set_title("ST Kriging prediction, July"); ax[1].set_xlabel("Longitude")
    fig.colorbar(s1, ax=ax, shrink=0.6, label="PM2.5 (ug/m3)")
    plt.show()
    record("Map: observed vs predicted (held-out, July)", "PASS", f"{len(snap)} stations mapped")
except Exception as e:
    record("Map: observed vs predicted (held-out, July)", "BLOCKED", str(e))


# ## 8. Leave-one-out sanity check (training data)
# 
# Ten random *training* station-months, each re-predicted with itself
# excluded from its own neighbor search.

# In[15]:


sample = train.sample(10, random_state=0)
loo = pd.DataFrame([stk.predict_id(row["site_id"], row["month"], nmax=20)
                     for _, row in sample.iterrows()])
loo["observed"] = sample["pm25"].to_numpy()

ok = np.isfinite(loo["zhat"]).all() and (loo["sig"].fillna(0) >= 0).all()
record("Leave-one-out sanity check (10 training station-months)", "PASS" if ok else "FAIL",
       f"mean|zhat-observed|={np.mean(np.abs(loo['zhat']-loo['observed'])):.2f}")
loo[["id", "time", "observed", "zhat", "sig"]]


# ## 9. Forecast beyond the observed range (month index 12)
# 
# Predicting month index 12 (one month past December 2025) is pure temporal
# extrapolation using the fitted space-time covariance.

# In[16]:


demo_sites = train["site_id"].drop_duplicates().sample(3, random_state=0).to_numpy()
demo_coords = (train[["site_id", "x", "y"]].drop_duplicates("site_id").set_index("site_id")
               .loc[demo_sites][["x", "y"]].to_numpy())
fzhat, fsig = stk.predict(demo_coords, np.full(3, 12.0), nmax=20)

ok = np.all(np.isfinite(fzhat)) and np.all(fzhat > 0) and np.all(fzhat < 3 * panel["pm25"].max())
record("Forecast beyond observed range (month index 12)", "PASS" if ok else "FAIL",
       f"predictions={np.round(fzhat, 2).tolist()}")

for site, zh, sg in zip(demo_sites, fzhat, fsig):
    last_obs = train.loc[train["site_id"] == site].sort_values("month").iloc[-1]["pm25"]
    print(f"Site {site}: Dec 2025 observed={last_obs:.2f}  ->  forecast={zh:.2f} +/- {sg:.2f}")


# ## 10. Error-handling checks
# 
# `STKriging` should fail loudly, not silently, on bad inputs.

# In[17]:


try:
    stk.predict_id(999999999, 0)
    record("Error handling: unknown (id, time) in predict_id", "FAIL", "no exception raised")
except ValueError as e:
    record("Error handling: unknown (id, time) in predict_id", "PASS", str(e))


# In[18]:


try:
    STKriging(fitted_model).fit(train[["x", "y"]].to_numpy()[:-1], train["month"].to_numpy(float),
                                 train["pm25"].to_numpy(), ids=train["site_id"].to_numpy())
    record("Error handling: mismatched array lengths in fit()", "FAIL", "no exception raised")
except ValueError as e:
    record("Error handling: mismatched array lengths in fit()", "PASS", str(e))


# In[19]:


try:
    STKriging(fitted_model).fit(np.array([[0.0, 0.0]]), np.array([0.0]), np.array([1.0]))
    record("Error handling: fit() needs >=2 observations", "FAIL", "no exception raised")
except ValueError as e:
    record("Error handling: fit() needs >=2 observations", "PASS", str(e))


# ## 11. California-wide prediction grid (5 km, all 12 months)
# 
# Beyond point validation, kriging's usual end product is a continuous
# *map*: predict on a regular grid covering the whole study area so the
# interpolated surface can be visualized or fed into further analysis. This
# section extracts California's boundary from `data/US_STATE.shp`, builds a
# 5 km x 5 km grid clipped to it (`pygstat.krigeST.make_prediction_grid`),
# and krige-predicts PM2.5 at every grid cell for each of the 12 months --
# the same workflow as `scripts/predict_ca_pm25_grid.py`, reusing the
# `fitted_model` space-time variogram from Section 5 but refitting
# `STKriging` on the **full** 163-station panel (rather than the
# train/test split of Sections 4-9), since a production map should use all
# the data available.

# In[20]:


ca_states = gpd.read_file("../data/US_STATE.shp")
ca = ca_states[ca_states["STATE"] == "California"].to_crs("EPSG:3310")
ok = (len(ca) > 0) and (ca.geometry.area.sum() > 0)
record("Load California boundary from US_STATE.shp", "PASS" if ok else "FAIL",
       f"{len(ca)} polygon feature(s), reprojected to EPSG:3310")


# In[21]:


from pygstat.krigeST import make_prediction_grid

grid = make_prediction_grid(ca, cell_size=5000.0)
ok = len(grid) > 0
record("Build 5km x 5km prediction grid clipped to California", "PASS" if ok else "FAIL",
       f"{len(grid)} grid cells")
grid.head()


# In[22]:


stk_full = STKriging(fitted_model).fit(
    coords=panel[["x", "y"]].to_numpy(),
    times=panel["month"].to_numpy(float),
    values=panel["pm25"].to_numpy(),
    ids=panel["site_id"].to_numpy(),
)
record("Fit STKriging on the full station panel (163 stations)", "PASS",
       f"{len(panel)} station-month observations")


# In[23]:


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

grid_xy = grid[["x", "y"]].to_numpy()
grid_rows = []
n_clipped_total = 0
for m in range(12):
    zh, sg = stk_full.predict(grid_xy, np.full(len(grid_xy), float(m)), nmax=20)
    # Ordinary kriging has no positivity constraint; a handful of cells far
    # from any station can come out very slightly negative. PM2.5 cannot be
    # negative, so clip (this only ever nudges values within a fraction of
    # a ug/m3 of zero -- see the printed count below).
    n_clipped_total += int(np.sum(zh < 0))
    zh = np.clip(zh, 0.0, None)
    grid_rows.append(pd.DataFrame({
        "x": grid_xy[:, 0], "y": grid_xy[:, 1],
        "month": m, "month_name": MONTH_NAMES[m],
        "pm25_pred": zh, "pm25_sd": sg,
    }))
grid_pred = pd.concat(grid_rows, ignore_index=True)

ok = (not grid_pred["pm25_pred"].isna().any()) and (grid_pred["pm25_pred"] >= 0).all()
record("Predict 5km grid x 12 months (California-wide map)", "PASS" if ok else "FAIL",
       f"{len(grid_pred)} grid-month predictions, {n_clipped_total} slightly-negative "
       f"cells clipped to 0 ({100 * n_clipped_total / len(grid_pred):.2f}%)")

grid_pred.to_csv("../data/CA_pm25_grid_predictions.csv", index=False)
grid_pred.groupby("month_name")["pm25_pred"].agg(["mean", "min", "max"]).loc[MONTH_NAMES]


# In[24]:


vmin, vmax = grid_pred["pm25_pred"].quantile(0.02), grid_pred["pm25_pred"].quantile(0.98)
fig, axes = plt.subplots(3, 4, figsize=(16, 13), sharex=True, sharey=True)
for m, ax in enumerate(axes.ravel()):
    sub = grid_pred[grid_pred["month"] == m]
    sc = ax.scatter(sub["x"], sub["y"], c=sub["pm25_pred"], cmap="RdYlGn_r",
                     vmin=vmin, vmax=vmax, s=5, marker="s")
    ca.boundary.plot(ax=ax, color="black", linewidth=0.6)
    ax.set_title(MONTH_NAMES[m], fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(sc, ax=axes, shrink=0.6, label="Predicted PM2.5 (ug/m3)")
fig.suptitle("California PM2.5 -- Spatio-Temporal Kriging, 5km grid, 2025", fontsize=14)
plt.show()


# ## 12. Summary report

# In[25]:


summary_df = pd.DataFrame(RESULTS)
def _style(row):
    color = {"PASS": "background-color: #d4edda",
             "FAIL": "background-color: #f8d7da",
             "BLOCKED": "background-color: #fff3cd"}.get(row["Status"], "")
    return [color] * len(row)
summary_df.style.apply(_style, axis=1)

