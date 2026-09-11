#!/usr/bin/env python
# coding: utf-8

# # Spatial-Temporal Poisson Kriging
# 
# Testing `pygstat.STPoisson_kriging` against the North-Atlantic-seaboard US
# county **lung/bronchus cancer (LBC) mortality rate** panel (`lbc_atlantic.csv`,
# 666 counties x 1998-2012) and the `COUNTY_ATLANTIC.shp` county polygons.
# 
# > **Goovaerts, P. (2006).** *Geostatistical analysis of disease data: accounting
# > for spatial support and population density in the isopleth mapping of cancer
# > mortality risk.* International Journal of Health Geographics, 5(1).
# >
# > **Pebesma, E. (2012).** *spacetime: Spatio-Temporal Data in R.* Journal of
# > Statistical Software, 51(7). (source of the `krigeST` space-time kriging
# > algorithm this module is a Python port of, and of this Atlantic county
# > dataset.)

# ## 1. Theory
# 
# ### 1.1 Why *Poisson* kriging?
# 
# Each county's mortality rate is computed from a count over a population at
# risk,
# 
# $$
# z(\mathbf u_i) = B\,\frac{d(\mathbf u_i)}{n(\mathbf u_i)}, \qquad
# d(\mathbf u_i)\sim\mathrm{Poisson}\!\big(n(\mathbf u_i)\,R(\mathbf u_i)\big),
# $$
# 
# where $d$ is the observed count, $n$ the population at risk, $R$ the
# underlying (latent) risk, and $B$ a rate base (e.g. $100{,}000$). Because a
# Poisson count has $\mathrm{Var}(d)=\mathbb E[d]$, the rate itself is
# heteroscedastic:
# 
# $$
# \mathrm{Var}\big(z(\mathbf u_i)\big) \;\approx\; B\,\frac{m^\ast}{n(\mathbf u_i)},
# \qquad
# m^\ast=\frac{\sum_i d(\mathbf u_i)}{\sum_i n(\mathbf u_i)}
#       =\frac{\sum_i n(\mathbf u_i)\,z(\mathbf u_i)/B}{\sum_i n(\mathbf u_i)} .
# $$
# 
# A county with a small population has a far noisier rate than one with a
# large population, even if their *true* underlying risk is identical. Kriging
# the raw rate ignores this and lets small, noisy counties distort the map.
# **Poisson Kriging** (Goovaerts, 2006) corrects for this by folding
# $\mathrm{Var}(z(\mathbf u_i))$ into the kriging system itself, as a
# population-dependent term on the diagonal, so noisy (small-population)
# areas are automatically down-weighted.
# 
# ### 1.2 Ordinary kriging with a Poisson reliability term
# 
# The risk estimator is linear, $\hat R(\mathbf u_0,t_0)=\sum_i\lambda_i\,z(\mathbf u_i,t_i)$,
# with the usual unbiasedness constraint $\sum_i\lambda_i=1$. The weights solve
# 
# $$
# \sum_{j=1}^{N}\lambda_j\Big[C\big((\mathbf u_i,t_i)-(\mathbf u_j,t_j)\big)
#      +\delta_{ij}\,B\,\frac{m^\ast}{n(\mathbf u_i)}\Big] + \mu
#      = C\big((\mathbf u_i,t_i)-(\mathbf u_0,t_0)\big), \quad i=1,\dots,N,
# $$
# 
# or, in matrix form with Lagrange multiplier $\mu$,
# 
# $$
# \begin{bmatrix} \mathbf V + B\,\mathrm{diag}\!\left(\dfrac{m^\ast}{n_i}\right) & \mathbf 1 \\[4pt]
#                  \mathbf 1^\top & 0 \end{bmatrix}
# \begin{bmatrix} \boldsymbol\lambda \\ \mu \end{bmatrix}
# =
# \begin{bmatrix} \mathbf v_0 \\ 1 \end{bmatrix},
# $$
# 
# with prediction $\hat R=\boldsymbol\lambda^\top\mathbf z$ and kriging variance
# $\sigma^2 = C(0) - \boldsymbol\lambda^\top\mathbf v_0 - \mu$. This is exactly
# `pygstat.poisson_kriging.PoissonKriging`, restricted to areas observed once
# (no time dimension).
# 
# ### 1.3 Adding time: the space-time covariance
# 
# An area-year panel (the same 666 counties, each observed every year from
# 1998-2012) needs $C$ as a function of *both* a spatial lag $h_s=\lVert
# \mathbf u_i-\mathbf u_j\rVert$ and a temporal lag $h_t=|t_i-t_j|$. This
# notebook uses the **metric** space-time model (Pebesma, 2012), the simplest
# choice when there is no strong prior for separate spatial/temporal
# structure: space and time are folded into one joint (anisotropic) distance
# 
# $$
# h_{st} = \sqrt{h_s^2 + (a\,h_t)^2},
# $$
# 
# where $a$ (`stAni`, units of space per unit time, e.g. meters/year) makes
# space and time commensurable, and a single ordinary variogram/covariance
# model $C(h_{st})$ is fit to it. `pygstat.st_variogram_models` (a Python port
# of gstat's `stVariogramModels.R`) also implements `separable`, `productSum`,
# and `sumMetric` space-time models for cases with distinct spatial/temporal
# structure.
# 
# The empirical semivariogram is the Matheron estimator, binned jointly by
# spatial and temporal lag:
# 
# $$
# \hat\gamma(h_s,h_t) = \frac{1}{2\,|N(h_s,h_t)|}
#    \sum_{(i,a),(j,b)\,\in\,N(h_s,h_t)} \big(z(\mathbf u_i,t_a)-z(\mathbf u_j,t_b)\big)^2,
# $$
# 
# where $N(h_s,h_t)$ collects area-year pairs whose spatial distance falls in
# the $h_s$ bin and whose time difference equals $h_t$.
# 
# ### 1.4 Spatio-Temporal Poisson Kriging
# 
# `STPoissonKriging` (this repo's `pygstat/STPoisson_kriging.py`) combines
# 1.2 and 1.3: the kriging system of Section 1.2, but with $C$ the fitted
# space-time covariance of Section 1.3 (via
# `pygstat.krige_st.cov_fn_st`), evaluated on a local neighborhood of the
# `number_of_neighbors` nearest area-years in a scaled $(x, y, a\cdot t)$
# space. It reduces to `PoissonKriging` when every area has one time stamp,
# and to plain space-time ordinary kriging when every population is equal.

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
from pygstat import STPoissonKriging, empirical_st_variogram, fit_metric_st_variogram
from pygstat.st_variogram_models import vgm, vgm_st, variogram_line

# --- running test log, rendered as a styled summary table at the end ---
RESULTS = []
def record(test, status, details=""):
    RESULTS.append({"Test": test, "Status": status, "Details": details})
    print(f"[{status}] {test} -- {details}")


# ## 3. Data
# 
# * `data/lbc_atlantic.csv` -- wide panel, one row per county (`FIPS`), one
#   column per year 1998-2012, age-adjusted lung/bronchus cancer mortality
#   rate per 100,000.
# * `data/COUNTY_ATLANTIC.shp` (+ `.dbf`/`.shx`/`.prj`) -- the 666 county
#   polygons, with `FIPS` and projected centroid attributes `x`, `y` (Albers
#   Equal Area, meters).
# * `data/lbc_rate.csv` -- supplies each county's **population** (`pop`),
#   which `COUNTY_ATLANTIC.shp` does not carry. This is the same source
#   `tests/test_poisson_kriging.py` already draws population from for the
#   spatial-only `PoissonKriging` tests.
# 
# **Population caveat.** `lbc_rate.csv` gives one population figure per county
# (its approximate population over the study period), not a year-by-year
# series. True Poisson Kriging needs population *at the time of each count*;
# lacking that, population is treated as constant across 1998-2012 for every
# county -- a limitation of the available data, not of the method.

# In[2]:


atlantic = pd.read_csv("../data/lbc_atlantic.csv")
year_cols = [c for c in atlantic.columns if c != "FIPS"]
ok = (atlantic.shape[0] == 666) and (not atlantic.isna().any().any())
record("Load lbc_atlantic.csv (wide rate panel)", "PASS" if ok else "FAIL",
       f"shape={atlantic.shape}, years={year_cols[0]}-{year_cols[-1]}")
atlantic.head()


# In[3]:


counties = gpd.read_file("../data/COUNTY_ATLANTIC.shp")
counties["FIPS"] = counties["FIPS"].astype(int)
ok = (len(counties) == 666) and (set(counties["FIPS"]) == set(atlantic["FIPS"]))
record("Load COUNTY_ATLANTIC.shp county polygons", "PASS" if ok else "FAIL",
       f"{len(counties)} polygons, FIPS match={set(counties['FIPS'])==set(atlantic['FIPS'])}")
counties[["FIPS", "x", "y", "STATE", "COUNTY"]].head()


# In[4]:


rate_meta = pd.read_csv("../data/lbc_rate.csv")
counties_pop = counties.merge(rate_meta[["FIPS", "pop"]], on="FIPS", how="left")
ok = counties_pop["pop"].notna().all() and (counties_pop["pop"] > 0).all()
record("Merge per-county population from lbc_rate.csv", "PASS" if ok else "FAIL",
       f"pop range=({counties_pop['pop'].min():.0f}, {counties_pop['pop'].max():.0f})")


# In[5]:


long = atlantic.melt(id_vars="FIPS", value_vars=year_cols, var_name="year", value_name="rate")
long["year"] = long["year"].astype(int)
long = long.merge(counties_pop[["FIPS", "x", "y", "pop"]], on="FIPS", how="left")
long = long.sort_values(["FIPS", "year"]).reset_index(drop=True)
ok = (long.shape[0] == 666 * 15) and (not long.isna().any().any())
record("Build long area-year panel", "PASS" if ok else "FAIL", f"shape={long.shape}")
long.head()


# ### 3.1 Exploratory maps

# In[6]:


mean_rate = long.groupby("FIPS")["rate"].mean().rename("mean_rate")
map_df = counties_pop.merge(mean_rate, on="FIPS")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
map_df.plot("mean_rate", cmap="RdBu_r", legend=True, ax=ax[0], edgecolor="k", linewidth=0.2,
            legend_kwds={"label": "rate / 100k", "shrink": .7})
ax[0].set_title("Mean LBC rate, 1998-2012")
map_df.plot("pop", cmap="viridis", legend=True, ax=ax[1], edgecolor="k", linewidth=0.2,
            legend_kwds={"label": "population", "shrink": .7})
ax[1].set_title("County population")
for a in ax:
    a.set_axis_off()
plt.tight_layout()
plt.show()


# In[7]:


fig, ax = plt.subplots(figsize=(7, 4))
sample_fips = long["FIPS"].drop_duplicates().sample(6, random_state=0)
for fips in sample_fips:
    sub = long[long["FIPS"] == fips].sort_values("year")
    ax.plot(sub["year"], sub["rate"], marker="o", ms=3, label=str(fips))
ax.set_xlabel("year"); ax.set_ylabel("rate / 100,000")
ax.set_title("LBC rate time series, 6 random counties")
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.show()


# ## 4. Train / test split
# 
# To validate genuine spatial-temporal *extrapolation*, 60 counties (~9%) are
# held out **entirely** -- none of their years are seen during training -- and
# predicted purely from their space-time neighbors among the remaining 606
# counties.

# In[8]:


rng = np.random.default_rng(0)
fips_all = long["FIPS"].unique()
test_fips = rng.choice(fips_all, size=60, replace=False)
is_test = long["FIPS"].isin(test_fips)
train = long.loc[~is_test].reset_index(drop=True)
test = long.loc[is_test].reset_index(drop=True)
print(f"train: {train['FIPS'].nunique()} counties, {len(train)} rows")
print(f"test:  {test['FIPS'].nunique()} counties (held out), {len(test)} rows")


# ## 5. Empirical semivariogram and space-time model fit
# 
# `empirical_st_variogram` needs a *complete* area x time matrix (every county
# observed every year); this holds for the training panel since whole
# counties -- not scattered county-years -- were held out.

# In[9]:


train_wide = train.pivot(index="FIPS", columns="year", values="rate").sort_index()
train_coords = (train[["FIPS", "x", "y"]].drop_duplicates("FIPS").set_index("FIPS")
                 .loc[train_wide.index][["x", "y"]].to_numpy())
years = np.sort(train["year"].unique()).astype(float)

emp = empirical_st_variogram(train_coords, years, train_wide.to_numpy(), n_space_bins=12)
ok = (len(emp["dist"]) > 0) and np.all(np.isfinite(emp["gamma"])) and np.all(emp["gamma"] >= 0)
record("Empirical space-time variogram estimation", "PASS" if ok else "FAIL",
       f"{len(emp['dist'])} (space-bin, time-lag) cells from {emp['np'].sum():.0f} pairs")


# In[10]:


fitted_model = fit_metric_st_variogram(emp)
joint = fitted_model["joint"]
nugget, sill, prange = joint[0]["psill"], joint[1]["psill"], joint[1]["range"]
st_ani = fitted_model["stAni"]
ok = (nugget >= 0) and (sill > 0) and (prange > 0) and (st_ani > 0) and np.isfinite(fitted_model["MSE"])
record("Fit 'metric' space-time variogram model", "PASS" if ok else "FAIL",
       f"nugget={nugget:.2f} sill={sill:.2f} range={prange:.0f}m stAni={st_ani:.0f}m/yr "
       f"MSE={fitted_model['MSE']:.3f}")


# In[11]:


fig, ax = plt.subplots(figsize=(6, 4))
show_lags = sorted(set(emp["timelag"]))[:5]
colors = plt.cm.viridis(np.linspace(0, 1, len(show_lags)))
h = np.linspace(0, emp["dist"].max(), 100)
for lag, c in zip(show_lags, colors):
    m = emp["timelag"] == lag
    ax.scatter(emp["dist"][m], emp["gamma"][m], s=15, color=c, label=f"lag={lag:.0f}yr")
    hh = np.sqrt(h ** 2 + (st_ani * lag) ** 2)
    ax.plot(h, variogram_line(fitted_model["joint"], hh, covariance=False), "--", color=c)
ax.set_xlabel("spatial lag (m)"); ax.set_ylabel("semivariance")
ax.set_title("Empirical (points) vs fitted metric model (dashed)")
ax.legend(fontsize=7)
plt.tight_layout()
plt.show()


# ## 6. Fit Spatio-Temporal Poisson Kriging

# In[12]:


stpk = STPoissonKriging(fitted_model).fit(
    coords=train[["x", "y"]].to_numpy(),
    times=train["year"].to_numpy(float),
    values=train["rate"].to_numpy(),
    populations=train["pop"].to_numpy(),
    ids=train["FIPS"].to_numpy(),
)
ok = stpk.rate_base == 100000
record("Fit STPoissonKriging (auto-detected rate_base)", "PASS" if ok else "FAIL",
       f"rate_base={stpk.rate_base:.0f}")


# ## 7. Validation: held-out counties
# 
# Predict every county-year of the 60 held-out counties from their nearest
# space-time neighbors among the 606 training counties, and compare against
# a naive baseline (the overall training mean rate, i.e. no spatial or
# temporal information at all).

# In[13]:


zhat, sig = stpk.predict(test[["x", "y"]].to_numpy(), test["year"].to_numpy(float),
                          number_of_neighbors=30)
test = test.assign(zhat=zhat, sig=sig)

resid = test["rate"] - test["zhat"]
rmse = float(np.sqrt(np.mean(resid ** 2)))
mae = float(np.mean(np.abs(resid)))
r2 = 1.0 - float(np.sum(resid ** 2)) / float(np.sum((test["rate"] - test["rate"].mean()) ** 2))
baseline_rmse = float(np.sqrt(np.mean((test["rate"] - train["rate"].mean()) ** 2)))

ok = rmse < baseline_rmse
record("Held-out county prediction accuracy (60 counties)", "PASS" if ok else "FAIL",
       f"RMSE={rmse:.2f} vs naive-mean RMSE={baseline_rmse:.2f}, R2={r2:.3f}, "
       f"mean kriging sd={test['sig'].mean():.2f}")

print(f"RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}  naive-mean RMSE={baseline_rmse:.3f}")


# In[14]:


fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(test["rate"], test["zhat"], s=10, alpha=0.5)
lims = [min(test["rate"].min(), test["zhat"].min()), max(test["rate"].max(), test["zhat"].max())]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Observed rate (per 100,000)")
ax.set_ylabel("ST Poisson Kriging prediction")
ax.set_title(f"Held-out counties (RMSE={rmse:.2f})")
plt.tight_layout()
plt.show()


# In[15]:


snap = test[test["year"] == 2012].merge(counties[["FIPS", "geometry"]], on="FIPS")
snap = gpd.GeoDataFrame(snap, geometry="geometry", crs=counties.crs)
try:
    vmin, vmax = snap[["rate", "zhat"]].min().min(), snap[["rate", "zhat"]].max().max()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    snap.plot("rate", ax=ax[0], legend=True, vmin=vmin, vmax=vmax, cmap="RdBu_r",
              edgecolor="k", linewidth=0.3)
    ax[0].set_title("Observed, 2012"); ax[0].set_axis_off()
    snap.plot("zhat", ax=ax[1], legend=True, vmin=vmin, vmax=vmax, cmap="RdBu_r",
              edgecolor="k", linewidth=0.3)
    ax[1].set_title("ST Poisson Kriging, 2012"); ax[1].set_axis_off()
    plt.tight_layout()
    plt.show()
    record("Choropleth map: observed vs predicted (held-out, 2012)", "PASS",
           f"{len(snap)} counties mapped")
except Exception as e:
    record("Choropleth map: observed vs predicted (held-out, 2012)", "BLOCKED", str(e))


# ## 8. Leave-one-out sanity check (training data)
# 
# Ten random *training* area-years, each re-predicted with itself excluded
# from its own neighbor search -- a basic sanity check that predictions track
# observed values without exactly reproducing them (they should be shrunk
# toward the local space-time mean, not copied).

# In[16]:


sample = train.sample(10, random_state=0)
loo = pd.DataFrame([stpk.predict_id(row["FIPS"], row["year"], number_of_neighbors=30)
                     for _, row in sample.iterrows()])
loo["observed"] = sample["rate"].to_numpy()

ok = np.isfinite(loo["zhat"]).all() and (loo["sig"].fillna(0) >= 0).all()
record("Leave-one-out sanity check (10 training area-years)", "PASS" if ok else "FAIL",
       f"mean|zhat-observed|={np.mean(np.abs(loo['zhat']-loo['observed'])):.2f}")
loo[["id", "year", "observed", "zhat", "sig"]]


# ## 9. Forecast beyond the observed range (2013)
# 
# Predicting a year past the observed 1998-2012 window is pure temporal
# extrapolation using the fitted space-time covariance -- the space-time
# analogue of kriging a location past the edge of a spatial survey.

# In[17]:


demo_fips = train["FIPS"].drop_duplicates().sample(3, random_state=0).to_numpy()
demo_coords = (train[["FIPS", "x", "y"]].drop_duplicates("FIPS").set_index("FIPS")
               .loc[demo_fips][["x", "y"]].to_numpy())
fzhat, fsig = stpk.predict(demo_coords, np.full(3, 2013.0), number_of_neighbors=30)

ok = np.all(np.isfinite(fzhat)) and np.all(fzhat > 0) and np.all(fzhat < 3 * long["rate"].max())
record("Forecast beyond observed range (2013)", "PASS" if ok else "FAIL",
       f"predictions={np.round(fzhat, 2).tolist()}")

for fips, zh, sg in zip(demo_fips, fzhat, fsig):
    last_obs = train.loc[train["FIPS"] == fips].sort_values("year").iloc[-1]["rate"]
    print(f"FIPS {fips}: 2012 observed={last_obs:.2f}  ->  2013 forecast={zh:.2f} +/- {sg:.2f}")


# ## 10. Error-handling checks
# 
# `STPoissonKriging` should fail loudly, not silently, on bad inputs.

# In[18]:


try:
    stpk.predict_id(999999, 2000)
    record("Error handling: unknown (id, time) in predict_id", "FAIL", "no exception raised")
except ValueError as e:
    record("Error handling: unknown (id, time) in predict_id", "PASS", str(e))


# In[19]:


try:
    bad_pop = train["pop"].to_numpy().copy()
    bad_pop[0] = 0
    STPoissonKriging(fitted_model).fit(train[["x", "y"]].to_numpy(), train["year"].to_numpy(float),
                                        train["rate"].to_numpy(), bad_pop,
                                        ids=train["FIPS"].to_numpy())
    record("Error handling: non-positive population in fit()", "FAIL", "no exception raised")
except ValueError as e:
    record("Error handling: non-positive population in fit()", "PASS", str(e))


# In[20]:


try:
    stpk.predict(np.array([[0.0, 0.0]]), np.array([2005.0]), number_of_neighbors=1)
    record("Error handling: too few neighbors raises ValueError", "FAIL", "no exception raised")
except ValueError as e:
    record("Error handling: too few neighbors raises ValueError", "PASS", str(e))


# ## 11. Summary report

# In[21]:


summary_df = pd.DataFrame(RESULTS)
def _style(row):
    color = {"PASS": "background-color: #d4edda",
             "FAIL": "background-color: #f8d7da",
             "BLOCKED": "background-color: #fff3cd"}.get(row["Status"], "")
    return [color] * len(row)
summary_df.style.apply(_style, axis=1)

