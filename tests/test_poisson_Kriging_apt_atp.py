#!/usr/bin/env python
# coding: utf-8

# # Poisson Kriging of Areal Disease Data
# ## Centroid-based, Area-to-Area (ATA), and Area-to-Point (ATP)
# 
# The three Poisson-kriging estimators of Goovaerts (2006), applied to Northeastern-US
# breast-cancer mortality. This notebook supplies the surrounding *variogram
# inference* (model, population-weighted risk semivariogram, deconvolution), loads the
# data, runs the three estimators, and validates them.
# 
# > **Goovaerts, P. (2006).** *Geostatistical analysis of disease data: accounting for
# > spatial support and population density in the isopleth mapping of cancer mortality
# > risk using area-to-point Poisson kriging.* **Int. J. Health Geographics, 5:52.**
# > https://doi.org/10.1186/1476-072X-5-52
# 
# **Module API used here**
# 
# | class | role | key methods |
# |-------|------|-------------|
# | `PoissonKriging(variogram, rate_base=None)` | centroid-based (Eq. 2–4) | `fit`, `predict_id`, `predict` |
# | `AreaPoissonKriging(variogram, rate_base=None)` | ATA + ATP (Eq. 6–14) | `fit`, `predict_area`, `predict_points` |
# | `PointSupport(coords, values, block_ids)` | groups census points by county | `coords`, `values`, `total`, `centroid` |
# 
# `rate_base` (the *per-B* multiplier of the rate) is **auto-detected** from the rate
# magnitude; the Poisson diagonal term is `rate_base · m*/n(uᵢ)`, which is what
# down-weights small-population areas. `predict_points` solves a per-point kriging
# system so the **coherence constraint** (Eq. 15) holds as a population-weighted
# average.
# 
# > **To run this notebook, keep `poisson_kriging_1.py` in the same folder.**

# ## 0. Environment and module import

# In[1]:


import matplotlib
matplotlib.use("Agg")
import sys, os
sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import pandas as pd
import geopandas as gpd
import pygstat
display = print
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
import warnings; warnings.filterwarnings("ignore")

# --- the kriging engine (importable module) ---
from pygstat import poisson_kriging as pk
from pygstat.poisson_kriging import (
    PoissonKriging,
    AreaPoissonKriging,
    PointSupport,
    _weighted_semivariance as wgamma,
    _within_support_semivariance as within_gamma,
    _detect_rate_base,
)


plt.rcParams.update({"figure.dpi":110, "font.size":10,
                     "axes.spines.top":False, "axes.spines.right":False})
np.set_printoptions(suppress=True)
print("pygstat", pygstat.__version__, "| geopandas", gpd.__version__)

RESULTS = []
def record(section, status, note=""):
    RESULTS.append({"Feature": section, "Status": status, "Notes": note})
    marker = {"PASS": "\u2705", "FAIL": "\u274c", "BLOCKED": "\u26a0\ufe0f"}.get(status, "?")
    print(f"{marker} [{status}] {section}  {note}")

record("poisson_kriging: module import", "PASS",
       f"PoissonKriging, AreaPoissonKriging, PointSupport from pygstat {pygstat.__version__}")


# ## 1. Data and point support
# 
# 217 county polygons (`FIPS`, age-adjusted `rate` per 100,000) and 5,419 census
# population points (`POP10`). A rate from a small population is noisy (the
# *small-number problem*), so each county needs its **point support** and **total
# population**. Points are assigned to counties by a `within` join, with stragglers
# snapped to the nearest county; the module's `PointSupport` then groups them.

# In[2]:


areas = gpd.read_file("../data/cancer_areas.shp"); areas["FIPS"] = areas["FIPS"].astype(int)
pts   = gpd.read_file("../data/cancer_data.gpkg", layer="points").set_crs(areas.crs, allow_override=True)

j = gpd.sjoin(pts, areas[["FIPS","geometry"]], how="left", predicate="within")
miss = j["FIPS"].isna()
if miss.any():
    jn = gpd.sjoin_nearest(pts[miss.values], areas[["FIPS","geometry"]], how="left")
    j.loc[miss.values, "FIPS"] = jn["FIPS"].values
j["FIPS"] = j["FIPS"].astype(int)

pxy = np.column_stack([j.geometry.x, j.geometry.y])
ps  = PointSupport(pxy, j["POP10"].values, j["FIPS"].values)   # <-- module class

pop_area = j.groupby("FIPS")["POP10"].sum()
areas["pop"] = areas["FIPS"].map(pop_area)
fips  = areas["FIPS"].values
rates = areas["rate"].values
pops  = areas["pop"].values
cent  = np.array([ps.centroid(int(f)) for f in fips])          # pop-weighted centroids
areas["cx"], areas["cy"] = cent[:,0], cent[:,1]

print(f"{len(areas)} counties | {len(j)} points ({miss.sum()} snapped) | {len(ps)} supports")
print(f"population/county: {pops.min():,.0f}–{pops.max():,.0f} (median {np.median(pops):,.0f})")
print(f"rate/100k: {rates.min():.1f}–{rates.max():.1f} (mean {rates.mean():.1f})")
ok_n = len(areas) == 217 and len(ps) == 217 and np.all(np.isfinite(cent))
record("Data + PointSupport: 217 counties with finite pop-weighted centroids",
       "PASS" if ok_n else "FAIL",
       f"{len(areas)} counties | {len(j)} points ({int(miss.sum())} snapped) | {len(ps)} supports")


# ### 1.1 Exploratory maps

# In[3]:


fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
areas.plot("rate", cmap="RdBu_r", legend=True, ax=ax[0], edgecolor="k", linewidth=.2,
           legend_kwds={"label":"rate / 100k","shrink":.7}); ax[0].set_title("Raw mortality rate")
areas.plot("pop", cmap="viridis", legend=True, ax=ax[1], edgecolor="k", linewidth=.2,
           norm=Normalize(), legend_kwds={"label":"population","shrink":.7})
ax[1].scatter(cent[:,0], cent[:,1], s=3, c="red"); ax[1].set_title("Population at risk")
areas["npts"] = areas["FIPS"].map(j.groupby("FIPS").size())
areas.plot("npts", cmap="magma", legend=True, ax=ax[2], edgecolor="k", linewidth=.2,
           legend_kwds={"label":"# support points","shrink":.7}); ax[2].set_title("Point-support density")
for a in ax: a.set_axis_off()
plt.tight_layout(); plt.show()


# ## 2. The three estimators (theory)
# 
# Target: the latent **risk** $R$ behind $z(v_\alpha)=d(v_\alpha)/n(v_\alpha)$, with
# $D\sim\mathrm{Poisson}(nR)$. All are linear, $\hat R=\sum_i\lambda_i z_i$, with a
# population penalty on the kriging diagonal.
# 
# **Centroid (Eq. 3):** $\sum_j\lambda_j\!\big[C_R(\mathbf u_i-\mathbf u_j)+\delta_{ij}\tfrac{m^\ast}{n(\mathbf u_i)}\big]+\mu=C_R(\mathbf u_i-\mathbf u_\alpha)$.
# 
# **ATA (Eq. 7):** point covariances become block covariances
# $\bar C_R(v_i,v_j)=\dfrac{\sum_{s,s'} n_s n_{s'} C(\mathbf u_s-\mathbf u_{s'})}{\sum_{s,s'} n_s n_{s'}}$.
# 
# **ATP (Eq. 13–14):** same LHS, RHS uses area-to-point covariances $\bar C_R(v_i,\mathbf u_s)$; solve at every grid node → continuous surface.
# 
# **Coherence (Eq. 15):** $\sum_s \frac{n(\mathbf u_s)}{\sum_{s'}n(\mathbf u_{s'})}\hat R(\mathbf u_s)=\hat R(v_\alpha)$.
# 
# > The module scales the diagonal penalty by the **rate base** (here $10^5$, since
# > rates are per 100,000). Without it the penalty is ~$10^5\times$ too small and
# > ATA/ATP diverge; the module auto-detects it.

# ## 3. Semivariogram model (the object the module consumes)
# 
# `AreaPoissonKriging` / `PoissonKriging` expect a `variogram` with
# `fitted_params = [nugget, partial_sill, range]` that is callable as
# `variogram(h) → γ(h)`. We supply this small `VariogramModel`; the range is capped at
# the largest lag so it cannot rail past the domain.

# In[4]:


def _spherical(h, n, ps_, r):
    h = np.asarray(h, float); g = np.where(h <= r, ps_*(1.5*h/r - 0.5*(h/r)**3), ps_)
    return n + np.where(h == 0, 0.0, g)
def _exponential(h, n, ps_, r):
    h = np.asarray(h, float); return n + np.where(h == 0, 0.0, ps_*(1-np.exp(-3*h/r)))
_MODELS = {"spherical":_spherical, "exponential":_exponential}

class VariogramModel:
    """Callable gamma(h) with fitted_params = [nugget, psill, range]."""
    def __init__(self, model="spherical", nugget=0.0, psill=1.0, rng=1.0):
        self.model = model; self.fitted_params = [nugget, psill, rng]
    def __call__(self, h):
        return _MODELS[self.model](h, *self.fitted_params)
    @property
    def sill(self):
        return self.fitted_params[0] + self.fitted_params[1]
    def fit(self, lags, gamma, weights=None):
        lags, gamma = np.asarray(lags, float), np.asarray(gamma, float)
        w = np.ones_like(gamma) if weights is None else np.asarray(weights, float)
        s0 = gamma.max()
        def resid(p):
            return (_MODELS[self.model](lags, *p) - gamma)*np.sqrt(w)
        sol = least_squares(resid, [gamma.min(), s0, lags[len(lags)//2]],
                            bounds=([0,0,lags.min()], [s0, 3*s0, lags.max()]))
        self.fitted_params = list(sol.x); return self
_vm = VariogramModel("spherical", 1, 10, 100)
ok_vm = (len(_vm.fitted_params) == 3
         and np.isclose(_vm.sill, 11)
         and np.isclose(_vm(100), 11)
         and np.isclose(_vm(200), 11))
record("VariogramModel: callable gamma(h) with [nugget, psill, range]",
       "PASS" if ok_vm else "FAIL",
       f"sill={_vm.sill:.1f}, gamma(range)={float(_vm(100)):.1f}, gamma(2*range)={float(_vm(200)):.1f}")


# ## 4. Population-weighted distance (Eq. 1) and risk semivariogram (Eq. 5)
# 
# The Monestiez/Goovaerts estimator down-weights unreliable pairs and removes the
# Poisson noise variance:
# $$\hat\gamma_R(h)=\frac{1}{2\sum_\alpha w_\alpha}\sum_\alpha w_\alpha\Big[(z_\alpha-z_{\alpha+h})^2-m^\ast\Big],\quad w_\alpha=\frac{n_\alpha n_{\alpha+h}}{n_\alpha+n_{\alpha+h}}.$$

# In[5]:


def pop_weighted_distance(ca, va, cb, vb):
    """Eq. 1: population-weighted mean distance between two point supports."""
    d = cdist(ca, cb); w = np.outer(va, vb); tw = w.sum()
    return float((d*w).sum()/tw) if tw > 0 else float(d.mean())

def risk_semivariogram(coords, rates, pops, n_lags=12, max_dist=None):
    """Eq. 5: population-weighted, Poisson-corrected experimental risk semivariogram."""
    d = cdist(coords, coords); iu = np.triu_indices(len(coords), 1)
    dd, zi, zj = d[iu], rates[iu[0]], rates[iu[1]]
    ni, nj = pops[iu[0]], pops[iu[1]]
    mstar = np.sum(rates*pops)/np.sum(pops)
    w    = ni*nj/(ni+nj)
    incr = (zi-zj)**2 - mstar
    if max_dist is None: max_dist = dd.max()/2
    edges = np.linspace(0, max_dist, n_lags+1)
    L, G, N = [], [], []
    for k in range(len(edges)-1):
        m = (dd > edges[k]) & (dd <= edges[k+1])
        if m.sum() < 5: continue
        L.append(dd[m].mean()); G.append(np.sum(w[m]*incr[m])/(2*np.sum(w[m]))); N.append(int(m.sum()))
    return np.array(L), np.clip(np.array(G), 0, None), np.array(N)

dmax = cdist(cent, cent).max()
lags, gam, npair = risk_semivariogram(cent, rates, pops, n_lags=12, max_dist=dmax*0.5)
vg_area = VariogramModel("spherical").fit(lags, gam, weights=npair)
print(f"areal risk model: nugget={vg_area.fitted_params[0]:.1f}  sill={vg_area.sill:.1f}  "
      f"range={vg_area.fitted_params[2]/1000:.0f} km")

hh = np.linspace(0, lags.max(), 200)
fig, ax = plt.subplots(figsize=(7,4))
ax.scatter(lags/1000, gam, s=npair/npair.max()*120, c="#c0392b", alpha=.7,
           label="experimental (size ∝ #pairs)")
ax.plot(hh/1000, vg_area(hh), "b-", lw=2, label="fitted spherical (areal)")
ax.set_xlabel("distance (km)"); ax.set_ylabel(r"$\gamma_R(h)$")
ax.set_title("Population-weighted risk semivariogram (Eq. 5)"); ax.legend()
plt.tight_layout(); plt.show()
ok_vg = np.all(np.isfinite(gam)) and vg_area.sill > 0 and vg_area.fitted_params[2] > 0
record("Eq. 5: population-weighted risk semivariogram fitted",
       "PASS" if ok_vg else "FAIL",
       f"nugget={vg_area.fitted_params[0]:.1f}  sill={vg_area.sill:.1f}  "
       f"range={vg_area.fitted_params[2]/1000:.0f} km  n_lags={len(lags)}")


# ## 5. Deconvolution (Eq. 18)
# 
# ATA/ATP need the **point-support** model. We invert
# $\gamma_v(h)=\bar\gamma(v,v_h)-\bar\gamma_h(v,v)$ iteratively, **reusing the module's
# own block-semivariance helpers** (`_weighted_semivariance`,
# `_within_support_semivariance`) so the regularisation is consistent with the kriging
# engine. The result is a point-support model with a higher sill than the areal one.

# In[6]:


def regularize(vg_pt, ps, ids, centd, lags, lag_tol):
    """Regularise a point-support model to a theoretical areal gamma_v(h) (Eq. 18)."""
    within = np.array([within_gamma(vg_pt, ps.coords(int(b)), ps.values(int(b))) for b in ids])
    iu = np.triu_indices(len(ids), 1)
    gv = np.full(len(lags), np.nan)
    for li, h in enumerate(lags):
        m = (centd[iu] > h-lag_tol) & (centd[iu] <= h+lag_tol)
        ii, jj = iu[0][m], iu[1][m]
        if len(ii) < 3: continue
        vals = [wgamma(vg_pt, ps.coords(int(ids[a])), ps.values(int(ids[a])),
                       ps.coords(int(ids[b])), ps.values(int(ids[b]))) - 0.5*(within[a]+within[b])
                for a, b in zip(ii, jj)]
        gv[li] = np.mean(vals)
    return gv

def deconvolute(vg_area, ps, ids, centd, lags, exp_gamma, lag_tol, max_iter=12, tol=0.01):
    vg_pt = VariogramModel(vg_area.model, *vg_area.fitted_params)
    reg = regularize(vg_pt, ps, ids, centd, lags, lag_tol)
    good = np.isfinite(reg) & np.isfinite(exp_gamma)
    dev = lambda a, b: np.mean(np.abs(a-b)/np.where(b>0, b, 1))
    bestD, hist = dev(reg[good], exp_gamma[good]), []
    best = vg_pt; hist.append(bestD)
    for _ in range(max_iter):
        r = np.where((reg > 0) & np.isfinite(reg), exp_gamma/reg, 1.0)
        s = float(np.clip(np.nanmean(r), 0.5, 2.0))
        n, psl, rng = best.fitted_params
        cand = VariogramModel(best.model, n*s, psl*s, rng)
        reg2 = regularize(cand, ps, ids, centd, lags, lag_tol)
        g2 = np.isfinite(reg2) & np.isfinite(exp_gamma)
        D = dev(reg2[g2], exp_gamma[g2]); hist.append(D)
        if D < bestD: best, bestD, reg = cand, D, reg2
        if len(hist) > 3 and abs(hist[-2]-hist[-1]) < tol*hist[0]: break
    return best, bestD, hist

centd   = cdist(cent, cent)
lag_tol = (lags[1]-lags[0])/2
vg_pt, Dstat, hist = deconvolute(vg_area, ps, fips, centd, lags, gam, lag_tol)
reg_final = regularize(vg_pt, ps, fips, centd, lags, lag_tol)
print(f"point-support model: nugget={vg_pt.fitted_params[0]:.1f}  sill={vg_pt.sill:.1f}  "
      f"range={vg_pt.fitted_params[2]/1000:.0f} km  (D={Dstat:.3f}, {len(hist)} iters)")
print(f"areal sill={vg_area.sill:.1f} -> point-support sill={vg_pt.sill:.1f} (higher, as expected)")

fig, ax = plt.subplots(figsize=(7,4))
ax.scatter(lags/1000, gam, s=40, c="#c0392b", label="experimental areal")
ax.plot(hh/1000, vg_area(hh), "r--", lw=1.5, label="fitted areal")
ax.plot(hh/1000, vg_pt(hh),  "g-",  lw=2,   label="deconvoluted point-support")
ax.plot(lags/1000, reg_final, "b:o", ms=4, lw=1.5, label="regularised point-support")
ax.set_xlabel("distance (km)"); ax.set_ylabel(r"$\gamma(h)$")
ax.set_title("Deconvolution (Eq. 18)"); ax.legend(); plt.tight_layout(); plt.show()
ok_dec = np.isfinite(Dstat) and vg_pt.sill > vg_area.sill
record("Eq. 18: deconvolution recovers higher point-support sill",
       "PASS" if ok_dec else "FAIL",
       f"areal sill={vg_area.sill:.1f} -> point sill={vg_pt.sill:.1f} (D={Dstat:.3f})")


# ## 6. Run the three estimators (from the module)
# 
# The estimators come straight from `poisson_kriging_1`. We pass the deconvoluted
# point-support model and let `rate_base` auto-detect. Centroid and ATA run
# leave-one-out (each county predicted from the others) so predictions can be scored
# against observed rates.

# In[7]:


K = 16
pk_cent = PoissonKriging(vg_pt).fit(cent, rates, pops, ids=fips)          # module class
apk     = AreaPoissonKriging(vg_pt).fit(ps, pd.Series(rates, index=fips)) # module class
print(f"auto-detected rate_base: centroid={pk_cent.rate_base:.0f}  area={apk.rate_base:.0f}")

cpk = [pk_cent.predict_id(int(f), number_of_neighbors=K) for f in fips]
ata = [apk.predict_area(int(f), number_of_neighbors=K)   for f in fips]

zc = np.array([r["zhat"] for r in cpk]); sc = np.array([r["sig"] for r in cpk])
za = np.array([r["zhat"] for r in ata]); sa = np.array([r["sig"] for r in ata])
areas["z_centroid"], areas["s_centroid"] = zc, sc
areas["z_ata"],      areas["s_ata"]      = za, sa

def st(z): return (z-rates).mean(), np.abs(z-rates).mean(), z.var()
pd.DataFrame({
    "ME":  [np.nan, st(zc)[0], st(za)[0]],
    "MAE": [np.nan, st(zc)[1], st(za)[1]],
    "variance": [rates.var(), zc.var(), za.var()],
    "mean kriging sd": [np.nan, np.nanmean(sc), np.nanmean(sa)],
}, index=["raw rates","Centroid PK","ATA PK"]).round(3)
ok_rb = pk_cent.rate_base == 100000 and apk.rate_base == 100000
ok_cent = np.all(np.isfinite(zc)) and np.all(np.isfinite(sc))
ok_ata = np.all(np.isfinite(za)) and np.all(np.isfinite(sa))
ok_sd = np.nanmean(sa) < np.nanmean(sc)
record("rate_base auto-detected as 1e5", "PASS" if ok_rb else "FAIL",
       f"centroid={pk_cent.rate_base:.0f}  area={apk.rate_base:.0f}")
record("Centroid PK: leave-one-out predictions finite",
       "PASS" if ok_cent else "FAIL",
       f"ME={st(zc)[0]:.3f} MAE={st(zc)[1]:.3f} var={zc.var():.2f}")
record("ATA PK: leave-one-out predictions finite",
       "PASS" if ok_ata else "FAIL",
       f"ME={st(za)[0]:.3f} MAE={st(za)[1]:.3f} var={za.var():.2f}")
record("ATA kriging sd smaller than centroid (change of support)",
       "PASS" if ok_sd else "FAIL",
       f"mean sd centroid={np.nanmean(sc):.2f}  ATA={np.nanmean(sa):.2f}")

display(pd.DataFrame({
    "ME":  [np.nan, st(zc)[0], st(za)[0]],
    "MAE": [np.nan, st(zc)[1], st(za)[1]],
    "variance": [rates.var(), zc.var(), za.var()],
    "mean kriging sd": [np.nan, np.nanmean(sc), np.nanmean(sa)],
}, index=["raw rates","Centroid PK","ATA PK"]).round(3))


# Centroid and ATA agree on the risk (similar ME/MAE), but ATA's **kriging
# standard deviation is far smaller** — the change-of-support pay-off of respecting
# each county's area and population. Both are much smoother than the raw map.

# In[8]:


fig, ax = plt.subplots(2, 2, figsize=(12, 9))
vmin, vmax = np.percentile(rates, 2), np.percentile(rates, 98)
for a, col, ttl in [(ax[0,0],"rate","Raw rate"), (ax[0,1],"z_centroid","Centroid PK risk"),
                    (ax[1,0],"z_ata","ATA PK risk")]:
    areas.plot(col, cmap="RdBu_r", ax=a, vmin=vmin, vmax=vmax, edgecolor="k",
               linewidth=.2, legend=True, legend_kwds={"shrink":.7}); a.set_title(ttl); a.set_axis_off()
ax[1,1].scatter(rates, zc, s=14, alpha=.6, label="Centroid")
ax[1,1].scatter(rates, za, s=14, alpha=.6, label="ATA")
lim=[rates.min()-5, rates.max()+5]; ax[1,1].plot(lim, lim, "k--", lw=1)
ax[1,1].set_xlabel("observed"); ax[1,1].set_ylabel("LOO prediction")
ax[1,1].set_title("Leave-one-out CV"); ax[1,1].legend(); ax[1,1].set_aspect("equal"); ax[1,1].grid(alpha=.3)
plt.tight_layout(); plt.show()


# ## 7. Area-to-Point surface via `predict_points(..., targets=grid)`
# 
# The module's `predict_points` accepts arbitrary `targets`, so we evaluate ATP on a
# 12 km grid. Each node inherits the block-kriging system of its containing county
# (only the area-to-point RHS is recomputed), giving a continuous **isopleth** surface.

# In[9]:


step = 12_000
minx, miny, maxx, maxy = areas.total_bounds
gx, gy = np.meshgrid(np.arange(minx, maxx, step), np.arange(miny, maxy, step))
grid = gpd.GeoDataFrame(geometry=[Point(x,y) for x,y in zip(gx.ravel(), gy.ravel())], crs=areas.crs)
gj = gpd.sjoin(grid, areas[["FIPS","geometry"]], how="inner", predicate="within")

parts = []
for fid, sub in gj.groupby("FIPS"):
    tgt = np.column_stack([sub.geometry.x, sub.geometry.y])
    d = apk.predict_points(int(fid), number_of_neighbors=K, targets=tgt)   # <-- module ATP
    d["FIPS"] = int(fid); parts.append(d)
atp = pd.concat(parts, ignore_index=True)
print(f"ATP surface: {len(atp)} nodes | risk {atp.zhat.min():.1f}–{atp.zhat.max():.1f} "
      f"| variance {atp.zhat.var():.1f} (ATA {za.var():.1f}, less smoothed)")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
vmin, vmax = np.percentile(rates, 2), np.percentile(rates, 98)
s1 = ax[0].scatter(atp.x, atp.y, c=atp.zhat, cmap="RdBu_r", s=16, vmin=vmin, vmax=vmax, marker="s")
areas.boundary.plot(ax=ax[0], color="k", linewidth=.25); plt.colorbar(s1, ax=ax[0], shrink=.8, label="risk/100k")
ax[0].set_title("ATP isopleth risk surface (12 km)")
s2 = ax[1].scatter(atp.x, atp.y, c=atp.sig, cmap="magma", s=16, marker="s")
areas.boundary.plot(ax=ax[1], color="w", linewidth=.25); plt.colorbar(s2, ax=ax[1], shrink=.8, label="kriging sd")
ax[1].set_title("ATP kriging standard deviation")
for a in ax: a.set_axis_off()
plt.tight_layout(); plt.show()
ok_atp = (len(atp) > 0 and np.all(np.isfinite(atp.zhat))
          and atp.zhat.var() > za.var())
record("ATP: predict_points on 12 km grid (less smoothed than ATA)",
       "PASS" if ok_atp else "FAIL",
       f"{len(atp)} nodes | risk {atp.zhat.min():.1f}–{atp.zhat.max():.1f} | "
       f"var {atp.zhat.var():.1f} (ATA {za.var():.1f})")


# ### 7.1 Coherence constraint (Eq. 15)
# 
# With the module's corrected ATP, the **population-weighted average** of a county's
# point estimates equals its ATA estimate — verified to machine precision.

# In[10]:


rows = []
for fid in fips[:8]:
    a  = apk.predict_area(int(fid), number_of_neighbors=K)
    pp = apk.predict_points(int(fid), number_of_neighbors=K)   # county's own support points
    wavg = np.sum(pp.zhat*pp.population)/pp.population.sum()
    rows.append([int(fid), a["zhat"], wavg, abs(a["zhat"]-wavg), len(pp)])
coh = pd.DataFrame(rows, columns=["FIPS","ATA estimate","pop-wtd avg of ATP","abs diff","# points"])
print("max coherence error:", f"{coh['abs diff'].max():.2e}")
coh.round(4)
ok_coh = coh["abs diff"].max() < 1e-6
record("Eq. 15: ATP/ATA coherence to machine precision",
       "PASS" if ok_coh else "FAIL",
       f"max abs diff={coh['abs diff'].max():.2e} (n={len(coh)} counties)")

display(coh.round(4))


# ## 8. Kriging-variance comparison and summary

# In[11]:


fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
smax = np.nanpercentile(np.r_[sc, sa], 98)
areas.plot("s_centroid", cmap="magma", ax=ax[0], vmin=0, vmax=smax, edgecolor="k",
           linewidth=.2, legend=True, legend_kwds={"shrink":.7})
ax[0].set_title(f"Centroid PK sd (mean {np.nanmean(sc):.2f})")
areas.plot("s_ata", cmap="magma", ax=ax[1], vmin=0, vmax=smax, edgecolor="k",
           linewidth=.2, legend=True, legend_kwds={"shrink":.7})
ax[1].set_title(f"ATA PK sd (mean {np.nanmean(sa):.2f})")
for a in ax: a.set_axis_off()
plt.tight_layout(); plt.show()

print("Summary")
print(f"  raw-rate variance ......... {rates.var():8.2f}")
print(f"  Centroid PK variance ...... {zc.var():8.2f}  ({100*zc.var()/rates.var():.0f}% of raw)")
print(f"  ATA PK variance ........... {za.var():8.2f}  ({100*za.var()/rates.var():.0f}% of raw)")
print(f"  ATP surface variance ...... {atp.zhat.var():8.2f}  (less smoothed than ATA)")
print(f"  mean sd  centroid / ATA ... {np.nanmean(sc):.2f} / {np.nanmean(sa):.2f}")
print(f"  max coherence error ....... {coh['abs diff'].max():.1e}")
ok_smooth = zc.var() < rates.var() and za.var() < rates.var()
record("Centroid and ATA smoother than raw rates",
       "PASS" if ok_smooth else "FAIL",
       f"raw var={rates.var():.1f}  centroid={zc.var():.1f}  ATA={za.var():.1f}")


# ## 9. Summary
# 
# | Estimator | Module call | Support | Kriging variance |
# |-----------|-------------|---------|------------------|
# | **Centroid PK** | `PoissonKriging.predict_id` | point → point | inflated (ignores area/shape) |
# | **ATA PK** | `AreaPoissonKriging.predict_area` | area → area | correct, smaller (change of support) |
# | **ATP PK** | `AreaPoissonKriging.predict_points` | area → point | population-aware; coherent with ATA |
# 
# **Findings on this dataset**
# 
# - Deconvolution recovers a point-support model with a **higher sill** than the areal
#   model.
# - Centroid and ATA predict very similar risk, but **ATA reports markedly smaller
#   uncertainty**.
# - ATP yields a smooth continuous surface, **less over-smoothed than ATA**, and
#   satisfies the **coherence constraint to machine precision** (Eq. 15).
# - `rate_base` auto-detects to $10^5$; the correctly-scaled Poisson diagonal is what
#   keeps ATA/ATP stable.
# 
# All kriging is done by `poisson_kriging_1.py`; the notebook supplies only the
# variogram inference (model, Eq. 5 estimator, Eq. 18 deconvolution). Natural
# extensions: full lag-specific deconvolution (Goovaerts 2007) and p-field simulation
# for uncertainty propagation.

# ## 10. Test summary
# 

# In[12]:


summary_df = pd.DataFrame(RESULTS)
def _style(row):
    color = {"PASS": "background-color: #d4edda",
             "FAIL": "background-color: #f8d7da",
             "BLOCKED": "background-color: #fff3cd"}.get(row["Status"], "")
    return [color] * len(row)
summary_df.style.apply(_style, axis=1)

