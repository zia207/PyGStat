#!/usr/bin/env python
# coding: utf-8

# # Sequential Indicator Simulation (`sisim`) 
# 
# `src/pygstat/sisim.py` is a Python port of GSLIB's `sisim.for` (Sequential
# Indicator Simulation), ported from the **actual Fortran source**
# (`Helper_packages/Gslib90/gslib90sc/sisim.for` + the shared `gslib/` library
# routines it calls: `ordrel.for`, `beyond.for`, `powint.for`, `cova3.for`).
# 
# **Deliberate scope reductions** (documented in the module docstring, not
# silent): 3D -> 2D (matching the rest of pygstat); the super-block/multi-grid
# search strategies are pure performance optimizations of the Fortran's own
# brute-force search, replaced here with `sgsim.py`'s octant search
# (mathematically equivalent, just not sped up for huge grids); only
# single-structure variograms per threshold; median-IK weight reuse, soft/
# Markov-Bayes data, and the tabulated-quantiles tail options (which need an
# external reference file) are not implemented.
# 
# This notebook validates the real properties SIS is supposed to have:
# 
# 1. **Exact conditioning** at data locations, in every realization.
# 2. **Global CDF reproduction** -- across many realizations, `P(Z <= threshold)`
#    at each threshold should be close to the input `global_cdf` used to drive
#    the simulation.
# 3. **Order-relations correction** (`ordrel`) actually fixes non-monotonic
#    local CDFs -- tested directly against the ported function, not just
#    indirectly through the full simulation.
# 4. **CDF interpolation/tail draw** (`beyond`) is tested directly: known
#    cdfval -> known zval, checked against hand-computed expected values for
#    the linear/power/hyperbolic tail models.
# 5. **Threshold consistency**: the indicator convention (`I(Z <= threshold)`)
#    matches this repo's own `fit_indicator_variogram`.
# 6. **Reproducibility**: same seed -> identical output.
# 

# ## 0. Setup

# In[1]:


import matplotlib
matplotlib.use("Agg")
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pygstat
from pygstat import sisim
print("pygstat version:", pygstat.__version__)

RESULTS = []
def record(section, status, note=""):
    RESULTS.append({"Feature": section, "Status": status, "Notes": note})
    marker = {"PASS": "\u2705", "FAIL": "\u274c", "BLOCKED": "\u26a0\ufe0f"}.get(status, "?")
    print(f"{marker} [{status}] {section}  {note}")


# In[2]:


df = pd.read_csv("../data/meuse.csv")
zinc = df["zinc"].values
coords = df[["x", "y"]].values

thresholds = np.percentile(zinc, [10, 25, 50, 75, 90]).tolist()
global_cdf = [0.10, 0.25, 0.50, 0.75, 0.90]
print("thresholds (zinc):", [round(t, 1) for t in thresholds])
print("global_cdf:", global_cdf)

xmin, xmax = df.x.min(), df.x.max()
ymin, ymax = df.y.min(), df.y.max()


# Rather than guessing indicator-variogram parameters, fit each threshold's
# empirical indicator variogram with `fit_indicator_variogram` (already in
# `pygstat`) and an exponential curve fit -- one real variogram per threshold,
# in the GSLIB practical-range convention `sisim`/`sgsim` share.

# In[3]:


from pygstat import fit_indicator_variogram
from scipy.optimize import curve_fit

def _exp_gamma(h, nugget, sill, rng_):
    return nugget + sill * (1.0 - np.exp(-3.0 * h / rng_))

vario = {}
indicator_vario_fits = {}  # threshold -> (lags, gamma, popt), kept for plotting below
for t in thresholds:
    lags, gamma = fit_indicator_variogram(coords, zinc, threshold=t, n_lags=10)
    mask = ~np.isnan(gamma)
    popt, _ = curve_fit(_exp_gamma, lags[mask], gamma[mask],
                         p0=[0.01, gamma[mask].max(), lags.max() / 2],
                         bounds=([0, 1e-6, 1e-3], [1, 1, lags.max() * 3]), maxfev=5000)
    nugget, psill, rng_ = popt
    # sisim/sgsim expect total C(0) = nugget + partial sill as vario[4]
    vario[t] = [0, nugget, rng_, rng_, nugget + psill, "exponential"]
    indicator_vario_fits[t] = (lags, gamma, popt)
    print(f"t={t:7.1f}: nugget={nugget:.3f} psill={psill:.3f} "
          f"total sill={nugget + psill:.3f} range={rng_:.0f}")


# ### Indicator variograms by CDF level
# 
# One empirical + fitted indicator variogram per threshold, labeled by the CDF
# probability it corresponds to (`I(Z <= threshold)`, so its sill is the
# indicator's variance `p(1-p)` at that cutoff -- variograms for cutoffs near
# the median (p close to 0.5) should have the largest sill; cutoffs near the
# tails (p close to 0 or 1) should have small sills, since the indicator is
# nearly constant (mostly 0s or mostly 1s) out there.

# In[4]:


fig, axes = plt.subplots(1, len(thresholds), figsize=(4.2 * len(thresholds), 4), sharey=True)
for ax, t, p in zip(axes, thresholds, global_cdf):
    lags, gamma, popt = indicator_vario_fits[t]
    h = np.linspace(0, lags.max(), 200)
    ax.scatter(lags, gamma, color="black", s=25, label="empirical")
    ax.plot(h, _exp_gamma(h, *popt), color="crimson", label="fitted")
    ax.axhline(p * (1 - p), color="gray", ls=":", lw=1, label="p(1-p)")
    ax.set_title(f"t={t:.0f}  (P(Z\u2264t)={p:.2f})")
    ax.set_xlabel("lag distance")
axes[0].set_ylabel("indicator semivariance")
axes[0].legend(fontsize=8)
fig.suptitle("Indicator variograms by CDF level", y=1.04, fontsize=13)
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(6.5, 5))
fitted_sills = [indicator_vario_fits[t][2][0] + indicator_vario_fits[t][2][1]
                for t in thresholds]  # total C(0) = nugget + partial sill
p_grid = np.linspace(0.01, 0.99, 100)
ax2.plot(p_grid, p_grid * (1 - p_grid), color="gray", ls="--", label="theoretical p(1-p)")
ax2.plot(global_cdf, fitted_sills, "o-", color="crimson", label="fitted sill")
for t, p, s in zip(thresholds, global_cdf, fitted_sills):
    ax2.annotate(f"t={t:.0f}", (p, s), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax2.set_xlabel("global CDF probability p = P(Z \u2264 t)")
ax2.set_ylabel("fitted indicator variogram sill")
ax2.set_title("Fitted sill vs. the indicator-variance envelope p(1-p)")
ax2.legend()
plt.tight_layout()
plt.show()

sill_follows_envelope = all(s <= p * (1 - p) + 0.05 for s, p in zip(fitted_sills, global_cdf))
record("Indicator variogram sills stay within the p(1-p) envelope",
       "PASS" if sill_follows_envelope else "FAIL",
       f"fitted sills={[round(s,3) for s in fitted_sills]}, "
       f"p(1-p) envelope={[round(p*(1-p),3) for p in global_cdf]}")


# ## 1. `ordrel` -- order relations correction (tested directly)
# 
# Feed a deliberately non-monotonic ccdf and confirm the ported `_ordrel`
# produces a properly non-decreasing, [0,1]-clipped result.

# In[5]:


from pygstat.sisim import _ordrel

bad_ccdf = np.array([0.05, 0.30, 0.22, 0.71, 1.15])  # violates monotonicity and [0,1]
fixed = _ordrel(bad_ccdf)
print("input :", bad_ccdf)
print("fixed :", np.round(fixed, 4))

monotonic = np.all(np.diff(fixed) >= -1e-12)
in_range = np.all((fixed >= 0) & (fixed <= 1))
record("ordrel: fixes monotonicity violations", "PASS" if monotonic else "FAIL",
       f"output diffs: {np.round(np.diff(fixed), 4)}")
record("ordrel: clips to [0,1]", "PASS" if in_range else "FAIL",
       f"range=[{fixed.min():.3f}, {fixed.max():.3f}]")


# ## 2. `beyond` -- CDF interpolation/tail draw (tested directly)
# 
# Check the ported `_beyond_zval` against hand-computed values for each tail
# model, using a simple 3-threshold ccdf.

# In[6]:


from pygstat.sisim import _beyond_zval, _powint

ccut = np.array([100.0, 500.0, 1000.0])
ccdf = np.array([0.2, 0.6, 0.9])
zmin, zmax = 0.0, 2000.0

# --- middle, linear: cdfval=0.4 is halfway between ccdf[0]=0.2 and ccdf[1]=0.6
z_mid = _beyond_zval(0.4, ccut, ccdf, zmin, zmax, "linear", 1.0, "linear", 1.0, "linear", 1.0)
expected_mid = _powint(0.2, 0.6, 100.0, 500.0, 0.4, 1.0)
record("beyond: middle linear interpolation", "PASS" if np.isclose(z_mid, expected_mid) else "FAIL",
       f"got {z_mid:.3f}, expected {expected_mid:.3f}")

# --- lower tail, linear: cdfval=0.1 is below ccdf[0]=0.2
z_low = _beyond_zval(0.1, ccut, ccdf, zmin, zmax, "linear", 1.0, "linear", 1.0, "linear", 1.0)
expected_low = _powint(0.0, 0.2, zmin, 100.0, 0.1, 1.0)
record("beyond: lower tail linear extrapolation", "PASS" if np.isclose(z_low, expected_low) else "FAIL",
       f"got {z_low:.3f}, expected {expected_low:.3f}")

# --- upper tail, hyperbolic: cdfval=0.95 is above ccdf[-1]=0.9
utpar = 2.0
lam = (ccut[-1] ** utpar) * (1.0 - ccdf[-1])
expected_up = (lam / (1.0 - 0.95)) ** (1.0 / utpar)
z_up = _beyond_zval(0.95, ccut, ccdf, zmin, zmax, "linear", 1.0, "linear", 1.0, "hyperbolic", utpar)
record("beyond: upper tail hyperbolic extrapolation", "PASS" if np.isclose(z_up, expected_up) else "FAIL",
       f"got {z_up:.3f}, expected {expected_up:.3f}")

# --- clipping to [zmin, zmax]
z_clip = _beyond_zval(0.999999, ccut, ccdf, 0.0, 950.0, "linear", 1.0, "linear", 1.0, "linear", 1.0)
record("beyond: clips to zmax", "PASS" if z_clip == 950.0 else "FAIL", f"got {z_clip}")


# ## 3. Exact conditioning
# 
# Simulate on a grid including real meuse sample locations plus new points,
# across several realizations. Every realization must reproduce the true
# value exactly at real locations.

# In[7]:


n_cond_check = 15
cond_idx = np.random.default_rng(0).choice(len(coords), size=n_cond_check, replace=False)
cond_pts = coords[cond_idx]
new_pts = np.column_stack([
    np.random.default_rng(1).uniform(xmin, xmax, 20),
    np.random.default_rng(2).uniform(ymin, ymax, 20),
])
grid = np.vstack([cond_pts, new_pts])

sims = sisim(grid, df, "x", "y", "zinc", thresholds, global_cdf, vario,
             num_points=16, radius=3000, kriging_type="ordinary",
             nsim=5, seed=42, quiet=True)

true_vals = zinc[cond_idx]
exact = all(np.allclose(sims[r, :n_cond_check], true_vals, atol=1e-6) for r in range(sims.shape[0]))
record("Exact conditioning at data locations", "PASS" if exact else "FAIL",
       f"max abs error across {sims.shape[0]} realizations = "
       f"{np.max(np.abs(sims[:, :n_cond_check] - true_vals)):.2e}")

varies = not np.allclose(sims[:, n_cond_check:], sims[0, n_cond_check:])
record("Realizations vary at unconditioned nodes", "PASS" if varies else "FAIL",
       f"std across realizations at new points: mean={sims[:, n_cond_check:].std(axis=0).mean():.1f}")


# ## 4. Global CDF reproduction
# 
# Run many realizations and compare the pooled fraction of simulated values at
# or below each threshold to the input `global_cdf` that drove the simulation.
# 
# **A first version of this check used a uniform grid over meuse's bounding
# box and found a ~0.15-0.2 gap.** Root-caused below (Section 4a) before
# trusting it as a real bug: meuse's zinc concentration has a strong spatial
# trend (highest near the river, decreasing with distance -- the same `dist`
# covariate used in the regression-kriging tests elsewhere in this repo), and
# the 155 samples are themselves clustered near the river. A uniform
# bounding-box grid covers a lot of area far from the river that the sample
# -- and therefore `global_cdf`, computed as percentiles of the sample -- barely
# represents. That's a **test methodology mismatch**, not a `sisim.py` bug:
# Section 4b confirms it by rerunning the identical check on the real (masked)
# `meuse_grid.csv` domain, which matches the actual sampled area.

# ### 4a. Root cause: raw kriging estimates, isolated from `ordrel`/`beyond`
# 
# Before assuming the gap was in the simulation draw, the two other moving
# parts were checked in isolation: `_beyond_zval`'s inverse-CDF sampling (given
# a *fixed* local ccdf, do random draws reproduce it exactly? yes) and whether
# `_ordrel`'s correction shifts the *pooled average* ccdf (it does, modestly --
# that's expected of a nonlinear monotonicity fix, since raw kriging estimates
# occasionally violate order relations). The dominant effect, though, is
# already present in the raw per-node kriging estimate itself, averaged over a
# grid whose spatial coverage doesn't match the sample's -- `ordrel`'s own
# contribution is clearly secondary to that.

# In[8]:


from pygstat.sisim import _beyond_zval as _bz_check, _indicator_ccdf, _ordrel
from pygstat.sgsim import _octant_search

# Check 1: for a FIXED local ccdf, does random cdfval sampling reproduce it exactly?
_ccut = thresholds
_ccdf = np.array([0.114, 0.244, 0.375, 0.546, 0.795])
_rng = np.random.default_rng(0)
_draws = np.array([_bz_check(_rng.uniform(), _ccut, _ccdf, float(zinc.min()), float(zinc.max()),
                              "linear", 1.0, "linear", 1.0, "linear", 1.0) for _ in range(50000)])
_empirical = [np.mean(_draws <= t) for t in _ccut]
_bz_ok = np.allclose(_empirical, _ccdf, atol=0.01)
record("Root cause check: beyond() sampling matches a fixed local ccdf exactly",
       "PASS" if _bz_ok else "FAIL", f"target={list(_ccdf)}, empirical={[round(e,3) for e in _empirical]}")

# Check 2: does ordrel's correction shift the POOLED AVERAGE ccdf across many nodes?
_data = df.rename(columns={"x": "X", "y": "Y", "zinc": "Z"})[["X", "Y", "Z"]]
_rng2 = np.random.default_rng(1)
_targets = np.column_stack([_rng2.uniform(xmin, xmax, 400), _rng2.uniform(ymin, ymax, 400)])
_raw, _corr = [], []
for _t in _targets:
    try:
        _near = _octant_search(3000, 16, _t, _data)
    except ValueError:
        continue
    _r = _indicator_ccdf(_t, _near[:, :2], _near[:, 2], thresholds, np.array(global_cdf), vario, "ordinary")
    _raw.append(_r); _corr.append(_ordrel(_r))
_raw, _corr = np.array(_raw), np.array(_corr)
_raw_mean, _corr_mean = _raw.mean(axis=0), _corr.mean(axis=0)
_ordrel_shift = np.max(np.abs(_raw_mean - _corr_mean))
_raw_vs_target_gap = np.max(np.abs(_raw_mean - np.array(global_cdf)))
# ordrel does shift the pooled average somewhat (it's a nonlinear correction,
# so that's expected when raw estimates occasionally violate monotonicity)
# -- the diagnostic question is whether that shift is the dominant source of
# the gap versus target, or clearly secondary to it.
record("Root cause check: ordrel's shift is secondary to the raw-vs-target gap",
       "PASS" if _ordrel_shift < 0.5 * _raw_vs_target_gap else "FAIL",
       f"ordrel shift={_ordrel_shift:.3f} vs raw-to-target gap={_raw_vs_target_gap:.3f} "
       f"(ordrel accounts for {100*_ordrel_shift/_raw_vs_target_gap:.0f}% of it)")
print("mean raw ccdf over a uniform bounding-box grid:", np.round(_raw_mean, 3))
print("mean corrected (post-ordrel) ccdf:              ", np.round(_corr_mean, 3))
print("target global_cdf (from the clustered, river-biased sample):", global_cdf)
print("-> most of the gap is already present in the raw kriging estimate, "
      "before ordrel/beyond ever run; ordrel contributes some but is not the dominant cause.")


# ### 4b. The fair test: pooled reproduction on the real `meuse_grid.csv` domain
# 
# Same check, same variograms, but on a random subsample of the actual (masked)
# sampled area instead of an unrepresentative uniform grid.

# In[9]:


grid_df = pd.read_csv("../data/meuse_grid.csv")
real_domain_sample = grid_df.sample(n=500, random_state=0)[["x", "y"]].values

t0 = time.time()
sims_full = sisim(real_domain_sample, df, "x", "y", "zinc", thresholds, global_cdf, vario,
                   num_points=16, radius=1200, kriging_type="ordinary",
                   nsim=30, seed=7, quiet=True)
elapsed = time.time() - t0
print(f"{sims_full.shape[0]} realizations x {sims_full.shape[1]} nodes (real domain) in {elapsed:.1f}s")

pooled = sims_full.ravel()
reproduced_cdf = [np.mean(pooled <= t) for t in thresholds]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(thresholds, global_cdf, "o-", label="input global_cdf", color="steelblue")
ax.plot(thresholds, reproduced_cdf, "s--", label="reproduced (pooled SIS, real domain)", color="darkorange")
ax.set_xlabel("threshold (zinc)"); ax.set_ylabel("P(Z <= threshold)")
ax.legend(); ax.set_title("SIS reproduces the target global CDF (real sampled domain)")
plt.tight_layout(); plt.show()

max_diff = max(abs(a - b) for a, b in zip(global_cdf, reproduced_cdf))
record("Global CDF reproduction (real meuse_grid.csv domain)", "PASS" if max_diff < 0.12 else "FAIL",
       f"input={global_cdf}, reproduced={[round(c,3) for c in reproduced_cdf]}, "
       f"max diff={max_diff:.3f} (~{30*500} pooled draws)")


# ## 5. E-type vs. ordinary indicator kriging
# 
# The mean over many realizations at each node should broadly track what a
# single-shot `pygstat.IndicatorKriging` + E-type estimate gives at the same
# nodes -- simulation adds variability around the same underlying local
# distributions, it doesn't change their central tendency.

# In[10]:


from pygstat import IndicatorKriging, compute_etype_from_probabilities

ik = IndicatorKriging(cov_model="spherical").fit(coords, zinc)
# Use the fitted (nugget, sill, range) per threshold instead of guessed values
# IndicatorKriging covariance uses (nugget, partial sill, range)
variogram_params = {t: (vario[t][1], vario[t][4] - vario[t][1], vario[t][2]) for t in thresholds}
probs = ik.predict(real_domain_sample, thresholds=thresholds, variogram_params=variogram_params)
ik_etype = compute_etype_from_probabilities(thresholds, [probs[t] for t in thresholds])

sis_etype = sims_full.mean(axis=0)

# Compare on nodes where the IK E-type is well inside the data range (the
# partial-expectation E-type formula only integrates between thresholds, so
# it systematically underestimates far outside that span -- restrict the
# comparison to where both estimators are actually estimating the same thing)
mask = (ik_etype > thresholds[0]) & (ik_etype < thresholds[-1])
corr = np.corrcoef(sis_etype[mask], ik_etype[mask])[0, 1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc0 = axes[0].scatter(real_domain_sample[:, 0], real_domain_sample[:, 1], c=sis_etype, s=20, cmap="viridis")
axes[0].set_title(f"SIS E-type ({sims_full.shape[0]} realizations)"); plt.colorbar(sc0, ax=axes[0])
axes[1].scatter(ik_etype[mask], sis_etype[mask], s=15, alpha=0.6)
lims = [thresholds[0], thresholds[-1]]
axes[1].plot(lims, lims, "r--", lw=1)
axes[1].set_xlabel("IndicatorKriging E-type"); axes[1].set_ylabel("SIS E-type")
axes[1].set_title(f"correlation = {corr:.3f}")
plt.tight_layout(); plt.show()

record("SIS E-type correlates with IndicatorKriging E-type", "PASS" if corr > 0.5 else "FAIL",
       f"correlation={corr:.3f} (n={mask.sum()} nodes in comparison range)")


# ## 6. Reproducibility

# In[11]:


sims_a = sisim(grid, df, "x", "y", "zinc", thresholds, global_cdf, vario,
               num_points=16, radius=3000, nsim=2, seed=99, quiet=True)
sims_b = sisim(grid, df, "x", "y", "zinc", thresholds, global_cdf, vario,
               num_points=16, radius=3000, nsim=2, seed=99, quiet=True)
sims_c = sisim(grid, df, "x", "y", "zinc", thresholds, global_cdf, vario,
               num_points=16, radius=3000, nsim=2, seed=123, quiet=True)

record("Reproducibility: same seed -> identical output",
       "PASS" if np.allclose(sims_a, sims_b) else "FAIL", "")
record("Reproducibility: different seed -> different output",
       "PASS" if not np.allclose(sims_a, sims_c) else "FAIL", "")


# ## 7. Prediction on the real `meuse_grid.csv`, with a per-threshold vario dict
# 
# Also exercises the `vario` dict form (one variogram per threshold, instead
# of a single one reused for all) and `kriging_type='simple'`.

# In[12]:


grid_df = pd.read_csv("../data/meuse_grid.csv")
mgrid = grid_df[["x", "y"]].values
print(f"meuse_grid.csv: {len(mgrid)} points")

vario_per_threshold = {
    t: [0, 0.05, 900.0, 900.0, p * (1 - p), "exponential"]
    for t, p in zip(thresholds, global_cdf)
}

t0 = time.time()
sims_grid = sisim(mgrid, df, "x", "y", "zinc", thresholds, global_cdf, vario_per_threshold,
                   num_points=16, radius=1200, kriging_type="simple",
                   nsim=6, seed=2024, quiet=True)
elapsed = time.time() - t0
print(f"{sims_grid.shape[0]} realizations x {len(mgrid)} points in {elapsed/60:.1f} min")

finite = np.all(np.isfinite(sims_grid))
in_range = sims_grid.min() >= 0 and sims_grid.max() <= zinc.max() * 1.01
record("sisim on real meuse_grid.csv (per-threshold vario, simple kriging)",
       "PASS" if finite and in_range else "FAIL",
       f"range=[{sims_grid.min():.1f}, {sims_grid.max():.1f}], {elapsed/60:.1f} min")

etype_grid = sims_grid.mean(axis=0)
pick_rng = np.random.default_rng(2024)
chosen = np.sort(pick_rng.choice(sims_grid.shape[0], size=3, replace=False))

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
vmin, vmax = np.percentile(sims_grid, [1, 99])
for ax, r in zip(axes[:3], chosen):
    sc = ax.scatter(mgrid[:, 0], mgrid[:, 1], c=sims_grid[r], s=6, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(f"Realization #{r}"); ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax)
sc = axes[3].scatter(mgrid[:, 0], mgrid[:, 1], c=etype_grid, s=6, cmap="viridis", vmin=vmin, vmax=vmax)
axes[3].set_title(f"E-type ({sims_grid.shape[0]} realizations)"); axes[3].set_aspect("equal")
plt.colorbar(sc, ax=axes[3])
fig.suptitle("Sequential Indicator Simulation on meuse_grid.csv (zinc)", fontsize=13)
plt.tight_layout(); plt.show()


# ## Summary

# In[13]:


summary_df = pd.DataFrame(RESULTS)
def _style(row):
    color = {"PASS": "background-color: #d4edda",
             "FAIL": "background-color: #f8d7da",
             "BLOCKED": "background-color: #fff3cd"}.get(row["Status"], "")
    return [color] * len(row)
summary_df.style.apply(_style, axis=1)


# ## Notes
# 
# - This is a faithful port of the **actual Fortran source**
#   (`sisim.for` + `gslib/ordrel.for`, `gslib/beyond.for`, `gslib/powint.for`,
#   `gslib/cova3.for`), unlike `sgsim.py` (whose `.exe` had no available
#   source). The math in `_ordrel`, `_beyond_zval`, and `_powint` was checked
#   line-by-line against the Fortran and is tested directly above (Sections 1-2),
#   not just indirectly through the full simulation.
# - Reuses `sgsim.py`'s covariance/rotation/octant-search engine, confirmed
#   against `cova3.for` to use the identical practical-range convention.
# - **Not implemented** (documented in the module docstring): 3D (2D only,
#   matching pygstat elsewhere), super-block/multi-grid search speedups
#   (replaced with octant search -- same results, not optimized for huge
#   grids), multi-structure variograms per threshold, median-IK weight reuse,
#   soft/Markov-Bayes data, and the tabulated-quantiles tail options
#   (`ltail/middle/utail = 3` in the original, which need an external
#   reference-distribution file).
# 
