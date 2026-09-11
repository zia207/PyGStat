#!/usr/bin/env python
# coding: utf-8

# # Multivariate Poisson Cokriging for Disease Count Data
# ### Prediction and noise-filtering of areal disease risk with correlated auxiliary variables
# 
# ---
# 
# **Abstract.** Mapping disease risk from areal counts is hampered by three things:
# counts are discrete and often over-dispersed or zero-inflated; dividing counts by
# population gives heavy-tailed, unreliable *rates* (the small-number problem); and
# a single disease rarely carries enough signal on its own. **Multivariate Poisson
# cokriging** addresses all three by (i) treating each count as Poisson, (ii)
# modelling several spatially co-regionalised diseases jointly through a **Linear
# Model of Coregionalization (LMC)**, and (iii) predicting or smoothing a target
# disease's risk by borrowing strength from correlated auxiliary diseases. This
# notebook develops the mathematics, implements it in the importable module
# **`multi_poisson_kriging.py`**, validates it on a controlled simulation, and
# applies it to **diabetes in the US Northeast** using physical-inactivity and
# obesity as auxiliaries.
# 
# **Method reference**
# 
# > **Payares-Garcia, D., Osei, F., Mateu, J., Stein, A. (2024).** *Multivariate
# > Poisson cokriging: a geostatistical model for health count data.* **Statistical
# > Methods in Medical Research** 33(10): 1637–1656. doi:10.1177/09622802241268488
# > (PMC11500483). Building on the bivariate Poisson cokriging of Payares-Garcia
# > et al. (*Spatial Statistics*, 2023) and Goovaerts-style Poisson kriging.
# 
# **Contents**
# 
# 1. **Mathematical model** — Poisson data model, population-weighted direct/cross
#    semivariogram estimators, the LMC, the cokriging predictor and system, and the
#    prediction error variance, each mapped to the code that implements it.
# 2.–6. **Simulation study** — recover a known multivariate risk field and quantify
#    the MSPE gain from auxiliaries (mirroring the paper's simulation).
# 7. **Application** — diabetes risk across the US Northeast.
# 8. **Summary.**
# 
# **Software.** All kriging is performed by `multi_poisson_kriging.py` (a Python
# conversion of the author's R code). Keep that module — and, for Section 7, the
# Northeast shapefile `data/diabetes_northeast.shp` (with `.shx`, `.dbf`, `.prj`,
# `.cpg` and `diabetes_northeast.csv`) — to run this notebook. All application
# maps are **county polygons**, not point scatter.

# ## 0. Environment

# In[1]:


import matplotlib
matplotlib.use("Agg")
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import pandas as pd
import geopandas as gpd
import pygstat
display = print
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from scipy.optimize import least_squares
from pygstat import multi_poisson_kriging as mpk

plt.rcParams.update({"figure.dpi":110, "font.size":10,
                     "axes.spines.top":False, "axes.spines.right":False})
RNG = np.random.default_rng(512)     # R used seed = 512
POP_RATE = 100_000
print("pygstat", pygstat.__version__, "| geopandas", gpd.__version__)
print("module functions:", ", ".join(mpk.__all__))

RESULTS = []
def record(section, status, note=""):
    RESULTS.append({"Feature": section, "Status": status, "Notes": note})
    marker = {"PASS": "\u2705", "FAIL": "\u274c", "BLOCKED": "\u26a0\ufe0f"}.get(status, "?")
    print(f"{marker} [{status}] {section}  {note}")

record("multi_poisson_kriging: module import", "PASS",
       f"pck_variogram, fit_lmc, poisson_krige, poisson_cokrige from pygstat {pygstat.__version__}")


# ## 1. Mathematical model
# 
# We follow the formulation of Payares-Garcia et al. (2024). Let there be
# $\mathcal{L}$ diseases indexed by $\ell,\kappa\in\{1,\dots,\mathcal{L}\}$, observed
# over $N$ areal units with (population-weighted) centroids $s_1,\dots,s_N$. Unit
# $s_i$ has a population at risk $n_\ell(s_i)$ — here a common denominator
# $n(s_i)$ shared across diseases. Write $h=\lVert s_i-s_j\rVert$ and let $\varrho$
# be the rate base (e.g. $\varrho=10^5$ for rates per 100,000).
# 
# ### 1.1 Poisson data model
# 
# Each count is conditionally Poisson given the latent **risk** $R_\ell$:
# 
# $$Y_\ell(s_i)\,\big|\,R_\ell(s_i)\;\sim\;\mathrm{Poisson}\!\big(n(s_i)\,R_\ell(s_i)\big),\qquad
# z_\ell(s_i)=\varrho\,\frac{Y_\ell(s_i)}{n(s_i)} .$$
# 
# The sample rate $z_\ell$ is an unbiased but noisy estimator of $\varrho R_\ell$,
# and — crucially — its sampling variance is inversely proportional to the
# population,
# 
# $$\mathbb{E}[z_\ell(s_i)]=\varrho R_\ell(s_i),\qquad
# \mathrm{Var}[z_\ell(s_i)]\;\propto\;\frac{R_\ell(s_i)}{n(s_i)} .$$
# 
# Small populations therefore give unreliable rates. The goal is to predict the
# target risk $R_t$ at an unsampled $s_0$, or to *smooth* it at observed units,
# filtering this Poisson noise.

# ### 1.2 Population-weighted (cross-)semivariogram estimators
# 
# Spatial dependence within and between diseases is measured with
# **population-weighted, bias-corrected** estimators (eqs. 12–13 of the paper). For
# pairs $(i,j)$ whose separation falls in lag class $N(h)$, and with the
# inverse-variance weights
# 
# $$w_{ij}=\frac{n(s_i)\,n(s_j)}{n(s_i)+n(s_j)},$$
# 
# the **direct** semivariogram of disease $\ell$ is
# 
# $$\hat\gamma_{\ell\ell}(h)=\frac{1}{2\sum_{N(h)}w_{ij}}
# \sum_{N(h)}\Big[\,w_{ij}\big(z_\ell(s_i)-z_\ell(s_j)\big)^2-m^{*}_{\ell}\,\Big],
# \qquad m^{*}_{\ell}=\varrho\,\frac{\sum_i Y_\ell(s_i)}{\sum_i n(s_i)},$$
# 
# and the **cross** semivariogram between $\ell$ and $\kappa$ is
# 
# $$\hat\gamma_{\ell\kappa}(h)=\frac{1}{2\sum_{N(h)}w_{ij}}
# \sum_{N(h)}\Big[\,w_{ij}\big(z_\ell(s_i)-z_\ell(s_j)\big)\big(z_\kappa(s_i)-z_\kappa(s_j)\big)-m_{\ell\kappa}\,\Big].$$
# 
# The weights $w_{ij}$ homogenise the variance of the sample rates, while the
# **bias terms** $m^{*}_{\ell}$ and $m_{\ell\kappa}$ (mean shared risk,
# §1.4) subtract the Poisson noise induced by population variability — without them
# the estimated sill is inflated by pure sampling noise. This is exactly what
# `pck_variogram` and `pck_crossvariogram` compute.

# ### 1.3 Linear Model of Coregionalization
# 
# To obtain a **valid** joint covariance for every pair of diseases, the direct and
# cross structures are fitted simultaneously with an LMC: a sum of $S$ basic,
# licit correlation structures $g_u$ (the paper uses a nested **exponential +
# spherical**; this module uses a single exponential, $S=1$),
# 
# $$C_{\ell\kappa}(h)=\sum_{u=1}^{S} b^{u}_{\ell\kappa}\,g_u(h),
# \qquad g_u(0)=1,$$
# 
# with **coregionalization matrices** $B_u=\big[b^{u}_{\ell\kappa}\big]_{\mathcal L\times\mathcal L}$.
# The model is a valid multivariate covariance **iff every $B_u$ is symmetric
# positive semidefinite**; `fit_lmc` estimates $B_u$ by weighted least squares and
# then projects it onto the nearest PSD matrix. Covariance and semivariogram are
# linked by $\gamma_{\ell\kappa}(h)=C_{\ell\kappa}(0)-C_{\ell\kappa}(h)$, and for a
# single structure the sill is $C_{\ell\kappa}(0)=b_{\ell\kappa}=B_{\ell\kappa}$.
# Basic structures used here:
# 
# $$g^{\text{exp}}(h)=e^{-h/a},\qquad
# g^{\text{sph}}(h)=\Big(1-\tfrac{3h}{2a}+\tfrac{h^{3}}{2a^{3}}\Big)\mathbf{1}_{\{h\le a\}} .$$

# ### 1.4 Poisson cokriging predictor
# 
# The target risk at $s_0$ is a linear combination of **all** diseases' sample
# rates (eq. 7):
# 
# $$\boxed{\;\hat R_t(s_0)=\sum_{\ell=1}^{\mathcal L}\sum_{i=1}^{k_\ell}
# \lambda_{\ell}(s_i)\,z_\ell(s_i)\;}$$
# 
# where $k_\ell$ is the number of $\ell$-observations used and $\lambda_\ell(s_i)$
# are the cokriging weights (negative predictions are truncated at $0$). The
# **Poisson shared risk** that couples diseases $\ell$ and $\kappa$ is modelled, in
# the spirit of Kawamura's bivariate Poisson, as
# 
# $$R_{\ell\kappa}(s_i)=\rho_{\ell\kappa}\sqrt{R_\ell(s_i)\,R_\kappa(s_i)},\qquad
# m_{\ell\kappa}=\overline{R_{\ell\kappa}} ,$$
# 
# with $\rho_{\ell\kappa}$ the Poisson correlation (`shared_risk`).

# ### 1.5 Cokriging system with population-bias terms
# 
# The weights minimise the prediction variance subject to unbiasedness. This yields
# a linear system of $\big(\sum_{\ell}k_\ell+\mathcal L\big)$ equations in the
# weights $\lambda_\ell(s_i)$ and $\mathcal L$ Lagrange multipliers $\mu_\ell$
# (eq. 9): for every disease $\ell$ and observation $s_i$,
# 
# $$\sum_{\kappa=1}^{\mathcal L}\sum_{j=1}^{k_\kappa}
# \lambda_{\kappa}(s_j)\Big[\,C_{\ell\kappa}(s_i,s_j)+\Delta_{\ell\kappa}(s_i,s_j)\,\Big]
# +\mu_\ell \;=\; C_{t\ell}(s_i,s_0),$$
# 
# subject to the **unbiasedness constraints**
# 
# $$\sum_{i=1}^{k_t}\lambda_t(s_i)=1,\qquad
# \sum_{i=1}^{k_\ell}\lambda_\ell(s_i)=0\quad(\ell\neq t).$$
# 
# The target disease's weights sum to one; each auxiliary's weights sum to zero.
# The **bias / reliability terms** $\Delta_{\ell\kappa}$, added to the block
# diagonal, are the multivariate Poisson error variance and encode
# $\mathrm{Var}[z]\propto 1/n$:
# 
# $$\Delta_{\ell\ell}(s_i,s_i)=\frac{1}{n(s_i)}\sum_{\kappa=1}^{\mathcal L}m_{\ell\kappa}\;\;\big(\text{here }m^{*}_\ell/n(s_i)\big),
# \qquad
# \Delta_{\ell\kappa}(s_i,s_i)=\frac{m_{\ell\kappa}}{n(s_i)}\ \ (\ell\neq\kappa),$$
# 
# and $\Delta=0$ off the diagonal / for non-co-located pairs. Assembling the blocks
# $C_{\ell\kappa}$ (with these terms) plus the constraint rows/columns gives the
# matrix that `poisson_cokrige` builds and inverts.

# ### 1.6 Prediction error variance
# 
# With $c_0$ the right-hand-side covariance vector and $\lambda$ the solved weights
# (including the target Lagrange multiplier $\mu_t$), the cokriging variance is
# (eqs. 10–11)
# 
# $$\sigma^{2}_t(s_0)=C_{tt}(0)-\sum_{\ell=1}^{\mathcal L}\sum_{i=1}^{k_\ell}
# \lambda_\ell(s_i)\,C_{t\ell}(s_i,s_0)-\mu_t
# \;=\;B_{tt}-\lambda^{\!\top}c_0 .$$
# 
# Because auxiliaries add information, this variance is systematically **smaller**
# than the univariate Poisson-kriging variance.
# 
# ### 1.7 Special cases
# 
# * **Univariate Poisson kriging** ($\mathcal L=1$): the system collapses to the
#   ordinary Poisson-kriging equations with a single reliability term
#   $m^{*}/n(s_i)$ and one Lagrange multiplier (`poisson_krige`) — the baseline.
# * **Smoothing (noise filtering):** taking the prediction locations to be the
#   observed units and removing each in turn (leave-one-out) yields denoised risk
#   estimates instead of interpolated ones.
# 
# ### 1.8 Equations → code
# 
# | Equation | Quantity | Module implementation |
# |----------|----------|-----------------------|
# | §1.2 (12) | direct Poisson semivariogram $\hat\gamma_{\ell\ell}$ | `pck_variogram` |
# | §1.2 (13) | cross Poisson semivariogram $\hat\gamma_{\ell\kappa}$ | `pck_crossvariogram` |
# | §1.4 | shared risk $R_{\ell\kappa}=\rho_{\ell\kappa}\sqrt{R_\ell R_\kappa}$ | `shared_risk` |
# | §1.3 | LMC $C_{\ell\kappa}(h)=\sum_u b^u_{\ell\kappa}g_u(h)$, $B_u\succeq0$ | `fit_lmc`, `LMC`, `c_exp`, `c_sph` |
# | §1.4–1.6 (7,9,10) | cokriging predictor, system, variance | `poisson_cokrige` |
# | §1.7 | univariate Poisson kriging | `poisson_krige` |
# 
# The remainder of the notebook exercises each of these: a simulation study
# (Sections 2–6) and a real-data application to diabetes (Section 7).

# ## 2. Simulate a multivariate dataset
# 
# Mirroring the paper's simulation: draw $P=4$ co-regionalised latent Gaussian
# risk fields on an irregular set of locations (via $\mathrm{Cov}=B\otimes\rho$),
# convert to per-100k risks, and draw Poisson counts on a shared population. The
# target is an **HIV-incidence-like** disease with very low counts (deliberately
# noisy — the case cokriging is designed to help).

# In[2]:


N = 217
cent = RNG.uniform([0, 0], [600, 400], size=(N, 2))          # locations (km)
D = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))
P = 4
names = ["HIV Inc.", "HIV Prev.", "Chlamydia", "Gonorrhea"]
target = 0

range_true = 180.0
Rho = np.exp(-D / range_true)
# PSD coregionalization matrix (strong positive cross-dependence)
Lb = np.array([[1.0,0,0,0],[0.8,0.6,0,0],[0.7,0.4,0.5,0],[0.6,0.3,0.4,0.5]])
B_true = Lb @ Lb.T

big = np.kron(B_true, Rho)
Z = (np.linalg.cholesky(big + 1e-8*np.eye(P*N)) @ RNG.standard_normal(P*N)).reshape(P, N)

pop = np.clip(np.round(np.exp(RNG.normal(10.5, 0.9, N))), 500, None)   # shared denom
baseline = np.array([15.0, 260.0, 400.0, 150.0])                       # per-100k
rates_true = baseline[:, None] * np.exp(Z - 0.5*np.diag(B_true)[:, None])
counts = RNG.poisson(pop[None, :] * rates_true / POP_RATE).astype(float)
rates_obs = counts / pop[None, :] * POP_RATE

print("mean counts / disease:", np.round(counts.mean(1), 1))
print("observed-rate corr with target:",
      np.round([np.corrcoef(rates_obs[0], rates_obs[k])[0,1] for k in range(P)], 2))

ok_sim = (N == 217 and P == 4
          and np.all(np.isfinite(counts)) and np.all(counts >= 0)
          and np.all(np.linalg.eigvalsh(B_true) >= -1e-8))
record("Simulation: 4 co-regionalised Poisson count fields",
       "PASS" if ok_sim else "FAIL",
       f"N={N} | mean counts={np.round(counts.mean(1), 1).tolist()} | "
       f"corr with target={np.round([np.corrcoef(rates_obs[0], rates_obs[k])[0,1] for k in range(P)], 2).tolist()}")


# ### 2.1 Simulated risk surfaces and their correlation

# In[3]:


fig = plt.figure(figsize=(14, 4.2))
gs = GridSpec(1, 5, width_ratios=[1,1,1,1,0.9], wspace=0.25)
for l in range(P):
    ax = fig.add_subplot(gs[0, l])
    s = ax.scatter(cent[:,0], cent[:,1], c=rates_obs[l], s=28, cmap="RdBu_r",
                   vmin=np.percentile(rates_obs[l],5), vmax=np.percentile(rates_obs[l],95))
    ax.set_title(names[l], fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(s, ax=ax, shrink=.75)
axc = fig.add_subplot(gs[0, 4])
C = np.corrcoef(rates_obs)
im = axc.imshow(C, cmap="viridis", vmin=0, vmax=1)
axc.set_xticks(range(P)); axc.set_yticks(range(P))
axc.set_xticklabels([n.split()[0] for n in names], rotation=45, ha="right", fontsize=7)
axc.set_yticklabels([n.split()[0] for n in names], fontsize=7)
for i in range(P):
    for k in range(P):
        axc.text(k, i, f"{C[i,k]:.2f}", ha="center", va="center",
                 color="w" if C[i,k] < 0.6 else "k", fontsize=7)
axc.set_title("observed-rate corr", fontsize=9)
plt.tight_layout(); plt.show()


# ## 3. Empirical direct and cross Poisson semivariograms
# 
# `pck_variogram` (direct) and `pck_crossvariogram` (cross) implement the
# population-weighted, Poisson-bias-corrected estimators. Shared risks
# $R_{lk}=\rho_{lk}\sqrt{R_lR_k}$ feed the cross-variogram bias term.

# In[4]:


maxd = D.max() * 0.5
direct_vars = [mpk.pck_variogram(cent, counts[l], pop, POP_RATE, maxd, 12)
               for l in range(P)]
cross_pairs = [(l, k) for l in range(P) for k in range(l+1, P)]
rho_lk, shared_mean, cross_vars = {}, {}, []
for (l, k) in cross_pairs:
    rho = max(np.corrcoef(rates_obs[l], rates_obs[k])[0,1], 0.01)
    r = mpk.shared_risk(rates_obs[l], rates_obs[k], rho)
    rho_lk[(l,k)] = rho; shared_mean[(l,k)] = float(np.mean(r))
    cross_vars.append(mpk.pck_crossvariogram(cent, counts[l], counts[k], pop, r,
                                             POP_RATE, maxd, 12))

def fit_exp(lags, gam, npv):
    ok = np.isfinite(gam) & (gam >= 0); lags, gam, npv = lags[ok], gam[ok], npv[ok]
    s0 = gam.max()
    sol = least_squares(lambda p: (p[0]*(1-np.exp(-lags/p[1]))-gam)*np.sqrt(npv),
                        [s0, lags[len(lags)//2]], bounds=([0,lags.min()],[3*s0,lags.max()]))
    return float(sol.x[0]), float(sol.x[1])
range_common = fit_exp(*direct_vars[target])[1]
print(f"common range fit to target direct variogram: {range_common:.0f} km")

ok_dv = all(np.any(np.isfinite(dv[1])) and (dv[1][np.isfinite(dv[1])] >= 0).any()
            for dv in direct_vars)
ok_cv = all(np.any(np.isfinite(cv[1])) for cv in cross_vars)
ok_rng = np.isfinite(range_common) and range_common > 0
record("pck_variogram: 4 finite population-weighted direct semivariograms",
       "PASS" if ok_dv else "FAIL",
       f"{P} direct | n_lags={[len(dv[0]) for dv in direct_vars]}")
record("pck_crossvariogram: 6 finite Poisson-bias-corrected cross semivariograms",
       "PASS" if ok_cv else "FAIL",
       f"{len(cross_pairs)} pairs")
record("Exponential range fitted to target direct variogram",
       "PASS" if ok_rng else "FAIL",
       f"range={range_common:.0f} km")


# ## 4. Fit the Linear Model of Coregionalization
# 
# `fit_lmc` estimates each $B_{lk}$ by weighted least squares at the common range,
# then projects $B$ onto the nearest positive-semidefinite matrix (a valid LMC).
# The classic coregionalization display shows direct semivariograms on the
# diagonal and cross semivariograms off it, with the fitted LMC overlaid.

# In[5]:


lmc = mpk.fit_lmc(direct_vars, cross_vars, cross_pairs, P,
                  rng=range_common, model="Exp", fit_range=False)
print("fitted", lmc, "\ncoregionalization matrix B:\n", np.round(lmc.B, 1))

fig, ax = plt.subplots(P, P, figsize=(12, 10), sharex=True)
hh = np.linspace(0, maxd, 200)
cross_lookup = {pair: cv for pair, cv in zip(cross_pairs, cross_vars)}
for l in range(P):
    for k in range(P):
        a = ax[l, k]
        if l == k:
            lags, gam, _ = direct_vars[l]
            a.plot(lags, gam, "o", ms=4, color="#c0392b")
            a.plot(hh, lmc.B[l,l]*(1-np.exp(-hh/lmc.range)), "b-", lw=1.8)
            a.set_title(f"$\\gamma_{{{l+1}{l+1}}}$  {names[l]}", fontsize=8)
        elif k > l:
            lags, gam, _ = cross_lookup[(l, k)]
            a.plot(lags, gam, "o", ms=4, color="#8e44ad")
            a.plot(hh, lmc.B[l,k]*(1-np.exp(-hh/lmc.range)), "g-", lw=1.8)
            a.set_title(f"$\\gamma_{{{l+1}{k+1}}}$", fontsize=8)
        else:
            a.axis("off")
        if l == P-1: a.set_xlabel("dist (km)", fontsize=7)
plt.tight_layout(); plt.show()

evals = np.linalg.eigvalsh(lmc.B)
ok_lmc = (np.all(np.isfinite(lmc.B)) and np.all(evals >= -1e-8)
          and np.isfinite(lmc.range) and lmc.range > 0)
record("fit_lmc: PSD coregionalization matrix B at common range",
       "PASS" if ok_lmc else "FAIL",
       f"B min eig={evals.min():.2f} | range={lmc.range:.0f} km | model={lmc.model}")


# ## 5. Prediction: cokriging vs univariate kriging
# 
# Hold out 30% of the target's locations and predict them with univariate Poisson
# kriging (PK) and bi/tri/tetra-variate cokriging (PCK). Following the paper's
# **simulation** protocol, predictions are scored against the **known latent
# risk** (which filters out the irreducible Poisson noise that dominates a rare
# outcome like HIV incidence).

# In[6]:


n_hold = int(0.30 * N)
hold = np.sort(RNG.choice(N, n_hold, replace=False))
mask = np.ones(N, bool); mask[hold] = False
coords_hold = cent[hold]
truth = rates_true[target][hold]           # known latent risk (simulation)

# univariate PK baseline, fit to the target's own direct variogram
sill_pk, range_pk = fit_exp(*mpk.pck_variogram(cent[mask], counts[target][mask],
                                               pop[mask], POP_RATE, maxd, 12))
pk = mpk.poisson_krige(dict(x=cent[mask,0], y=cent[mask,1],
                            cases=counts[target][mask], pop=pop[mask]),
                       sill=sill_pk, rng=range_pk, model="Exp",
                       pop_rate=POP_RATE, coords_pred=coords_hold)

def train_sets(variables):
    ds = []
    for l in variables:
        if l == target:
            ds.append(dict(x=cent[mask,0], y=cent[mask,1],
                           cases=counts[l][mask], pop=pop[mask]))
        else:
            ds.append(dict(x=cent[:,0], y=cent[:,1], cases=counts[l], pop=pop))
    return ds

def metrics(pred, label):
    e = pred - truth
    return dict(method=label, MSPE=np.mean(e**2), MAE=np.mean(np.abs(e)),
                cor=np.corrcoef(pred, truth)[0,1])

rows = [dict(**metrics(pk["pred"], "PK (univariate)"), mean_var=pk["var"].mean())]
preds = {"PK (univariate)": pk["pred"]}
for label, variables in [("PCK bivariate", [0,1]),
                         ("PCK trivariate", [0,1,2]),
                         ("PCK tetravariate", [0,1,2,3])]:
    sm = {}
    for a in range(len(variables)):
        for b in range(a+1, len(variables)):
            la, lb = variables[a], variables[b]
            sm[(a,b)] = shared_mean[(la,lb) if (la,lb) in shared_mean else (lb,la)]
    sub = mpk.LMC(lmc.B[np.ix_(variables, variables)], lmc.range, lmc.model)
    res = mpk.poisson_cokrige(train_sets(variables), 0, sub, POP_RATE, sm,
                              coords_pred=coords_hold)
    rows.append(dict(**metrics(res["pred"], label), mean_var=res["var"].mean()))
    preds[label] = res["pred"]

import pandas as pd
base = rows[0]["MSPE"]
tbl = pd.DataFrame(rows).set_index("method")
tbl["MSPE drop %"] = (100*(base - tbl["MSPE"])/base).round(0)
best_sim = min(rows[1:], key=lambda r: r["MSPE"])
ok_pk = np.all(np.isfinite(pk["pred"])) and np.all(np.isfinite(pk["var"]))
ok_pck = all(np.all(np.isfinite(preds[lab])) for lab in preds)
ok_gain = best_sim["MSPE"] < base
record("poisson_krige: held-out univariate predictions finite",
       "PASS" if ok_pk else "FAIL",
       f"MSPE={rows[0]['MSPE']:.3f}  cor={rows[0]['cor']:.3f}  mean var={rows[0]['mean_var']:.3f}")
record("poisson_cokrige: bi/tri/tetra-variate held-out predictions finite",
       "PASS" if ok_pck else "FAIL",
       f"methods={list(preds.keys())}")
record("Simulation: cokriging MSPE lower than univariate PK",
       "PASS" if ok_gain else "FAIL",
       f"PK MSPE={base:.3f}  best PCK={best_sim['method']} MSPE={best_sim['MSPE']:.3f}  "
       f"drop={100*(base-best_sim['MSPE'])/base:.0f}%")

tbl.round(3)


# In[7]:


fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
labels = list(preds)
colors = ["#7f8c8d", "#5dade2", "#2e86c1", "#1b4f72"]
ax[0].bar(range(len(rows)), [r["MSPE"] for r in rows], color=colors)
ax[0].axhline(base, ls="--", c="k", lw=1)
ax[0].set_xticks(range(len(rows))); ax[0].set_xticklabels(
    [l.replace(" ", "\n") for l in labels], fontsize=8)
ax[0].set_ylabel("MSPE (vs true risk)"); ax[0].set_title("Prediction error")
for i, r in enumerate(rows):
    ax[0].text(i, r["MSPE"], f'-{100*(base-r["MSPE"])/base:.0f}%' if i else "base",
               ha="center", va="bottom", fontsize=8)

ax[1].bar(range(len(rows)), [r["mean_var"] for r in rows], color=colors)
ax[1].set_xticks(range(len(rows))); ax[1].set_xticklabels(
    [l.replace(" ", "\n") for l in labels], fontsize=8)
ax[1].set_ylabel("mean kriging variance"); ax[1].set_title("Prediction uncertainty")

lim = [truth.min()-2, truth.max()+2]
ax[2].plot(lim, lim, "k--", lw=1)
ax[2].scatter(truth, preds["PK (univariate)"], s=18, alpha=.6, label="PK", color="#7f8c8d")
ax[2].scatter(truth, preds["PCK tetravariate"], s=18, alpha=.7, label="PCK 4-var", color="#1b4f72")
ax[2].set_xlabel("true risk"); ax[2].set_ylabel("predicted"); ax[2].set_aspect("equal")
ax[2].set_title("Held-out predictions"); ax[2].legend(); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.show()

best = min(rows[1:], key=lambda r: r["MSPE"])
print(f"Best: {best['method']} -> MSPE reduced {100*(base-best['MSPE'])/base:.0f}% "
      f"vs univariate PK, correlation {rows[0]['cor']:.2f} -> {best['cor']:.2f} "
      f"(paper: up to ~50% MSPE reduction in simulation).")


# ### 5.1 Spatial view: reconstructed target risk
# 
# Training locations show the observed (noisy) rate; held-out locations show the
# tetravariate cokriging prediction. The cokriged field recovers the smooth latent
# structure the single-disease rates obscure.

# In[8]:


fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
vmin, vmax = np.percentile(rates_true[target], 5), np.percentile(rates_true[target], 95)
ax[0].scatter(cent[:,0], cent[:,1], c=rates_true[target], s=30, cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax[0].set_title("True latent risk (HIV Inc.)")
ax[1].scatter(cent[:,0], cent[:,1], c=rates_obs[target], s=30, cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax[1].set_title("Observed noisy rate")
recon = rates_obs[target].copy(); recon[hold] = preds["PCK tetravariate"]
ax[2].scatter(cent[mask,0], cent[mask,1], c=rates_obs[target][mask], s=30, cmap="RdBu_r", vmin=vmin, vmax=vmax)
sc = ax[2].scatter(cent[hold,0], cent[hold,1], c=preds["PCK tetravariate"], s=60,
                   cmap="RdBu_r", vmin=vmin, vmax=vmax, edgecolor="k", linewidth=.8)
ax[2].set_title("PCK 4-var (○ = predicted hold-out)")
for a in ax: a.set_xticks([]); a.set_yticks([])
plt.colorbar(sc, ax=ax, shrink=.7, label="risk / 100k")
plt.show()


# ## 6. Leave-one-out smoothing
# 
# When the prediction locations equal the data locations, `poisson_cokrige`
# switches to leave-one-out **smoothing** (filtering the Poisson noise at observed
# units). Near-zero mean residual confirms unbiasedness.

# In[9]:


sm4 = {(a,b): shared_mean[(a,b)] for (a,b) in cross_pairs}
datasets = [dict(x=cent[:,0], y=cent[:,1], cases=counts[l], pop=pop) for l in range(P)]
smooth = mpk.poisson_cokrige(datasets, target, lmc, POP_RATE, sm4, smooth=True)

print(f"mean residual (bias)     = {np.nanmean(smooth['residual']):+.3f}")
print(f"MSPE vs observed         = {np.nanmean(smooth['residual']**2):.2f}")
print(f"MSPE vs TRUE latent risk = {np.nanmean((smooth['rate_pred']-rates_true[target])**2):.2f}"
      f"  (raw obs vs truth: {np.nanmean((rates_obs[target]-rates_true[target])**2):.2f})")
print(f"cor(smoothed, true risk) = {np.corrcoef(smooth['rate_pred'], rates_true[target])[0,1]:.3f}"
      f"  (raw obs: {np.corrcoef(rates_obs[target], rates_true[target])[0,1]:.3f})")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
vmin, vmax = np.percentile(rates_true[target], 5), np.percentile(rates_true[target], 95)
ax[0].scatter(cent[:,0], cent[:,1], c=rates_obs[target], s=30, cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax[0].set_title("Observed rate")
ax[1].scatter(cent[:,0], cent[:,1], c=smooth["rate_pred"], s=30, cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax[1].set_title("Cokriging-smoothed risk")
sv = ax[2].scatter(cent[:,0], cent[:,1], c=smooth["rate_var"], s=30, cmap="magma")
ax[2].set_title("Smoothing variance"); plt.colorbar(sv, ax=ax[2], shrink=.75)
for a in ax: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()

mspe_sm = np.nanmean((smooth["rate_pred"] - rates_true[target]) ** 2)
mspe_raw = np.nanmean((rates_obs[target] - rates_true[target]) ** 2)
bias = float(np.nanmean(smooth["residual"]))
ok_sm_finite = np.all(np.isfinite(smooth["rate_pred"]))
record("poisson_cokrige(smooth=True): leave-one-out smoothing finite",
       "PASS" if ok_sm_finite else "FAIL",
       f"bias={bias:+.3f}  MSPE vs true={mspe_sm:.2f} (raw {mspe_raw:.2f})")
record("Smoothing MSPE vs true latent risk beats raw observed rates",
       "PASS" if mspe_sm < mspe_raw else "FAIL",
       f"smoothed={mspe_sm:.2f}  raw={mspe_raw:.2f}")


# ## 7. Application: diabetes risk in the US Northeast
# 
# We now apply the method to **real county data** — diagnosed diabetes across the
# nine-state US Census **Northeast region** (New England + NJ, NY, PA; 217
# counties). Diabetes is the Poisson target ($Y=$ `Diabetes_count`, $n=$
# `POP_Total`); the two strongest correlated risk factors, **physical inactivity**
# and **obesity**, are the auxiliaries. Because they are reported as prevalence
# percentages over the *same* county population, we turn them into pseudo-counts
# $Y_l = \mathrm{round}(\text{pct}_l/100 \times n)$, so all three variables are
# co-located Poisson counts sharing the denominator $n$ — exactly what the
# estimator needs.
# 
# County boundaries and attributes are loaded from `data/diabetes_northeast.shp`
# (NAD 1983 Contiguous USA Albers / EPSG:5070; sidecars `.shx`, `.dbf`, `.prj`,
# `.cpg`) joined to `data/diabetes_northeast.csv`. Every map in this section is a
# **polygon choropleth** of the 217 counties.
# 
# Unlike the simulation, there is no known latent risk here, so — as in the
# paper's real-data analysis — hold-out predictions are scored against the
# **observed** diabetes rate.

# In[10]:


# County polygons (NAD 1983 Contiguous USA Albers / EPSG:5070) + full attribute table.
# Shapefile sidecars: .shx, .dbf, .prj, .cpg. DBF truncates names, so join the CSV.
ne_poly = gpd.read_file("../data/diabetes_northeast.shp")
dia = pd.read_csv("../data/diabetes_northeast.csv")
ne_poly["FIPS"] = ne_poly["FIPS"].astype(int)
dia["FIPS"] = dia["FIPS"].astype(int)

ne = ne_poly[["FIPS", "geometry"]].merge(dia, on="FIPS", how="inner")
ne_gdf = ne

print(f"ne_gdf: {len(ne)} polygons from ../data/diabetes_northeast.shp | "
      f"{ne.geom_type.value_counts().to_dict()}")
print(f"CRS: {ne.crs}")

coordsD = np.column_stack([ne.X.values, ne.Y.values]) / 1000.0     # km
popD = ne.POP_Total.values.astype(float)
dia_names = ["Diabetes", "Physical Inactivity", "Obesity"]
yD = [ne.Diabetes_count.values.astype(float),
      np.round(ne.Physical_Inactivity.values / 100 * popD),
      np.round(ne.Obesity.values / 100 * popD)]
PD, targetD = 3, 0
ratesD = [yD[l] / popD * POP_RATE for l in range(PD)]
DD = np.sqrt(((coordsD[:, None, :] - coordsD[None, :, :]) ** 2).sum(-1))

print(f"{len(ne)} Northeast counties | diabetes rate/100k: "
      f"{ratesD[0].min():.0f}–{ratesD[0].max():.0f}")
print("rate correlation, diabetes vs [phys.inact., obesity]:",
      [round(np.corrcoef(ratesD[0], ratesD[k])[0, 1], 2) for k in (1, 2)])

def mapper(values, title, ax, cmap="RdBu_r", vmin=None, vmax=None, label=""):
    """Choropleth of county polygons (never a point scatter)."""
    gg = ne_gdf.merge(pd.DataFrame({"FIPS": ne.FIPS.values, "_v": values}), on="FIPS")
    gg.plot("_v", cmap=cmap, ax=ax, vmin=vmin, vmax=vmax, edgecolor="white",
            linewidth=.2, legend=True, legend_kwds={"shrink":.7, "label":label})
    ax.set_title(title); ax.set_axis_off()

ok_ne = (len(ne) == 217 and np.all(np.isfinite(popD)) and np.all(popD > 0)
         and np.all(np.isfinite(ratesD[0])))
record("Northeast diabetes shapefile: 217 counties with finite rates",
       "PASS" if ok_ne else "FAIL",
       f"{len(ne)} polygons | CRS={ne.crs} | diabetes/100k {ratesD[0].min():.0f}–{ratesD[0].max():.0f} | "
       f"corr vs phys.inact./obesity="
       f"{[round(np.corrcoef(ratesD[0], ratesD[k])[0, 1], 2) for k in (1, 2)]}")

ne


# ### 7.1 Diabetes and its risk factors across the Northeast
# 
# The spatial patterns are strikingly concordant — Appalachian Pennsylvania and
# inland/northern counties carry higher diabetes, physical inactivity, and obesity
# together — which is exactly the cross-correlation cokriging exploits.

# In[11]:


fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for idx, (vals, title, cmap, label) in enumerate([
    (ratesD[0]/1000, "Diabetes (per 1,000)", "RdBu_r", "per 1,000"),
    (ne.Physical_Inactivity.values, "Physical inactivity (%)", "RdBu_r", "%"),
    (ne.Obesity.values, "Obesity (%)", "RdBu_r", "%")
]):
    mapper(vals, title, ax[idx], cmap, label=label)
plt.tight_layout()
plt.show()


# ### 7.2 Direct and cross Poisson semivariograms + LMC

# In[12]:


maxdD = DD.max() * 0.5
dvD = [mpk.pck_variogram(coordsD, yD[l], popD, POP_RATE, maxdD, 12) for l in range(PD)]
pairsD = [(l, k) for l in range(PD) for k in range(l+1, PD)]
rhoD, sharedD, cvD = {}, {}, []
for (l, k) in pairsD:
    r = max(np.corrcoef(ratesD[l], ratesD[k])[0, 1], 0.01)
    rr = mpk.shared_risk(ratesD[l], ratesD[k], r)
    rhoD[(l,k)] = r; sharedD[(l,k)] = float(np.mean(rr))
    cvD.append(mpk.pck_crossvariogram(coordsD, yD[l], yD[k], popD, rr, POP_RATE, maxdD, 12))

rngD = fit_exp(*dvD[targetD])[1]                       # range from target direct variogram
lmcD = mpk.fit_lmc(dvD, cvD, pairsD, PD, rng=rngD, model="Exp", fit_range=False)
print(f"fitted {lmcD} | common range = {lmcD.range:.0f} km")

fig, ax = plt.subplots(PD, PD, figsize=(11, 8.5), sharex=True)
hhD = np.linspace(0, maxdD, 200)
clook = {p: c for p, c in zip(pairsD, cvD)}
for l in range(PD):
    for k in range(PD):
        a = ax[l, k]
        if l == k:
            lags, gam, _ = dvD[l]
            a.plot(lags, gam, "o", ms=4, color="#c0392b")
            a.plot(hhD, lmcD.B[l,l]*(1-np.exp(-hhD/lmcD.range)), "b-", lw=1.8)
            a.set_title(f"$\\gamma_{{{l+1}{l+1}}}$  {dia_names[l]}", fontsize=8)
        elif k > l:
            lags, gam, _ = clook[(l, k)]
            a.plot(lags, gam, "o", ms=4, color="#8e44ad")
            a.plot(hhD, lmcD.B[l,k]*(1-np.exp(-hhD/lmcD.range)), "g-", lw=1.8)
            a.set_title(f"$\\gamma_{{{l+1}{k+1}}}$", fontsize=8)
        else:
            a.axis("off")
        if l == PD-1: a.set_xlabel("dist (km)", fontsize=7)
plt.tight_layout(); plt.show()

evalsD = np.linalg.eigvalsh(lmcD.B)
ok_lmcD = np.all(np.isfinite(lmcD.B)) and np.all(evalsD >= -1e-8) and lmcD.range > 0
record("Application: LMC fitted to diabetes + inactivity + obesity",
       "PASS" if ok_lmcD else "FAIL",
       f"range={lmcD.range:.0f} km | B min eig={evalsD.min():.2f} | model={lmcD.model}")


# ### 7.3 Predicting held-out diabetes rates: kriging vs cokriging
# 
# 30% of counties are held out and predicted from the rest with univariate Poisson
# kriging and with bi/tri-variate cokriging that adds the risk factors.

# In[13]:


nhD = int(0.30 * len(ne))
holdD = np.sort(RNG.choice(len(ne), nhD, replace=False))
maskD = np.ones(len(ne), bool); maskD[holdD] = False
truthD = ratesD[0][holdD]                              # observed held-out rate

spD, rpD = fit_exp(*mpk.pck_variogram(coordsD[maskD], yD[0][maskD], popD[maskD],
                                      POP_RATE, maxdD, 12))
pkD = mpk.poisson_krige(dict(x=coordsD[maskD,0], y=coordsD[maskD,1],
                             cases=yD[0][maskD], pop=popD[maskD]),
                        sill=spD, rng=rpD, model="Exp", pop_rate=POP_RATE,
                        coords_pred=coordsD[holdD])

def setsD(vs):
    ds = []
    for l in vs:
        if l == 0: ds.append(dict(x=coordsD[maskD,0], y=coordsD[maskD,1],
                                  cases=yD[l][maskD], pop=popD[maskD]))
        else:      ds.append(dict(x=coordsD[:,0], y=coordsD[:,1], cases=yD[l], pop=popD))
    return ds

def metD(p, lab):
    e = p - truthD
    return dict(method=lab, MSPE=np.mean(e**2), MAE=np.mean(np.abs(e)),
                cor=np.corrcoef(p, truthD)[0,1])

rowsD = [dict(**metD(pkD["pred"], "PK (univariate)"), mean_var=pkD["var"].mean())]
predsD = {"PK (univariate)": pkD["pred"]}
for lab, vs in [("PCK + Phys.Inact.", [0,1]), ("PCK + Phys.Inact. + Obesity", [0,1,2])]:
    sm = {}
    for a in range(len(vs)):
        for b in range(a+1, len(vs)):
            la, lb = vs[a], vs[b]
            sm[(a,b)] = sharedD[(la,lb) if (la,lb) in sharedD else (lb,la)]
    sub = mpk.LMC(lmcD.B[np.ix_(vs, vs)], lmcD.range, lmcD.model)
    res = mpk.poisson_cokrige(setsD(vs), 0, sub, POP_RATE, sm, coords_pred=coordsD[holdD])
    rowsD.append(dict(**metD(res["pred"], lab), mean_var=res["var"].mean()))
    predsD[lab] = res["pred"]

baseD = rowsD[0]["MSPE"]
tblD = pd.DataFrame(rowsD).set_index("method")
tblD["MSPE drop %"] = (100*(baseD - tblD["MSPE"])/baseD).round(0)
bestD_row = min(rowsD[1:], key=lambda r: r["MSPE"])
ok_pkD = np.all(np.isfinite(pkD["pred"]))
ok_pckD = all(np.all(np.isfinite(predsD[lab])) for lab in predsD)
ok_gainD = bestD_row["MSPE"] < baseD
record("Application: univariate PK held-out predictions finite",
       "PASS" if ok_pkD else "FAIL",
       f"n_hold={nhD}  MSPE={rowsD[0]['MSPE']:.2f}  cor={rowsD[0]['cor']:.3f}")
record("Application: PCK with inactivity/obesity held-out predictions finite",
       "PASS" if ok_pckD else "FAIL",
       f"methods={list(predsD.keys())}")
record("Application: cokriging MSPE lower than univariate PK",
       "PASS" if ok_gainD else "FAIL",
       f"PK MSPE={baseD:.2f}  best={bestD_row['method']} MSPE={bestD_row['MSPE']:.2f}  "
       f"drop={100*(baseD-bestD_row['MSPE'])/baseD:.0f}%")

tblD.round(2)


# In[14]:


fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
labsD = list(predsD); colsD = ["#7f8c8d", "#2e86c1", "#1b4f72"]
ax[0].bar(range(len(rowsD)), [r["MSPE"] for r in rowsD], color=colsD)
ax[0].set_xticks(range(len(rowsD)))
ax[0].set_xticklabels([l.replace(" + ", "\n+") for l in labsD], fontsize=8)
ax[0].set_ylabel("MSPE (vs observed rate)"); ax[0].set_title("Held-out prediction error")
for i, r in enumerate(rowsD):
    ax[0].text(i, r["MSPE"], "base" if i == 0 else f'-{100*(baseD-r["MSPE"])/baseD:.0f}%',
               ha="center", va="bottom", fontsize=8)
lim = [truthD.min()*0.95, truthD.max()*1.05]
ax[1].plot(lim, lim, "k--", lw=1)
ax[1].scatter(truthD, predsD["PK (univariate)"], s=18, alpha=.6, color="#7f8c8d", label="PK")
ax[1].scatter(truthD, predsD[labsD[-1]], s=18, alpha=.75, color="#1b4f72", label="PCK (3-var)")
ax[1].set_xlabel("observed diabetes rate /100k"); ax[1].set_ylabel("predicted")
ax[1].set_aspect("equal"); ax[1].set_title("Held-out counties"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()

bestD = min(rowsD[1:], key=lambda r: r["MSPE"])
print(f"Best: {bestD['method']} -> MSPE reduced {100*(baseD-bestD['MSPE'])/baseD:.0f}% "
      f"vs univariate PK; correlation {rowsD[0]['cor']:.2f} -> {bestD['cor']:.2f}.")


# ### 7.4 Smoothed diabetes risk map
# 
# Leave-one-out cokriging smoothing filters the county-level Poisson noise using
# both risk factors, giving a cleaner risk surface and its uncertainty.

# In[15]:


smD = {(l, k): sharedD[(l, k)] for (l, k) in pairsD}
dsetsD = [dict(x=coordsD[:,0], y=coordsD[:,1], cases=yD[l], pop=popD) for l in range(PD)]
smoothD = mpk.poisson_cokrige(dsetsD, targetD, lmcD, POP_RATE, smD, smooth=True)
print(f"mean residual (bias) = {np.nanmean(smoothD['residual']):+.1f} per 100k "
      f"| smoothing MSPE = {np.nanmean(smoothD['residual']**2):.0f} "
      f"| cor(smoothed, observed) = "
      f"{np.corrcoef(smoothD['rate_pred'], smoothD['observed'])[0,1]:.3f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
vlo, vhi = np.percentile(ratesD[0]/1000, 3), np.percentile(ratesD[0]/1000, 97)
mapper(ratesD[0]/1000, "Observed diabetes", ax[0], "RdBu_r", vlo, vhi, "per 1,000")
mapper(smoothD["rate_pred"]/1000, "Cokriging-smoothed", ax[1], "RdBu_r", vlo, vhi, "per 1,000")
mapper(np.sqrt(smoothD["rate_var"])/1000, "Smoothing std. error", ax[2], "magma", label="per 1,000")
plt.tight_layout(); plt.show()

ok_smD = (np.all(np.isfinite(smoothD["rate_pred"]))
          and np.all(np.isfinite(smoothD["rate_var"])))
biasD = float(np.nanmean(smoothD["residual"]))
corD = float(np.corrcoef(smoothD["rate_pred"], smoothD["observed"])[0, 1])
record("Application: leave-one-out cokriging smoothing finite",
       "PASS" if ok_smD else "FAIL",
       f"bias={biasD:+.1f}/100k  MSPE={np.nanmean(smoothD['residual']**2):.0f}  cor={corD:.3f}")


# In[16]:


# Polygon choropleths of observed vs cokriging-smoothed diabetes rates
gg = ne_gdf.merge(
    pd.DataFrame({
        "FIPS": ne.FIPS.values,
        "obs": ratesD[0] / 1000,
        "cokriged": smoothD["rate_pred"] / 1000,
    }),
    on="FIPS",
)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
vlo, vhi = np.percentile(gg["obs"].dropna(), 3), np.percentile(gg["obs"].dropna(), 97)
norm = Normalize(vmin=vlo, vmax=vhi)
cmap = "RdBu_r"
smap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

gg.plot("obs", ax=axes[0], cmap=cmap, norm=norm, edgecolor="k", linewidth=0.15)
axes[0].set_title("Observed Diabetes Rate (per 1,000)")
axes[0].axis("off")
fig.colorbar(smap, ax=axes[0], orientation="vertical", fraction=0.034, pad=0.01)

gg.plot("cokriged", ax=axes[1], cmap=cmap, norm=norm, edgecolor="k", linewidth=0.15)
axes[1].set_title("Cokriging-Smoothed Diabetes Rate (per 1,000)")
axes[1].axis("off")
fig.colorbar(smap, ax=axes[1], orientation="vertical", fraction=0.034, pad=0.01)
plt.tight_layout()
plt.show()


# ## 8. Summary
# 
# | Estimator | Module call | Uses auxiliaries | Prediction variance |
# |-----------|-------------|:---:|---|
# | Univariate PK | `poisson_krige` | no | baseline |
# | Poisson cokriging | `poisson_cokrige` | yes | markedly smaller |
# 
# **Findings**
# 
# - *Simulation* (Sections 2–6): adding correlated auxiliary diseases **cuts the
#   MSPE** of the target-risk prediction and raises the correlation with the true
#   risk; the prediction variance drops once auxiliaries enter; gains taper as
#   weakly-correlated variables are added.
# - *Real data — Northeast diabetes* (Section 7): using physical inactivity and
#   obesity as auxiliaries **substantially reduces the held-out prediction MSPE**
#   versus univariate Poisson kriging and sharply improves agreement with observed
#   rates — the same benefit the paper reports (up to ~50% in simulation, 74% on
#   the Pennsylvania STD data).
# - Leave-one-out **smoothing** filters county-level Poisson noise with near-zero
#   bias, yielding a cleaner diabetes risk surface and its standard error.
# 
# All kriging is performed by `multi_poisson_kriging.py`; the notebook supplies the
# simulation, the real data, and the plots. Natural extensions: add more risk
# factors (SVI, food environment), a second LMC structure (exponential + spherical,
# as in the paper), and prediction onto a fine grid for isopleth mapping.

# ## 9. Test summary
# 

# In[17]:


summary_df = pd.DataFrame(RESULTS)
def _style(row):
    color = {"PASS": "background-color: #d4edda",
             "FAIL": "background-color: #f8d7da",
             "BLOCKED": "background-color: #fff3cd"}.get(row["Status"], "")
    return [color] * len(row)
summary_df.style.apply(_style, axis=1)

