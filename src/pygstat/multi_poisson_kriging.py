"""
multi_poisson_kriging.py
========================

Multivariate Poisson (co)kriging for areal disease-count data.

Python conversion of the two R scripts
``Multivariate-Poisson-cokriging.R`` / ``-utils.R`` implementing the method of

    Payares-Garcia D., Osei F., Mateu J., Stein A. (2024).
    "Multivariate Poisson cokriging: a geostatistical model for health count
    data." Statistical Methods in Medical Research. doi:10.1177/09622802241268488

The model predicts/smooths a target disease *risk* from noisy areal counts by
borrowing strength across several spatially co-regionalised diseases. Each
count is treated as Poisson, ``Y_l(v_i) ~ Poisson(n_i R_l(v_i))``; the latent
risks ``R_l`` are jointly modelled with a Linear Model of Coregionalization
(LMC). This module converts, and generalises to an arbitrary number ``P`` of
variables, the R functions:

    pck.variogram          -> pck_variogram         (direct Poisson semivariogram)
    pck.crossvariogram     -> pck_crossvariogram    (cross Poisson semivariogram)
    rlk                    -> shared_risk           (Poisson shared/joint risk)
    eb.risk                -> eb_risk               (empirical-Bayes rates)
    Cexp/Csph              -> c_exp / c_sph          (covariance models)
    lmc.poisson.cokrige.*  -> fit_lmc               (LMC coregionalization fit)
    poisson.cokrige.{bi,three,four} + *.pred.*
                           -> poisson_cokrige       (N-variate co-kriging)
    poisson.krige.one + .pred.one
                           -> poisson_krige         (univariate Poisson kriging)

Both prediction (new locations) and leave-one-out smoothing (prediction
locations == data locations) are supported, matching the R behaviour.

Conventions (as in the R code)
------------------------------
* ``cov`` functions return the covariance C(h) directly (sill at h=0), NOT the
  semivariance. The LMC covariance for a pair (l,k) is ``B[l,k] * rho(h)``.
* A rate is ``pop_rate * cases / population`` (e.g. ``pop_rate = 100000``).
* Poisson reliability terms are added to the (co)kriging matrix diagonal:
  direct block ``m*_l * pop_rate / n_i`` with ``m*_l = sum(y_l)/sum(n_l)``;
  cross block at co-located points ``mean(R_lk) / n_i`` with ``R_lk`` the
  Poisson shared risk. This is the multivariate extension of the univariate
  Poisson-kriging error variance.

GPU (CuPy) support
------------------
:func:`poisson_cokrige` in **k-nearest-neighbor** mode
(``number_of_neighbors=k``) batches every local ``(P k + P) × (P k + P)``
system into one stacked ``xp.linalg.solve`` of shape
``(n_pred, P k + P, P k + P)`` — NumPy on CPU, CuPy on GPU. Pass
``use_gpu='auto'|True|False``. The default **global** path (every known
area in the system) and leave-one-out ``smooth=True`` keep the original
per-target CPU loop; block size grows with ``N × P`` and is not a
fixed-``k`` stack.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist, squareform

from .utils.backend import (
    as_gpu_array,
    get_array_module,
    resolve_cupy_use_gpu,
    to_numpy,
)

__all__ = [
    "c_exp", "c_sph", "covariance_model",
    "shared_risk", "eb_risk",
    "pck_variogram", "pck_crossvariogram",
    "LMC", "fit_lmc",
    "poisson_cokrige", "poisson_krige",
]

# ---------------------------------------------------------------------------
# Covariance models  (R: Cexp, Csph, covariance.model)
# ---------------------------------------------------------------------------
def c_exp(h, rng, sill):
    """Exponential covariance C(h) = sill * exp(-h / range)."""
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    return sill * xp.exp(-h / rng)

def c_sph(h, rng, sill):
    """Spherical covariance C(h) = sill*(1 - 1.5 h/a + 0.5 (h/a)^3), 0 beyond a."""
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    hr = xp.clip(h / rng, 0, 1)
    return xp.where(h <= rng, sill * (1 - 1.5 * hr + 0.5 * hr ** 3), 0.0)

def _unit_correlation(h, rng, model, xp):
    """Unit-sill spatial correlation ρ(h) on ``xp`` (NumPy or CuPy)."""
    if model == "Exp":
        return xp.exp(-h / rng)
    hr = xp.clip(h / rng, 0, 1)
    return xp.where(h <= rng, 1.0 - 1.5 * hr + 0.5 * hr ** 3, 0.0)

def covariance_model(name):
    """Map 'Exp'/'Sph' to the covariance function (R: covariance.model)."""
    return {"Exp": c_exp, "Sph": c_sph}[name]

# ---------------------------------------------------------------------------
# Shared (joint) risk and empirical-Bayes rates
# ---------------------------------------------------------------------------
def shared_risk(Rl, Rk, rho, log=False):
    """
    Poisson shared risk between two diseases (R: rlk).

    ``rho * sqrt(Rl * Rk)`` (or the log-scale variant). ``Rl``/``Rk`` are the
    (per-``pop_rate``) rates of the two diseases and ``rho`` their Poisson
    correlation. Used to build the cross-semivariogram bias term and the
    co-located cross reliability term.
    """
    Rl, Rk = np.asarray(Rl, float), np.asarray(Rk, float)
    if log:
        return np.log(rho * np.sqrt(np.exp(Rl) * np.exp(Rk)))
    return rho * np.sqrt(Rl * Rk)

def eb_risk(cases, pop, pop_rate=100000.0):
    """Empirical-Bayes standardized rate (R: eb.risk)."""
    r = np.asarray(cases, float)
    n = np.asarray(pop, float)
    x = r / n
    N = len(r)
    m = np.nansum(r) / np.nansum(n)
    s2 = np.nansum(n * (x - m) ** 2) / np.nansum(n)
    a = s2 - (m / (np.nansum(n) / N))
    if a < (m / (np.nansum(n) / N)):
        a = 0.0
    v = a + (m / n)
    return (x - m) / np.sqrt(v)

# ---------------------------------------------------------------------------
# Empirical population-weighted Poisson (cross-)semivariograms
# ---------------------------------------------------------------------------
def _bin_semivariogram(d, wdiff_minus_bias, w, maxd, nbins):
    """Shared binning: gamma(bin) = sum(w*diff - bias) / (2 sum(w))."""
    if maxd is None:
        maxd = d.max() / 2.0
    edges = np.linspace(0, maxd, nbins + 1)
    idx = np.digitize(d, edges) - 1
    lags, gam, npair = [], [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() < 1:
            continue
        sw = w[m].sum()
        if sw <= 0:
            continue
        lags.append(d[m].mean())
        gam.append(wdiff_minus_bias[m].sum() / (2.0 * sw))
        npair.append(int(m.sum()))
    return (np.array(lags), np.array(gam), np.array(npair, dtype=float))

def pck_variogram(coords, cases, pop, pop_rate=100000.0, maxd=None, nbins=15):
    """
    Population-weighted **direct** Poisson semivariogram (R: pck.variogram).

    For unique location pairs (i<j)::

        diff = (pop_rate*(y_i/n_i - y_j/n_j))^2
        w    = n_i n_j / (n_i + n_j)
        bias = pop_rate * sum(y)/sum(n)            # m* * pop_rate
        gamma(h) = [ sum_pairs(w*diff) - n_pairs*bias ] / [ 2 sum_pairs(w) ]

    Returns ``(lags, gamma, npairs)``; negative gammas are clipped to 0.
    """
    coords = np.asarray(coords, float)
    y = np.asarray(cases, float)
    n = np.asarray(pop, float)
    iu = np.triu_indices(len(coords), 1)
    d = squareform(pdist(coords))[iu]
    diff = (pop_rate * (y[iu[0]] / n[iu[0]] - y[iu[1]] / n[iu[1]])) ** 2
    w = (n[iu[0]] * n[iu[1]]) / (n[iu[0]] + n[iu[1]])
    bias = pop_rate * y.sum() / n.sum()
    lags, gam, npair = _bin_semivariogram(d, w * diff - bias, w, maxd, nbins)
    return lags, np.clip(gam, 0, None), npair

def pck_crossvariogram(coords, cases_l, cases_k, pop, r_lk,
                       pop_rate=100000.0, maxd=None, nbins=15):
    """
    Population-weighted **cross** Poisson semivariogram (R: pck.crossvariogram).

    ``r_lk`` is the Poisson shared-risk vector (see :func:`shared_risk`);
    ``bias = pop_rate * mean(r_lk)``. Cross gammas may be negative; returned as
    given (NaNs dropped).
    """
    coords = np.asarray(coords, float)
    yl = np.asarray(cases_l, float)
    yk = np.asarray(cases_k, float)
    n = np.asarray(pop, float)
    iu = np.triu_indices(len(coords), 1)
    d = squareform(pdist(coords))[iu]
    dl = pop_rate * (yl[iu[0]] / n[iu[0]] - yl[iu[1]] / n[iu[1]])
    dk = pop_rate * (yk[iu[0]] / n[iu[0]] - yk[iu[1]] / n[iu[1]])
    diff = dl * dk
    w = (n[iu[0]] * n[iu[1]]) / (n[iu[0]] + n[iu[1]])
    bias = pop_rate * np.nanmean(np.asarray(r_lk, float))
    lags, gam, npair = _bin_semivariogram(d, w * diff - bias, w, maxd, nbins)
    return lags, gam, npair

# ---------------------------------------------------------------------------
# Linear Model of Coregionalization
# ---------------------------------------------------------------------------
def _nearest_psd(B):
    """Project a symmetric matrix onto the nearest positive-semidefinite one."""
    B = 0.5 * (B + B.T)
    w, V = np.linalg.eigh(B)
    w = np.clip(w, 0, None)
    return (V * w) @ V.T

class LMC:
    """
    Fitted Linear Model of Coregionalization: a single spatial structure with
    common ``range`` and a P x P positive-semidefinite coregionalization sill
    matrix ``B``. The covariance between variables l and k is
    ``cov(l, k, h) = B[l, k] * model(h, range, 1)``.
    """
    def __init__(self, B, rng, model="Exp"):
        self.B = np.asarray(B, float)
        self.range = float(rng)
        self.model = model
        self._cov = covariance_model(model)
        self.P = self.B.shape[0]

    def cov(self, l, k, h):
        return self.B[l, k] * self._cov(h, self.range, 1.0)

    def sill(self, l):
        return self.B[l, l]

    def __repr__(self):
        return f"LMC(model={self.model}, range={self.range:.3g}, P={self.P})"

def fit_lmc(direct_vars, cross_vars, cross_pairs, P, rng, model="Exp",
            fit_range=False):
    """
    Fit an LMC to empirical direct and cross semivariograms (R: fit.lmc via
    lmc.poisson.cokrige.*).

    Parameters
    ----------
    direct_vars : list of (lags, gamma, npairs), length P
        Direct semivariograms, variable ``l`` at index ``l``.
    cross_vars : list of (lags, gamma, npairs)
        Cross semivariograms, aligned with ``cross_pairs``.
    cross_pairs : list of (l, k)
        Variable index pairs (l < k) for each entry of ``cross_vars``.
    P : int
        Number of variables.
    rng : float
        Common range (fixed unless ``fit_range=True``, matching the R default
        ``fit.ranges = F``).
    model : {"Exp", "Sph"}
    fit_range : bool
        If True, also optimise the common range by a small 1-D search.

    Returns
    -------
    LMC
        With a PSD-projected coregionalization matrix ``B``.
    """
    covfun = covariance_model(model)

    def solve_B(a):
        """WLS estimate of each B[l,k] with fixed range a, then PSD-project."""
        B = np.zeros((P, P))
        # semivariance of unit-sill structure: g(h) = 1 - cov(h,a,1)/1 = 1-exp(...)
        def gfit(lags):
            return covfun(0.0, a, 1.0) - covfun(np.asarray(lags), a, 1.0)
        # direct
        for l in range(P):
            lags, gam, npv = direct_vars[l]
            g = gfit(lags); wsum = np.sum(npv * g * g)
            B[l, l] = np.sum(npv * g * gam) / wsum if wsum > 0 else max(gam.max(), 0.0)
        # cross
        for (l, k), (lags, gam, npv) in zip(cross_pairs, cross_vars):
            ok = np.isfinite(gam)
            if ok.sum() == 0:
                B[l, k] = B[k, l] = 0.0; continue
            g = gfit(lags[ok]); wsum = np.sum(npv[ok] * g * g)
            b = np.sum(npv[ok] * g * gam[ok]) / wsum if wsum > 0 else 0.0
            B[l, k] = B[k, l] = b
        return _nearest_psd(B)

    if not fit_range:
        return LMC(solve_B(rng), rng, model)

    # crude range search minimising weighted SSE over all semivariograms
    def sse(a):
        B = solve_B(a); tot = 0.0
        def gpred(l, k, lags):
            return B[l, k] * (covfun(0.0, a, 1.0) - covfun(np.asarray(lags), a, 1.0))
        for l in range(P):
            lags, gam, npv = direct_vars[l]
            tot += np.sum(npv * (gpred(l, l, lags) - gam) ** 2)
        for (l, k), (lags, gam, npv) in zip(cross_pairs, cross_vars):
            ok = np.isfinite(gam)
            tot += np.sum(npv[ok] * (gpred(l, k, lags[ok]) - gam[ok]) ** 2)
        return tot
    grid = np.linspace(0.25 * rng, 4 * rng, 40)
    a_best = grid[int(np.argmin([sse(a) for a in grid]))]
    return LMC(solve_B(a_best), a_best, model)

# ---------------------------------------------------------------------------
# N-variate Poisson cokriging
# (R: poisson.cokrige.{bi,three,four} + poisson.cokrige.pred.*)
# ---------------------------------------------------------------------------
def _cokrige_solve(datasets, target, lmc, pop_rate, shared_mean, coords_pred):
    """
    Build the co-kriging system for the given ``datasets`` (list of dicts with
    keys x, y, cases, pop) and predict the target risk at ``coords_pred``.
    Returns an array with columns [x, y, pred, var].
    """
    P = len(datasets)
    coords = [np.column_stack([d["x"], d["y"]]).astype(float) for d in datasets]
    y = [np.asarray(d["cases"], float) for d in datasets]
    n = [np.asarray(d["pop"], float) for d in datasets]
    sizes = [len(c) for c in coords]
    off = np.concatenate([[0], np.cumsum(sizes)])
    N = off[-1]

    # ---- block covariance matrix with Poisson reliability terms
    C = np.zeros((N, N))
    for l in range(P):
        for k in range(l, P):
            D = cdist(coords[l], coords[k])
            block = lmc.cov(l, k, D)
            if l == k:
                # direct reliability: m*_l * pop_rate / n_i on the diagonal
                mstar = y[l].sum() / n[l].sum()
                block = block + np.diag(mstar * pop_rate / n[l])
            else:
                # cross reliability at co-located points: mean(R_lk)/n_i
                rlk = shared_mean.get((l, k), shared_mean.get((k, l), 0.0))
                co = np.isclose(D, 0.0)
                if co.any():
                    ii, jj = np.where(co)
                    block = block.copy()
                    block[ii, jj] = block[ii, jj] + rlk / n[l][ii]
            C[off[l]:off[l+1], off[k]:off[k+1]] = block
            if l != k:
                C[off[k]:off[k+1], off[l]:off[l+1]] = block.T

    # ---- unbiasedness constraints (one Lagrange multiplier per variable)
    Cbias = np.zeros((N, P))
    for l in range(P):
        Cbias[off[l]:off[l+1], l] = 1.0
    K = np.zeros((N + P, N + P))
    K[:N, :N] = C
    K[:N, N:] = Cbias
    K[N:, :N] = Cbias.T
    Cinv = np.linalg.inv(K)

    rates_all = np.concatenate([y[l] / n[l] * pop_rate for l in range(P)])
    out = np.empty((len(coords_pred), 4))
    for r, x0 in enumerate(np.atleast_2d(coords_pred)):
        rhs = np.empty(N + P)
        for k in range(P):
            dk = np.sqrt(((coords[k] - x0) ** 2).sum(1))
            rhs[off[k]:off[k+1]] = lmc.cov(target, k, dk)   # cross-cov to target
        rhs[N:] = 0.0
        rhs[N + target] = 1.0                                # target constraint
        lam = Cinv @ rhs
        pred = float((lam[:N] * rates_all).sum())
        if pred < 0:
            pred = 0.0
        var = lmc.sill(target) - float((lam * rhs).sum())
        out[r] = [x0[0], x0[1], pred, var]
    return out


def _cokrige_solve_knn_batch(datasets, target, lmc, pop_rate, shared_mean,
                             coords_pred, number_of_neighbors, use_gpu):
    """
    Local k-NN Poisson cokriging: one stacked ``xp.linalg.solve`` of shape
    ``(n_pred, P k + P, P k + P)``. Variables must be co-located (same
    coordinates), matching :func:`poisson_cokrige`.
    """
    P = len(datasets)
    coords0 = np.column_stack([datasets[0]["x"], datasets[0]["y"]]).astype(float)
    for d in datasets[1:]:
        other = np.column_stack([d["x"], d["y"]]).astype(float)
        if other.shape != coords0.shape or not np.allclose(other, coords0):
            raise ValueError(
                "k-NN Poisson cokriging requires co-located variables "
                "(the same x, y for every dataset)."
            )
    y = [np.asarray(d["cases"], float) for d in datasets]
    n = [np.asarray(d["pop"], float) for d in datasets]
    n_train = len(coords0)
    k = int(number_of_neighbors)
    coords_pred = np.atleast_2d(np.asarray(coords_pred, dtype=float))
    if k < 1:
        raise ValueError("number_of_neighbors must be >= 1")
    if k > n_train:
        raise ValueError(
            "Not enough known areas to satisfy number_of_neighbors "
            f"({k}); only {n_train} available."
        )

    tree = cKDTree(coords0)
    _, idx = tree.query(coords_pred, k=k)
    idx = np.asarray(idx, dtype=int)
    if k == 1:
        idx = idx.reshape(-1, 1)
    else:
        idx = np.atleast_2d(idx)
        if idx.shape[1] != k:
            idx = idx.reshape(len(coords_pred), k)

    n_pred = len(coords_pred)
    nc = coords0[idx]                                      # (n_pred, k, 2)
    y_n = np.stack([y[l][idx] for l in range(P)], axis=1)  # (n_pred, P, k)
    n_n = np.stack([n[l][idx] for l in range(P)], axis=1)
    Rlk = np.zeros((P, P))
    for l in range(P):
        for p in range(l + 1, P):
            val = shared_mean.get((l, p), shared_mean.get((p, l), 0.0))
            Rlk[l, p] = Rlk[p, l] = val

    if use_gpu:
        nc = as_gpu_array(nc)
        tg = as_gpu_array(coords_pred)
        y_n = as_gpu_array(y_n)
        n_n = as_gpu_array(n_n)
        B = as_gpu_array(lmc.B)
        Rlk_xp = as_gpu_array(Rlk)
    else:
        tg = coords_pred
        B = np.asarray(lmc.B, dtype=float)
        Rlk_xp = Rlk
    xp = get_array_module(nc)
    mstar = xp.sum(y_n, axis=2) / xp.sum(n_n, axis=2)  # (n_pred, P)

    dmat = xp.sqrt(xp.sum((nc[:, :, None, :] - nc[:, None, :, :]) ** 2, axis=-1))
    d0 = xp.sqrt(xp.sum((nc - tg[:, None, :]) ** 2, axis=-1))
    rho_nn = _unit_correlation(dmat, lmc.range, lmc.model, xp)
    rho_0 = _unit_correlation(d0, lmc.range, lmc.model, xp)

    m = P * k
    dim = m + P
    C = xp.zeros((n_pred, m, m))
    diag = xp.arange(k)
    for l in range(P):
        sl = slice(l * k, (l + 1) * k)
        for p in range(P):
            sp = slice(p * k, (p + 1) * k)
            block = B[l, p] * rho_nn
            if l == p:
                block[:, diag, diag] = block[:, diag, diag] + (
                    mstar[:, l, None] * pop_rate / n_n[:, l, :]
                )
            else:
                block[:, diag, diag] = block[:, diag, diag] + (
                    Rlk_xp[l, p] / n_n[:, l, :]
                )
            C[:, sl, sp] = block

    K = xp.zeros((n_pred, dim, dim))
    K[:, :m, :m] = C
    ones = xp.ones((n_pred, k))
    for l in range(P):
        sl = slice(l * k, (l + 1) * k)
        K[:, sl, m + l] = ones
        K[:, m + l, sl] = ones

    rhs = xp.zeros((n_pred, dim, 1))
    for p in range(P):
        sp = slice(p * k, (p + 1) * k)
        rhs[:, sp, 0] = B[target, p] * rho_0
    rhs[:, m + target, 0] = 1.0

    try:
        w = xp.linalg.solve(K, rhs)[:, :, 0]
    except xp.linalg.LinAlgError:
        w = xp.linalg.lstsq(K, rhs, rcond=None)[0][:, :, 0]

    rates_flat = (y_n / n_n * pop_rate).reshape(n_pred, m)
    pred = xp.sum(w[:, :m] * rates_flat, axis=1)
    pred = xp.maximum(pred, 0.0)
    var = B[target, target] - xp.sum(w * rhs[:, :, 0], axis=1)

    pred = to_numpy(pred)
    var = to_numpy(var)
    out = np.empty((n_pred, 4))
    out[:, 0] = coords_pred[:, 0]
    out[:, 1] = coords_pred[:, 1]
    out[:, 2] = pred
    out[:, 3] = var
    return out


def poisson_cokrige(datasets, target, lmc, pop_rate=100000.0,
                    shared_mean=None, coords_pred=None, smooth=None,
                    number_of_neighbors=None, use_gpu="auto"):
    """
    N-variate Poisson cokriging (generalises poisson.cokrige.bi/three/four).

    Parameters
    ----------
    datasets : list of dict
        One entry per variable, each with arrays ``x, y, cases, pop``. Element
        ``target`` is the variable being predicted; the rest are auxiliary.
        All variables must share the same locations/populations (co-located),
        as required by the cross-semivariogram estimator.
    target : int
        Index of the target variable in ``datasets``.
    lmc : LMC
        Fitted coregionalization model.
    pop_rate : float
        Rate base (e.g. 100000).
    shared_mean : dict
        ``{(l, k): mean shared risk R_lk}`` for auxiliary reliability terms.
        Missing pairs default to 0.
    coords_pred : array (m, 2), optional
        Prediction locations. If omitted (or equal to the target locations),
        leave-one-out **smoothing** is performed instead.
    smooth : bool, optional
        Force smoothing (True) or prediction (False). Auto-detected when None.
    number_of_neighbors : int, optional
        If set, each target uses its ``k`` nearest co-located areas (local
        k-NN) and all ``n_pred`` systems are stacked into one batched
        ``xp.linalg.solve``. If omitted, the original **global** system
        (every known area) is used. Required for GPU acceleration.
    use_gpu : bool or {'auto'}, default='auto'
        Solve the batched k-NN systems on the GPU via CuPy when usable.
        Ignored for the global path and for ``smooth=True``.

    Returns
    -------
    dict of numpy arrays
        Prediction mode: ``x, y, pred, var``.
        Smoothing mode: ``rate_pred, rate_var, observed, residual, zscore``.
    """
    shared_mean = shared_mean or {}
    tgt = datasets[target]
    tcoords = np.column_stack([tgt["x"], tgt["y"]]).astype(float)

    if smooth is None:
        smooth = coords_pred is None or (
            np.shape(coords_pred) == tcoords.shape and np.allclose(coords_pred, tcoords))

    if not smooth:
        if number_of_neighbors is not None:
            use_gpu_flag = resolve_cupy_use_gpu(use_gpu)
            res = _cokrige_solve_knn_batch(
                datasets, target, lmc, pop_rate, shared_mean, coords_pred,
                number_of_neighbors, use_gpu_flag,
            )
        else:
            res = _cokrige_solve(datasets, target, lmc, pop_rate, shared_mean, coords_pred)
        return {"x": res[:, 0], "y": res[:, 1], "pred": res[:, 2], "var": res[:, 3]}

    # ---- leave-one-out smoothing of the target variable
    m = len(tcoords)
    rate_pred = np.full(m, np.nan); rate_var = np.full(m, np.nan)
    observed = tgt["cases"] / tgt["pop"] * pop_rate
    for i in range(m):
        keep = np.ones(m, bool); keep[i] = False
        ds = []
        for l, d in enumerate(datasets):
            if l == target:
                ds.append({k: np.asarray(v)[keep] for k, v in
                           dict(x=d["x"], y=d["y"], cases=d["cases"], pop=d["pop"]).items()})
            else:
                ds.append(dict(x=d["x"], y=d["y"], cases=d["cases"], pop=d["pop"]))
        r = _cokrige_solve(ds, target, lmc, pop_rate, shared_mean,
                           tcoords[i:i+1])
        rate_pred[i], rate_var[i] = r[0, 2], r[0, 3]
    residual = rate_pred - observed
    zscore = residual / np.sqrt(rate_var)
    return {"rate_pred": rate_pred, "rate_var": rate_var, "observed": observed,
            "residual": residual, "zscore": zscore}

# ---------------------------------------------------------------------------
# Univariate Poisson kriging  (R: poisson.krige.one + .pred.one)
# ---------------------------------------------------------------------------
def poisson_krige(data, sill, rng, nugget=0.0, model="Exp",
                  pop_rate=100000.0, coords_pred=None, smooth=None):
    """
    Ordinary univariate Poisson kriging (R: poisson.krige.one).

    ``data`` is a dict with ``x, y, cases, pop``. The kriging matrix adds the
    Poisson error variance ``m* * pop_rate / n_i`` and the nugget to the
    diagonal. Prediction vs leave-one-out smoothing is auto-detected from
    ``coords_pred`` (see :func:`poisson_cokrige`).
    """
    covfun = covariance_model(model)
    x = np.asarray(data["x"], float); yc = np.asarray(data["y"], float)
    y = np.asarray(data["cases"], float); n = np.asarray(data["pop"], float)
    coords = np.column_stack([x, yc])

    if smooth is None:
        smooth = coords_pred is None or (
            np.shape(coords_pred) == coords.shape and np.allclose(coords_pred, coords))

    def solve(coords_tr, y_tr, n_tr, targets):
        size = len(coords_tr)
        D = squareform(pdist(coords_tr)) if size > 1 else np.zeros((1, 1))
        mstar = y_tr.sum() / n_tr.sum()
        C = covfun(D, rng, sill)
        C = C + np.diag(mstar * pop_rate / n_tr) + np.diag(np.full(size, nugget))
        A = np.zeros((size + 1, size + 1))
        A[:size, :size] = C; A[:size, size] = 1; A[size, :size] = 1
        Ainv = np.linalg.inv(A)
        rates = y_tr / n_tr * pop_rate
        out = np.empty((len(targets), 4))
        for r, x0 in enumerate(np.atleast_2d(targets)):
            d0 = np.sqrt(((coords_tr - x0) ** 2).sum(1))
            c0 = covfun(d0, rng, sill)
            pos = np.isclose(d0, 0.0)
            if pos.any():
                c0 = c0.copy(); c0[pos] = c0[pos] + nugget
            rhs = np.append(c0, 1.0)
            lam = Ainv @ rhs
            pred = float((lam[:size] * rates).sum())
            if pred < 0:
                pred = 0.0
            var = (covfun(0.0, rng, sill) + nugget) - float((lam * rhs).sum())
            out[r] = [x0[0], x0[1], pred, var]
        return out

    if not smooth:
        res = solve(coords, y, n, coords_pred)
        return {"x": res[:, 0], "y": res[:, 1], "pred": res[:, 2], "var": res[:, 3]}

    m = len(coords)
    rate_pred = np.full(m, np.nan); rate_var = np.full(m, np.nan)
    observed = y / n * pop_rate
    for i in range(m):
        keep = np.ones(m, bool); keep[i] = False
        r = solve(coords[keep], y[keep], n[keep], coords[i:i+1])
        rate_pred[i], rate_var[i] = r[0, 2], r[0, 3]
    residual = rate_pred - observed
    zscore = residual / np.sqrt(rate_var)
    return {"rate_pred": rate_pred, "rate_var": rate_var, "observed": observed,
            "residual": residual, "zscore": zscore}
