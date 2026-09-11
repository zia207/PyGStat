"""
Cokriging with a Linear Model of Coregionalization (LMC).

Includes:
- ``CrossVariogram`` — empirical / theoretical cross-variogram
- ``Cokriging`` — ordinary cokriging of one primary and one secondary
- ``MultivariateCokriging`` — ordinary cokriging of one primary and any
  number of secondaries (full cokriging; heterotopic sampling allowed)
- ``ColocatedCokriging`` — Markov-model (MM1) colocated cokriging
"""

from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from .core.variogram_models import MODEL_FUNCS
from .core.kriging import _cov_from_gamma
from .utils.backend import get_array_module, as_gpu_array, to_numpy, resolve_cupy_use_gpu, get_linalg_module


def _c0(variogram):
    params = getattr(variogram, "fitted_params", None)
    if params is None:
        raise RuntimeError("Variogram / cross-variogram is not fitted.")
    return float(params[0] + params[1])


def _jitter_diag(C, rel=1e-8):
    """Add a tiny ridge so near-singular LMC matrices remain solvable.
    Array-module-agnostic (numpy or cupy `C`, via get_array_module)."""
    xp = get_array_module(C)
    scale = float(xp.nanmax(xp.abs(xp.diag(C))))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    C = xp.array(C, dtype=float, copy=True)
    idx = xp.arange(C.shape[0])
    C[idx, idx] += rel * scale
    return C


def _pairwise_dist(A, B, xp):
    """Euclidean distance matrix between rows of A and rows of B, on
    whichever backend `xp` is (mirrors the same helper in
    pygstat.core.kriging / pygstat.sgsim)."""
    if xp is np:
        return cdist(A, B)
    diff = A[:, None, :] - B[None, :, :]
    return xp.sqrt(xp.sum(diff ** 2, axis=2))


def check_lmc_validity(sill_matrix, nugget_matrix=None, tol=1e-10):
    """
    Cauchy–Schwarz / PSD check for LMC coregionalization matrices.

    Parameters
    ----------
    sill_matrix : ndarray, shape (p, p)
        Partial-sill (structure) matrix B₁.
    nugget_matrix : ndarray, shape (p, p), optional
        Nugget matrix B₀.
    tol : float
        Eigenvalues above ``-tol`` are treated as non-negative.

    Returns
    -------
    dict
        ``valid``, ``sill_eigs``, ``nugget_eigs`` (latter is None if unused).
    """
    B1 = 0.5 * (np.asarray(sill_matrix, dtype=float) + np.asarray(sill_matrix, dtype=float).T)
    eigs1 = np.linalg.eigvalsh(B1)
    ok = bool(np.all(eigs1 >= -tol))
    eigs0 = None
    if nugget_matrix is not None:
        B0 = 0.5 * (np.asarray(nugget_matrix, dtype=float) + np.asarray(nugget_matrix, dtype=float).T)
        eigs0 = np.linalg.eigvalsh(B0)
        ok = ok and bool(np.all(eigs0 >= -tol))
    return {"valid": ok, "sill_eigs": eigs1, "nugget_eigs": eigs0}


class CrossVariogram:
    """
    Empirical and theoretical cross-variogram.

    The theoretical model is γ₁₂(h) = nugget + c12 * g(h), where g(h) is
    the same basic structure as the marginal variograms (default: exponential).

    ``fitted_params`` is ``[nugget, c12, range, ...]`` where ``c12`` is the
    partial cross-sill, so C₁₂(0) = nugget + c12.

    Unlike auto-variograms, ``c12`` (and the cross-nugget) may be **negative**.
    """

    def __init__(self, coords, primary, secondary, maxlag=None, n_lags=10, model="exponential",
                 use_gpu="auto"):
        self.coords = np.asarray(coords, dtype=float)
        self.primary = np.asarray(primary, dtype=float)
        self.secondary = np.asarray(secondary, dtype=float)
        self.maxlag = maxlag
        self.n_lags = n_lags
        if model not in MODEL_FUNCS:
            raise ValueError(f"model must be one of {list(MODEL_FUNCS)}")
        self.model = model
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)
        self.lags = None
        self.experimental = None
        self.fitted_params = None  # [nugget, c12, range, ...]
        self.fit_success_ = None

    def fit(self, method="empirical"):
        """
        Compute the empirical cross-variogram, optionally fitting a model.

        The O(n^2) pairwise-distance/cross-difference step runs on GPU via
        CuPy when `self.use_gpu` (same convention as
        `pygstat.core.variogram.Variogram`'s empirical semivariogram);
        `fit_model`'s small nonlinear least-squares fit always runs on CPU.

        Parameters
        ----------
        method : {'empirical', 'auto'}
            ``empirical`` (default) only fills ``lags`` / ``experimental`` so
            existing callers can set ``fitted_params`` themselves.  ``auto``
            also calls :meth:`fit_model`.
        """
        if len(self.primary) != len(self.secondary):
            raise ValueError("Primary and secondary must be co-located (same length).")
        if len(self.coords) != len(self.primary):
            raise ValueError("coords must have the same length as primary/secondary.")

        if self.use_gpu:
            coords, primary, secondary = as_gpu_array(self.coords), as_gpu_array(self.primary), as_gpu_array(self.secondary)
        else:
            coords, primary, secondary = self.coords, self.primary, self.secondary
        xp = get_array_module(coords)

        dists = _pairwise_dist(coords, coords, xp)
        diff1 = primary[:, None] - primary[None, :]
        diff2 = secondary[:, None] - secondary[None, :]
        cross_diff = 0.5 * diff1 * diff2

        positive = dists[dists > 0]
        if positive.size == 0:
            raise ValueError("All locations are coincident; cannot estimate a cross-variogram")
        if self.maxlag is None:
            self.maxlag = float(xp.percentile(positive, 50))

        bins = xp.linspace(0, self.maxlag, self.n_lags + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        cross_vario = []

        for i in range(self.n_lags):
            mask = (dists > bins[i]) & (dists <= bins[i + 1])
            if xp.any(mask):
                cross_vario.append(float(xp.nanmean(cross_diff[mask])))
            else:
                cross_vario.append(np.nan)

        self.lags = to_numpy(bin_centers)
        self.experimental = np.array(cross_vario)
        if method == "auto":
            self.fit_model()
        elif method != "empirical":
            raise ValueError("method must be 'empirical' or 'auto'")
        return self

    def set_params(self, nugget, c12, range_, nu=None, alpha=None):
        """Set theoretical cross-variogram parameters (c12 may be negative)."""
        params = [float(nugget), float(c12), float(range_)]
        if self.model == "matern":
            if nu is None:
                raise ValueError("matern requires nu")
            params.append(float(nu))
        elif self.model == "stable":
            if alpha is None:
                raise ValueError("stable requires alpha")
            params.append(float(alpha))
        elif nu is not None or alpha is not None:
            raise ValueError(f"{self.model} does not take nu or alpha")
        self.fitted_params = params
        return self

    def fit_model(self, range_=None, nu=None, alpha=None):
        """
        Least-squares fit of the theoretical cross-variogram to the empirical
        points.  ``c12`` is unbounded (cross-correlation may be negative).

        Parameters
        ----------
        range_ : float, optional
            If given, the range is locked (LMC shared-range fit).
        nu, alpha : float, optional
            Locked extra parameter for Matérn / stable models.
        """
        if self.experimental is None:
            raise RuntimeError("Call fit() before fit_model().")

        obs = self.experimental
        mask = ~np.isnan(obs)
        if not np.any(mask):
            raise RuntimeError("Empirical cross-variogram has no finite lags.")

        cov0 = float(np.cov(self.primary, self.secondary)[0, 1])
        nugget0 = 0.0
        c12_0 = cov0
        range0 = float(range_) if range_ is not None else self.maxlag / 2

        if self.model == "matern":
            nu0 = 1.5 if nu is None else float(nu)
            x0 = [nugget0, c12_0, range0, nu0]
            bounds = [(None, None), (None, None), (1e-6, self.maxlag * 3), (0.1, 10)]
        elif self.model == "stable":
            a0 = 1.0 if alpha is None else float(alpha)
            x0 = [nugget0, c12_0, range0, a0]
            bounds = [(None, None), (None, None), (1e-6, self.maxlag * 3), (0.01, 2.0)]
        else:
            x0 = [nugget0, c12_0, range0]
            bounds = [(None, None), (None, None), (1e-6, self.maxlag * 3)]

        lock_range = range_ is not None
        lock_extra = (self.model == "matern" and nu is not None) or (
            self.model == "stable" and alpha is not None
        )

        def unpack(params):
            p = list(params)
            if lock_range:
                p[2] = range0
            if lock_extra and len(p) > 3:
                p[3] = x0[3]
            return p

        def obj(params):
            p = unpack(params)
            pred = MODEL_FUNCS[self.model](self.lags, *p)
            return float(np.sum((pred[mask] - obs[mask]) ** 2))

        res = minimize(obj, x0, bounds=bounds, method="L-BFGS-B")
        self.fitted_params = unpack(res.x if res.success else x0)
        self.fit_success_ = bool(res.success)
        return self

    def __call__(self, h):
        """Evaluate theoretical cross-variogram γ₁₂(h)."""
        if self.fitted_params is None:
            raise RuntimeError("Call .fit() and set .fitted_params first.")
        return MODEL_FUNCS[self.model](h, *self.fitted_params)


def fit_lmc(coords, values_list, names=None, model="spherical", n_lags=12,
            maxlag=None, estimator="matheron", shared_range=True, use_gpu="auto"):
    """
    Fit auto- and cross-variograms under a 1-structure LMC.

    Auto-variograms are fitted first.  If ``shared_range`` is True, a common
    range (median of the auto ranges) is imposed on every auto- and
    cross-variogram and the sills / cross-sills are refitted.

    Parameters
    ----------
    use_gpu : bool or 'auto', default='auto'
        Passed through to every `Variogram`/`CrossVariogram` this fits, so
        each one's O(n^2) empirical-variogram/cross-variogram step runs on
        GPU via CuPy when usable (see `pygstat.core.variogram.Variogram`).

    Returns
    -------
    variograms : list of Variogram
    cross_variograms : dict[(i, j), CrossVariogram] with i < j
    info : dict
        ``range``, ``sill_matrix``, ``nugget_matrix``, ``lmc_check``.
    """
    from .core.variogram import Variogram

    coords = np.asarray(coords, dtype=float)
    values_list = [np.asarray(v, dtype=float) for v in values_list]
    p = len(values_list)
    if names is None:
        names = [f"Z{i}" for i in range(p)]
    if len(names) != p:
        raise ValueError("names must have one entry per variable")

    variograms = []
    for vals in values_list:
        vg = Variogram(
            coords, vals, model=model, estimator=estimator,
            maxlag=maxlag, n_lags=n_lags, use_gpu=use_gpu,
        )
        vg.fit()
        variograms.append(vg)

    if maxlag is None:
        maxlag = float(np.median([v.maxlag for v in variograms]))

    common_range = float(np.median([v.fitted_params[2] for v in variograms]))
    extra_kw = {}
    if model == "matern":
        extra_kw["nu"] = float(np.median([v.fitted_params[3] for v in variograms]))
    elif model == "stable":
        extra_kw["alpha"] = float(np.median([v.fitted_params[3] for v in variograms]))

    if shared_range:
        for vg, vals in zip(variograms, values_list):
            vg.set_params(
                nugget=float(vg.fitted_params[0]),
                sill=float(vg.fitted_params[1]),
                range_=common_range,
                **extra_kw,
            )
            # Refit nugget + sill with the locked range.
            obs = vg.experimental
            mask = ~np.isnan(obs)
            lags = vg.lags

            def obj(ns, _vg=vg, _mask=mask, _lags=lags):
                nugget, sill = ns
                pred = MODEL_FUNCS[model](_lags, nugget, sill, common_range, **extra_kw)
                return float(np.sum((pred[_mask] - obs[_mask]) ** 2))

            res = minimize(
                obj,
                x0=[float(vg.fitted_params[0]), float(vg.fitted_params[1])],
                bounds=[(0, None), (0, None)],
                method="L-BFGS-B",
            )
            nugget, sill = res.x if res.success else (vg.fitted_params[0], vg.fitted_params[1])
            vg.set_params(nugget=float(nugget), sill=float(sill), range_=common_range, **extra_kw)

    cross = {}
    for i, j in combinations(range(p), 2):
        cv = CrossVariogram(
            coords, values_list[i], values_list[j],
            maxlag=maxlag, n_lags=n_lags, model=model, use_gpu=use_gpu,
        )
        cv.fit(method="empirical")
        if shared_range:
            cv.fit_model(range_=common_range, **extra_kw)
        else:
            cv.fit_model(**extra_kw)
        # Cauchy–Schwarz clip of the partial cross-sill.
        s_i = abs(float(variograms[i].fitted_params[1]))
        s_j = abs(float(variograms[j].fitted_params[1]))
        cap = np.sqrt(s_i * s_j)
        nug_i = abs(float(variograms[i].fitted_params[0]))
        nug_j = abs(float(variograms[j].fitted_params[0]))
        nug_cap = np.sqrt(nug_i * nug_j)
        nugget = float(np.clip(cv.fitted_params[0], -nug_cap, nug_cap))
        c12 = float(np.clip(cv.fitted_params[1], -cap, cap))
        rng = common_range if shared_range else float(cv.fitted_params[2])
        cv.set_params(nugget=nugget, c12=c12, range_=rng, **extra_kw)
        cross[(i, j)] = cv

    B0 = np.zeros((p, p))
    B1 = np.zeros((p, p))
    for k, vg in enumerate(variograms):
        B0[k, k] = float(vg.fitted_params[0])
        B1[k, k] = float(vg.fitted_params[1])
    for (i, j), cv in cross.items():
        B0[i, j] = B0[j, i] = float(cv.fitted_params[0])
        B1[i, j] = B1[j, i] = float(cv.fitted_params[1])

    info = {
        "names": list(names),
        "model": model,
        "range": common_range if shared_range else None,
        "sill_matrix": B1,
        "nugget_matrix": B0,
        "lmc_check": check_lmc_validity(B1, B0),
    }
    return variograms, cross, info


class Cokriging:
    """
    Ordinary cokriging (two unbiasedness constraints).

    Parameters
    ----------
    primary_var : Variogram
        Fitted variogram for the primary variable.
    secondary_var : Variogram
        Fitted variogram for the secondary variable.
    cross_var : CrossVariogram
        Cross-variogram with fitted_params = [nugget, c12, range, ...]
        where C₁₂(0) = nugget + c12.
    use_gpu : bool or 'auto', default='auto'
        Builds the full (n_primary+n_secondary)^2 covariance matrix and
        solves the cokriging system on GPU via CuPy when usable -- the
        same convention (and Matérn CPU-only caveat) as
        `pygstat.core.kriging.OrdinaryKriging`.
    """

    def __init__(self, primary_var, secondary_var, cross_var, use_gpu="auto"):
        self.primary_var = primary_var
        self.secondary_var = secondary_var
        self.cross_var = cross_var
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)

    def fit(self, coords_primary, primary, coords_secondary, secondary):
        """Store training data."""
        self.coords_primary = np.asarray(coords_primary, dtype=float)
        self.primary = np.asarray(primary, dtype=float)
        self.coords_secondary = np.asarray(coords_secondary, dtype=float)
        self.secondary = np.asarray(secondary, dtype=float)

        if len(self.coords_primary) != len(self.primary):
            raise ValueError("coords_primary and primary must have same length.")
        if len(self.coords_secondary) != len(self.secondary):
            raise ValueError("coords_secondary and secondary must have same length.")

        self.all_coords = np.vstack([self.coords_primary, self.coords_secondary])
        self.all_values = np.hstack([self.primary, self.secondary])
        self.n_primary = len(self.primary)
        self.n_secondary = len(self.secondary)
        return self

    def _build_covariance_matrix(self, coords_primary, coords_secondary, xp):
        """Build full covariance matrix using LMC."""
        n = self.n_primary + self.n_secondary
        C = xp.zeros((n, n))

        dists_pp = _pairwise_dist(coords_primary, coords_primary, xp)
        C11_0 = _c0(self.primary_var)
        C[:self.n_primary, :self.n_primary] = _cov_from_gamma(
            self.primary_var(dists_pp), C11_0, dists_pp
        )

        dists_ss = _pairwise_dist(coords_secondary, coords_secondary, xp)
        C22_0 = _c0(self.secondary_var)
        C[self.n_primary:, self.n_primary:] = _cov_from_gamma(
            self.secondary_var(dists_ss), C22_0, dists_ss
        )

        dists_ps = _pairwise_dist(coords_primary, coords_secondary, xp)
        C12_0 = _c0(self.cross_var)
        C_ps = _cov_from_gamma(self.cross_var(dists_ps), C12_0, dists_ps)
        C[:self.n_primary, self.n_primary:] = C_ps
        C[self.n_primary:, :self.n_primary] = C_ps.T

        return _jitter_diag(C)

    def _build_rhs(self, coords_pred, coords_primary, coords_secondary, xp):
        """Build right-hand side vector for prediction."""
        n_pred = len(coords_pred)
        rhs = xp.zeros((n_pred, self.n_primary + self.n_secondary))

        dists_p = _pairwise_dist(coords_pred, coords_primary, xp)
        C11_0 = _c0(self.primary_var)
        rhs[:, :self.n_primary] = _cov_from_gamma(
            self.primary_var(dists_p), C11_0, dists_p
        )

        dists_s = _pairwise_dist(coords_pred, coords_secondary, xp)
        C12_0 = _c0(self.cross_var)
        rhs[:, self.n_primary:] = _cov_from_gamma(
            self.cross_var(dists_s), C12_0, dists_s
        )
        return rhs

    def predict(self, coords_pred, return_variance=False):
        """Predict primary variable at new locations."""
        coords_pred = np.asarray(coords_pred, dtype=float)
        n_pred = len(coords_pred)

        if self.use_gpu:
            coords_primary = as_gpu_array(self.coords_primary)
            coords_secondary = as_gpu_array(self.coords_secondary)
            coords_pred_d = as_gpu_array(coords_pred)
            all_values = as_gpu_array(self.all_values)
        else:
            coords_primary, coords_secondary = self.coords_primary, self.coords_secondary
            coords_pred_d, all_values = coords_pred, self.all_values
        xp = get_array_module(coords_primary)

        C = self._build_covariance_matrix(coords_primary, coords_secondary, xp)
        rhs = self._build_rhs(coords_pred_d, coords_primary, coords_secondary, xp)

        # Ordinary cokriging: Σ λ₁ = 1, Σ λ₂ = 0
        n = self.n_primary + self.n_secondary
        F = xp.zeros((n, 2))
        F[:self.n_primary, 0] = 1.0
        F[self.n_primary:, 1] = 1.0

        K = xp.vstack([
            xp.hstack([C, F]),
            xp.hstack([F.T, xp.zeros((2, 2))]),
        ])
        rhs_aug = xp.hstack([rhs, xp.ones((n_pred, 1)), xp.zeros((n_pred, 1))])

        try:
            weights = xp.linalg.solve(K, rhs_aug.T).T
        except xp.linalg.LinAlgError as e:
            raise xp.linalg.LinAlgError(
                "Singular matrix in cokriging. Check for duplicate coordinates, "
                "unfitted variograms, or invalid cross-variogram parameters."
            ) from e

        pred = weights[:, :n] @ all_values

        if not return_variance:
            return to_numpy(pred)

        C11_0 = _c0(self.primary_var)
        var = C11_0 - xp.einsum("ij,ij->i", weights[:, :n], rhs) - weights[:, n]
        var = xp.clip(var, 0, None)
        return to_numpy(pred), to_numpy(xp.sqrt(var))


class MultivariateCokriging:
    """
    Ordinary **full cokriging** of a primary variable using one or more
    secondaries and an LMC.

    Heterotopic sampling is allowed: each variable may be observed at a
    different set of locations.  The predictor of variable 0 (primary) is

    .. math::

        \\hat{Z}_0(x_0) = \\sum_{k=0}^{p-1}\\sum_{i=1}^{n_k}
        \\lambda_{k,i}\\, Z_k(x_{k,i})

    with unbiasedness constraints :math:`\\sum_i \\lambda_{0,i}=1` and
    :math:`\\sum_i \\lambda_{k,i}=0` for every secondary :math:`k\\ge 1`.

    Parameters
    ----------
    variograms : sequence of Variogram
        Auto-variograms, index 0 = primary.
    cross_variograms : dict
        ``{(i, j): CrossVariogram}`` for every pair ``i < j``.
    use_gpu : bool or 'auto', default='auto'
        Builds the full coregionalization matrix and factors it (and
        solves at `predict()` time) on GPU via CuPy when usable -- same
        convention as `Cokriging`. The factorization is computed once, at
        `fit()`, and reused (on whichever device it lives on) across every
        `predict()` call, exactly like the CPU path already did.
    """

    def __init__(self, variograms, cross_variograms, use_gpu="auto"):
        self.variograms = list(variograms)
        self.n_var = len(self.variograms)
        if self.n_var < 2:
            raise ValueError("Need at least a primary and one secondary.")
        self.cross_variograms = dict(cross_variograms)
        for i, j in combinations(range(self.n_var), 2):
            if (i, j) not in self.cross_variograms and (j, i) not in self.cross_variograms:
                raise ValueError(f"Missing cross-variogram for pair {(i, j)}")
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)

    def _cross(self, i, j):
        if i == j:
            raise ValueError("Use an auto-variogram for i == j")
        if (i, j) in self.cross_variograms:
            return self.cross_variograms[(i, j)]
        return self.cross_variograms[(j, i)]

    def fit(self, coords_list, values_list):
        """
        Store training data.

        Parameters
        ----------
        coords_list : sequence of array-like, each shape (n_k, 2)
        values_list : sequence of array-like, each shape (n_k,)
        """
        if len(coords_list) != self.n_var or len(values_list) != self.n_var:
            raise ValueError("coords_list and values_list must match n_var")
        self.coords_list = [np.asarray(c, dtype=float) for c in coords_list]
        self.values_list = [np.asarray(v, dtype=float) for v in values_list]
        self.n_per_var = [len(v) for v in self.values_list]
        for c, v, n in zip(self.coords_list, self.values_list, self.n_per_var):
            if len(c) != n:
                raise ValueError("Each coords/values pair must have the same length.")
        self.all_values = np.hstack(self.values_list)
        self._factor_system()
        return self

    def _cov_block(self, i, j, coords_i, coords_j, xp):
        d = _pairwise_dist(coords_i, coords_j, xp)
        if i == j:
            vg = self.variograms[i]
            return _cov_from_gamma(vg(d), _c0(vg), d)
        cv = self._cross(i, j)
        return _cov_from_gamma(cv(d), _c0(cv), d)

    def _factor_system(self):
        # Training coordinates are moved to GPU (if applicable) once here
        # and reused for every predict() call, rather than re-transferring
        # them each time.
        if self.use_gpu:
            self._coords_d = [as_gpu_array(c) for c in self.coords_list]
            self._all_values_d = as_gpu_array(self.all_values)
        else:
            self._coords_d = list(self.coords_list)
            self._all_values_d = self.all_values
        xp = get_array_module(self._coords_d[0])

        n = int(np.sum(self.n_per_var))
        C = xp.zeros((n, n))
        offset = np.cumsum([0] + self.n_per_var)
        for i in range(self.n_var):
            for j in range(i, self.n_var):
                block = self._cov_block(
                    i, j, self._coords_d[i], self._coords_d[j], xp
                )
                C[offset[i]:offset[i + 1], offset[j]:offset[j + 1]] = block
                if i != j:
                    C[offset[j]:offset[j + 1], offset[i]:offset[i + 1]] = block.T
        C = _jitter_diag(C)

        F = xp.zeros((n, self.n_var))
        for k in range(self.n_var):
            F[offset[k]:offset[k + 1], k] = 1.0

        K = xp.vstack([
            xp.hstack([C, F]),
            xp.hstack([F.T, xp.zeros((self.n_var, self.n_var))]),
        ])
        self._offset = offset
        self._n = n
        self._xp = xp
        self._lu = get_linalg_module(xp).lu_factor(K)

    def _build_rhs(self, coords_pred, xp):
        n_pred = len(coords_pred)
        rhs = xp.zeros((n_pred, self._n))
        for k in range(self.n_var):
            sl = slice(self._offset[k], self._offset[k + 1])
            if k == 0:
                vg = self.variograms[0]
                d = _pairwise_dist(coords_pred, self._coords_d[0], xp)
                rhs[:, sl] = _cov_from_gamma(vg(d), _c0(vg), d)
            else:
                cv = self._cross(0, k)
                d = _pairwise_dist(coords_pred, self._coords_d[k], xp)
                rhs[:, sl] = _cov_from_gamma(cv(d), _c0(cv), d)
        return rhs

    def predict(self, coords_pred, return_variance=False):
        """Predict the primary variable at ``coords_pred``."""
        if not hasattr(self, "_lu"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        coords_pred = np.asarray(coords_pred, dtype=float)
        n_pred = len(coords_pred)
        xp = self._xp
        coords_pred_d = as_gpu_array(coords_pred) if self.use_gpu else coords_pred
        rhs = self._build_rhs(coords_pred_d, xp)

        constraints = xp.zeros((n_pred, self.n_var))
        constraints[:, 0] = 1.0
        rhs_aug = xp.hstack([rhs, constraints])
        weights = get_linalg_module(xp).lu_solve(self._lu, rhs_aug.T).T
        pred = weights[:, :self._n] @ self._all_values_d

        if not return_variance:
            return to_numpy(pred)

        C11_0 = _c0(self.variograms[0])
        var = C11_0 - xp.einsum("ij,ij->i", weights[:, :self._n], rhs) - weights[:, self._n]
        var = xp.clip(var, 0, None)
        return to_numpy(pred), to_numpy(xp.sqrt(var))


class ColocatedCokriging:
    """
    Markov-model **colocated cokriging** (MM1; Xu et al. 1992).

    The screening hypothesis approximates the cross-covariance by the
    primary auto-covariance:

    .. math::

        C_{0k}(h) = \\frac{C_{0k}(0)}{C_{00}(0)}\\, C_{00}(h).

    Only the **colocated** secondary values at the prediction location enter
    the kriging system, together with all primary samples.  Secondary means
    are treated as known, so a single ordinary-kriging constraint
    :math:`\\sum \\lambda_i = 1` is used on the primary weights.

    Parameters
    ----------
    primary_var : Variogram
        Fitted auto-variogram of the primary.
    secondary_vars : sequence of Variogram
        Auto-variograms of each secondary (used for :math:`C_{kk}(0)`).
    cross_vars : sequence of CrossVariogram
        Primary–secondary cross-variograms (used for :math:`C_{0k}(0)`).
    secondary_cross_vars : dict, optional
        ``{(i, j): CrossVariogram}`` among secondaries (0-based secondary
        indices).  If omitted, lag-0 secondary cross-covariances are taken
        as 0 (secondaries treated as uncorrelated given the primary).
    mm1 : bool, default True
        If False, :math:`C_{0k}(h)` is evaluated from the fitted
        cross-variogram instead of the MM1 product.
    use_gpu : bool or 'auto', default='auto'
        Factors the primary kriging system on GPU via CuPy when usable
        (cached at `fit()`, reused across `predict()` calls, like
        `MultivariateCokriging`). The per-prediction-point secondary
        (colocated) system is also solved for every point *at once* via a
        single batched `xp.linalg.solve` rather than a Python loop -- a
        real speedup on CPU too, and part of what makes GPU dispatch
        worthwhile here.
    """

    def __init__(
        self,
        primary_var,
        secondary_vars,
        cross_vars,
        secondary_cross_vars=None,
        mm1=True,
        use_gpu="auto",
    ):
        self.primary_var = primary_var
        self.secondary_vars = list(secondary_vars)
        self.cross_vars = list(cross_vars)
        self.n_sec = len(self.secondary_vars)
        if len(self.cross_vars) != self.n_sec:
            raise ValueError("Need one primary–secondary cross-variogram per secondary.")
        self.secondary_cross_vars = {} if secondary_cross_vars is None else dict(secondary_cross_vars)
        self.mm1 = bool(mm1)
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)

    def fit(self, coords_primary, primary, secondary_means):
        """
        Store primary samples and secondary means.

        Parameters
        ----------
        coords_primary : array-like, shape (n, 2)
        primary : array-like, shape (n,)
        secondary_means : sequence of float
            Known means of each secondary (typically training means).
        """
        self.coords_primary = np.asarray(coords_primary, dtype=float)
        self.primary = np.asarray(primary, dtype=float)
        if len(self.coords_primary) != len(self.primary):
            raise ValueError("coords_primary and primary must have the same length.")
        self.secondary_means = np.asarray(secondary_means, dtype=float).reshape(-1)
        if self.secondary_means.size != self.n_sec:
            raise ValueError("secondary_means must have one entry per secondary.")
        self.n_primary = len(self.primary)
        self.mean_primary = float(np.mean(self.primary))

        self.C00 = _c0(self.primary_var)
        self.C0k = np.array([_c0(cv) for cv in self.cross_vars], dtype=float)
        self.Ckk = np.array([_c0(vg) for vg in self.secondary_vars], dtype=float)

        Cyy = np.diag(self.Ckk)
        for (i, j), cv in self.secondary_cross_vars.items():
            Cyy[i, j] = Cyy[j, i] = _c0(cv)
        self._Cyy = Cyy

        # Training data is moved to GPU (if applicable) once here and
        # reused for every predict() call.
        if self.use_gpu:
            self._coords_primary_d = as_gpu_array(self.coords_primary)
            self._primary_d = as_gpu_array(self.primary)
        else:
            self._coords_primary_d = self.coords_primary
            self._primary_d = self.primary
        xp = get_array_module(self._coords_primary_d)
        self._xp = xp

        dist_pp = _pairwise_dist(self._coords_primary_d, self._coords_primary_d, xp)
        Czz = _cov_from_gamma(self.primary_var(dist_pp), self.C00, dist_pp)
        Czz = _jitter_diag(Czz)
        n = self.n_primary
        A = xp.zeros((n + 1, n + 1))
        A[:n, :n] = Czz
        A[:n, n] = 1.0
        A[n, :n] = 1.0
        self._A_lu = get_linalg_module(xp).lu_factor(A)
        return self

    def _c0k_of_h(self, dist_p, xp):
        """Primary–secondary covariance C_{0k}(|x_i − x_0|) for each secondary."""
        n = dist_p.shape[0]
        m = dist_p.shape[1]
        out = xp.zeros((self.n_sec, n, m))
        C00_h = _cov_from_gamma(self.primary_var(dist_p), self.C00, dist_p)
        for k, cv in enumerate(self.cross_vars):
            if self.mm1:
                out[k] = (self.C0k[k] / self.C00) * C00_h
            else:
                out[k] = _cov_from_gamma(cv(dist_p), self.C0k[k], dist_p)
        return out

    def predict(self, coords_pred, secondary_at_pred, return_variance=False):
        """
        Predict the primary at ``coords_pred`` using colocated secondaries.

        Parameters
        ----------
        coords_pred : array-like, shape (m, 2)
        secondary_at_pred : array-like, shape (m, n_sec)
            Secondary values known at each prediction location.
        return_variance : bool
        """
        if not hasattr(self, "coords_primary"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        coords_pred = np.asarray(coords_pred, dtype=float)
        Y = np.asarray(secondary_at_pred, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        m = len(coords_pred)
        if Y.shape != (m, self.n_sec):
            raise ValueError(
                f"secondary_at_pred must have shape ({m}, {self.n_sec}), got {Y.shape}"
            )

        n = self.n_primary
        k = self.n_sec
        xp = self._xp
        lin = get_linalg_module(xp)
        coords_pred_d = as_gpu_array(coords_pred) if self.use_gpu else coords_pred

        dist_p0 = _pairwise_dist(coords_pred_d, self._coords_primary_d, xp)  # (m, n)
        C00_p0 = _cov_from_gamma(self.primary_var(dist_p0), self.C00, dist_p0)
        C0k_h = self._c0k_of_h(dist_p0, xp)  # (k, m, n)
        Czy = xp.transpose(C0k_h, (2, 0, 1))  # (n, k, m)

        rhs_a = xp.zeros((n + 1, m))
        rhs_a[:n, :] = C00_p0.T
        rhs_a[n, :] = 1.0
        w_a0 = lin.lu_solve(self._A_lu, rhs_a)  # (n+1, m)

        B_flat = xp.zeros((n + 1, k * m))
        B_flat[:n, :] = Czy.reshape(n, k * m)
        P = lin.lu_solve(self._A_lu, B_flat).reshape(n + 1, k, m)

        Cyy = xp.asarray(self._Cyy) if self.use_gpu else self._Cyy
        C0k = xp.asarray(self.C0k) if self.use_gpu else self.C0k
        Y_d = as_gpu_array(Y) if self.use_gpu else Y
        Y_res = Y_d - xp.asarray(self.secondary_means)[None, :] if self.use_gpu \
            else Y_d - self.secondary_means[None, :]

        # Every prediction point's (k x k) secondary system is solved at
        # once via a single batched xp.linalg.solve, instead of a Python
        # loop over m points -- a real speedup on CPU too (see class
        # docstring), and what makes GPU dispatch worthwhile here.
        S_batch = Cyy[None, :, :] - xp.einsum("iat,ibt->tab", Czy, P[:n])
        rhs_y_batch = C0k[None, :] - xp.einsum("iat,it->ta", Czy, w_a0[:n])
        try:
            w_y_batch = xp.linalg.solve(S_batch, rhs_y_batch[..., None])[..., 0]
        except xp.linalg.LinAlgError:
            # Rare (a genuinely singular colocated system at some point);
            # fall back to solving each point's system individually.
            w_y_batch = xp.stack([
                xp.linalg.lstsq(S_batch[t], rhs_y_batch[t], rcond=None)[0]
                for t in range(m)
            ])
        w_a_batch = w_a0.T - xp.einsum("abt,tb->ta", P, w_y_batch)  # (m, n+1)

        preds = xp.einsum("ta,a->t", w_a_batch[:, :n], self._primary_d) \
            + xp.einsum("tb,tb->t", w_y_batch, Y_res)

        if not return_variance:
            return to_numpy(preds)

        var = (
            self.C00
            - xp.einsum("ta,ta->t", w_a_batch[:, :n], C00_p0)
            - xp.einsum("tb,b->t", w_y_batch, C0k)
            - w_a_batch[:, n]
        )
        se = xp.sqrt(xp.clip(var, 0, None))
        return to_numpy(preds), to_numpy(se)
