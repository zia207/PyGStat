# src/pygstat/disjunctive_kriging.py
"""
Disjunctive Kriging (DK) -- Matheron (1976); Hermite-polynomial formulation
following Rivoirard (1994) / Chilès & Delfiner.

DK is a **nonlinear** estimator: where ordinary/simple kriging give the best
*linear* combination of the data, DK gives the best estimator that is a sum
of separate functions of each datum, :math:`\\hat Z(x_0)=\\sum_i f_i(Z(x_i))`
-- a strictly richer class, made tractable by one specific model.

Recipe
------
1. **Gaussian anamorphosis**: transform ``Z`` to a standard Gaussian ``Y``
   via the normal-score transform already used for sequential simulation
   (:func:`pygstat.nscore.nscore_forward`): ``Z(x) = phi(Y(x))``.
2. **Hermite expansion** of the anamorphosis:
   ``phi(y) = sum_n phi_n * H_n(y)``, where ``H_n`` are the *normalized*
   (orthonormal) probabilists' Hermite polynomials,
   ``E[H_n(Y) H_m(Y)] = delta_nm`` for ``Y ~ N(0,1)``, and
   ``phi_n = E[Z H_n(Y)]`` is estimated empirically as
   ``mean(Z_i * H_n(Y_i))`` over the data.
3. **Bi-Gaussian model**: assumes every pair ``(Y(x), Y(x'))`` is bivariate
   Gaussian with correlation ``rho(h)``. Under this model, Hermite
   polynomials diagonalize the spatial covariance across *and within*
   locations: ``E[H_n(Y(x)) H_m(Y(x'))] = delta_nm * rho(h)**n``.
4. **Krige each order separately**: this orthogonality means the optimal
   combination reduces to kriging ``H_1(Y(x))..H_N(Y(x))`` *independently*,
   each via :class:`pygstat.core.kriging.SimpleKriging` (GPU-dispatching
   when usable) with covariance ``rho(h)**n`` -- one small adapter per
   order (:class:`_OrderVariogram`) is all that's needed to reuse
   `SimpleKriging` unmodified. Order 0 (``H_0=1``) is just the constant
   mean, not kriged.
5. **Recombine**: ``Ẑ_DK(x0) = sum_n phi_n * [H_n(Y(x0))]*_SK``. The *same*
   kriged Hermite components are reused, with different coefficients, to
   estimate any nonlinear functional of ``Z(x0)`` -- most usefully the
   exceedance probability ``P(Z(x0) > z_c)`` (see :meth:`predict_proba`),
   via the indicator function's own Hermite expansion
   (:func:`hermite_indicator_coefficients`).

See the project's disjunctive-kriging theory notes for the full derivation
(orthogonality proof, indicator-coefficient derivation via integration by
parts, and the additive-variance property used by ``return_variance=True``
below). Tested against ``data/meuse.csv``/``data/meuse_grid.csv`` -- see
``tests/test_disjunctive_kriging.py``.

Authors
-------
Zia Ahmed, PhD (zia207@gmail.com)
"""

import numpy as np
from scipy.stats import norm

from .nscore import nscore_forward
from .core.variogram import Variogram
from .core.variogram_models import MODEL_FUNCS
from .core.kriging import SimpleKriging
from .utils.backend import resolve_cupy_use_gpu

__all__ = ["DisjunctiveKriging", "hermite_polynomials", "hermite_indicator_coefficients"]


def hermite_polynomials(y, n_max):
    """
    Normalized (orthonormal) probabilists' Hermite polynomials
    :math:`H_0, \\dots, H_{n_{max}}` evaluated at `y`:
    :math:`H_0=1,\\ H_1(y)=y,\\ H_n(y)=\\frac{1}{\\sqrt n}\\big(y H_{n-1}(y)-\\sqrt{n-1}\\,H_{n-2}(y)\\big)`,
    satisfying :math:`E[H_n(Y)H_m(Y)]=\\delta_{nm}` for :math:`Y\\sim N(0,1)`
    (verified numerically to ~1e-8 in this module's tests).

    Parameters
    ----------
    y : array-like
    n_max : int

    Returns
    -------
    ndarray, shape ``(n_max+1,) + np.shape(y)``
    """
    y = np.asarray(y, dtype=float)
    H = np.empty((n_max + 1,) + y.shape, dtype=float)
    H[0] = 1.0
    if n_max >= 1:
        H[1] = y
    for n in range(2, n_max + 1):
        H[n] = (y * H[n - 1] - np.sqrt(n - 1) * H[n - 2]) / np.sqrt(n)
    return H


def hermite_indicator_coefficients(yc, n_max):
    """
    Hermite coefficients of the indicator function :math:`\\mathbf 1_{Y>y_c}`:

    .. math::
        \\mathbf 1_{Y>y_c} = \\sum_{n=0}^\\infty \\chi_n(y_c) H_n(Y), \\qquad
        \\chi_0(y_c) = 1-\\Phi(y_c), \\qquad
        \\chi_n(y_c) = \\frac{g(y_c)\\,H_{n-1}(y_c)}{\\sqrt n}\\ (n\\ge 1),

    where :math:`g` is the standard normal density -- derived via
    integration by parts from :math:`H_n' \\propto H_{n-1}` (see module docs).

    Parameters
    ----------
    yc : float
        Cutoff in Gaussian-score units.
    n_max : int

    Returns
    -------
    ndarray, shape (n_max+1,)
    """
    chi = np.empty(n_max + 1, dtype=float)
    chi[0] = 1.0 - norm.cdf(yc)
    if n_max >= 1:
        Hyc = hermite_polynomials(np.asarray(yc, dtype=float), n_max - 1)
        pdf_yc = norm.pdf(yc)
        for n in range(1, n_max + 1):
            chi[n] = pdf_yc * Hyc[n - 1] / np.sqrt(n)
    return chi


class _OrderVariogram:
    """
    Adapter exposing just the interface `pygstat.core.kriging.SimpleKriging`
    needs (`.fitted_params`, `.anisotropy`, `__call__`) for the covariance
    of the order-`n` Hermite transform :math:`H_n(Y(x))`:
    :math:`\\rho_Y(h)^n`, where :math:`\\rho_Y` is the correlation function
    of the fitted Y-variogram. By Hermite orthonormality,
    :math:`\\mathrm{Var}[H_n(Y(x))]=1` at every :math:`x`, so this order's
    :math:`C(0)=1` always -- hence ``fitted_params=[0.0, 1.0]``
    (nugget=0, sill=1). `__call__` reuses `MODEL_FUNCS`, which is already
    array-module-agnostic (NumPy or CuPy `h`), so this needs no GPU-specific
    code of its own.
    """

    def __init__(self, base_model, base_params, C0, order, anisotropy=None):
        self.base_model = base_model
        self.base_params = base_params
        self.C0 = C0
        self.order = order
        self.fitted_params = [0.0, 1.0]
        self.anisotropy = anisotropy

    def __call__(self, h):
        gamma_y = MODEL_FUNCS[self.base_model](h, *self.base_params)
        rho = 1.0 - gamma_y / self.C0
        return 1.0 - rho ** self.order


class DisjunctiveKriging:
    """
    Disjunctive Kriging: a nonlinear estimator built from a Gaussian
    anamorphosis + Hermite polynomial expansion under the bi-Gaussian
    model. See the module docstring for the full recipe.

    Parameters
    ----------
    model : str, default='spherical'
        Variogram model fitted to the Gaussian-transformed data ``Y``.
    n_hermite : int, default=20
        Number of Hermite orders (1..n_hermite, plus the constant order 0)
        used in the expansion. Higher orders capture sharper nonlinearity
        (useful for indicator/probability functionals near a cutoff) but
        the *anamorphosis* coefficients ``phi_n`` become noisier at high
        order with a small sample -- see `explained_variance_ratio_`.
    tails : {'linear', 'none'}, default='linear'
        Passed to the normal-score transform (`pygstat.nscore`); also used
        when mapping a new raw cutoff to Gaussian-score units in
        `predict_proba`.
    use_gpu : bool or 'auto', default='auto'
        Passed to the Y-variogram's empirical-semivariogram step and to
        every per-order `SimpleKriging` -- same convention (verified
        usable, not just import-checked, automatic CPU fallback) as
        `pygstat.core.kriging`.
    variogram_kwargs : dict, optional
        Extra kwargs for `pygstat.core.variogram.Variogram` (e.g.
        ``estimator``, ``n_lags``, ``maxlag``, ``anisotropy``).

    Attributes set by fit()
    ------------------------
    y_variogram_ : Variogram
        Fitted variogram of the Gaussian-transformed data.
    phi_ : ndarray, shape (n_hermite+1,)
        Anamorphosis (Hermite) coefficients of Z; ``phi_[0]`` is the mean.
    explained_variance_ratio_ : float
        ``sum(phi_[1:]**2) / Var(Z)`` -- fraction of Z's variance captured
        by the truncated expansion (-> 1 as ``n_hermite`` -> infinity).
    krige_by_order_ : list of SimpleKriging
        One fitted SimpleKriging per order n=1..n_hermite.

    Examples
    --------
    >>> dk = DisjunctiveKriging(model="spherical", n_hermite=20)
    >>> dk.fit(coords, values)
    >>> z_hat, se = dk.predict(coords_grid, return_variance=True)
    >>> prob_exceed = dk.predict_proba(coords_grid, cutoffs=[500, 1000])
    """

    def __init__(
        self,
        model="spherical",
        n_hermite=20,
        tails="linear",
        use_gpu="auto",
        variogram_kwargs=None,
    ):
        if n_hermite < 1:
            raise ValueError("n_hermite must be >= 1")
        if tails not in ("linear", "none"):
            raise ValueError("tails must be 'linear' or 'none'")
        self.model = model
        self.n_hermite = int(n_hermite)
        self.tails = tails
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)
        self.variogram_kwargs = variogram_kwargs or {}
        self.is_fitted_ = False

    def fit(self, coords, values):
        """
        Fit the anamorphosis, the Y-variogram, and one SimpleKriging per
        Hermite order.

        Parameters
        ----------
        coords : array-like, shape (n, 2)
        values : array-like, shape (n,)
        """
        coords = np.asarray(coords, dtype=float)
        values = np.asarray(values, dtype=float)
        if len(coords) != len(values):
            raise ValueError("coords and values must have the same length")
        if len(values) < 2:
            raise ValueError("at least 2 observations are required")

        # 1. Gaussian anamorphosis (normal-score transform).
        y, values_sorted, nscore_sorted = nscore_forward(values, tails=self.tails)
        self.values_sorted_ = values_sorted
        self.nscore_sorted_ = nscore_sorted
        self.mean_z_ = float(np.mean(values))
        self.var_z_ = float(np.var(values))

        # 2. Anamorphosis (Hermite) coefficients: phi_n = E[Z H_n(Y)].
        H_data = hermite_polynomials(y, self.n_hermite)  # (n_hermite+1, n)
        self.phi_ = (H_data * values[None, :]).mean(axis=1)
        self.explained_variance_ratio_ = (
            float(np.sum(self.phi_[1:] ** 2) / self.var_z_) if self.var_z_ > 0 else float("nan")
        )

        # 3. One variogram for Y -- the shared spatial-correlation shape
        #    every Hermite order's covariance rho_Y(h)**n is built from.
        vg_kwargs = dict(model=self.model, use_gpu=self.use_gpu)
        vg_kwargs.update(self.variogram_kwargs)
        y_variogram = Variogram(coords, y, **vg_kwargs)
        y_variogram.fit()
        self.y_variogram_ = y_variogram
        base_params = tuple(float(p) for p in y_variogram.fitted_params)
        C0_y = base_params[0] + base_params[1]
        anisotropy = getattr(y_variogram, "anisotropy", None)

        # 4. Krige H_1(Y)..H_N(Y) independently, each via SimpleKriging
        #    with covariance rho_Y(h)**order (order 0 is the constant mean).
        self.krige_by_order_ = []
        for order in range(1, self.n_hermite + 1):
            order_vg = _OrderVariogram(self.model, base_params, C0_y, order, anisotropy)
            sk = SimpleKriging(order_vg, mean=0.0, use_gpu=self.use_gpu)
            sk.fit(coords, H_data[order])
            self.krige_by_order_.append(sk)

        self.coords_, self.values_ = coords, values
        self.is_fitted_ = True
        return self

    def _krige_hermite_components(self, coords_pred, return_variance=False):
        """Krige H_1(Y0)..H_N(Y0) at `coords_pred`; H_0 is the known
        constant 1. Returns shape ``(n_hermite+1, n_pred)`` [+ variance]."""
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        coords_pred = np.asarray(coords_pred, dtype=float)
        n_pred = len(coords_pred)
        comp = np.empty((self.n_hermite + 1, n_pred))
        comp[0] = 1.0
        var = np.zeros((self.n_hermite + 1, n_pred)) if return_variance else None
        for i, sk in enumerate(self.krige_by_order_, start=1):
            if return_variance:
                pred, se = sk.predict(coords_pred, return_variance=True)
                comp[i], var[i] = pred, se ** 2
            else:
                comp[i] = sk.predict(coords_pred)
        return (comp, var) if return_variance else comp

    def predict(self, coords_pred, return_variance=False):
        """
        Disjunctive kriging point estimate:
        :math:`\\hat Z_{DK}(x_0)=\\sum_n \\varphi_n\\,[H_n(Y(x_0))]^*_{SK}`.

        If `return_variance`, also returns the DK estimation-error standard
        deviation, using the fact that different-order kriging errors are
        mutually uncorrelated under the bi-Gaussian model:
        :math:`\\mathrm{Var}[Z_0-\\hat Z_{DK}] = \\sum_n \\varphi_n^2\\,
        \\mathrm{Var}[H_n(Y_0)-[H_n(Y_0)]^*_{SK}]`.
        """
        if return_variance:
            comp, var = self._krige_hermite_components(coords_pred, return_variance=True)
            pred = (self.phi_[:, None] * comp).sum(axis=0)
            dk_var = (self.phi_[:, None] ** 2 * var).sum(axis=0)
            return pred, np.sqrt(np.clip(dk_var, 0, None))
        comp = self._krige_hermite_components(coords_pred)
        return (self.phi_[:, None] * comp).sum(axis=0)

    def _z_to_y(self, z):
        """Map raw cutoff(s) `z` to Gaussian-score units, using the same
        lookup table and tail convention as the fitted forward transform
        (mirrors `pygstat.nscore.nscore_forward`'s own interpolation)."""
        v_sorted, ns_sorted = self.values_sorted_, self.nscore_sorted_
        z = np.atleast_1d(np.asarray(z, dtype=float))
        out = np.interp(z, v_sorted, ns_sorted)
        if self.tails == "linear" and len(v_sorted) > 1:
            slope_low = (ns_sorted[1] - ns_sorted[0]) / (v_sorted[1] - v_sorted[0])
            slope_high = (ns_sorted[-1] - ns_sorted[-2]) / (v_sorted[-1] - v_sorted[-2])
            out = np.where(z < v_sorted[0], ns_sorted[0] + slope_low * (z - v_sorted[0]), out)
            out = np.where(z > v_sorted[-1], ns_sorted[-1] + slope_high * (z - v_sorted[-1]), out)
        else:
            out = np.clip(out, ns_sorted[0], ns_sorted[-1])
        return out

    def predict_proba(self, coords_pred, cutoffs):
        """
        Exceedance probability :math:`P(Z(x_0) > z_c \\mid \\text{data})`
        for one or more raw-unit cutoffs `cutoffs`, reusing the *same*
        kriged Hermite components as `predict()` (computed once here).

        Returns
        -------
        dict
            ``{cutoff: probability_array}``, each array clipped to
            ``[0, 1]`` -- the truncated Hermite series can slightly over-
            or under-shoot right at a cutoff (the classic Gibbs phenomenon
            of approximating a discontinuous target function; unaffected
            away from the cutoff, and shrinks as `n_hermite` grows).
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        cutoffs = np.atleast_1d(cutoffs).astype(float)
        comp = self._krige_hermite_components(coords_pred)  # (N+1, n_pred)

        out = {}
        for zc in cutoffs:
            yc = float(self._z_to_y(zc)[0])
            chi = hermite_indicator_coefficients(yc, self.n_hermite)
            prob = (chi[:, None] * comp).sum(axis=0)
            out[float(zc)] = np.clip(prob, 0.0, 1.0)
        return out
