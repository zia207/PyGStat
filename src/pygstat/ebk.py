# src/pygstat/ebk.py
"""
Empirical Bayesian Kriging (EBK) -- Krivoruchko (2012).

Ordinary/simple/universal kriging estimate the semivariogram parameters
:math:`\\theta` once (e.g. by weighted least squares or REML) and then
krige *as if* :math:`\\theta` were known exactly. The reported kriging
variance only reflects spatial-interpolation error given
:math:`\\theta=\\hat\\theta`; it silently ignores the sampling error in
:math:`\\hat\\theta` itself, so it tends to be optimistic -- especially
with modest sample sizes or a semivariogram whose range/nugget is hard to
pin down.

The full Bayesian fix is

.. math::
    p(Z(x_0)\\mid Z) = \\int p(Z(x_0)\\mid Z,\\theta)\\, p(\\theta\\mid Z)\\, d\\theta,

i.e. krige for every plausible :math:`\\theta`, weighted by how consistent
it is with the data -- but that needs a prior and MCMC (Handcock & Stein,
1993). EBK is *empirical* Bayesian: instead of specifying a prior and
sampling the posterior, it builds an approximation to :math:`p(\\theta\\mid Z)`
directly from the data via a simulate-and-refit loop, in the spirit of
empirical-Bayes methods generally (estimate the hyperparameter
distribution from the data rather than a priori):

1. Fit one semivariogram :math:`\\hat\\theta_0` to the real data.
2. For :math:`k=1,\\dots,K`: simulate an unconditional Gaussian-field
   realization :math:`Z^*_k` at the *same* observation locations, using
   :math:`\\hat\\theta_0` (exact via Cholesky factorization of the implied
   covariance matrix -- cheap since it only involves the training
   locations, not the prediction grid). Re-estimate a fresh semivariogram
   :math:`\\theta_k` from :math:`Z^*_k`. Across many :math:`k`, the scatter
   of :math:`\\theta_k` approximates the sampling distribution of the
   fitting procedure around the truth -- an empirical stand-in for
   :math:`p(\\theta\\mid Z)`.
3. Weight each :math:`\\theta_k` by how well it explains the spatial
   structure of *its own* simulated realization, via leave-one-out
   cross-validation (:func:`pygstat.validation.loo_cv`) -- an in-sample
   check would be meaningless, since ordinary/simple kriging interpolate
   their own training data exactly regardless of :math:`\\theta`. Weights
   are inverse-LOO-MSE (precision) weights, normalized to sum to 1.
4. Predict by kriging the *real* data once per :math:`\\theta_k` and
   combining across the ensemble:

   .. math::
       \\hat Z_{EBK}(x_0) = \\sum_k w_k\\, \\hat Z_k(x_0)

   .. math::
       \\mathrm{Var}[Z(x_0)] = \\underbrace{\\sum_k w_k\\,\\sigma_k^2(x_0)}_{\\text{within-model}}
       + \\underbrace{\\sum_k w_k\\,(\\hat Z_k(x_0)-\\hat Z_{EBK}(x_0))^2}_{\\text{between-model}},

   the law-of-total-variance / Rubin's-rules decomposition: within-model
   variance is the usual kriging error given :math:`\\theta_k`; the
   between-model term is new -- it is exactly the semivariogram-parameter
   uncertainty that plain kriging ignores.

Each ensemble member's kriging model is an ordinary, unmodified
:class:`pygstat.core.kriging.OrdinaryKriging` or
:class:`pygstat.core.kriging.SimpleKriging` -- including their GPU
dispatch (`use_gpu='auto'|True|False`) -- so predicting a large grid with
a large ensemble benefits from GPU acceleration exactly like a single
kriging model would. The simulate-and-refit loop itself runs on CPU (it
only ever touches the training locations, which are typically far fewer
than the prediction grid, so this is not the part worth accelerating --
the same reasoning `pygstat.sgsim`/`pygstat.sisim` use for defaulting
`use_gpu=False`).

Tested against ``data/meuse.csv``/``data/meuse_grid.csv`` -- see
``tests/test_ebk.py``.

Authors
-------
Zia Ahmed, PhD (zia207@gmail.com)
"""

import numpy as np
from scipy.stats import norm

from .core.variogram import Variogram
from .core.kriging import (
    OrdinaryKriging,
    SimpleKriging,
    _apply_variogram_anisotropy,
    _pairwise_dist,
    _cov_from_gamma,
)
from .validation import loo_cv

__all__ = ["EmpiricalBayesianKriging"]


class EmpiricalBayesianKriging:
    """
    Empirical Bayesian Kriging: an ensemble of kriging models built from
    simulate-and-refit semivariograms, combined to propagate
    semivariogram-parameter uncertainty into the prediction variance and
    into a genuine mixture-of-Gaussians predictive distribution. See the
    module docstring for the full algorithm.

    Parameters
    ----------
    model : str, default='spherical'
        Variogram model used for every fit (the base fit and every
        realization's refit).
    kriging_type : {'ordinary', 'simple'}, default='ordinary'
        Which kriging predictor each ensemble member uses.
    mean : float, optional
        Only meaningful when ``kriging_type='simple'``; passed straight
        through to every member's `SimpleKriging`. Left as ``None`` (the
        default), each member estimates its own mean from whatever data
        it is fit on -- `SimpleKriging`'s own default behavior.
    n_realizations : int, default=100
        Ensemble size K (ESRI's Geostatistical Analyst default is also
        100). Each realization costs one semivariogram fit plus a
        leave-one-out cross-validation over the training locations, so
        cost scales as ``O(K * n_train)`` kriging solves -- reduce for
        very large training sets.
    use_gpu : bool or 'auto', default='auto'
        Passed through to every ensemble member's kriging model (used at
        `predict()` time); the fit-time simulate-and-refit loop itself
        always runs on CPU (see module docstring).
    variogram_kwargs : dict, optional
        Extra kwargs for `pygstat.core.variogram.Variogram` (e.g.
        ``estimator``, ``n_lags``, ``maxlag``, ``anisotropy``).
    random_state : int, `numpy.random.Generator`, or None
        Seeds the unconditional-simulation draws.
    jitter : float, default=1e-8
        Diagonal regularization (as a fraction of C(0)) added before the
        Cholesky factorization used for unconditional simulation --
        numerical stability only, doubled automatically on failure.

    Attributes set by fit()
    ------------------------
    base_variogram_ : Variogram
        The single semivariogram fitted directly to the real data
        (:math:`\\hat\\theta_0` above) -- drives the simulation step only,
        never used directly for prediction.
    variograms_ : list of Variogram, length `n_realizations`
        Each realization's refit semivariogram :math:`\\theta_k`.
    weights_ : ndarray, shape (n_realizations,)
        Inverse-LOO-MSE precision weights, summing to 1.
    krige_models_ : list of OrdinaryKriging/SimpleKriging
        Each :math:`\\theta_k`'s kriging model, fitted on the *real* data
        -- what `predict()` actually uses.
    loo_rmse_ : ndarray, shape (n_realizations,)
        Each realization's leave-one-out RMSE (diagnostic).

    Examples
    --------
    >>> ebk = EmpiricalBayesianKriging(model="spherical", n_realizations=100)
    >>> ebk.fit(coords, values)
    >>> z_hat, se = ebk.predict(coords_grid, return_variance=True)
    >>> prob_exceed = ebk.predict_proba(coords_grid, cutoffs=[500, 1000])
    """

    def __init__(
        self,
        model="spherical",
        kriging_type="ordinary",
        mean=None,
        n_realizations=100,
        use_gpu="auto",
        variogram_kwargs=None,
        random_state=None,
        jitter=1e-8,
    ):
        if kriging_type not in ("ordinary", "simple"):
            raise ValueError("kriging_type must be 'ordinary' or 'simple'")
        if n_realizations < 2:
            raise ValueError("n_realizations must be >= 2")
        self.model = model
        self.kriging_type = kriging_type
        self.mean = mean
        self.n_realizations = int(n_realizations)
        self.use_gpu = use_gpu
        self.variogram_kwargs = variogram_kwargs or {}
        self.random_state = random_state
        self.jitter = jitter
        self.is_fitted_ = False

    def _make_kriging(self, variogram):
        if self.kriging_type == "simple":
            return SimpleKriging(variogram, mean=self.mean, use_gpu=self.use_gpu)
        return OrdinaryKriging(variogram, use_gpu=self.use_gpu)

    def _simulate_unconditional(self, coords, variogram, mean, rng):
        """One unconditional Gaussian-field realization at `coords`, exact
        via Cholesky factorization of the covariance matrix implied by
        `variogram` (small n_train x n_train system -- the training
        locations only, never the prediction grid)."""
        X = _apply_variogram_anisotropy(variogram, coords)
        dist = _pairwise_dist(X, X, np)
        C0 = float(variogram.fitted_params[0] + variogram.fitted_params[1])
        C = _cov_from_gamma(variogram(dist), C0, dist)

        jitter = self.jitter
        for _ in range(5):
            try:
                L = np.linalg.cholesky(C + jitter * C0 * np.eye(len(coords)))
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        else:
            raise np.linalg.LinAlgError(
                "Covariance matrix is not positive definite even after regularization."
            )
        return mean + L @ rng.standard_normal(len(coords))

    def fit(self, coords, values):
        """
        Run the simulate-refit-weight loop and fit one kriging model per
        ensemble member on the real data.

        Parameters
        ----------
        coords : array-like, shape (n, 2)
        values : array-like, shape (n,)
        """
        coords = np.asarray(coords, dtype=float)
        values = np.asarray(values, dtype=float)
        if len(coords) != len(values):
            raise ValueError("coords and values must have the same length")
        if len(values) < 4:
            raise ValueError("at least 4 observations are required")

        rng = np.random.default_rng(self.random_state)
        mean0 = float(np.mean(values))

        vg_kwargs = dict(model=self.model, use_gpu=self.use_gpu)
        vg_kwargs.update(self.variogram_kwargs)

        base_vg = Variogram(coords, values, **vg_kwargs)
        base_vg.fit()
        self.base_variogram_ = base_vg

        K = self.n_realizations
        self.variograms_ = []
        self.krige_models_ = []
        rmse = np.empty(K)

        for k in range(K):
            z_star = self._simulate_unconditional(coords, base_vg, mean0, rng)

            vg_k = Variogram(coords, z_star, **vg_kwargs)
            vg_k.fit()
            self.variograms_.append(vg_k)

            # Weight theta_k by how well it explains the spatial structure
            # of its OWN simulated realization, via leave-one-out CV -- an
            # in-sample check is meaningless here, since SK/OK interpolate
            # their own training data exactly regardless of theta.
            cv = loo_cv(self._make_kriging(vg_k), coords, z_star)
            r = cv["rmse"]
            rmse[k] = r if np.isfinite(r) and r > 0 else np.inf

            # The model actually used for prediction is fit on the REAL data.
            km = self._make_kriging(vg_k)
            km.fit(coords, values)
            self.krige_models_.append(km)

        self.loo_rmse_ = rmse
        inv_mse = 1.0 / np.clip(rmse, 1e-300, None) ** 2
        if not np.isfinite(inv_mse).any() or inv_mse.sum() == 0:
            # Degenerate fallback (every realization's LOO blew up): weight uniformly.
            inv_mse = np.ones(K)
        inv_mse = np.where(np.isfinite(inv_mse), inv_mse, 0.0)
        self.weights_ = inv_mse / inv_mse.sum()

        self.coords_, self.values_ = coords, values
        self.is_fitted_ = True
        return self

    def _predict_ensemble(self, coords_pred, return_variance=False):
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        coords_pred = np.asarray(coords_pred, dtype=float)
        n_pred = len(coords_pred)
        K = self.n_realizations
        preds = np.empty((K, n_pred))
        if not return_variance:
            for k, km in enumerate(self.krige_models_):
                preds[k] = km.predict(coords_pred)
            return preds, None
        var = np.empty((K, n_pred))
        for k, km in enumerate(self.krige_models_):
            p, se = km.predict(coords_pred, return_variance=True)
            preds[k], var[k] = p, se ** 2
        return preds, var

    def predict(self, coords_pred, return_variance=False):
        """
        Weighted-ensemble point estimate:
        :math:`\\hat Z_{EBK}(x_0)=\\sum_k w_k\\, \\hat Z_k(x_0)`.

        If `return_variance`, also returns the total predictive standard
        deviation combining within-model kriging variance and
        between-model variance from semivariogram-parameter uncertainty
        (see module docstring) -- this is why it is typically larger than
        a single `OrdinaryKriging`/`SimpleKriging` fit's reported standard
        error on the same data.
        """
        preds, var = self._predict_ensemble(coords_pred, return_variance=return_variance)
        w = self.weights_[:, None]
        mean_pred = (w * preds).sum(axis=0)
        if not return_variance:
            return mean_pred
        within = (w * var).sum(axis=0)
        between = (w * (preds - mean_pred) ** 2).sum(axis=0)
        return mean_pred, np.sqrt(np.clip(within + between, 0, None))

    def predict_proba(self, coords_pred, cutoffs, greater_than=True):
        """
        ``P(Z(x0) > cutoff | data)`` (or ``<`` if `greater_than=False`) for
        one or more cutoffs, by treating each ensemble member's predictive
        distribution as :math:`N(\\hat Z_k(x_0), \\sigma_k^2(x_0))` and
        mixing across k with weight :math:`w_k` -- a genuine
        mixture-of-Gaussians predictive distribution (not a single
        Gaussian built from the combined mean/variance), so it reflects
        the extra spread coming from semivariogram-parameter uncertainty
        rather than smoothing it away.

        Returns
        -------
        dict
            ``{cutoff: probability_array}``, each clipped to ``[0, 1]``.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        preds, var = self._predict_ensemble(coords_pred, return_variance=True)
        sds = np.sqrt(np.clip(var, 1e-300, None))

        cutoffs = np.atleast_1d(cutoffs).astype(float)
        w = self.weights_[:, None]
        out = {}
        for zc in cutoffs:
            tail = norm.sf(zc, loc=preds, scale=sds) if greater_than else norm.cdf(
                zc, loc=preds, scale=sds
            )
            prob = (w * tail).sum(axis=0)
            out[float(zc)] = np.clip(prob, 0.0, 1.0)
        return out
