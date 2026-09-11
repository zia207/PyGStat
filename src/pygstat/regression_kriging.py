# src/pygstat/regression_kriging.py
"""
Regression Kriging = Regression + Kriging of Residuals.

The regression step models the large-scale trend from auxiliary predictors
(covariates); kriging then interpolates the spatially-correlated residuals left
over, and the two are added back together at prediction time. Any scikit-learn
-compatible regressor works for the trend model -- including **ensembles** and a
few other useful **statistical models**, which this module makes first-class,
convenient options via the `regressor` argument:

    RegressionKriging(regressor="random_forest")     # ENSEMBLE_REGRESSORS
    RegressionKriging(regressor="gradient_boosting")
    RegressionKriging(regressor="bagging")
    RegressionKriging(regressor="stacking")

    RegressionKriging(regressor="gam")                # STATISTICAL_REGRESSORS
    RegressionKriging(regressor="glm")
    RegressionKriging(regressor="bayesian_ridge")
    RegressionKriging(regressor="quantile")

Each shortcut builds an estimator with sensible defaults (see
`AVAILABLE_REGRESSORS` and `_build_regressor`); pass `regressor_kwargs` to override
any of its hyperparameters, or simply pass an already-constructed scikit-learn (or
otherwise `.fit(X, y)`/`.predict(X)`-compatible) estimator for full control --
including estimators this module has no shortcut for.

``regressor='gam'`` requires the optional ``pygam`` package
(``pip install pygam``, or ``pip install pygstat[gam]``); every other shortcut
only needs ``scikit-learn``, already a core dependency.
"""

import numpy as np
from .core.variogram import Variogram
from .core.kriging import OrdinaryKriging
from .utils.backend import resolve_cupy_use_gpu

__all__ = [
    "RegressionKriging",
    "GAMRegressor",
    "ENSEMBLE_REGRESSORS",
    "STATISTICAL_REGRESSORS",
    "AVAILABLE_REGRESSORS",
    "enforce_quantile_monotonicity",
]

# String shortcuts accepted by the `regressor` argument, in addition to passing
# an already-constructed scikit-learn-compatible estimator instance.
ENSEMBLE_REGRESSORS = ("linear", "random_forest", "gradient_boosting", "bagging", "stacking")
STATISTICAL_REGRESSORS = ("gam", "glm", "bayesian_ridge", "quantile")
AVAILABLE_REGRESSORS = ENSEMBLE_REGRESSORS + STATISTICAL_REGRESSORS

_DEFAULT_PARAMS = {
    "random_forest": {"n_estimators": 300, "max_depth": None, "n_jobs": -1, "random_state": 42},
    "gradient_boosting": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "random_state": 42},
    "bagging": {"n_estimators": 50, "random_state": 42},
    # glm (TweedieRegressor) and bayesian_ridge (BayesianRidge) use scikit-learn's
    # own defaults unchanged -- both are already sensible, general-purpose choices.
    "quantile": {"quantile": 0.5, "alpha": 0.0},
}


def _default_stacking_estimator(**overrides):
    """
    A sensible default `StackingRegressor`: diverse base learners (random
    forest, gradient boosting, and a plain linear model, so the ensemble isn't
    just three flavors of tree) combined by a ridge meta-learner. Pass
    `estimators=[...]` and/or `final_estimator=...` in `regressor_kwargs` to
    override either piece; everything else is passed through to
    `StackingRegressor` as-is (e.g. `cv=`, `passthrough=`).
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import LinearRegression, RidgeCV

    overrides = dict(overrides)
    estimators = overrides.pop("estimators", None) or [
        ("random_forest", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ("gradient_boosting", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
        ("linear", LinearRegression()),
    ]
    final_estimator = overrides.pop("final_estimator", None) or RidgeCV()
    return StackingRegressor(estimators=estimators, final_estimator=final_estimator, **overrides)


class GAMRegressor:
    """
    scikit-learn-compatible wrapper around `pyGAM <https://pygam.readthedocs.io>`_,
    fitting a purely additive smooth-term model
    :math:`f(X) = \\beta_0 + \\sum_j s_j(X_j)` -- one spline smooth per feature.

    pyGAM's ``GAM`` classes need their term structure (``s(0) + s(1) + ...``)
    specified before the number of features is known, which doesn't fit
    scikit-learn's "configure at construction, fit later" convention. This
    wrapper defers building the actual ``pygam.GAM`` until :meth:`fit`, when
    ``X.shape[1]`` is available, so it can be used as a drop-in
    ``regressor`` for :class:`RegressionKriging` (and anywhere else expecting
    ``.fit(X, y)`` / ``.predict(X)`` and scikit-learn's clone protocol).

    Parameters
    ----------
    distribution : str, default='normal'
        Exponential-family noise model (passed to ``pygam.GAM``); e.g.
        ``'normal'``, ``'poisson'``, ``'gamma'``.
    link : str, default='identity'
        Link function (e.g. ``'identity'``, ``'log'``).
    n_splines : int, default=10
        Number of splines per smooth term.
    lam : float, default=0.6
        Smoothing-penalty strength per term (higher = smoother).

    Requires the optional ``pygam`` package (``pip install pygam``).
    """

    def __init__(self, distribution="normal", link="identity", n_splines=10, lam=0.6):
        self.distribution = distribution
        self.link = link
        self.n_splines = n_splines
        self.lam = lam
        self.gam_ = None

    def get_params(self, deep=True):
        return {
            "distribution": self.distribution,
            "link": self.link,
            "n_splines": self.n_splines,
            "lam": self.lam,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        try:
            from pygam import GAM, s
        except ImportError as e:
            raise ImportError(
                "regressor='gam' requires the optional 'pygam' package. "
                "Install it with: pip install pygam  (or: pip install pygstat[gam])"
            ) from e

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1]
        if n_features < 1:
            raise ValueError("GAMRegressor needs at least one feature column in X.")

        terms = s(0, n_splines=self.n_splines, lam=self.lam)
        for j in range(1, n_features):
            terms = terms + s(j, n_splines=self.n_splines, lam=self.lam)

        self.gam_ = GAM(terms, distribution=self.distribution, link=self.link)
        self.gam_.fit(X, y)
        self.n_features_in_ = n_features
        return self

    def predict(self, X):
        if self.gam_ is None:
            raise RuntimeError("GAMRegressor is not fitted yet. Call fit() first.")
        return self.gam_.predict(np.asarray(X, dtype=float))

    def __repr__(self):
        return (
            f"GAMRegressor(distribution='{self.distribution}', link='{self.link}', "
            f"n_splines={self.n_splines}, lam={self.lam})"
        )


def _build_regressor(regressor, regressor_kwargs):
    """Resolve `regressor` into a concrete scikit-learn-compatible estimator.

    `regressor` may be: None (-> LinearRegression), an already-constructed
    estimator instance (returned as-is, `regressor_kwargs` ignored), or one of
    the strings in `AVAILABLE_REGRESSORS`.
    """
    if regressor is None:
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**(regressor_kwargs or {}))

    if not isinstance(regressor, str):
        return regressor

    key = regressor.lower()
    kwargs = dict(regressor_kwargs or {})

    if key in ("linear", "ols"):
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**kwargs)

    if key in ("random_forest", "randomforest", "rf"):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**{**_DEFAULT_PARAMS["random_forest"], **kwargs})

    if key in ("gradient_boosting", "gradientboosting", "gbm", "gbr"):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**{**_DEFAULT_PARAMS["gradient_boosting"], **kwargs})

    if key == "bagging":
        from sklearn.ensemble import BaggingRegressor
        return BaggingRegressor(**{**_DEFAULT_PARAMS["bagging"], **kwargs})

    if key == "stacking":
        return _default_stacking_estimator(**kwargs)

    if key == "gam":
        return GAMRegressor(**kwargs)

    if key == "glm":
        from sklearn.linear_model import TweedieRegressor
        return TweedieRegressor(**{**_DEFAULT_PARAMS.get("glm", {}), **kwargs})

    if key in ("bayesian_ridge", "bayesianridge", "bayesian", "bayes"):
        from sklearn.linear_model import BayesianRidge
        return BayesianRidge(**{**_DEFAULT_PARAMS.get("bayesian_ridge", {}), **kwargs})

    if key == "quantile":
        from sklearn.linear_model import QuantileRegressor
        return QuantileRegressor(**{**_DEFAULT_PARAMS["quantile"], **kwargs})

    raise ValueError(
        f"Unknown regressor '{regressor}'. Pass a scikit-learn-compatible "
        f"estimator instance, or one of {list(AVAILABLE_REGRESSORS)}."
    )


class RegressionKriging:
    """
    Regression Kriging = Regression + Kriging of Residuals.

    Parameters
    ----------
    regressor : sklearn regressor, str, or None, default=None
        The trend model. One of:

        - ``None`` -- ``LinearRegression()`` (the original default).
        - an already-constructed scikit-learn-compatible estimator (any
          object with ``.fit(X, y)`` / ``.predict(X)``) -- used as-is, giving
          full control over hyperparameters, custom pipelines, etc.
        - a string naming one of pygstat's shortcuts,
          ``AVAILABLE_REGRESSORS = ENSEMBLE_REGRESSORS + STATISTICAL_REGRESSORS``:

          **Ensembles** (``ENSEMBLE_REGRESSORS``):

          - ``'random_forest'`` -- ``RandomForestRegressor``: averages many
            de-correlated trees (bagging + random feature subsets). Robust,
            hard to overfit badly, minimal tuning required.
          - ``'gradient_boosting'`` -- ``GradientBoostingRegressor``: builds
            trees sequentially, each correcting the previous ensemble's
            residual error. Often the most accurate on structured/tabular
            data, but more sensitive to `learning_rate`/`n_estimators`.
          - ``'bagging'`` -- ``BaggingRegressor``: averages the same base
            learner (a decision tree, by default) trained on bootstrap
            resamples. The simplest variance-reduction ensemble.
          - ``'stacking'`` -- ``StackingRegressor``: trains several diverse
            base learners (random forest + gradient boosting + linear, by
            default) and a meta-learner (ridge, by default) that combines
            their out-of-fold predictions. See
            :func:`_default_stacking_estimator`.
          - ``'linear'`` -- ``LinearRegression()`` (explicit spelling of the
            default).

          **Statistical models** (``STATISTICAL_REGRESSORS``):

          - ``'gam'`` -- :class:`GAMRegressor`, a `pyGAM
            <https://pygam.readthedocs.io>`_-backed Generalized Additive
            Model: a smooth, flexible, per-feature nonlinear trend
            (:math:`\\sum_j s_j(X_j)`) that stays interpretable feature-by
            -feature, unlike a black-box ensemble. Requires the optional
            ``pygam`` package.
          - ``'glm'`` -- ``TweedieRegressor``, a Generalized Linear Model
            (exponential-family noise + link function); set
            ``regressor_kwargs={"power": 1}`` for a Poisson-like GLM,
            ``{"power": 2}`` for Gamma, etc. -- useful for non-negative,
            skewed targets (counts, concentrations, rates).
          - ``'bayesian_ridge'`` -- ``BayesianRidge``: a linear model with a
            Gaussian prior on the coefficients, automatically balancing fit
            against regularization from the data (no `alpha` to tune by
            hand). ``rk.regressor.predict(X, return_std=True)`` additionally
            exposes the regression model's own predictive uncertainty.
          - ``'quantile'`` -- ``QuantileRegressor``: models a chosen
            *quantile* of the target (median by default) rather than its
            mean -- set ``regressor_kwargs={"quantile": 0.9}`` for a 90th
            -percentile (upper-bound risk) surface instead of an average one.
            Fitting several quantile levels as separate `RegressionKriging`
            models does **not** guarantee they stay ordered at every location
            ("quantile crossing", a known property of quantile regression in
            general, not specific to kriging) -- see
            :func:`enforce_quantile_monotonicity` to fix that when it matters.
    regressor_kwargs : dict, optional
        Constructor keyword arguments for the shortcut named by `regressor`
        (ignored if `regressor` is already an estimator instance). For
        ``'stacking'``, the special keys ``estimators`` and
        ``final_estimator`` override the default base/meta learners; anything
        else is passed straight to the underlying constructor
        (e.g. ``regressor_kwargs={"n_estimators": 500, "max_depth": 8}`` for
        ``'random_forest'``, or ``regressor_kwargs={"quantile": 0.9}`` for
        ``'quantile'``).
    kriging_type : {'ordinary'}, default='ordinary'
        Only 'ordinary' is supported.
    variogram_model : str, default='spherical'
        Used only if `variogram` is not provided.
    variogram_kwargs : dict, optional
        Arguments for Variogram (e.g., maxlag, n_lags).
    variogram : Variogram, optional
        Pre-fitted variogram object. If provided, auto-fitting is skipped.
    use_gpu : bool or 'auto', default='auto'
        Passed through to residual `Variogram` fitting and to
        `OrdinaryKriging` of the residuals. The scikit-learn trend itself
        always runs on CPU.

    Examples
    --------
    >>> rk = RegressionKriging(regressor="random_forest",
    ...                        regressor_kwargs={"n_estimators": 500})
    >>> rk.fit(X_train, y_train, coords_train)
    >>> y_pred = rk.predict(X_test, coords_test)
    >>> rk.feature_importances_
    array([...])

    >>> rk_stack = RegressionKriging(regressor="stacking")
    >>> rk_stack.fit(X_train, y_train, coords_train)

    >>> rk_gam = RegressionKriging(regressor="gam")  # needs `pip install pygam`
    >>> rk_gam.fit(X_train, y_train, coords_train)

    >>> rk_p90 = RegressionKriging(regressor="quantile", regressor_kwargs={"quantile": 0.9})
    >>> rk_p90.fit(X_train, y_train, coords_train)  # an upper-bound risk surface
    """
    def __init__(
        self,
        regressor=None,
        kriging_type='ordinary',
        variogram_model='spherical',
        variogram_kwargs=None,
        variogram=None,
        regressor_kwargs=None,
        use_gpu="auto",
    ):
        self.regressor = _build_regressor(regressor, regressor_kwargs)
        self.regressor_kwargs = regressor_kwargs
        self.kriging_type = kriging_type
        self.variogram_model = variogram_model
        self.variogram_kwargs = variogram_kwargs or {}
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)
        self._user_provided_variogram = variogram is not None
        if self._user_provided_variogram:
            self.variogram_ = variogram
        self.is_fitted_ = False

    def fit(self, X, y, coords):
        """
        Fit regression model and kriging on residuals.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        coords : array-like, shape (n_samples, 2)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        coords = np.asarray(coords)

        # 1. Fit regression
        self.regressor.fit(X, y)
        y_pred_reg = self.regressor.predict(X)
        residuals = y - y_pred_reg

        # 2. Fit or use variogram
        if not self._user_provided_variogram:
            vg_kwargs = {
                'model': self.variogram_model,
                'estimator': 'matheron',
            }
            vg_kwargs.update(self.variogram_kwargs)
            vg_kwargs['use_gpu'] = self.use_gpu
            self.variogram_ = Variogram(coords, residuals, **vg_kwargs)
            self.variogram_.fit()
        elif getattr(self.variogram_, "fitted_params", None) is None:
            raise RuntimeError(
                "User-provided variogram is not fitted. Call fit() or set_params() first."
            )

        # 3. Fit kriging on residuals
        if self.kriging_type == 'ordinary':
            self.krige_ = OrdinaryKriging(self.variogram_, use_gpu=self.use_gpu)
        else:
            raise ValueError("Only 'ordinary' kriging is supported.")
        self.krige_.fit(coords, residuals)

        self.coords_ = coords
        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True
        return self

    def predict(self, X_pred, coords_pred, return_std=False):
        """
        Predict target variable at new locations.
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        X_pred = np.asarray(X_pred)
        coords_pred = np.asarray(coords_pred)

        reg_pred = self.regressor.predict(X_pred)
        if return_std:
            resid_pred, resid_std = self.krige_.predict(coords_pred, return_variance=True)
            return reg_pred + resid_pred, resid_std
        else:
            resid_pred = self.krige_.predict(coords_pred)
            return reg_pred + resid_pred

    @property
    def feature_importances_(self):
        """
        Feature importances from the fitted trend regressor, if it exposes
        them -- tree-based ensembles (``'random_forest'``,
        ``'gradient_boosting'``, ``'bagging'`` of trees) do; plain linear
        regressors and most stacking configurations do not.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        AttributeError
            If the fitted regressor does not expose ``feature_importances_``.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        if hasattr(self.regressor, "feature_importances_"):
            return self.regressor.feature_importances_
        raise AttributeError(
            f"The fitted regressor ({type(self.regressor).__name__}) does not "
            "expose feature_importances_ (only tree-based ensembles do)."
        )

    def __repr__(self):
        if self._user_provided_variogram:
            model_name = self.variogram_.model
        else:
            model_name = self.variogram_model
        return (
            f"RegressionKriging(regressor={type(self.regressor).__name__}, "
            f"variogram_model='{model_name}')"
        )


def enforce_quantile_monotonicity(predictions):
    """
    Fix "quantile crossing": given predictions from several *independently*
    fitted ``regressor='quantile'`` models (e.g. quantile levels 0.1, 0.5,
    0.9) at the same locations, sort them point-by-point so a lower quantile
    is never predicted above a higher one.

    This is a real, well-known phenomenon (Bassett & Koenker, 1982; He, 1997)
    with quantile regression in general, independent of kriging: separately
    fitting one model per quantile level offers no built-in guarantee they
    stay ordered. Combining quantile regression with kriging of that model's
    own residuals does not fix it either -- each quantile's residual field has
    its own, generally *different* spatial pattern, so the kriged correction
    at a given point can differ enough between quantile levels to reverse
    their order there. The standard remedy is the **rearrangement** of
    Chernozhukov, Fernandez-Val & Galichon (2010): simply sort the predicted
    quantiles at each point, which is exactly what this function does.

    Parameters
    ----------
    predictions : dict {quantile_level: array-like} or array-like, shape (n_quantiles, n_points)
        Predictions from `RegressionKriging(regressor='quantile', ...).predict(...)`
        at several quantile levels, at the *same* locations. A dict's keys
        are taken as the quantile levels and sorted numerically; an array's
        rows must already be ordered by increasing quantile level.

    Returns
    -------
    Same type and shape as `predictions`, with values sorted along the
    quantile axis at every point so they are monotonically non-decreasing.

    Examples
    --------
    >>> preds = {
    ...     0.1: rk_p10.predict(X_test, coords_test),
    ...     0.5: rk_p50.predict(X_test, coords_test),
    ...     0.9: rk_p90.predict(X_test, coords_test),
    ... }
    >>> fixed = enforce_quantile_monotonicity(preds)
    >>> (fixed[0.1] <= fixed[0.5]).all() and (fixed[0.5] <= fixed[0.9]).all()
    True
    """
    if isinstance(predictions, dict):
        levels = sorted(predictions.keys())
        stacked = np.stack([np.asarray(predictions[q], dtype=float) for q in levels], axis=0)
        sorted_stacked = np.sort(stacked, axis=0)
        return {q: sorted_stacked[i] for i, q in enumerate(levels)}
    return np.sort(np.asarray(predictions, dtype=float), axis=0)
