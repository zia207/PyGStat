# src/pygstat/regression_kriging_h2o.py
"""
Regression Kriging with H2O-3 as the trend model.

The regression step models the large-scale trend from auxiliary predictors
(covariates) using any of H2O's supervised learning algorithms; kriging then
interpolates the spatially-correlated residuals, and the two are added back
together at prediction time.

`model_type` selects the trend model:

    RegressionKrigingH2O(model_type="automl")           # H2OAutoML (default, original behavior)
    RegressionKrigingH2O(model_type="deep_learning")     # H2ODeepLearningEstimator (neural network)
    RegressionKrigingH2O(model_type="drf")               # H2ORandomForestEstimator (Distributed Random Forest)
    RegressionKrigingH2O(model_type="glm")               # H2OGeneralizedLinearEstimator
    RegressionKrigingH2O(model_type="gam")               # H2OGeneralizedAdditiveEstimator
    RegressionKrigingH2O(model_type="gbm")               # H2OGradientBoostingEstimator
    RegressionKrigingH2O(model_type="xgboost")           # H2OXGBoostEstimator
    RegressionKrigingH2O(model_type="uplift_drf")        # H2OUpliftRandomForestEstimator (needs `treatment=` in fit())
    RegressionKrigingH2O(model_type="stacked_ensemble")  # H2OStackedEnsembleEstimator (see _build_default_stacked_ensemble)

For any `model_type` other than `'automl'`, pass `tune_hyperparameters=True` to run
an H2O grid search over `hyper_params` (a sensible default grid is used if not
given) and keep the best-scoring model -- see :func:`_default_hyper_params` and
:meth:`RegressionKrigingH2O.fit`.

Pass `categorical_columns` for factor covariates (e.g. land-cover class) and
`X_valid` / `y_valid` to `fit()` to use an H2O validation frame for early
stopping and hyperparameter ranking.
"""

import numpy as np
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from h2o.grid.grid_search import H2OGridSearch
from .core.variogram import Variogram
from .core.kriging import OrdinaryKriging
from .utils.backend import resolve_cupy_use_gpu

__all__ = ["RegressionKrigingH2O", "H2O_MODEL_TYPES"]

# Model-type shortcuts accepted by `model_type`.
H2O_MODEL_TYPES = (
    "automl",
    "deep_learning",
    "drf",
    "glm",
    "gam",
    "gbm",
    "xgboost",
    "uplift_drf",
    "stacked_ensemble",
)

_MODEL_TYPE_ALIASES = {
    "dl": "deep_learning",
    "neural_network": "deep_learning",
    "neural_net": "deep_learning",
    "random_forest": "drf",
    "distributed_random_forest": "drf",
    "gradient_boosting": "gbm",
    "gradient_boosting_machine": "gbm",
    "xgb": "xgboost",
    "extreme_gradient_boosting": "xgboost",
    "uplift": "uplift_drf",
    "uplift_random_forest": "uplift_drf",
    "stacking": "stacked_ensemble",
    "stack": "stacked_ensemble",
    "generalized_additive_model": "gam",
    "generalized_linear_model": "glm",
}


def _normalize_model_type(model_type):
    key = model_type.lower()
    key = _MODEL_TYPE_ALIASES.get(key, key)
    if key not in H2O_MODEL_TYPES:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose one of {list(H2O_MODEL_TYPES)}.")
    return key


def _get_estimator_class(model_type):
    """Lazily import and return the H2O estimator class for `model_type`
    ('automl' and 'stacked_ensemble' are handled separately by the caller)."""
    if model_type == "deep_learning":
        from h2o.estimators import H2ODeepLearningEstimator
        return H2ODeepLearningEstimator
    if model_type == "drf":
        from h2o.estimators import H2ORandomForestEstimator
        return H2ORandomForestEstimator
    if model_type == "glm":
        from h2o.estimators import H2OGeneralizedLinearEstimator
        return H2OGeneralizedLinearEstimator
    if model_type == "gam":
        from h2o.estimators import H2OGeneralizedAdditiveEstimator
        return H2OGeneralizedAdditiveEstimator
    if model_type == "gbm":
        from h2o.estimators import H2OGradientBoostingEstimator
        return H2OGradientBoostingEstimator
    if model_type == "xgboost":
        from h2o.estimators import H2OXGBoostEstimator
        return H2OXGBoostEstimator
    if model_type == "uplift_drf":
        from h2o.estimators import H2OUpliftRandomForestEstimator
        return H2OUpliftRandomForestEstimator
    raise ValueError(f"No individual estimator class for model_type='{model_type}'.")


def _default_hyper_params(model_type):
    """A modest, fast-to-search default hyperparameter grid per model type,
    used by grid-search tuning (`tune_hyperparameters=True`) when the caller
    doesn't supply their own `hyper_params`."""
    return {
        "drf": {"ntrees": [50, 100, 200], "max_depth": [10, 20, 30]},
        "gbm": {"ntrees": [50, 100, 200], "max_depth": [3, 5, 7], "learn_rate": [0.05, 0.1, 0.2]},
        "xgboost": {"ntrees": [50, 100, 200], "max_depth": [3, 5, 7], "learn_rate": [0.05, 0.1, 0.2]},
        "glm": {"alpha": [0.0, 0.25, 0.5, 0.75, 1.0]},
        "gam": {"lambda_": [0.0, 0.01, 0.1, 1.0]},
        "deep_learning": {
            "hidden": [[32, 32], [64, 64], [128, 64, 32]],
            "epochs": [10, 20],
        },
        "uplift_drf": {"ntrees": [50, 100], "max_depth": [5, 10]},
    }.get(model_type, {})


class RegressionKrigingH2O:
    """
    Regression Kriging with H2O-3 for the trend model.

    Parameters
    ----------
    model_type : str, default='automl'
        The trend model, one of `H2O_MODEL_TYPES` (aliases like `'dl'`,
        `'random_forest'`, `'gradient_boosting'` are also accepted):

        - ``'automl'`` -- H2OAutoML searches over several algorithms and
          returns the best leader model (the original behavior of this
          class; see `max_runtime_secs`, `max_models`, `include_algos`).
        - ``'deep_learning'`` -- ``H2ODeepLearningEstimator``, a
          feed-forward neural network.
        - ``'drf'`` -- ``H2ORandomForestEstimator``, Distributed Random Forest.
        - ``'glm'`` -- ``H2OGeneralizedLinearEstimator``.
        - ``'gam'`` -- ``H2OGeneralizedAdditiveEstimator``; by default every
          feature column is smoothed (``gam_columns``), overridable via
          `model_kwargs`.
        - ``'gbm'`` -- ``H2OGradientBoostingEstimator``.
        - ``'xgboost'`` -- ``H2OXGBoostEstimator``, H2O's wrapper around the
          XGBoost library. Distinct from `'gbm'` (H2O's own native gradient
          boosting implementation): usually comparable accuracy, sometimes
          faster and more tunable, but needs a platform XGBoost supports
          (unavailable on some H2O builds/architectures -- ``'gbm'`` is the
          always-available fallback with the same job).
        - ``'uplift_drf'`` -- ``H2OUpliftRandomForestEstimator``: models the
          *causal effect* of a binary treatment on a binary response, not a
          general regression trend. Requires passing `treatment=` (a 0/1
          array-like) to :meth:`fit`, and `y` must itself be binary.
        - ``'stacked_ensemble'`` -- ``H2OStackedEnsembleEstimator`` combining
          DRF + GBM + GLM base learners trained with cross-validation (see
          :func:`_build_default_stacked_ensemble`), unless `model_kwargs`
          supplies its own ``base_models``.
    model_kwargs : dict, optional
        Constructor keyword arguments for the chosen `model_type`'s
        estimator (ignored for `'automl'`; for `'gam'`, the special key
        ``gam_columns`` overrides which columns get spline smoothing).
    tune_hyperparameters : bool, default=False
        If True (and `model_type` is not `'automl'`), run an H2O grid search
        over `hyper_params` and keep the best model by cross-validated
        `stopping_metric`. Ignored for `'automl'`, which already searches a
        model space itself.
    hyper_params : dict, optional
        Hyperparameter grid for tuning, e.g. ``{"ntrees": [50, 100, 200]}``.
        Defaults to a sensible per-`model_type` grid (:func:`_default_hyper_params`)
        if not given.
    search_criteria : dict, optional
        H2O grid search strategy, e.g. ``{"strategy": "RandomDiscrete",
        "max_models": 20}``. Defaults to an exhaustive ``{"strategy":
        "Cartesian"}`` search over `hyper_params`.
    nfolds : int, default=5
        Cross-validation folds used for grid-search scoring and for the
        default stacked-ensemble's base learners.
    stopping_metric : str, default='RMSE'
        Metric used to rank models during grid search.
    max_runtime_secs : int, default=300
        AutoML only (`model_type='automl'`).
    max_models : int, default=20
        AutoML only.
    seed : int, default=42
    include_algos : list, optional
        AutoML only; default ``['XGBoost', 'GBM', 'GLM', 'DRF',
        'DeepLearning', 'StackedEnsemble']``.
    categorical_columns : list of str or int, optional
        Feature columns to treat as H2O factors (e.g. land-cover class
        ``['NLCD']``). Integer entries are interpreted as column indices
        into ``X``. Required for any non-numeric covariate: H2O will
        otherwise treat the codes as a continuous scale.
    variogram_model : str, default='spherical'
    variogram_kwargs : dict, optional
    variogram : Variogram, optional
    use_gpu : bool or 'auto', default='auto'
        Passed through to residual `Variogram` fitting and to
        `OrdinaryKriging` of the residuals. The H2O trend itself
        always runs on the JVM / CPU.

    Examples
    --------
    >>> rk = RegressionKrigingH2O(model_type="gbm", model_kwargs={"ntrees": 200})
    >>> rk.fit(X_train, y_train, coords_train)
    >>> y_pred = rk.predict(X_test, coords_test)

    >>> rk_tuned = RegressionKrigingH2O(
    ...     model_type="drf", tune_hyperparameters=True, categorical_columns=["NLCD"])
    >>> rk_tuned.fit(X_train, y_train, coords_train, X_valid=X_valid, y_valid=y_valid)
    >>> rk_tuned.get_best_model_metrics()
    """

    def __init__(
        self,
        model_type="automl",
        model_kwargs=None,
        tune_hyperparameters=False,
        hyper_params=None,
        search_criteria=None,
        nfolds=5,
        stopping_metric="RMSE",
        max_runtime_secs=300,
        max_models=20,
        seed=42,
        include_algos=None,
        categorical_columns=None,
        variogram_model='spherical',
        variogram_kwargs=None,
        variogram=None,
        use_gpu="auto",
    ):
        self.model_type = _normalize_model_type(model_type)
        self.model_kwargs = model_kwargs or {}
        self.tune_hyperparameters = tune_hyperparameters
        self.hyper_params = hyper_params
        self.search_criteria = search_criteria
        self.nfolds = nfolds
        self.stopping_metric = stopping_metric
        self.max_runtime_secs = max_runtime_secs
        self.max_models = max_models
        self.seed = seed
        self.include_algos = include_algos or [
            'XGBoost', 'GBM', 'GLM', 'DRF', 'DeepLearning', 'StackedEnsemble'
        ]
        self.categorical_columns = list(categorical_columns) if categorical_columns else []
        self.variogram_model = variogram_model
        self.variogram_kwargs = variogram_kwargs or {}
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)
        self._user_provided_variogram = variogram is not None
        if self._user_provided_variogram:
            self.variogram_ = variogram
        self.is_fitted_ = False

    # ------------------------------------------------------------------
    # Trend-model builders
    # ------------------------------------------------------------------

    def _ensure_cluster(self):
        # h2o.connection() returns None when there's no active connection --
        # it does not raise H2OConnectionError, so that used to never trigger
        # h2o.init() and every fit() failed with "Not connected to a cluster".
        if h2o.connection() is None:
            h2o.init(max_mem_size="32G", nthreads=-1)

    def _feature_frame(self, X, feature_names=None):
        """Return ``(DataFrame, names)`` without casting categoricals to float."""
        if isinstance(X, pd.DataFrame):
            names = list(feature_names) if feature_names is not None else list(X.columns)
            return X.loc[:, names].copy(), names
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if feature_names is not None:
            names = list(feature_names)
        elif getattr(self, "feature_names_", None):
            names = list(self.feature_names_)
        else:
            names = [f"X{i}" for i in range(X.shape[1])]
        if len(names) != X.shape[1]:
            names = [f"X{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=names), names

    def _resolve_categorical_columns(self, feature_names):
        resolved = []
        for col in self.categorical_columns:
            if isinstance(col, (int, np.integer)):
                resolved.append(feature_names[int(col)])
            else:
                resolved.append(str(col))
        missing = [c for c in resolved if c not in feature_names]
        if missing:
            raise ValueError(
                f"categorical_columns {missing} are not among the feature names {feature_names}."
            )
        return resolved

    def _to_h2o_frame(self, df, y=None, treatment=None):
        frame_df = df.copy()
        if y is not None:
            frame_df["target"] = np.asarray(y)
        treatment_column = None
        if treatment is not None:
            treatment_column = "treatment"
            frame_df[treatment_column] = np.asarray(treatment)
        frame = h2o.H2OFrame(frame_df)
        for col in getattr(self, "categorical_columns_", []):
            if col in frame.names:
                frame[col] = frame[col].asfactor()
        if treatment_column is not None:
            frame[treatment_column] = frame[treatment_column].asfactor()
            # UpliftDRF is a binomial-classification-only algorithm in H2O: it
            # rejects a numeric response outright, even one that only takes
            # values 0/1 -- the response column must be an explicit factor.
            if self.model_type == "uplift_drf":
                frame["target"] = frame["target"].asfactor()
        return frame, treatment_column

    def _train_kwargs(self, feature_names, train, valid=None):
        kwargs = dict(x=feature_names, y="target", training_frame=train)
        if valid is not None:
            kwargs["validation_frame"] = valid
        return kwargs

    def _fit_automl(self, feature_names, train, valid=None):
        aml = H2OAutoML(
            max_runtime_secs=self.max_runtime_secs,
            max_models=self.max_models,
            seed=self.seed,
            include_algos=self.include_algos
        )
        kwargs = self._train_kwargs(feature_names, train, valid)
        if valid is not None:
            kwargs["leaderboard_frame"] = valid
        aml.train(**kwargs)
        self.automl_ = aml
        self.leaderboard_ = aml.leaderboard.as_data_frame(use_pandas=True)
        return aml.leader

    def _fit_single_model(self, feature_names, train, treatment_column=None, valid=None):
        estimator_cls = _get_estimator_class(self.model_type)
        kwargs = dict(self.model_kwargs)
        kwargs.setdefault("seed", self.seed)
        if self.model_type == "gam" and "gam_columns" not in kwargs:
            kwargs["gam_columns"] = [c for c in feature_names if c not in self.categorical_columns_]
            if not kwargs["gam_columns"]:
                kwargs["gam_columns"] = feature_names
        if self.model_type == "uplift_drf":
            if treatment_column is None and "treatment_column" not in kwargs:
                raise ValueError(
                    "model_type='uplift_drf' requires a treatment column: pass "
                    "treatment=<0/1 array-like> to fit()."
                )
            kwargs.setdefault("treatment_column", treatment_column)

        train_kwargs = self._train_kwargs(feature_names, train, valid)

        if not self.tune_hyperparameters:
            model = estimator_cls(**kwargs)
            model.train(**train_kwargs)
            return model

        hyper_params = self.hyper_params or _default_hyper_params(self.model_type)
        if not hyper_params:
            raise ValueError(
                f"tune_hyperparameters=True but no default hyper_params are known for "
                f"model_type='{self.model_type}'; pass hyper_params=... explicitly."
            )
        # An explicit validation frame is the score set for hyperparameter
        # selection; n-fold CV on the training set is only used when no
        # validation data is supplied.
        if valid is None:
            kwargs.setdefault("nfolds", self.nfolds)
        elif self.model_type in ("drf", "gbm", "xgboost", "deep_learning", "uplift_drf"):
            kwargs.setdefault("stopping_rounds", 5)
            kwargs.setdefault("stopping_metric", self.stopping_metric)
        base_model = estimator_cls(**kwargs)
        grid = H2OGridSearch(
            model=base_model,
            hyper_params=hyper_params,
            search_criteria=self.search_criteria or {"strategy": "Cartesian"},
        )
        grid.train(**train_kwargs)
        sorted_grid = grid.get_grid(sort_by=self.stopping_metric.lower(), decreasing=False)
        if len(sorted_grid.models) == 0:
            raise RuntimeError("Grid search did not produce any models.")
        self.grid_ = sorted_grid
        self.grid_results_ = sorted_grid.sorted_metric_table()
        return sorted_grid.models[0]

    def _fit_stacked_ensemble(self, feature_names, train, valid=None):
        from h2o.estimators import (
            H2OGeneralizedLinearEstimator,
            H2OGradientBoostingEstimator, H2ORandomForestEstimator,
            H2OStackedEnsembleEstimator,
        )

        kwargs = dict(self.model_kwargs)
        base_models = kwargs.pop("base_models", None)
        train_kwargs = self._train_kwargs(feature_names, train, valid)

        if base_models is None:
            # StackedEnsemble metalearners need out-of-fold predictions from
            # the base learners, so nfolds stays even when a validation frame
            # is also supplied (that frame is used for early stopping / scoring).
            cv_kwargs = dict(
                nfolds=self.nfolds, seed=self.seed,
                fold_assignment="Modulo", keep_cross_validation_predictions=True,
            )
            drf = H2ORandomForestEstimator(**cv_kwargs)
            drf.train(**train_kwargs)
            gbm = H2OGradientBoostingEstimator(**cv_kwargs)
            gbm.train(**train_kwargs)
            glm = H2OGeneralizedLinearEstimator(**cv_kwargs)
            glm.train(**train_kwargs)
            self.base_models_ = [drf, gbm, glm]
            base_models = [m.model_id for m in self.base_models_]

        ensemble = H2OStackedEnsembleEstimator(base_models=base_models, **kwargs)
        ensemble.train(**train_kwargs)
        return ensemble

    # ------------------------------------------------------------------

    def fit(self, X, y, coords, treatment=None, X_valid=None, y_valid=None,
            treatment_valid=None):
        """
        Fit the H2O trend model + kriging on residuals.

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape (n_samples, n_features)
            Training covariates. A DataFrame keeps column names (and is
            required if `categorical_columns` holds names rather than
            integer indices).
        y : array-like, shape (n_samples,)
        coords : array-like, shape (n_samples, 2)
        treatment : array-like, shape (n_samples,), optional
            0/1 treatment indicator, required only for `model_type='uplift_drf'`.
        X_valid, y_valid : array-like, optional
            Held-out validation covariates/target used as H2O's
            ``validation_frame`` (and AutoML ``leaderboard_frame``) for
            early stopping and hyperparameter ranking. Residual kriging
            is still fit on the *training* residuals only.
        treatment_valid : array-like, optional
            Treatment indicator for `X_valid` (uplift DRF only).
        """
        self._ensure_cluster()

        y = np.asarray(y)
        coords = np.asarray(coords)
        X_df, feature_names = self._feature_frame(X)
        self.feature_names_ = feature_names
        self.categorical_columns_ = self._resolve_categorical_columns(feature_names)

        train, treatment_column = self._to_h2o_frame(X_df, y=y, treatment=treatment)

        valid = None
        if X_valid is not None:
            if y_valid is None:
                raise ValueError("X_valid was given but y_valid is missing.")
            X_valid_df, _ = self._feature_frame(X_valid, feature_names=feature_names)
            valid, _ = self._to_h2o_frame(
                X_valid_df, y=np.asarray(y_valid), treatment=treatment_valid
            )
            self.X_valid_ = X_valid_df
            self.y_valid_ = np.asarray(y_valid)
            self.treatment_valid_ = (
                None if treatment_valid is None else np.asarray(treatment_valid)
            )

        if self.model_type == "automl":
            self.best_model_ = self._fit_automl(feature_names, train, valid)
        elif self.model_type == "stacked_ensemble":
            self.best_model_ = self._fit_stacked_ensemble(feature_names, train, valid)
        else:
            self.best_model_ = self._fit_single_model(
                feature_names, train, treatment_column, valid
            )

        # Get residuals on the training locations (the spatial support for kriging)
        pred_frame = self.best_model_.predict(train)
        pred_df = pred_frame.as_data_frame(use_pandas=True)
        y_pred_reg = pred_df.iloc[:, 0].to_numpy().flatten()
        residuals = y - y_pred_reg

        # Fit or use variogram
        if not self._user_provided_variogram:
            vg_kwargs = {'model': self.variogram_model, 'estimator': 'matheron'}
            vg_kwargs.update(self.variogram_kwargs)
            vg_kwargs['use_gpu'] = self.use_gpu
            self.variogram_ = Variogram(coords, residuals, **vg_kwargs)
            self.variogram_.fit()
        # else: use user-provided variogram

        # Fit kriging
        self.krige_ = OrdinaryKriging(self.variogram_, use_gpu=self.use_gpu)
        self.krige_.fit(coords, residuals)

        self.coords_ = coords
        self.X_ = X_df
        self.y_ = y
        self.is_fitted_ = True
        return self

    def get_best_model_metrics(self):
        """Return trend-model performance metrics.

        For `model_type='automl'`, returns the AutoML leaderboard's row for
        the leader model. For every other `model_type`, queries
        `model_performance()` on the validation frame when `X_valid`
        was passed to :meth:`fit`, otherwise on the training frame.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet.")
        if self.model_type == "automl":
            best = self.leaderboard_.iloc[0]
            return {
                'model_id': best['model_id'],
                'rmse': best.get('rmse', np.nan),
                'r2': best.get('r2', np.nan),
                'mae': best.get('mae', np.nan),
            }
        valid_frame = None
        if getattr(self, "X_valid_", None) is not None:
            valid_frame, _ = self._to_h2o_frame(
                self.X_valid_,
                y=self.y_valid_,
                treatment=getattr(self, "treatment_valid_", None),
            )
        try:
            perf = self.best_model_.model_performance(valid_frame)
        except Exception:
            perf = self.best_model_.model_performance()
        metrics = {'model_id': self.best_model_.model_id}
        if self.model_type == "uplift_drf":
            # Uplift models are scored by AUUC (area under the uplift curve),
            # not RMSE/R2/MAE, which aren't meaningful for a treatment-effect
            # estimate.
            try:
                metrics['auuc'] = perf.auuc()
            except Exception:
                metrics['auuc'] = np.nan
            return metrics
        for name, fn in (('rmse', 'rmse'), ('r2', 'r2'), ('mae', 'mae')):
            try:
                metrics[name] = getattr(perf, fn)()
            except Exception:
                metrics[name] = np.nan
        return metrics

    def predict(self, X_pred, coords_pred, return_std=False):
        """Predict using the fitted trend model + kriged residuals."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet.")

        coords_pred = np.asarray(coords_pred)
        X_pred_df, _ = self._feature_frame(X_pred, feature_names=self.feature_names_)

        self._ensure_cluster()

        pred_h2o, _ = self._to_h2o_frame(X_pred_df)
        pred_df = self.best_model_.predict(pred_h2o).as_data_frame(use_pandas=True)
        reg_pred = pred_df.iloc[:, 0].to_numpy().flatten()

        if return_std:
            resid_pred, resid_std = self.krige_.predict(coords_pred, return_variance=True)
            result = (reg_pred + resid_pred, resid_std)
        else:
            resid_pred = self.krige_.predict(coords_pred)
            result = reg_pred + resid_pred

        # Do NOT shutdown — keep cluster alive for future calls
        return result

    def shutdown(self):
        """Manually shut down H2O cluster when done."""
        h2o.shutdown(prompt=False)

    def __repr__(self):
        if self._user_provided_variogram:
            model_name = self.variogram_.model
        else:
            model_name = self.variogram_model
        tuned = ", tuned" if (self.tune_hyperparameters and self.model_type != "automl") else ""
        return (
            f"RegressionKrigingH2O(model_type='{self.model_type}'{tuned}, "
            f"variogram_model='{model_name}')"
        )
