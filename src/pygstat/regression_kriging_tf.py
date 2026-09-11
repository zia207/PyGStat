# src/pygstat/regression_kriging_tf.py
"""
Deep Regression Kriging with TensorFlow/Keras as the trend model.

The regression step trains a feed-forward neural network (Keras `Sequential`)
on auxiliary predictors (covariates); kriging then interpolates the
spatially-correlated residuals left over, and the two are added back
together at prediction time.

Hyperparameter tuning
----------------------
Pass ``tune_hyperparameters=True`` to search the network's architecture and
optimizer settings with `Optuna <https://optuna.org>`_ before the final fit:

    DeepRegressionKrigingTF(tune_hyperparameters=True, n_trials=30, cv_folds=3)

By default this searches the number/width of hidden layers, dropout, learning
rate, weight decay, and batch size (:func:`_default_param_space`), scoring each
trial by K-fold cross-validated validation RMSE (with median pruning of
clearly weak trials), and refits one final model with the best hyperparameters
found on the *full* training set (using this instance's own
`max_epochs`/`patience`, not the smaller tuning budget). Pass your own
``param_space(trial) -> dict`` for a custom search space -- see
:func:`_default_param_space` for the expected shape. This mirrors
`pygstat.regression_kriging_pytorch_gpu.DeepRegressionKriging`'s tuning API
exactly, so the two backends can be swapped with the same calling code.
Requires the optional ``optuna`` package (``pip install optuna``, or
``pip install pygstat[optuna]``).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from .core.variogram import Variogram
from .core.kriging import OrdinaryKriging
from .utils.backend import resolve_cupy_use_gpu

__all__ = ["DeepRegressionKrigingTF"]


# ==========================================================
# Model construction and training (module-level so the Optuna
# tuning objective and the final production fit share one code path)
# ==========================================================

def _build_keras_mlp(input_dim, hidden_layers, dropout, learning_rate, weight_decay):
    init = keras.initializers.GlorotUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        hidden_layers[0], activation="relu", input_shape=(input_dim,),
        kernel_initializer=init, bias_initializer="zeros",
    ))
    model.add(keras.layers.Dropout(dropout))
    for units in hidden_layers[1:]:
        model.add(keras.layers.Dense(units, activation="relu", kernel_initializer=init, bias_initializer="zeros"))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, kernel_initializer=init, bias_initializer="zeros"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            weight_decay=(weight_decay if weight_decay else None),
            clipnorm=1.0,
        ),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _train_keras_mlp(
    X_train, y_train, X_val, y_val,
    hidden_layers, dropout, learning_rate, weight_decay, batch_size,
    max_epochs, patience, verbose=0,
):
    """
    Fit one Keras MLP (with early stopping on validation loss, if
    `X_val`/`y_val` are given). Returns ``(model, scaler, history_dict,
    best_val_loss)`` -- `scaler` is the `StandardScaler` fit on `X_train`
    only; `best_val_loss` is ``inf`` if no validation data was given.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = _build_keras_mlp(X_train.shape[1], hidden_layers, dropout, learning_rate, weight_decay)

    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
        history = model.fit(
            X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
            epochs=max_epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose,
        )
        best_val_loss = float(min(history.history["val_loss"]))
    else:
        callbacks = [keras.callbacks.EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True)]
        history = model.fit(
            X_train_scaled, y_train,
            epochs=max_epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose,
        )
        best_val_loss = float("inf")

    return model, scaler, history.history, best_val_loss


# ==========================================================
# Optuna hyperparameter search space
# ==========================================================

def _default_param_space(trial):
    """
    Default Optuna search space: 1-3 hidden layers (16-256 units each),
    dropout, learning rate and weight decay (log-scale), and batch size --
    identical shape to
    `pygstat.regression_kriging_pytorch_gpu._default_param_space`. Pass your
    own ``param_space(trial) -> dict`` with this same shape (``hidden_layers``,
    ``dropout``, ``learning_rate``, ``weight_decay``, ``batch_size``) to
    `DeepRegressionKrigingTF` to customize the search.
    """
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_layers = [
        trial.suggest_categorical(f"n_units_l{i}", [16, 32, 64, 128, 256])
        for i in range(n_layers)
    ]
    return {
        "hidden_layers": hidden_layers,
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    }


class DeepRegressionKrigingTF:
    """
    Deep Regression Kriging with TensorFlow/Keras for the trend model.

    Parameters
    ----------
    hidden_layers : list of int, default=[128, 64, 32]
        Hidden layer widths. Ignored if `tune_hyperparameters=True` (Optuna
        picks this instead; see `param_space`).
    dropout : float, default=0.2
    learning_rate : float, default=1e-3
    batch_size : int, default=32
    max_epochs : int, default=200
        Training budget for the *final* production model (after tuning, if
        `tune_hyperparameters=True`).
    patience : int, default=15
        Early-stopping patience for the final production model.
    use_gpu : bool or 'auto', default='auto'
        Passed through to residual `Variogram` fitting and to
        `OrdinaryKriging` of the residuals (CuPy). Independent of TensorFlow's
        own GPU: the Keras MLP can train on CUDA while residual OK uses
        CuPy, or vice versa.
    variogram_model : str, default='spherical'
    variogram_kwargs : dict, optional
    weight_decay : float, default=0.0
        L2 penalty for the Adam optimizer (``0.0`` disables it).

    tune_hyperparameters : bool, default=False
        If True, run an Optuna hyperparameter search (see module docstring)
        before the final fit, replacing `hidden_layers`, `dropout`,
        `learning_rate`, `weight_decay`, and `batch_size` with the best
        combination found.
    param_space : callable, optional
        ``param_space(trial) -> dict`` returning ``{"hidden_layers": [...],
        "dropout": ..., "learning_rate": ..., "weight_decay": ...,
        "batch_size": ...}`` for one Optuna trial. Defaults to
        :func:`_default_param_space` if not given.
    n_trials : int, default=30
        Number of Optuna trials.
    cv_folds : int, default=3
        Number of cross-validation folds used to score each trial. If <= 1,
        a single train/validation split (`tuning_val_fraction`) is used
        instead.
    tuning_val_fraction : float, default=0.2
        Validation fraction when `cv_folds` <= 1.
    tuning_max_epochs : int, default=100
        Training budget *per trial* (kept short so tuning stays fast; the
        final model is retrained with the full `max_epochs`/`patience`).
    tuning_patience : int, default=10
        Early-stopping patience *per trial*.
    tuning_direction : {'minimize', 'maximize'}, default='minimize'
        Trials are scored by validation RMSE, so this should stay
        `'minimize'` unless `param_space` changes what's returned/optimized.
    sampler, pruner : optuna Sampler / Pruner, optional
        Defaults to `optuna.samplers.TPESampler` and `optuna.pruners.MedianPruner`.
    tuning_timeout : float, optional
        Wall-clock budget (seconds) for the whole search, in addition to `n_trials`.
    tuning_seed : int, default=42
    tuning_verbose : bool, default=False
        If True, show Optuna's own logging/progress bar during the search.

    Attributes (after fitting with `tune_hyperparameters=True`)
    -------------------------------------------------------------
    study_ : optuna.Study
        The completed Optuna study (for `optuna.visualization.*`, custom
        analysis, etc.).
    best_params_ : dict
        The winning trial's resolved hyperparameters (same shape as
        `param_space`'s return value).
    best_score_ : float
        The winning trial's cross-validated RMSE.

    Examples
    --------
    >>> rk = DeepRegressionKrigingTF(tune_hyperparameters=True, n_trials=30, cv_folds=3)
    >>> rk.fit(X_train, y_train, coords_train)
    >>> rk.best_params_
    {'hidden_layers': [128, 64], 'dropout': 0.1, ...}
    >>> y_pred = rk.predict(X_test, coords_test)
    """

    def __init__(
        self,
        hidden_layers=[128, 64, 32],
        dropout=0.2,
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=200,
        patience=15,
        use_gpu="auto",
        variogram_model='spherical',
        variogram_kwargs=None,
        weight_decay=0.0,
        tune_hyperparameters=False,
        param_space=None,
        n_trials=30,
        cv_folds=3,
        tuning_val_fraction=0.2,
        tuning_max_epochs=100,
        tuning_patience=10,
        tuning_direction="minimize",
        sampler=None,
        pruner=None,
        tuning_timeout=None,
        tuning_seed=42,
        tuning_verbose=False,
    ):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)
        self.variogram_model = variogram_model
        self.variogram_kwargs = variogram_kwargs or {}
        self.weight_decay = weight_decay

        self.tune_hyperparameters = tune_hyperparameters
        self.param_space = param_space
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.tuning_val_fraction = tuning_val_fraction
        self.tuning_max_epochs = tuning_max_epochs
        self.tuning_patience = tuning_patience
        self.tuning_direction = tuning_direction
        self.sampler = sampler
        self.pruner = pruner
        self.tuning_timeout = tuning_timeout
        self.tuning_seed = tuning_seed
        self.tuning_verbose = tuning_verbose

        self.is_fitted_ = False

    def _build_model(self, input_dim):
        """Kept for backward compatibility; prefer the module-level `_build_keras_mlp`."""
        return _build_keras_mlp(input_dim, self.hidden_layers, self.dropout, self.learning_rate, self.weight_decay)

    # ------------------------------------------------------------------
    # Optuna hyperparameter search
    # ------------------------------------------------------------------

    def _cv_splits(self, n_samples):
        if self.cv_folds is None or self.cv_folds <= 1:
            from sklearn.model_selection import train_test_split
            idx = np.arange(n_samples)
            train_idx, val_idx = train_test_split(
                idx, test_size=self.tuning_val_fraction, random_state=self.tuning_seed
            )
            return [(train_idx, val_idx)]
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.tuning_seed)
        return list(kf.split(np.arange(n_samples)))

    def _run_optuna_search(self, X, y):
        try:
            import optuna
        except ImportError as e:
            raise ImportError(
                "tune_hyperparameters=True requires the optional 'optuna' package. "
                "Install it with: pip install optuna  (or: pip install pygstat[optuna])"
            ) from e

        param_space_fn = self.param_space or _default_param_space
        splits = self._cv_splits(len(X))

        def objective(trial):
            params = param_space_fn(trial)
            trial.set_user_attr("resolved_params", params)

            fold_rmses = []
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                keras.backend.clear_session()
                _, _, _, best_val_mse = _train_keras_mlp(
                    X[train_idx], y[train_idx], X[val_idx], y[val_idx],
                    hidden_layers=params["hidden_layers"], dropout=params["dropout"],
                    learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
                    batch_size=params["batch_size"], max_epochs=self.tuning_max_epochs,
                    patience=self.tuning_patience, verbose=0,
                )
                fold_rmses.append(np.sqrt(best_val_mse))

                # Report the running mean RMSE across folds completed so far,
                # so the pruner can cut a clearly weak trial short before
                # every fold is trained.
                trial.report(float(np.mean(fold_rmses)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(fold_rmses))

        sampler = self.sampler or optuna.samplers.TPESampler(seed=self.tuning_seed)
        pruner = self.pruner or optuna.pruners.MedianPruner()
        optuna.logging.set_verbosity(optuna.logging.INFO if self.tuning_verbose else optuna.logging.WARNING)

        study = optuna.create_study(direction=self.tuning_direction, sampler=sampler, pruner=pruner)
        study.optimize(
            objective, n_trials=self.n_trials, timeout=self.tuning_timeout,
            show_progress_bar=self.tuning_verbose,
        )

        self.study_ = study
        self.best_params_ = study.best_trial.user_attrs["resolved_params"]
        self.best_score_ = study.best_value
        return self.best_params_

    # ------------------------------------------------------------------

    def fit(self, X, y, coords, X_val=None, y_val=None, verbose=0):
        """
        Fit the Keras trend model (optionally Optuna-tuned first) + kriging
        on residuals.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        coords : array-like, shape (n_samples, 2)
        X_val, y_val : array-like, optional
            Held-out validation data for early stopping of the *final*
            production fit (independent of any Optuna tuning, which always
            uses its own `cv_folds`/`tuning_val_fraction` split of `X, y`).
        verbose : {0, 1, 2}, default=0
            Keras' own verbosity level for the final production fit
            (0=silent, 1=progress bar, 2=one line per epoch). Tuning trials
            always run silently regardless of this setting.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        coords = np.asarray(coords)

        print("TensorFlow GPU devices:", tf.config.list_physical_devices('GPU'))

        if self.tune_hyperparameters:
            params = self._run_optuna_search(X, y)
        else:
            params = {
                "hidden_layers": self.hidden_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
            }

        keras.backend.clear_session()
        self.model_, self.scaler_X_, self.history_, _ = _train_keras_mlp(
            X, y, X_val, y_val,
            hidden_layers=params["hidden_layers"], dropout=params["dropout"],
            learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
            batch_size=params["batch_size"], max_epochs=self.max_epochs, patience=self.patience,
            verbose=verbose,
        )
        self.fitted_params_ = params

        # Get residuals
        X_scaled = self.scaler_X_.transform(X)
        y_pred_reg = self.model_.predict(X_scaled, verbose=0).flatten()
        residuals = y - y_pred_reg

        # Fit variogram and kriging
        vg_kwargs = {'model': self.variogram_model, 'estimator': 'matheron'}
        vg_kwargs.update(self.variogram_kwargs)
        vg_kwargs['use_gpu'] = self.use_gpu
        self.variogram_ = Variogram(coords, residuals, **vg_kwargs)
        self.variogram_.fit()

        self.krige_ = OrdinaryKriging(self.variogram_, use_gpu=self.use_gpu)
        self.krige_.fit(coords, residuals)

        self.coords_ = coords
        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True
        return self

    def predict(self, X_pred, coords_pred, return_std=False):
        """Predict using DNN + kriged residuals."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet.")

        X_pred_scaled = self.scaler_X_.transform(X_pred)
        reg_pred = self.model_.predict(X_pred_scaled, verbose=0).flatten()

        if return_std:
            resid_pred, resid_std = self.krige_.predict(coords_pred, return_variance=True)
            return reg_pred + resid_pred, resid_std
        else:
            resid_pred = self.krige_.predict(coords_pred)
            return reg_pred + resid_pred

    def plot_losses(self):
        """Plot training and validation losses."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(self.history_['loss'], label='Training Loss')
        if 'val_loss' in self.history_:
            plt.plot(self.history_['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_optuna_history(self):
        """Plot the Optuna optimization history (requires `tune_hyperparameters=True`
        to have been used, and `optuna`'s matplotlib integration)."""
        if not hasattr(self, "study_"):
            raise RuntimeError(
                "No Optuna study to plot -- fit with tune_hyperparameters=True first."
            )
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history
        plot_optimization_history(self.study_)
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        hidden = self.fitted_params_["hidden_layers"] if hasattr(self, "fitted_params_") else self.hidden_layers
        tuned = f", tuned (best RMSE={self.best_score_:.4g})" if hasattr(self, "best_score_") else ""
        return f"DeepRegressionKrigingTF(hidden_layers={hidden}, variogram_model='{self.variogram_model}'{tuned})"
