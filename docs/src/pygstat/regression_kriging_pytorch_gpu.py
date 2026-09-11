# src/pygstat/regression_kriging_pytorch_gpu.py
"""
Deep Regression Kriging: a PyTorch MLP trend model + kriging of residuals,
optionally GPU-accelerated.

The regression step trains a feed-forward neural network on auxiliary
predictors (covariates); kriging then interpolates the spatially-correlated
residuals left over, and the two are added back together at prediction time.

Hyperparameter tuning
----------------------
Pass ``tune_hyperparameters=True`` to search the network's architecture and
optimizer settings with `Optuna <https://optuna.org>`_ before the final fit:

    DeepRegressionKriging(tune_hyperparameters=True, n_trials=30, cv_folds=3)

By default this searches the number/width of hidden layers, dropout, learning
rate, weight decay, and batch size (:func:`_default_param_space`), scoring each
trial by K-fold cross-validated validation RMSE (with median pruning of clearly
weak trials), and refits one final model with the best hyperparameters found on
the *full* training set (using this instance's own `max_epochs`/`patience`, not
the smaller tuning budget). Pass your own ``param_space(trial) -> dict`` for a
custom search space -- see :func:`_default_param_space` for the expected shape.
Requires the optional ``optuna`` package (``pip install optuna``, or
``pip install pygstat[optuna]``).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from .utils.backend import resolve_torch_device, resolve_cupy_use_gpu
from .core.variogram import Variogram
from .core.kriging import OrdinaryKriging
# matplotlib is imported lazily inside plot_losses()/plot_optuna_history()
# only -- it is not required for fitting/predicting/tuning, and is not a
# dependency of the `pytorch` extra (only of the separate `plot` extra).

__all__ = ["DeepRegressionKriging"]


# ==========================================================
# Model construction and training (module-level so the Optuna
# tuning objective and the final production fit share one code path)
# ==========================================================

def _build_mlp(input_dim, hidden_layers, dropout):
    layers = [nn.Linear(input_dim, hidden_layers[0]), nn.ReLU(), nn.Dropout(dropout)]
    for i in range(1, len(hidden_layers)):
        layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_layers[-1], 1))
    return nn.Sequential(*layers)


def _train_mlp(
    X_train, y_train, X_val, y_val, device,
    hidden_layers, dropout, learning_rate, weight_decay, batch_size,
    max_epochs, patience, verbose=True,
):
    """
    Fit one MLP (with early stopping on validation loss, if `X_val`/`y_val`
    are given). Returns ``(model, scaler, train_losses, val_losses,
    best_val_loss)`` -- `scaler` is the `StandardScaler` fit on `X_train`
    only, `val_losses`/`best_val_loss` are ``None``/``inf`` if no validation
    data was given.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

    model = _build_mlp(X_train.shape[1], hidden_layers, dropout).to(device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = model.state_dict()

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            # squeeze(-1), not squeeze(): a plain squeeze() also drops the
            # batch dimension when the last batch has exactly 1 sample,
            # silently mismatching batch_y's shape via broadcasting.
            outputs = model(batch_X).squeeze(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X).squeeze(-1)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            if avg_val_loss < best_val_loss - 1e-6:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if verbose:
            msg = f"Epoch {epoch + 1}/{max_epochs} - Train Loss: {avg_train_loss:.6f}"
            if avg_val_loss is not None:
                msg += f", Val Loss: {avg_val_loss:.6f}"
            print(msg)

    model.load_state_dict(best_model_state)
    return model, scaler, train_losses, (val_losses or None), best_val_loss


# ==========================================================
# Optuna hyperparameter search space
# ==========================================================

def _default_param_space(trial):
    """
    Default Optuna search space: 1-3 hidden layers (16-256 units each),
    dropout, learning rate and weight decay (log-scale), and batch size.
    Pass your own ``param_space(trial) -> dict`` with this same shape
    (``hidden_layers``, ``dropout``, ``learning_rate``, ``weight_decay``,
    ``batch_size``) to `DeepRegressionKriging` to customize the search.
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


class DeepRegressionKriging:
    """
    Deep Regression Kriging: a PyTorch MLP trend model + kriging of residuals.

    Parameters
    ----------
    hidden_layers : list of int, default=[128, 64, 32]
        Hidden layer widths. Ignored if `tune_hyperparameters=True` (Optuna
        picks this instead; see `param_space`).
    dropout : float, default=0.2
    learning_rate : float, default=1e-3
    weight_decay : float, default=0.0
        L2 penalty for the Adam optimizer.
    batch_size : int, default=64
    max_epochs : int, default=500
        Training budget for the *final* production model (after tuning, if
        `tune_hyperparameters=True`).
    patience : int, default=20
        Early-stopping patience for the final production model.
    device : {'auto', 'cpu', 'cuda', ...}, default='auto'
        Device for the PyTorch MLP trend (training and inference).
    use_gpu : bool or 'auto', default='auto'
        Passed through to residual `Variogram` fitting and to
        `OrdinaryKriging` of the residuals (CuPy). Independent of
        `device`: the MLP can train on CUDA while residual OK uses
        CuPy, or vice versa.
    variogram_model : str, default='spherical'
    variogram_kwargs : dict, optional

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
    >>> rk = DeepRegressionKriging(tune_hyperparameters=True, n_trials=30, cv_folds=3)
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
        weight_decay=0.0,
        batch_size=64,
        max_epochs=500,
        patience=20,
        device='auto',
        use_gpu="auto",
        variogram_model='spherical',
        variogram_kwargs=None,
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
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)
        self.variogram_model = variogram_model
        self.variogram_kwargs = variogram_kwargs or {}

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
        """Kept for backward compatibility; prefer the module-level `_build_mlp`."""
        return _build_mlp(input_dim, self.hidden_layers, self.dropout)

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
                _, _, _, _, best_val_mse = _train_mlp(
                    X[train_idx], y[train_idx], X[val_idx], y[val_idx], self.device_,
                    hidden_layers=params["hidden_layers"], dropout=params["dropout"],
                    learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
                    batch_size=params["batch_size"], max_epochs=self.tuning_max_epochs,
                    patience=self.tuning_patience, verbose=False,
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

    def fit(self, X, y, coords, X_val=None, y_val=None):
        X = np.asarray(X)
        y = np.asarray(y)
        coords = np.asarray(coords)

        self.device_ = resolve_torch_device(self.device)

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

        self.model_, self.scaler_X_, self.train_losses_, self.val_losses_, _ = _train_mlp(
            X, y, X_val, y_val, self.device_,
            hidden_layers=params["hidden_layers"], dropout=params["dropout"],
            learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
            batch_size=params["batch_size"], max_epochs=self.max_epochs, patience=self.patience,
            verbose=True,
        )
        self.fitted_params_ = params

        # Get predictions and residuals
        self.model_.eval()
        with torch.no_grad():
            X_full_tensor = torch.FloatTensor(self.scaler_X_.transform(X)).to(self.device_)
            y_pred_reg = self.model_(X_full_tensor).cpu().numpy().flatten()
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
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet.")

        X_pred_scaled = self.scaler_X_.transform(X_pred)
        X_pred_tensor = torch.FloatTensor(X_pred_scaled).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            reg_pred = self.model_(X_pred_tensor).cpu().numpy().flatten()

        if return_std:
            resid_pred, resid_std = self.krige_.predict(coords_pred, return_variance=True)
            return reg_pred + resid_pred, resid_std
        else:
            resid_pred = self.krige_.predict(coords_pred)
            return reg_pred + resid_pred

    def plot_losses(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses_, label='Training Loss')
        if self.val_losses_ is not None:
            plt.plot(self.val_losses_, label='Validation Loss')
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
        return (
            f"DeepRegressionKriging(hidden_layers={hidden}, "
            f"variogram_model='{self.variogram_model}'{tuned})"
        )
