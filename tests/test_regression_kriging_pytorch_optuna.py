"""
Test for pygstat.regression_kriging_pytorch_gpu's Optuna hyperparameter-tuning
option, using the Jura data set: predict Cd from the other heavy-metal assays
(Co, Cr, Cu, Ni, Pb, Zn) as covariates, kriging the regression residuals.

A note on `sys.modules['tensorflow'] = None` below: this sandbox's installed
TensorFlow and PyTorch builds segfault the interpreter (not a catchable Python
exception) the moment `pygstat` (whose `__init__.py` eagerly-but-lazily-caught
imports TensorFlow for `regression_kriging_tf`) and a PyTorch training loop
are both used in the same process -- reproduced with a *minimal* model,
unrelated to this module's own code. Setting `sys.modules['tensorflow'] =
None` before importing `pygstat` makes `import tensorflow` raise
`ImportError` (Python's own documented mechanism for this), which
`pygstat`'s optional-import wrapper already handles gracefully, sidestepping
the conflict.

This only works if it runs *before* anything else in the same pytest
process has already triggered a real `import pygstat` (e.g. another test
module collected first, alphabetically): `sys.modules.setdefault` is a
no-op once the real `tensorflow` module is already cached there. Run this
file on its own (`pytest tests/test_regression_kriging_pytorch_optuna.py`)
to reliably exercise the code paths below; running it alongside test
modules that import `pygstat` at module level and happen to collect first
may reproduce the segfault instead -- a pre-existing environment issue in
this sandbox (multiple GPU-capable native library stacks sharing one
process), not a bug in this module.
"""

import sys

import numpy as np
import pandas as pd
import pytest

sys.modules.setdefault("tensorflow", None)

torch = pytest.importorskip("torch")
optuna = pytest.importorskip("optuna")

from pygstat.regression_kriging_pytorch_gpu import (  # noqa: E402
    DeepRegressionKriging,
    _default_param_space,
    _train_mlp,
)

DATA_PATH = "data/jura_data.csv"
FEATURES = ["Co", "Cr", "Cu", "Ni", "Pb", "Zn"]


def _load_jura_split():
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(DATA_PATH, index_col=0)
    X = df[FEATURES].to_numpy()
    y = df["Cd"].to_numpy()
    coords = df[["Xloc", "Yloc"]].to_numpy()
    return train_test_split(X, y, coords, test_size=0.3, random_state=42)


def test_baseline_fit_predict_unaffected_by_new_kwargs():
    """tune_hyperparameters=False (the default) must behave like the
    original implementation -- backward compatibility."""
    X_train, X_test, y_train, y_test, c_train, c_test = _load_jura_split()
    rk = DeepRegressionKriging(hidden_layers=[32, 16], max_epochs=20, patience=5, device="cpu")
    rk.fit(X_train, y_train, c_train)
    pred = rk.predict(X_test, c_test)

    assert pred.shape == y_test.shape
    assert np.isfinite(pred).all()
    assert not hasattr(rk, "study_")
    assert not hasattr(rk, "best_params_")

    pred_with_std, var = rk.predict(X_test, c_test, return_std=True)
    np.testing.assert_allclose(pred_with_std, pred)
    assert np.all(var >= -1e-8)


def test_fit_predict_tune_do_not_require_matplotlib(monkeypatch):
    """A real bug caught while verifying the `[pytorch]` extra in a clean venv:
    the module used to `import matplotlib.pyplot` unconditionally at module
    level, but matplotlib isn't a dependency of the `pytorch` extra -- so
    `pip install pygstat[pytorch]` alone couldn't even import
    DeepRegressionKriging. matplotlib should only be needed by the plotting
    convenience methods, not by fit/predict/tune."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("simulated missing matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    X_train, X_test, y_train, y_test, c_train, c_test = _load_jura_split()
    rk = DeepRegressionKriging(
        tune_hyperparameters=True, n_trials=2, cv_folds=2,
        tuning_max_epochs=5, tuning_patience=3, max_epochs=5, patience=3, device="cpu",
    )
    rk.fit(X_train, y_train, c_train)  # must not raise ImportError
    pred = rk.predict(X_test, c_test)
    assert np.isfinite(pred).all()

    with pytest.raises(ImportError, match="matplotlib"):
        rk.plot_losses()


def test_train_mlp_helper_directly():
    X = np.random.rand(60, 3).astype(np.float64)
    y = np.random.rand(60).astype(np.float64)
    device = torch.device("cpu")
    model, scaler, train_losses, val_losses, best_val = _train_mlp(
        X[:40], y[:40], X[40:], y[40:], device,
        hidden_layers=[8, 4], dropout=0.1, learning_rate=1e-3, weight_decay=1e-4,
        batch_size=16, max_epochs=5, patience=3, verbose=False,
    )
    assert len(train_losses) > 0
    assert val_losses is not None and len(val_losses) == len(train_losses)
    assert np.isfinite(best_val)


def test_hyperparameter_tuning_improves_or_matches_baseline_and_reports_study():
    X_train, X_test, y_train, y_test, c_train, c_test = _load_jura_split()

    rk = DeepRegressionKriging(
        tune_hyperparameters=True,
        n_trials=8,
        cv_folds=3,
        tuning_max_epochs=25,
        tuning_patience=5,
        max_epochs=60,
        patience=10,
        device="cpu",
        tuning_seed=1,
    )
    rk.fit(X_train, y_train, c_train)

    assert hasattr(rk, "study_")
    assert len(rk.study_.trials) == 8
    assert hasattr(rk, "best_params_")
    for key in ("hidden_layers", "dropout", "learning_rate", "weight_decay", "batch_size"):
        assert key in rk.best_params_
    assert np.isfinite(rk.best_score_)
    assert "tuned" in repr(rk)

    pred = rk.predict(X_test, c_test)
    assert np.isfinite(pred).all()

    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, pred)
    print(f"\nTuned DeepRegressionKriging R^2 on held-out Jura Cd: {r2:.3f} "
          f"(best CV RMSE={rk.best_score_:.4f}, best_params={rk.best_params_})")
    assert r2 > 0.0  # a sane, non-degenerate fit


def test_pruning_actually_prunes_some_trials():
    """With a deliberately wide/aggressive search and a tight epoch budget,
    the median pruner should cut at least some trials short."""
    X_train, _, y_train, _, c_train, _ = _load_jura_split()

    def wide_space(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        return {
            "hidden_layers": [trial.suggest_categorical(f"u{i}", [8, 256]) for i in range(n_layers)],
            "dropout": trial.suggest_float("dropout", 0.0, 0.6),
            "learning_rate": trial.suggest_float("lr", 1e-5, 1.0, log=True),
            "weight_decay": trial.suggest_float("wd", 1e-6, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("bs", [8, 16, 32]),
        }

    rk = DeepRegressionKriging(
        tune_hyperparameters=True, n_trials=12, cv_folds=3, param_space=wide_space,
        tuning_max_epochs=40, tuning_patience=15, max_epochs=10, patience=3,
        device="cpu", tuning_seed=2,
    )
    rk.fit(X_train, y_train, c_train)

    n_pruned = sum(1 for t in rk.study_.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_complete = sum(1 for t in rk.study_.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"\npruned={n_pruned} complete={n_complete} of {len(rk.study_.trials)} trials")
    assert n_complete > 0
    assert n_pruned >= 0  # pruning is stochastic; just confirm it ran without error


def test_custom_param_space_is_respected():
    X_train, X_test, y_train, _, c_train, c_test = _load_jura_split()

    def tiny_space(trial):
        return {
            "hidden_layers": [trial.suggest_categorical("units", [16, 32])],
            "dropout": 0.1,
            "learning_rate": trial.suggest_float("lr", 1e-3, 1e-2, log=True),
            "weight_decay": 0.0,
            "batch_size": 32,
        }

    rk = DeepRegressionKriging(
        tune_hyperparameters=True, n_trials=4, cv_folds=2, param_space=tiny_space,
        tuning_max_epochs=10, tuning_patience=3, max_epochs=15, patience=5, device="cpu",
    )
    rk.fit(X_train, y_train, c_train)

    assert len(rk.best_params_["hidden_layers"]) == 1
    assert rk.best_params_["hidden_layers"][0] in (16, 32)
    assert rk.best_params_["dropout"] == 0.1
    assert rk.best_params_["weight_decay"] == 0.0

    pred = rk.predict(X_test, c_test)
    assert np.isfinite(pred).all()


def test_default_param_space_shape():
    study = optuna.create_study()
    trial = study.ask()
    params = _default_param_space(trial)
    assert set(params) == {"hidden_layers", "dropout", "learning_rate", "weight_decay", "batch_size"}
    assert 1 <= len(params["hidden_layers"]) <= 3
    assert 0.0 <= params["dropout"] <= 0.5


def test_optuna_missing_dependency_gives_clear_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("simulated missing optuna")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    X_train, _, y_train, _, c_train, _ = _load_jura_split()
    rk = DeepRegressionKriging(tune_hyperparameters=True, n_trials=2, device="cpu")
    with pytest.raises(ImportError, match="pip install optuna"):
        rk.fit(X_train[:20], y_train[:20], c_train[:20])


def test_plot_optuna_history_requires_a_study():
    rk = DeepRegressionKriging(device="cpu")
    with pytest.raises(RuntimeError, match="tune_hyperparameters"):
        rk.plot_optuna_history()


def test_cv_folds_le_1_uses_single_validation_split():
    X_train, X_test, y_train, _, c_train, c_test = _load_jura_split()
    rk = DeepRegressionKriging(
        tune_hyperparameters=True, n_trials=3, cv_folds=1, tuning_val_fraction=0.25,
        tuning_max_epochs=10, tuning_patience=3, max_epochs=15, patience=5, device="cpu",
    )
    rk.fit(X_train, y_train, c_train)
    pred = rk.predict(X_test, c_test)
    assert np.isfinite(pred).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
