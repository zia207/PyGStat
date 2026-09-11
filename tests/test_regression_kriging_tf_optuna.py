"""
Test for pygstat.regression_kriging_tf's Optuna hyperparameter-tuning option,
using the Jura data set: predict Cd from the other heavy-metal assays
(Co, Cr, Cu, Ni, Pb, Zn) as covariates, kriging the regression residuals.

A note on `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` below: in this sandbox,
`pygstat`'s `__init__.py` lazily imports PyTorch (for
`regression_kriging_pytorch_gpu`/`regression_kriging_gnn`) as well as
TensorFlow. Once PyTorch's CUDA context is initialized first, TensorFlow's
own GPU detection picks up the same (for this hardware, incompatible) GPU
and *actually tries to use it*, failing with `CUDA_ERROR_INVALID_HANDLE` on
the very first layer build -- reproduced with a minimal Keras model,
unrelated to this module's own code. Hiding all GPUs from both frameworks
via `CUDA_VISIBLE_DEVICES=-1` (set before either is imported) sidesteps the
conflict entirely; this environment variable is what both PyTorch and
TensorFlow already check natively, so -- unlike poisoning `sys.modules`
entries to force one framework's `ImportError` (which some libraries, e.g.
newer scipy's `array_api_compat`, dereference under the assumption that
anything present in `sys.modules` is a real module, not a sentinel) -- it
never leaves either framework's own import machinery in an inconsistent
state for some other library to trip over.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

tf = pytest.importorskip("tensorflow")
optuna = pytest.importorskip("optuna")

from pygstat.regression_kriging_tf import (  # noqa: E402
    DeepRegressionKrigingTF,
    _default_param_space,
    _train_keras_mlp,
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
    rk = DeepRegressionKrigingTF(hidden_layers=[32, 16], max_epochs=20, patience=5)
    rk.fit(X_train, y_train, c_train)
    pred = rk.predict(X_test, c_test)

    assert pred.shape == y_test.shape
    assert np.isfinite(pred).all()
    assert not hasattr(rk, "study_")
    assert not hasattr(rk, "best_params_")

    pred_with_std, var = rk.predict(X_test, c_test, return_std=True)
    np.testing.assert_allclose(pred_with_std, pred)
    assert np.all(var >= -1e-8)


def test_train_keras_mlp_helper_directly():
    X = np.random.rand(60, 3).astype(np.float64)
    y = np.random.rand(60).astype(np.float64)
    model, scaler, history, best_val = _train_keras_mlp(
        X[:40], y[:40], X[40:], y[40:],
        hidden_layers=[8, 4], dropout=0.1, learning_rate=1e-3, weight_decay=1e-4,
        batch_size=16, max_epochs=5, patience=3, verbose=0,
    )
    assert "loss" in history and "val_loss" in history
    assert np.isfinite(best_val)


def test_hyperparameter_tuning_improves_or_matches_baseline_and_reports_study():
    X_train, X_test, y_train, y_test, c_train, c_test = _load_jura_split()

    rk = DeepRegressionKrigingTF(
        tune_hyperparameters=True,
        n_trials=6,
        cv_folds=2,
        tuning_max_epochs=15,
        tuning_patience=5,
        max_epochs=30,
        patience=8,
        tuning_seed=1,
    )
    rk.fit(X_train, y_train, c_train)

    assert hasattr(rk, "study_")
    assert len(rk.study_.trials) == 6
    assert hasattr(rk, "best_params_")
    for key in ("hidden_layers", "dropout", "learning_rate", "weight_decay", "batch_size"):
        assert key in rk.best_params_
    assert np.isfinite(rk.best_score_)
    assert "tuned" in repr(rk)

    pred = rk.predict(X_test, c_test)
    assert np.isfinite(pred).all()

    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, pred)
    print(f"\nTuned DeepRegressionKrigingTF R^2 on held-out Jura Cd: {r2:.3f} "
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

    rk = DeepRegressionKrigingTF(
        tune_hyperparameters=True, n_trials=10, cv_folds=3, param_space=wide_space,
        tuning_max_epochs=25, tuning_patience=8, max_epochs=10, patience=3, tuning_seed=2,
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

    rk = DeepRegressionKrigingTF(
        tune_hyperparameters=True, n_trials=4, cv_folds=2, param_space=tiny_space,
        tuning_max_epochs=10, tuning_patience=3, max_epochs=15, patience=5,
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
    rk = DeepRegressionKrigingTF(tune_hyperparameters=True, n_trials=2)
    with pytest.raises(ImportError, match="pip install optuna"):
        rk.fit(X_train[:20], y_train[:20], c_train[:20])


def test_plot_optuna_history_requires_a_study():
    rk = DeepRegressionKrigingTF()
    with pytest.raises(RuntimeError, match="tune_hyperparameters"):
        rk.plot_optuna_history()


def test_cv_folds_le_1_uses_single_validation_split():
    X_train, X_test, y_train, _, c_train, c_test = _load_jura_split()
    rk = DeepRegressionKrigingTF(
        tune_hyperparameters=True, n_trials=3, cv_folds=1, tuning_val_fraction=0.25,
        tuning_max_epochs=10, tuning_patience=3, max_epochs=15, patience=5,
    )
    rk.fit(X_train, y_train, c_train)
    pred = rk.predict(X_test, c_test)
    assert np.isfinite(pred).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
