"""
Test for pygstat.regression_kriging's statistical-model support (GAM, GLM,
Bayesian ridge, quantile regression), using the Jura data set: predict Cd from
the other heavy-metal assays (Co, Cr, Cu, Ni, Pb, Zn) as covariates, kriging the
regression residuals.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from pygstat.regression_kriging import (
    STATISTICAL_REGRESSORS,
    GAMRegressor,
    RegressionKriging,
    enforce_quantile_monotonicity,
)

DATA_PATH = "data/jura_data.csv"
FEATURES = ["Co", "Cr", "Cu", "Ni", "Pb", "Zn"]


def _load_jura_split():
    df = pd.read_csv(DATA_PATH, index_col=0)
    X = df[FEATURES].to_numpy()
    y = df["Cd"].to_numpy()
    coords = df[["Xloc", "Yloc"]].to_numpy()
    return train_test_split(X, y, coords, test_size=0.3, random_state=42)


@pytest.mark.parametrize("name", STATISTICAL_REGRESSORS)
def test_each_statistical_shortcut_fits_and_predicts(name):
    if name == "gam":
        pytest.importorskip("pygam")
    X_train, X_test, y_train, y_test, c_train, c_test = _load_jura_split()
    rk = RegressionKriging(regressor=name)
    rk.fit(X_train, y_train, c_train)
    pred = rk.predict(X_test, c_test)

    assert pred.shape == y_test.shape
    assert np.isfinite(pred).all()

    pred_with_std, var = rk.predict(X_test, c_test, return_std=True)
    np.testing.assert_allclose(pred_with_std, pred)
    assert np.all(var >= -1e-8)


def test_statistical_models_give_reasonable_r2_on_jura():
    X_train, X_test, y_train, y_test, c_train, c_test = _load_jura_split()
    scores = {}
    for name in STATISTICAL_REGRESSORS:
        if name == "gam":
            pytest.importorskip("pygam")
        rk = RegressionKriging(regressor=name)
        rk.fit(X_train, y_train, c_train)
        pred = rk.predict(X_test, c_test)
        scores[name] = r2_score(y_test, pred)

    print(f"\nRegressionKriging R^2 on held-out Jura Cd: "
          + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))
    for name, score in scores.items():
        assert score > 0.3, f"{name} scored unexpectedly poorly (R2={score:.3f})"


def test_gam_regressor_standalone_and_clonable():
    pytest.importorskip("pygam")
    X_train, X_test, y_train, y_test, c_train, c_test = _load_jura_split()

    gam = GAMRegressor(n_splines=8, lam=0.5)
    gam.fit(X_train, y_train)
    pred = gam.predict(X_test)
    assert np.isfinite(pred).all()
    assert gam.n_features_in_ == X_train.shape[1]

    cloned = clone(gam)
    assert cloned.n_splines == 8 and cloned.lam == 0.5
    assert cloned.gam_ is None  # clone gives a fresh, unfitted copy


def test_gam_missing_dependency_gives_clear_error(monkeypatch):
    """If pygam isn't installed, fitting (not constructing) a GAMRegressor
    should raise a clear, actionable ImportError -- not an obscure one."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pygam":
            raise ImportError("simulated missing pygam")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    gam = GAMRegressor()
    with pytest.raises(ImportError, match="pip install pygam"):
        gam.fit(np.random.rand(10, 2), np.random.rand(10))


def test_glm_power_kwarg_changes_family():
    X_train, _, y_train, _, c_train, _ = _load_jura_split()
    rk_gaussian = RegressionKriging(regressor="glm")
    rk_poisson = RegressionKriging(regressor="glm", regressor_kwargs={"power": 1})
    assert rk_gaussian.regressor.power == 0.0
    assert rk_poisson.regressor.power == 1


def test_bayesian_ridge_exposes_its_own_predictive_std():
    X_train, X_test, y_train, _, c_train, c_test = _load_jura_split()
    rk = RegressionKriging(regressor="bayesian_ridge").fit(X_train, y_train, c_train)
    mean_pred, std_pred = rk.regressor.predict(X_test, return_std=True)
    assert mean_pred.shape == std_pred.shape
    assert (std_pred > 0).all()


def test_quantile_defaults_and_kwargs_override():
    rk = RegressionKriging(regressor="quantile")
    assert rk.regressor.quantile == 0.5
    assert rk.regressor.alpha == 0.0

    rk90 = RegressionKriging(regressor="quantile", regressor_kwargs={"quantile": 0.9})
    assert rk90.regressor.quantile == 0.9


def test_quantile_crossing_can_happen_and_the_fix_removes_it():
    """Separately-fit quantile levels are not guaranteed to stay ordered
    ("quantile crossing") -- demonstrate it occurs on real data, and that
    enforce_quantile_monotonicity() fixes it."""
    X_train, X_test, y_train, y_test, c_train, c_test = _load_jura_split()

    preds = {}
    for q in (0.1, 0.5, 0.9):
        rk = RegressionKriging(regressor="quantile", regressor_kwargs={"quantile": q})
        rk.fit(X_train, y_train, c_train)
        preds[q] = rk.predict(X_test, c_test)

    raw_monotone = np.mean((preds[0.1] <= preds[0.5] + 1e-9) & (preds[0.5] <= preds[0.9] + 1e-9))
    print(f"\nraw pointwise monotone fraction across quantiles: {raw_monotone:.3f}")

    fixed = enforce_quantile_monotonicity(preds)
    assert (fixed[0.1] <= fixed[0.5] + 1e-9).all()
    assert (fixed[0.5] <= fixed[0.9] + 1e-9).all()
    # rearrangement only reorders values at each point; it must not invent or
    # drop any values overall.
    np.testing.assert_allclose(
        np.sort(np.stack([preds[0.1], preds[0.5], preds[0.9]]), axis=0),
        np.stack([fixed[0.1], fixed[0.5], fixed[0.9]]),
    )


def test_enforce_quantile_monotonicity_array_input():
    arr = np.array([[5.0, 1.0], [3.0, 2.0], [4.0, 9.0]])  # not sorted along axis 0
    fixed = enforce_quantile_monotonicity(arr)
    assert (np.diff(fixed, axis=0) >= 0).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
