"""
Test for pygstat.regression_kriging's ensemble-model support (random forest,
gradient boosting, bagging, stacking), using the Jura data set: predict Cd from
the other heavy-metal assays (Co, Cr, Cu, Ni, Pb, Zn) as covariates, kriging the
regression residuals.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from pygstat.regression_kriging import ENSEMBLE_REGRESSORS, RegressionKriging

DATA_PATH = "data/jura_data.csv"
FEATURES = ["Co", "Cr", "Cu", "Ni", "Pb", "Zn"]


def _load_jura_split():
    df = pd.read_csv(DATA_PATH, index_col=0)
    X = df[FEATURES].to_numpy()
    y = df["Cd"].to_numpy()
    coords = df[["Xloc", "Yloc"]].to_numpy()
    return train_test_split(X, y, coords, test_size=0.3, random_state=42)


@pytest.mark.parametrize("name", ENSEMBLE_REGRESSORS)
def test_each_ensemble_shortcut_fits_and_predicts(name):
    X_train, X_test, y_train, y_test, c_train, c_test = _load_jura_split()
    rk = RegressionKriging(regressor=name)
    rk.fit(X_train, y_train, c_train)
    pred = rk.predict(X_test, c_test)

    assert pred.shape == y_test.shape
    assert np.isfinite(pred).all()

    pred_with_std, var = rk.predict(X_test, c_test, return_std=True)
    np.testing.assert_allclose(pred_with_std, pred)
    assert np.all(var >= -1e-8)


def test_ensembles_beat_or_match_plain_linear_on_jura():
    """Not a strict requirement of every ensemble on every run, but at least
    one of them (and typically stacking, which can fall back toward its
    strongest base learner) should not do meaningfully worse than plain
    linear regression kriging on this real, moderately-nonlinear data set."""
    X_train, X_test, y_train, y_test, c_train, c_test = _load_jura_split()

    scores = {}
    for name in ENSEMBLE_REGRESSORS:
        rk = RegressionKriging(regressor=name)
        rk.fit(X_train, y_train, c_train)
        pred = rk.predict(X_test, c_test)
        scores[name] = r2_score(y_test, pred)

    print(f"\nRegressionKriging R^2 on held-out Jura Cd: "
          + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))

    assert max(scores[n] for n in ENSEMBLE_REGRESSORS if n != "linear") >= scores["linear"] - 0.05


def test_feature_importances_available_for_tree_ensembles():
    X_train, _, y_train, _, c_train, _ = _load_jura_split()
    for name in ("random_forest", "gradient_boosting"):
        rk = RegressionKriging(regressor=name, regressor_kwargs={"n_estimators": 50})
        rk.fit(X_train, y_train, c_train)
        importances = rk.feature_importances_
        assert len(importances) == len(FEATURES)
        assert np.isclose(importances.sum(), 1.0, atol=1e-6)


def test_feature_importances_raises_for_linear_and_unfitted():
    rk = RegressionKriging(regressor="linear")
    with pytest.raises(RuntimeError):
        rk.feature_importances_  # not fitted yet

    X_train, _, y_train, _, c_train, _ = _load_jura_split()
    rk.fit(X_train, y_train, c_train)
    with pytest.raises(AttributeError):
        rk.feature_importances_  # LinearRegression has no feature_importances_


def test_regressor_kwargs_override_defaults():
    rk = RegressionKriging(regressor="random_forest", regressor_kwargs={"n_estimators": 7, "max_depth": 2})
    assert rk.regressor.n_estimators == 7
    assert rk.regressor.max_depth == 2


def test_stacking_default_and_custom_final_estimator():
    rk = RegressionKriging(regressor="stacking")
    assert len(rk.regressor.estimators) == 3  # random_forest, gradient_boosting, linear
    assert type(rk.regressor.final_estimator).__name__ == "RidgeCV"

    rk_custom = RegressionKriging(regressor="stacking", regressor_kwargs={"final_estimator": Lasso()})
    assert isinstance(rk_custom.regressor.final_estimator, Lasso)


def test_passing_a_prebuilt_estimator_instance_still_works():
    """Backward compatibility: any scikit-learn-compatible estimator instance
    -- including ones with no pygstat shortcut -- must work exactly as before."""
    X_train, X_test, y_train, y_test, c_train, c_test = _load_jura_split()
    rk = RegressionKriging(regressor=ExtraTreesRegressor(n_estimators=50, random_state=1))
    rk.fit(X_train, y_train, c_train)
    pred = rk.predict(X_test, c_test)
    assert np.isfinite(pred).all()
    assert isinstance(rk.regressor, ExtraTreesRegressor)


def test_none_defaults_to_linear_regression():
    rk = RegressionKriging(regressor=None)
    assert isinstance(rk.regressor, LinearRegression)


def test_invalid_regressor_string_raises():
    with pytest.raises(ValueError, match="Unknown regressor"):
        RegressionKriging(regressor="not_a_real_model")


def test_regressor_is_clonable():
    """sklearn.base.clone is used by pygstat.spatial_cv_rk -- every shortcut
    must produce a clonable estimator."""
    for name in ENSEMBLE_REGRESSORS:
        rk = RegressionKriging(regressor=name)
        cloned = clone(rk.regressor)
        assert type(cloned) is type(rk.regressor)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
