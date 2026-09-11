"""
Test for pygstat.regression_kriging_h2o's model_type support (AutoML,
Deep Learning, DRF, GLM, GAM, GBM, XGBoost, Uplift DRF, Stacked Ensemble) and
hyperparameter-tuning option, using the Jura data set: predict Cd from the
other heavy-metal assays (Co, Cr, Cu, Ni, Pb, Zn) as covariates, kriging the
regression residuals.

Requires the optional `h2o` package and a working Java runtime (H2O starts a
local JVM-backed cluster); the whole module is skipped if either is missing
or a cluster fails to start.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

h2o = pytest.importorskip("h2o")

DATA_PATH = "data/jura_data.csv"
FEATURES = ["Co", "Cr", "Cu", "Ni", "Pb", "Zn"]


@pytest.fixture(scope="module")
def h2o_cluster():
    """Start (or reuse) one H2O cluster for the whole test module -- H2O
    startup dominates runtime, so tests should not each pay for it."""
    try:
        if h2o.connection() is None:
            h2o.init(max_mem_size="32G", nthreads=2)
    except Exception as e:
        pytest.skip(f"Could not start an H2O cluster (needs a working JVM/Java): {e}")
    yield
    # Leave the cluster running for any other test module in the same
    # session; nothing to tear down here.


@pytest.fixture(scope="module")
def jura_split():
    df = pd.read_csv(DATA_PATH, index_col=0)
    X = df[FEATURES].to_numpy()
    y = df["Cd"].to_numpy()
    coords = df[["Xloc", "Yloc"]].to_numpy()
    return train_test_split(X, y, coords, test_size=0.3, random_state=42)


@pytest.mark.parametrize("model_type,model_kwargs", [
    ("drf", {"ntrees": 30, "max_depth": 8}),
    ("glm", {}),
    ("gbm", {"ntrees": 30, "max_depth": 4}),
    ("gam", {}),
    ("deep_learning", {"hidden": [16, 16], "epochs": 5}),
    ("xgboost", {"ntrees": 30, "max_depth": 4}),
])
def test_individual_model_types_fit_and_predict(h2o_cluster, jura_split, model_type, model_kwargs):
    from pygstat.regression_kriging_h2o import RegressionKrigingH2O

    if model_type == "xgboost":
        from h2o.estimators import H2OXGBoostEstimator
        if not H2OXGBoostEstimator.available():
            pytest.skip("H2O's XGBoost backend is not available on this platform/build")

    X_train, X_test, y_train, y_test, c_train, c_test = jura_split
    rk = RegressionKrigingH2O(model_type=model_type, model_kwargs=model_kwargs, seed=1)
    rk.fit(X_train, y_train, c_train)
    pred = rk.predict(X_test, c_test)

    assert pred.shape == y_test.shape
    assert np.isfinite(pred).all()

    pred_with_std, var = rk.predict(X_test, c_test, return_std=True)
    np.testing.assert_allclose(pred_with_std, pred)
    assert np.all(var >= -1e-8)

    metrics = rk.get_best_model_metrics()
    assert "model_id" in metrics
    assert np.isfinite(metrics["rmse"])


def test_stacked_ensemble(h2o_cluster, jura_split):
    from pygstat.regression_kriging_h2o import RegressionKrigingH2O

    X_train, X_test, y_train, y_test, c_train, c_test = jura_split
    rk = RegressionKrigingH2O(model_type="stacked_ensemble", nfolds=3, seed=1)
    rk.fit(X_train, y_train, c_train)

    assert len(rk.base_models_) == 3  # default DRF + GBM + GLM base learners
    pred = rk.predict(X_test, c_test)
    assert np.isfinite(pred).all()


def test_automl_still_works_as_before(h2o_cluster, jura_split):
    """Backward compatibility: model_type='automl' (the default) must behave
    like the original AutoML-only implementation."""
    from pygstat.regression_kriging_h2o import RegressionKrigingH2O

    X_train, X_test, y_train, y_test, c_train, c_test = jura_split
    rk = RegressionKrigingH2O(max_runtime_secs=30, max_models=4, seed=1)  # model_type default
    assert rk.model_type == "automl"
    rk.fit(X_train, y_train, c_train)
    pred = rk.predict(X_test, c_test)
    assert np.isfinite(pred).all()

    metrics = rk.get_best_model_metrics()
    assert "model_id" in metrics
    assert hasattr(rk, "leaderboard_")
    assert hasattr(rk, "automl_")


def test_hyperparameter_tuning_picks_a_model_and_reports_grid(h2o_cluster, jura_split):
    from pygstat.regression_kriging_h2o import RegressionKrigingH2O

    X_train, X_test, y_train, y_test, c_train, c_test = jura_split
    rk = RegressionKrigingH2O(
        model_type="drf",
        tune_hyperparameters=True,
        hyper_params={"ntrees": [20, 40], "max_depth": [5, 10]},
        nfolds=3,
        seed=1,
    )
    rk.fit(X_train, y_train, c_train)

    assert hasattr(rk, "grid_")
    assert hasattr(rk, "grid_results_")
    assert len(rk.grid_.models) == 4  # 2 x 2 Cartesian grid

    pred = rk.predict(X_test, c_test)
    assert np.isfinite(pred).all()
    assert "tuned" in repr(rk)


def test_tuning_without_default_hyper_params_raises_clear_error(h2o_cluster, jura_split, monkeypatch):
    import pygstat.regression_kriging_h2o as rk_h2o_mod

    X_train, _, y_train, _, c_train, _ = jura_split
    # Every currently-supported model_type ships a default hyper_params grid
    # (see _default_hyper_params), so this guard is normally unreachable --
    # force it here to check the error path itself stays correct as new
    # model types are added in the future.
    monkeypatch.setattr(rk_h2o_mod, "_default_hyper_params", lambda model_type: {})
    rk = rk_h2o_mod.RegressionKrigingH2O(model_type="glm", tune_hyperparameters=True)
    with pytest.raises(ValueError, match="hyper_params"):
        rk.fit(X_train, y_train, c_train)


def test_uplift_drf_requires_treatment_and_binary_target(h2o_cluster, jura_split):
    from pygstat.regression_kriging_h2o import RegressionKrigingH2O

    X_train, X_test, y_train, y_test, c_train, c_test = jura_split

    rk_missing = RegressionKrigingH2O(model_type="uplift_drf")
    with pytest.raises(ValueError, match="treatment"):
        rk_missing.fit(X_train, y_train, c_train)  # no treatment= given

    rng = np.random.default_rng(0)
    treatment_train = rng.integers(0, 2, size=len(X_train))
    y_bin_train = (y_train > np.median(y_train)).astype(int)

    rk = RegressionKrigingH2O(model_type="uplift_drf", model_kwargs={"ntrees": 20, "max_depth": 5}, seed=1)
    rk.fit(X_train, y_bin_train, c_train, treatment=treatment_train)
    pred = rk.predict(X_test, c_test)
    assert np.isfinite(pred).all()

    metrics = rk.get_best_model_metrics()
    assert "auuc" in metrics


def test_model_type_aliases_and_invalid_value():
    from pygstat.regression_kriging_h2o import RegressionKrigingH2O

    assert RegressionKrigingH2O(model_type="random_forest").model_type == "drf"
    assert RegressionKrigingH2O(model_type="gradient_boosting").model_type == "gbm"
    assert RegressionKrigingH2O(model_type="dl").model_type == "deep_learning"
    assert RegressionKrigingH2O(model_type="stacking").model_type == "stacked_ensemble"
    assert RegressionKrigingH2O(model_type="xgb").model_type == "xgboost"
    assert RegressionKrigingH2O(model_type="extreme_gradient_boosting").model_type == "xgboost"

    with pytest.raises(ValueError, match="Unknown model_type"):
        RegressionKrigingH2O(model_type="not_a_real_model")


def test_dataframe_categoricals_and_validation_frame(h2o_cluster, jura_split):
    from pygstat.regression_kriging_h2o import H2O_MODEL_TYPES, RegressionKrigingH2O

    assert "glm" in H2O_MODEL_TYPES
    X_train, X_test, y_train, y_test, c_train, c_test = jura_split
    # Hold out a slice of the training split as an explicit validation frame.
    n_valid = max(8, len(X_train) // 5)
    X_tr, X_va = X_train[:-n_valid], X_train[-n_valid:]
    y_tr, y_va = y_train[:-n_valid], y_train[-n_valid:]
    c_tr = c_train[:-n_valid]

    features = FEATURES
    train_df = pd.DataFrame(X_tr, columns=features)
    valid_df = pd.DataFrame(X_va, columns=features)
    test_df = pd.DataFrame(X_test, columns=features)
    train_df["NLCD"] = np.where(train_df["Zn"] > np.median(train_df["Zn"]), "high", "low")
    valid_df["NLCD"] = np.where(valid_df["Zn"] > np.median(train_df["Zn"]), "high", "low")
    test_df["NLCD"] = np.where(test_df["Zn"] > np.median(train_df["Zn"]), "high", "low")

    rk = RegressionKrigingH2O(
        model_type="glm",
        categorical_columns=["NLCD"],
        tune_hyperparameters=True,
        hyper_params={"alpha": [0.0, 0.5, 1.0]},
        seed=1,
    )
    rk.fit(train_df, y_tr, c_tr, X_valid=valid_df, y_valid=y_va)

    assert rk.feature_names_[-1] == "NLCD"
    assert rk.categorical_columns_ == ["NLCD"]
    pred = rk.predict(test_df, c_test)
    assert pred.shape == y_test.shape
    assert np.isfinite(pred).all()


def test_x_valid_requires_y_valid(h2o_cluster, jura_split):
    from pygstat.regression_kriging_h2o import RegressionKrigingH2O

    X_train, _, y_train, _, c_train, _ = jura_split
    rk = RegressionKrigingH2O(model_type="glm", seed=1)
    with pytest.raises(ValueError, match="y_valid"):
        rk.fit(X_train, y_train, c_train, X_valid=X_train[:5])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
