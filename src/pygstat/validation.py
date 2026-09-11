"""
Cross-validation utilities for all pygstat estimators.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


def _clone_rk_model(model):
    """Clone RegressionKriging model (fresh regressor, no leaked fit state)."""
    from sklearn.base import clone
    from .regression_kriging import RegressionKriging
    return RegressionKriging(
        regressor=clone(model.regressor),
        kriging_type=model.kriging_type,
        variogram_model=model.variogram_model,
        variogram_kwargs=model.variogram_kwargs,
        use_gpu=model.use_gpu,
    )


def _clone_kriging_model(model):
    """Clone Ordinary/Simple/Universal Kriging by type, not duck-typed attributes."""
    from .core.kriging import OrdinaryKriging, SimpleKriging
    from .universal_kriging import UniversalKriging

    if isinstance(model, UniversalKriging):
        return UniversalKriging(
            variogram=model.variogram,
            degree=model.degree,
            use_gpu=model.use_gpu,
        )
    if isinstance(model, SimpleKriging):
        mean = model.mean if getattr(model, "_mean_provided", False) else None
        return SimpleKriging(
            variogram=model.variogram,
            mean=mean,
            use_gpu=model.use_gpu,
        )
    if isinstance(model, OrdinaryKriging):
        return OrdinaryKriging(
            variogram=model.variogram,
            use_gpu=model.use_gpu,
        )
    raise TypeError(
        f"loo_cv/kfold_cv cannot clone {type(model).__name__}. "
        "Use the matching spatial_cv_* helper for cokriging, indicator, "
        "or regression-kriging backends."
    )


def loo_cv(estimator, X, y, coords=None):
    """
    Leave-One-Out Cross-Validation for pygstat models.

    Parameters
    ----------
    estimator : kriging or RK model
    X : features (for RK) or coords (for kriging)
    y : target values
    coords : spatial coordinates (required only for RegressionKriging)

    Returns
    -------
    dict: {"rmse": float, "r2": float, "predictions": array}
    """
    from .regression_kriging import RegressionKriging

    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    predictions = np.empty(n)

    if isinstance(estimator, RegressionKriging):
        if coords is None:
            raise ValueError("coords is required for RegressionKriging cross-validation")
        coords = np.asarray(coords)
        for i in range(n):
            X_tr = np.delete(X, i, axis=0)
            y_tr = np.delete(y, i, axis=0)
            c_tr = np.delete(coords, i, axis=0)
            est = _clone_rk_model(estimator)
            est.fit(X_tr, y_tr, c_tr)
            predictions[i] = est.predict(X[i:i + 1], coords[i:i + 1])[0]
    else:
        coords = X
        for i in range(n):
            c_tr = np.delete(coords, i, axis=0)
            y_tr = np.delete(y, i, axis=0)
            est = _clone_kriging_model(estimator)
            est.fit(c_tr, y_tr)
            predictions[i] = est.predict(coords[i:i + 1])[0]

    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    return {"rmse": rmse, "r2": r2, "predictions": predictions}


def kfold_cv(estimator, X, y, coords=None, n_splits=5, random_state=None):
    """
    K-Fold Cross-Validation for pygstat models.

    Parameters
    ----------
    estimator : kriging or RK model
    X : features (for RK) or coords (for kriging)
    y : target values
    coords : spatial coordinates (required only for RegressionKriging)
    n_splits : int, default=5
    random_state : int, optional

    Returns
    -------
    dict: {"rmse": float, "r2": float, "predictions": array}
    """
    from .regression_kriging import RegressionKriging

    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    predictions = np.full(n, np.nan)

    if isinstance(estimator, RegressionKriging):
        if coords is None:
            raise ValueError("coords is required for RegressionKriging cross-validation")
        coords = np.asarray(coords)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for tr_idx, te_idx in kf.split(X):
            est = _clone_rk_model(estimator)
            est.fit(X[tr_idx], y[tr_idx], coords[tr_idx])
            pred = est.predict(X[te_idx], coords[te_idx])
            predictions[te_idx] = pred
    else:
        coords = X
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for tr_idx, te_idx in kf.split(coords):
            est = _clone_kriging_model(estimator)
            est.fit(coords[tr_idx], y[tr_idx])
            pred = est.predict(coords[te_idx])
            predictions[te_idx] = pred

    mask = ~np.isnan(predictions)
    rmse = np.sqrt(mean_squared_error(y[mask], predictions[mask]))
    r2 = r2_score(y[mask], predictions[mask])
    return {"rmse": rmse, "r2": r2, "predictions": predictions}
