"""K-fold and leave-one-out cross-validation for RegressionKriging."""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

class KFoldCV_RK:
    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def validate(self, rk_model, X, y, coords):
        X, y, coords = np.asarray(X), np.asarray(y), np.asarray(coords)
        n = len(y)
        predictions = np.full(n, np.nan)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in kf.split(X):
            # Clone model
            from sklearn.base import clone
            from .regression_kriging import RegressionKriging
            rk = RegressionKriging(
                regressor=clone(rk_model.regressor),
                kriging_type=rk_model.kriging_type,
                variogram_model=rk_model.variogram_model,
                variogram_kwargs=rk_model.variogram_kwargs,
                use_gpu=rk_model.use_gpu,
            )
            rk.fit(X[train_idx], y[train_idx], coords[train_idx])
            pred = rk.predict(X[test_idx], coords[test_idx])
            predictions[test_idx] = pred

        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        return {"rmse": rmse, "r2": r2, "predictions": predictions}

class LOOCV_RK:
    def validate(self, rk_model, X, y, coords):
        X, y, coords = np.asarray(X), np.asarray(y), np.asarray(coords)
        n = len(y)
        predictions = np.empty(n)

        from sklearn.base import clone
        from .regression_kriging import RegressionKriging
        for i in range(n):
            X_tr = np.delete(X, i, axis=0)
            y_tr = np.delete(y, i, axis=0)
            c_tr = np.delete(coords, i, axis=0)
            rk = RegressionKriging(
                regressor=clone(rk_model.regressor),
                kriging_type=rk_model.kriging_type,
                variogram_model=rk_model.variogram_model,
                variogram_kwargs=rk_model.variogram_kwargs,
                use_gpu=rk_model.use_gpu,
            )
            rk.fit(X_tr, y_tr, c_tr)
            predictions[i] = rk.predict(X[i:i+1], coords[i:i+1])[0]

        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        return {"rmse": rmse, "r2": r2, "predictions": predictions}