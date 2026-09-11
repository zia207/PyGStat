"""K-fold cross-validation for H2O regression-kriging wrappers (optional ``h2o`` extra)."""

import numpy as np
import pandas as pd
import h2o
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from .core.kriging import OrdinaryKriging
from .core.variogram import Variogram



# Helper class (ADD THIS)
class H2OResidualKriging:
    def __init__(self, h2o_model, kriging_model, feature_names):
        self.h2o_model = h2o_model
        self.krige = kriging_model
        self.feature_names = feature_names
    def predict(self, X, coords):
        pred_reg = self.h2o_model.predict(
            h2o.H2OFrame(pd.DataFrame(X, columns=self.feature_names))
        ).as_data_frame(use_pandas=True).values.flatten()
        resid = self.krige.predict(coords)
        return pred_reg + resid


class KFoldCV_H2O_RK:
    """K-Fold CV for H2O-based Regression Kriging wrappers."""
    
    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def validate(self, rk_wrapper, X, y, coords):
        X, y, coords = np.asarray(X), np.asarray(y), np.asarray(coords)
        n = len(y)
        predictions = np.full(n, np.nan)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in kf.split(X):
            # Clone by creating a new instance with same parameters
            rk = rk_wrapper.__class__(
                max_runtime_secs=getattr(rk_wrapper, "max_runtime_secs", 300),
                max_models=getattr(rk_wrapper, "max_models", 20),
                seed=getattr(rk_wrapper, "seed", 42),
                include_algos=getattr(rk_wrapper, "include_algos", None),
                variogram_model=rk_wrapper.variogram_model,
                variogram_kwargs=rk_wrapper.variogram_kwargs,
                use_gpu=getattr(rk_wrapper, "use_gpu", "auto"),
            )
            rk.fit(X[train_idx], y[train_idx], coords[train_idx])
            pred = rk.predict(X[test_idx], coords[test_idx])
            predictions[test_idx] = pred

        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        return {"rmse": rmse, "r2": r2, "predictions": predictions}


class KFoldCV_H2O_ResidualRK:
    """K-Fold CV for H2OResidualKriging wrappers."""
    
    def __init__(self, n_splits=5, random_state=None):
        # h2o.connection() returns None rather than raising when disconnected
        # (see note in regression_kriging_h2o.py.fit()).
        if h2o.connection() is None:
            h2o.init(max_mem_size="32G", nthreads=2)
        self.n_splits = n_splits
        self.random_state = random_state

    def validate(self, rk_wrapper, X, y, coords):
        """
        rk_wrapper must have: .h2o_model, .krige.variogram, .feature_names
        """
        X, y, coords = np.asarray(X), np.asarray(y), np.asarray(coords)
        n = len(y)
        predictions = np.full(n, np.nan)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in kf.split(X):
            # Get predictions from FULL H2O model
            X_train_fold = pd.DataFrame(X[train_idx], columns=rk_wrapper.feature_names)
            pred_reg_train = rk_wrapper.h2o_model.predict(
                h2o.H2OFrame(X_train_fold)
            ).as_data_frame(use_pandas=True).values.flatten()
            residuals_train = y[train_idx] - pred_reg_train
            
            # Fit variogram + kriging on FOLD residuals
            vg = Variogram(
                coords=coords[train_idx],
                values=residuals_train,
                model=rk_wrapper.krige.variogram.model,
                estimator='matheron',
                maxlag=rk_wrapper.krige.variogram.maxlag,
                n_lags=rk_wrapper.krige.variogram.n_lags
            )
            vg.fit()
            
            ok = OrdinaryKriging(
                vg, use_gpu=getattr(rk_wrapper.krige, "use_gpu", False)
            )
            ok.fit(coords[train_idx], residuals_train)
            
            # Predict on test fold
            X_test_fold = pd.DataFrame(X[test_idx], columns=rk_wrapper.feature_names)
            pred_reg_test = rk_wrapper.h2o_model.predict(
                h2o.H2OFrame(X_test_fold)
            ).as_data_frame(use_pandas=True).values.flatten()
            resid_pred = ok.predict(coords[test_idx])
            predictions[test_idx] = pred_reg_test + resid_pred

        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        return {"rmse": rmse, "r2": r2, "predictions": predictions}
    
    def shutdown(self):
        """Manually shut down H2O cluster when done."""
        h2o.shutdown(prompt=False)