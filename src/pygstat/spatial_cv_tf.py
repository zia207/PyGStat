"""K-fold cross-validation for TensorFlow deep regression kriging (optional ``tensorflow`` extra)."""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

class KFoldCV_TF_RK:
    """K-Fold CV for TensorFlow-based Regression Kriging."""
    
    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def validate(self, rk_wrapper, X, y, coords):
        X, y, coords = np.asarray(X), np.asarray(y), np.asarray(coords)
        n = len(y)
        predictions = np.full(n, np.nan)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in kf.split(X):
            rk = rk_wrapper.__class__(
                hidden_layers=rk_wrapper.hidden_layers,
                dropout=rk_wrapper.dropout,
                learning_rate=rk_wrapper.learning_rate,
                batch_size=rk_wrapper.batch_size,
                max_epochs=rk_wrapper.max_epochs,
                patience=rk_wrapper.patience,
                variogram_model=rk_wrapper.variogram_model,
                variogram_kwargs=rk_wrapper.variogram_kwargs,
                use_gpu=getattr(rk_wrapper, "use_gpu", "auto"),
            )
            rk.fit(X[train_idx], y[train_idx], coords[train_idx])
            pred = rk.predict(X[test_idx], coords[test_idx])
            predictions[test_idx] = pred

        nan_mask = np.isnan(predictions)
        if np.any(nan_mask):
            raise RuntimeError(
                f"{int(np.sum(nan_mask))} NaN predictions in TensorFlow RK CV"
            )

        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        return {"rmse": rmse, "r2": r2, "predictions": predictions}