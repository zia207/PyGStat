"""K-fold cross-validation for PyTorch deep regression kriging (optional ``pytorch`` extra)."""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

class KFoldCV_PyTorch_RK:
    """
    K-Fold Cross-Validation for PyTorch-based Regression Kriging.
    
    Handles NaN predictions gracefully and ensures all folds are processed.
    """
    
    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def validate(self, rk_wrapper, X, y, coords):
        """
        Perform K-Fold CV and return metrics + predictions.
        
        Parameters
        ----------
        rk_wrapper : DeepRegressionKriging instance
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        coords : array-like, shape (n_samples, 2)
        
        Returns
        -------
        dict : {"rmse": float, "r2": float, "predictions": array}
        """
        X, y, coords = np.asarray(X), np.asarray(y), np.asarray(coords)
        n = len(y)
        predictions = np.full(n, np.nan)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_errors = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            try:
                # Clone model with same parameters
                rk = rk_wrapper.__class__(
                    hidden_layers=rk_wrapper.hidden_layers,
                    dropout=rk_wrapper.dropout,
                    learning_rate=rk_wrapper.learning_rate,
                    batch_size=rk_wrapper.batch_size,
                    max_epochs=rk_wrapper.max_epochs,
                    patience=rk_wrapper.patience,
                    device=rk_wrapper.device,
                    use_gpu=getattr(rk_wrapper, "use_gpu", "auto"),
                    variogram_model=rk_wrapper.variogram_model,
                    variogram_kwargs=rk_wrapper.variogram_kwargs
                )
                # Fit on training fold
                rk.fit(X[train_idx], y[train_idx], coords[train_idx])
                # Predict on test fold
                pred = rk.predict(X[test_idx], coords[test_idx])
                predictions[test_idx] = pred
                
            except Exception as e:
                fold_errors.append(f"Fold {fold_idx}: {str(e)}")

        if fold_errors:
            raise RuntimeError(
                "KFoldCV_PyTorch_RK failed on one or more folds:\n  "
                + "\n  ".join(fold_errors)
            )

        nan_mask = np.isnan(predictions)
        if np.any(nan_mask):
            raise RuntimeError(
                f"{int(np.sum(nan_mask))} NaN predictions in PyTorch RK CV"
            )

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        
        return {
            "rmse": float(rmse),
            "r2": float(r2),
            "predictions": predictions
        }