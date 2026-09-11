"""Cross-validation for Indicator Kriging (Brier score)."""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss


class IndicatorKrigingCV:
    """K-fold cross-validation for IndicatorKriging using Brier score."""

    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def validate(self, ik_model, coords, values, thresholds, variogram_params_dict):
        """
        Run K-fold CV: fit on train folds, predict on test, aggregate Brier scores.

        Returns
        -------
        dict with keys "brier_scores" (per-threshold mean Brier), "predictions" (optional).
        """
        coords = np.asarray(coords)
        values = np.asarray(values)
        n = len(values)
        brier_scores = {}
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for threshold in thresholds:
            preds = np.full(n, np.nan)
            for train_idx, test_idx in kf.split(coords):
                ik = ik_model.__class__(
                    cov_model=ik_model.cov_model_name,
                    max_neighbors=ik_model.max_neighbors,
                    search_radius=ik_model.search_radius,
                    anisotropy_angle=ik_model.anisotropy_angle,
                    anisotropy_ratio=ik_model.anisotropy_ratio,
                    regularization=ik_model.regularization,
                    use_gpu=getattr(ik_model, "use_gpu", "auto"),
                )
                ik.fit(coords[train_idx], values[train_idx])
                params = variogram_params_dict.get(threshold) if isinstance(variogram_params_dict, dict) else variogram_params_dict
                vp = {threshold: params} if params is not None else None
                probs = ik.predict(coords[test_idx], [threshold], variogram_params=vp)
                preds[test_idx] = probs[threshold]

            # Indicator I(Z >= threshold), matching IndicatorKriging convention
            ind = (values >= threshold).astype(float)
            valid = np.isfinite(preds)
            if np.any(valid):
                brier_scores[threshold] = brier_score_loss(ind[valid], np.clip(preds[valid], 0, 1))
            else:
                brier_scores[threshold] = np.nan

        return {"brier_scores": brier_scores}