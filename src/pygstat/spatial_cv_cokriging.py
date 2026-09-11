"""K-fold cross-validation for two-variable Cokriging models."""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from .cokriging import Cokriging


class KFoldCV_Cokriging:
    """K-Fold Cross-Validation wrapper for Cokriging models.

    Assumes the secondary variable is fully observed and available at all locations
    during both training and prediction (i.e., no missing secondary data).

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds for cross-validation.
    random_state : int or None, default=None
        Controls the shuffling applied to the data before splitting.
    """

    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def validate(self, ck_model, coords_primary, primary, coords_secondary, secondary):
        """Perform K-Fold cross-validation on a cokriging model.

        Parameters
        ----------
        ck_model : Cokriging
            A fitted or unfitted Cokriging instance used as a template.
            Only its variogram models (primary_var, secondary_var, cross_var) are used.
        coords_primary : array-like, shape (n_primary, 2)
            Coordinates of primary variable observations.
        primary : array-like, shape (n_primary,)
            Values of the primary variable (e.g., soil property).
        coords_secondary : array-like, shape (n_secondary, 2)
            Coordinates of secondary variable (e.g., NDVI).
        secondary : array-like, shape (n_secondary,)
            Values of the secondary variable.

        Returns
        -------
        dict
            Dictionary containing:
            - 'rmse': Root Mean Squared Error
            - 'r2': Coefficient of determination
            - 'predictions': Array of cross-validated predictions (same length as `primary`)
        """
        coords_primary = np.asarray(coords_primary)
        primary = np.asarray(primary)
        coords_secondary = np.asarray(coords_secondary)
        secondary = np.asarray(secondary)

        n = len(primary)
        if n != coords_primary.shape[0]:
            raise ValueError("Length of `primary` must match number of `coords_primary`.")

        predictions = np.full(n, np.nan)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in kf.split(coords_primary):
            # Instantiate a new cokriging model using the same variogram structures
            ck = Cokriging(
                primary_var=ck_model.primary_var,
                secondary_var=ck_model.secondary_var,
                cross_var=ck_model.cross_var
            )

            # Fit on training subset of primary data; full secondary data assumed available
            ck.fit(
                coords_primary=coords_primary[train_idx],
                primary=primary[train_idx],
                coords_secondary=coords_secondary,
                secondary=secondary
            )

            # Predict on held-out test locations
            pred = ck.predict(coords_primary[test_idx])
            predictions[test_idx] = pred

        # Compute metrics only where predictions exist (should be all)
        valid_mask = ~np.isnan(predictions)
        if not np.all(valid_mask):
            raise RuntimeError("Some predictions are missing; check cokriging setup.")

        rmse = np.sqrt(mean_squared_error(primary[valid_mask], predictions[valid_mask]))
        r2 = r2_score(primary[valid_mask], predictions[valid_mask])

        return {
            "rmse": rmse,
            "r2": r2,
            "predictions": predictions
        }