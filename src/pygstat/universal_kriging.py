"""
Universal Kriging with polynomial drift.
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from .core.kriging import _apply_variogram_anisotropy, _cov_from_gamma, _pairwise_dist
from .utils.backend import get_array_module, as_gpu_array, to_numpy, resolve_cupy_use_gpu


class UniversalKriging:
    """
    Universal Kriging (kriging with a polynomial drift/trend). See
    `pygstat.core.kriging.OrdinaryKriging` for the `use_gpu` convention
    (CuPy-accelerated covariance-matrix build + solve, automatic CPU
    fallback). The polynomial drift matrix itself (`sklearn`'s
    `PolynomialFeatures`) is always built on CPU -- cheap relative to the
    O(n^3) kriging solve, which is where GPU acceleration actually matters
    -- then moved to the GPU alongside the covariance blocks.
    """

    def __init__(self, variogram, degree=1, use_gpu="auto"):
        self.variogram = variogram
        self.degree = degree
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)

    def fit(self, X, y):
        if getattr(self.variogram, "fitted_params", None) is None:
            raise RuntimeError("Variogram is not fitted. Call fit() or set_params() first.")
        self.X_train_orig = np.asarray(X, dtype=float)
        self.X_train = _apply_variogram_anisotropy(self.variogram, self.X_train_orig)
        self.y_train = np.asarray(y, dtype=float)
        # include_bias=True supplies the constant term so Σw = 1 plus trend constraints.
        self._poly = PolynomialFeatures(degree=self.degree, include_bias=True)
        self.F = self._poly.fit_transform(self.X_train_orig)
        return self

    def predict(self, X_pred, return_variance=False, block_size=None):
        if not hasattr(self, "X_train"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if block_size is not None:
            raise NotImplementedError(
                "block_size is not implemented for UniversalKriging; "
                "use OrdinaryKriging for block kriging, or predict at point support."
            )

        X_pred_orig = np.asarray(X_pred, dtype=float)
        X_pred_aniso = _apply_variogram_anisotropy(self.variogram, X_pred_orig)
        n, m = len(self.X_train), len(X_pred_orig)
        p = self.F.shape[1]

        F_pred = self._poly.transform(X_pred_orig)

        if self.use_gpu:
            X_train, X_pred_d = as_gpu_array(self.X_train), as_gpu_array(X_pred_aniso)
            y_train, F, F_pred = as_gpu_array(self.y_train), as_gpu_array(self.F), as_gpu_array(F_pred)
        else:
            X_train, X_pred_d, y_train = self.X_train, X_pred_aniso, self.y_train
            F = self.F
        xp = get_array_module(X_train)

        dist_train = _pairwise_dist(X_train, X_train, xp)
        dist_pred = _pairwise_dist(X_pred_d, X_train, xp)

        C0 = self.variogram.fitted_params[0] + self.variogram.fitted_params[1]
        C_train = _cov_from_gamma(self.variogram(dist_train), C0, dist_train)
        C_pred = _cov_from_gamma(self.variogram(dist_pred), C0, dist_pred)

        top = xp.hstack([C_train, F])
        bottom = xp.hstack([F.T, xp.zeros((p, p))])
        K = xp.vstack([top, bottom])
        C_pred_aug = xp.hstack([C_pred, F_pred])

        try:
            weights = xp.linalg.solve(K, C_pred_aug.T).T
        except xp.linalg.LinAlgError:
            weights = xp.linalg.lstsq(K, C_pred_aug.T, rcond=None)[0].T
        pred = weights[:, :n] @ y_train

        if not return_variance:
            return to_numpy(pred)

        var = C0 - xp.einsum("ij,ij->i", weights[:, :n], C_pred)
        var -= xp.einsum("ij,ij->i", weights[:, n:], F_pred)
        var = xp.clip(var, 0, None)
        return to_numpy(pred), to_numpy(xp.sqrt(var))
