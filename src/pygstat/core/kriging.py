"""
Kriging implementations: Ordinary, Simple, with point and block support.
"""

import numpy as np
from scipy.spatial.distance import cdist

from ..utils.backend import get_array_module, as_gpu_array, to_numpy, resolve_cupy_use_gpu
from ..utils.anisotropy import transform_aniso


def _apply_variogram_anisotropy(variogram, coords):
    """Transform coords into the same space used to fit ``variogram``."""
    coords = np.asarray(coords, dtype=float)
    aniso = getattr(variogram, "anisotropy", None)
    if aniso:
        return transform_aniso(coords, **aniso)
    return coords


def _pairwise_dist(A, B, xp):
    """Euclidean distance matrix between rows of A and rows of B, on
    whichever backend `xp` is (`numpy` uses `scipy.spatial.distance.cdist`;
    `cupy` has no `cdist`, so it's computed via broadcasting -- the same
    approach already used by `Variogram`'s GPU empirical-variogram path)."""
    if xp is np:
        return cdist(A, B)
    diff = A[:, None, :] - B[None, :, :]
    return xp.sqrt(xp.sum(diff ** 2, axis=2))


def _cov_from_gamma(gamma, C0, dist):
    """
    C(h) = C(0) − γ(h), restoring C(0) = nugget + sill on the origin.

    Theoretical models return γ(0) = nugget, so C0 − γ(0) would drop the
    nugget from the diagonal of the kriging matrix. Array-module-agnostic
    (works for a NumPy or CuPy `dist`/`gamma`, see
    `pygstat.utils.backend.get_array_module`).
    """
    xp = get_array_module(gamma, dist)
    dist = xp.asarray(dist)
    C = C0 - xp.asarray(gamma, dtype=float)
    C = xp.array(C, dtype=float, copy=True)
    C[dist == 0] = C0
    return C


class OrdinaryKriging:
    """
    Ordinary Kriging with GPU support (CuPy) and block kriging option.

    Parameters
    ----------
    variogram : Variogram
        Fitted variogram model.
    use_gpu : bool or 'auto', default='auto'
        Builds the kriging covariance matrix and solves the kriging system
        on the GPU via CuPy (`cupy.linalg.solve`) when available -- the
        same `CUPY_AVAILABLE`/`'auto'` convention as `Variogram`. `X_pred`
        (and `X`/`y` at `fit()`) may be a NumPy array, a CuPy array, or a
        cuDF Series/DataFrame; predictions are always returned as NumPy.
        Falls back to CPU (NumPy/SciPy) when CuPy is unavailable or
        `use_gpu=False`. One caveat: a Matérn variogram model always
        evaluates on the CPU internally regardless of `use_gpu` (SciPy's
        Bessel functions have no CuPy equivalent), transferring data over
        and back transparently -- correct either way, just not itself
        GPU-accelerated.
    """

    def __init__(self, variogram, use_gpu="auto"):
        self.variogram = variogram
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)

    def fit(self, X, y):
        """Store training data."""
        if getattr(self.variogram, "fitted_params", None) is None:
            raise RuntimeError("Variogram is not fitted. Call fit() or set_params() first.")
        self.X_train = _apply_variogram_anisotropy(self.variogram, X)
        self.y_train = np.asarray(y, dtype=float)
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X and y must have the same length")
        return self

    def predict(self, X_pred, return_variance=False, block_size=None):
        """
        Predict at points or over blocks.

        Parameters
        ----------
        X_pred : array-like, shape (n, 2)
        return_variance : bool
            If True, also return the kriging standard error (√variance).
        block_size : float, optional
            Side length of square block (meters) for block kriging.
        """
        if not hasattr(self, "X_train"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if block_size is not None:
            return self._predict_block(X_pred, return_variance, block_size)
        return self._predict_point(X_pred, return_variance)

    def _predict_point(self, X_pred, return_variance):
        """Point kriging implementation. Builds the covariance matrix and
        solves the kriging system on GPU (CuPy) when `self.use_gpu`, on
        CPU (NumPy/SciPy) otherwise -- one formula parametrized by the
        array module `xp`, so the two backends can't drift apart."""
        X_pred = _apply_variogram_anisotropy(self.variogram, X_pred)
        n, m = len(self.X_train), len(X_pred)

        if self.use_gpu:
            X_train = as_gpu_array(self.X_train)
            X_pred_d = as_gpu_array(X_pred)
            y_train = as_gpu_array(self.y_train)
        else:
            X_train, X_pred_d, y_train = self.X_train, np.asarray(X_pred, dtype=float), self.y_train
        xp = get_array_module(X_train)

        dist_train = _pairwise_dist(X_train, X_train, xp)
        dist_pred = _pairwise_dist(X_pred_d, X_train, xp)

        C0 = self.variogram.fitted_params[0] + self.variogram.fitted_params[1]
        C_train = _cov_from_gamma(self.variogram(dist_train), C0, dist_train)
        C_pred = _cov_from_gamma(self.variogram(dist_pred), C0, dist_pred)

        K = xp.vstack([
            xp.hstack([C_train, xp.ones((n, 1))]),
            xp.hstack([xp.ones((1, n)), xp.zeros((1, 1))]),
        ])
        C_pred_aug = xp.hstack([C_pred, xp.ones((m, 1))])

        try:
            weights = xp.linalg.solve(K, C_pred_aug.T).T
        except xp.linalg.LinAlgError:
            weights = xp.linalg.lstsq(K, C_pred_aug.T, rcond=None)[0].T
        pred = weights[:, :-1] @ y_train

        if not return_variance:
            return to_numpy(pred)

        var = C0 - xp.einsum("ij,ij->i", weights[:, :-1], C_pred) - weights[:, -1]
        var = xp.clip(var, 0, None)
        return to_numpy(pred), to_numpy(xp.sqrt(var))

    def _predict_block(self, X_centers, return_variance, block_size):
        """
        Approximate block kriging via Monte Carlo integration of point predictions.

        The reported block variance is the mean of the point kriging variances
        (a common approximation; it ignores within-block covariance).
        """
        n_blocks = len(X_centers)
        n_samples = 50

        block_preds = np.empty(n_blocks)
        block_vars = np.empty(n_blocks) if return_variance else None

        for i, center in enumerate(X_centers):
            offsets = np.random.uniform(-block_size / 2, block_size / 2, size=(n_samples, 2))
            block_points = np.asarray(center) + offsets
            if return_variance:
                preds, stds = self._predict_point(block_points, return_variance=True)
                block_preds[i] = np.mean(preds)
                block_vars[i] = np.mean(stds ** 2)
            else:
                preds = self._predict_point(block_points, return_variance=False)
                block_preds[i] = np.mean(preds)

        if return_variance:
            return block_preds, np.sqrt(block_vars)
        return block_preds


class SimpleKriging:
    """
    Simple Kriging with known mean. See `OrdinaryKriging` for the
    `use_gpu` convention (CuPy-accelerated covariance-matrix build + solve,
    NumPy/CuPy/cuDF input, NumPy output, automatic CPU fallback).
    """

    def __init__(self, variogram, mean=None, use_gpu="auto"):
        self.variogram = variogram
        self.mean = mean
        self._mean_provided = mean is not None
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)

    def fit(self, X, y):
        if getattr(self.variogram, "fitted_params", None) is None:
            raise RuntimeError("Variogram is not fitted. Call fit() or set_params() first.")
        self.X_train = _apply_variogram_anisotropy(self.variogram, X)
        self.y_train = np.asarray(y, dtype=float)
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X and y must have the same length")
        if not self._mean_provided:
            self.mean = float(np.mean(y))
        return self

    def predict(self, X_pred, return_variance=False):
        if not hasattr(self, "X_train"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_pred = _apply_variogram_anisotropy(self.variogram, X_pred)

        if self.use_gpu:
            X_train = as_gpu_array(self.X_train)
            X_pred_d = as_gpu_array(X_pred)
            y_train = as_gpu_array(self.y_train)
        else:
            X_train, X_pred_d, y_train = self.X_train, np.asarray(X_pred, dtype=float), self.y_train
        xp = get_array_module(X_train)

        dist_train = _pairwise_dist(X_train, X_train, xp)
        dist_pred = _pairwise_dist(X_pred_d, X_train, xp)

        C0 = self.variogram.fitted_params[0] + self.variogram.fitted_params[1]
        C_train = _cov_from_gamma(self.variogram(dist_train), C0, dist_train)
        C_pred = _cov_from_gamma(self.variogram(dist_pred), C0, dist_pred)

        try:
            weights = xp.linalg.solve(C_train, C_pred.T).T
        except xp.linalg.LinAlgError:
            weights = xp.linalg.lstsq(C_train, C_pred.T, rcond=None)[0].T
        mean = self.mean
        pred = mean + weights @ (y_train - mean)

        if not return_variance:
            return to_numpy(pred)

        var = C0 - xp.einsum("ij,ij->i", weights, C_pred)
        var = xp.clip(var, 0, None)
        return to_numpy(pred), to_numpy(xp.sqrt(var))
