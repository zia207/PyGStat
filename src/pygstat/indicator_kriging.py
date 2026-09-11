"""
Indicator Kriging implementation for pygstat.

This module provides tools for:
- Transforming continuous data into indicators
- Fitting per-threshold variograms
- Performing ordinary indicator kriging
- Computing E-type estimates

GPU (CuPy) support
-------------------
Unlike the sequential simulators (`pygstat.sgsim`/`pygstat.sisim`), every
prediction point here is independent of every other -- there is no
growing conditioning set to serialize on -- so the k-nearest-neighbor
mode (`search_radius=None`, the default) batches into *one* stacked
linear-algebra solve over all `n_pred` points at once
(`xp.linalg.solve` on a `(n_pred, k+1, k+1)` array), instead of looping
in Python and solving a tiny system per point. That's both a real CPU
speedup (no Python-level loop) and what makes GPU dispatch worthwhile
here: `use_gpu='auto'|True|False` runs that one batched solve on the GPU
via CuPy when usable, same convention as `pygstat.core.kriging`. The
`search_radius` mode (variable neighbor counts per point) isn't cleanly
batchable this way and keeps the original per-point loop (CPU only).

Author: Zia Ahmed
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import solve, LinAlgError
from typing import Union, List, Dict, Optional, Tuple

from .utils.backend import get_array_module, as_gpu_array, to_numpy, resolve_cupy_use_gpu

# ==========================================================
# Covariance Models (C(h) for OK system; see core.variogram_models for gamma(h) and covariance_from_variogram)
# ==========================================================
# Array-module-agnostic (`xp` = numpy or cupy, via get_array_module): the
# boolean-mask formulation broadcasts correctly whether `h` is a 1-D
# per-point array (old per-point loop / search_radius fallback) or an
# (n_pred, k, k) / (n_pred, k) batch (the vectorized k-nearest path).

def _spherical_covariance(h: np.ndarray, sill: float, rng: float, nugget: float = 0.0) -> np.ndarray:
    """Spherical covariance model."""
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    c = xp.zeros_like(h)
    mask = h < rng
    hr = h[mask] / rng
    c[mask] = sill * (1.0 - 1.5 * hr + 0.5 * hr ** 3)
    c[h == 0] += nugget
    return c

def _exponential_covariance(h: np.ndarray, sill: float, rng: float, nugget: float = 0.0) -> np.ndarray:
    """Exponential covariance model."""
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    c = sill * xp.exp(-h / rng)
    c[h == 0] += nugget
    return c

_COV_MODELS = {
    'spherical': _spherical_covariance,
    'exponential': _exponential_covariance
}

# ==========================================================
# Helper Functions
# ==========================================================

def _indicator_transform(values: np.ndarray, threshold: float) -> np.ndarray:
    """Convert continuous data to binary indicator I(Z >= threshold): 1 if value >= threshold else 0."""
    return (np.asarray(values) >= threshold).astype(float)

def _apply_anisotropy(
    coords: np.ndarray,
    angle: float = 0.0,
    ratio: float = 1.0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply geometric anisotropy via coordinate transformation."""
    if angle == 0.0 and ratio == 1.0:
        return coords, None
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])
    inv_scale = np.diag([1.0 / ratio, 1.0])
    transform_matrix = inv_scale @ rotation.T
    coords_transformed = coords @ transform_matrix.T
    return coords_transformed, transform_matrix

def _solve_ordinary_kriging_system(
    neighbor_coords: np.ndarray,
    target: np.ndarray,
    cov_model,
    sill: float,
    rng: float,
    nugget: float,
    regularization: float = 1e-8
) -> np.ndarray:
    """
    Solve the ordinary kriging system with Lagrange multiplier, for one
    point (used by the `search_radius` fallback path; the default
    k-nearest path uses `_solve_ok_batch` instead).

    Returns:
        weights: array of kriging weights (length = n_neighbors)
    """
    n = neighbor_coords.shape[0]
    if n == 0:
        raise ValueError("No neighbors found.")

    # Build covariance matrix K (n+1 x n+1)
    diff = neighbor_coords[:, None, :] - neighbor_coords[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = cov_model(dists, sill, rng, nugget)
    K[:n, n] = 1.0
    K[n, :n] = 1.0
    K[n, n] = 0.0

    # Regularization for numerical stability
    if regularization > 0:
        K[:n, :n] += np.eye(n) * regularization

    # Right-hand side vector
    d0 = np.sqrt(np.sum((neighbor_coords - target) ** 2, axis=1))
    rhs = np.zeros(n + 1)
    rhs[:n] = cov_model(d0, sill, rng, nugget)
    rhs[n] = 1.0

    # Solve system
    try:
        solution = solve(K, rhs, assume_a="sym")
    except LinAlgError:
        # Fallback to least squares
        solution = np.linalg.lstsq(K, rhs, rcond=None)[0]

    return solution[:n]


def _solve_ok_batch(
    neighbor_coords: np.ndarray,   # (n_pred, k, 2)
    targets: np.ndarray,           # (n_pred, 2)
    cov_model,
    sill: float,
    rng: float,
    nugget: float,
    regularization: float,
    use_gpu: bool,
):
    """
    Solve `n_pred` independent ordinary-kriging systems in one batched
    linear-algebra call: builds a `(n_pred, k+1, k+1)` stack of kriging
    matrices and solves them all at once via `xp.linalg.solve`'s support
    for stacked/batched matrices (a real NumPy/CuPy feature, not a
    pygstat-specific trick). Valid only when every point has the *same*
    neighbor count `k` -- true for the k-nearest-neighbor search mode,
    where `k` never depends on the point.

    Returns
    -------
    weights : ndarray, shape (n_pred, k) -- always NumPy, even if solved on GPU.
    """
    n_pred, k, _ = neighbor_coords.shape

    if use_gpu:
        nc = as_gpu_array(neighbor_coords)
        tg = as_gpu_array(targets)
    else:
        nc, tg = neighbor_coords, targets
    xp = get_array_module(nc)

    diff = nc[:, :, None, :] - nc[:, None, :, :]
    dists = xp.sqrt(xp.sum(diff ** 2, axis=-1))  # (n_pred, k, k)

    K = xp.zeros((n_pred, k + 1, k + 1))
    K[:, :k, :k] = cov_model(dists, sill, rng, nugget)
    K[:, :k, k] = 1.0
    K[:, k, :k] = 1.0
    if regularization > 0:
        K[:, :k, :k] += xp.eye(k) * regularization

    d0 = xp.sqrt(xp.sum((nc - tg[:, None, :]) ** 2, axis=-1))  # (n_pred, k)
    # `xp.linalg.solve` on a stacked (n_pred, k+1, k+1) `K` needs `rhs` to
    # be (n_pred, k+1, 1) -- a batch of single-column matrices, not (n_pred,
    # k+1) -- the latter is instead read as *one* (k+1, n_pred)-shaped
    # multi-RHS problem and mismatches K's core dimension.
    rhs = xp.ones((n_pred, k + 1, 1))
    rhs[:, :k, 0] = cov_model(d0, sill, rng, nugget)

    weights = xp.linalg.solve(K, rhs)[:, :, 0]
    return to_numpy(weights[:, :k])

# ==========================================================
# Main Class
# ==========================================================

class IndicatorKriging:
    """
    Ordinary Indicator Kriging with per-threshold variogram support.

    Parameters
    ----------
    cov_model : str, default='spherical'
        Covariance model ('spherical' or 'exponential').
    max_neighbors : int, default=12
        Maximum number of neighbors used in kriging.
    search_radius : float, optional
        Search radius for neighbors. If None, uses k-nearest.
    anisotropy_angle : float, default=0.0
        Rotation angle for anisotropy (degrees).
    anisotropy_ratio : float, default=1.0
        Anisotropy ratio (major axis / minor axis).
    regularization : float, default=1e-8
        Regularization parameter for singular systems.
    use_gpu : bool or 'auto', default='auto'
        Solve the batched k-nearest-neighbor kriging systems (see module
        docstring) on the GPU via CuPy when usable; verified the same way
        as `pygstat.core.kriging` (not just "did CuPy import"), with
        automatic CPU fallback. Has no effect in `search_radius` mode,
        which isn't batchable and always runs on CPU.

    Examples
    --------
    >>> from pygstat import IndicatorKriging
    >>> ik = IndicatorKriging()
    >>> ik.fit(coords, values)
    >>> prob = ik.predict(coords_pred, thresholds=[100, 200])
    """

    def __init__(
        self,
        cov_model: str = 'spherical',
        max_neighbors: int = 12,
        search_radius: Optional[float] = None,
        anisotropy_angle: float = 0.0,
        anisotropy_ratio: float = 1.0,
        regularization: float = 1e-8,
        use_gpu: Union[bool, str] = "auto",
    ):
        if cov_model not in _COV_MODELS:
            raise ValueError(f"cov_model must be one of {list(_COV_MODELS.keys())}")

        self.cov_model_name = cov_model
        self.cov_model = _COV_MODELS[cov_model]
        self.max_neighbors = max_neighbors
        self.search_radius = search_radius
        self.anisotropy_angle = anisotropy_angle
        self.anisotropy_ratio = anisotropy_ratio
        self.regularization = regularization
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)

        # Fitted attributes
        self.coords_ = None
        self.values_ = None
        self.coords_transformed_ = None
        self.transform_matrix_ = None
        self.tree_ = None
        self.is_fitted_ = False

    def fit(self, coords: np.ndarray, values: np.ndarray) -> 'IndicatorKriging':
        """
        Fit the indicator kriging model (store training data).

        Parameters
        ----------
        coords : array-like, shape (n_samples, 2)
            Spatial coordinates of training data.
        values : array-like, shape (n_samples,)
            Primary variable values.

        Returns
        -------
        self : IndicatorKriging
        """
        coords = np.asarray(coords)
        values = np.asarray(values)

        if coords.shape[0] != values.shape[0]:
            raise ValueError("coords and values must have the same length.")
        if coords.shape[1] != 2:
            raise ValueError("coords must be 2D (x, y).")

        self.coords_ = coords.copy()
        self.values_ = values.copy()
        self.coords_transformed_, self.transform_matrix_ = _apply_anisotropy(
            self.coords_, self.anisotropy_angle, self.anisotropy_ratio
        )
        self.tree_ = cKDTree(self.coords_transformed_)
        self.is_fitted_ = True
        return self

    def predict(
        self,
        coords_pred: np.ndarray,
        thresholds: Union[float, List[float]],
        variogram_params: Optional[Dict[float, Tuple[float, float, float]]] = None
    ) -> Dict[float, np.ndarray]:
        """
        Predict probabilities for given thresholds.

        Parameters
        ----------
        coords_pred : array-like, shape (n_pred, 2)
            Prediction coordinates.
        thresholds : float or list of float
            Cutoff values for indicator transformation.
        variogram_params : dict, optional
            Mapping from threshold to (nugget, sill, range).
            If None, uses default parameters (sill=0.25, range=50% of max distance, nugget=0.01).

        Returns
        -------
        probabilities : dict
            {threshold: probability_array}
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction.")

        coords_pred = np.asarray(coords_pred)
        if coords_pred.shape[1] != 2:
            raise ValueError("coords_pred must be 2D (x, y).")

        thresholds = np.atleast_1d(thresholds).tolist()
        probabilities = {}

        # Default variogram parameters (if not provided)
        if variogram_params is None:
            max_dist = np.percentile(
                np.sqrt(np.sum((self.coords_[:, None, :] - self.coords_[None, :, :]) ** 2, axis=2)),
                50
            )
            default_params = (0.01, 0.25, max_dist * 0.6)
            variogram_params = {t: default_params for t in thresholds}

        # Transform prediction coordinates
        if self.transform_matrix_ is not None:
            coords_pred_t = coords_pred @ self.transform_matrix_.T
        else:
            coords_pred_t = coords_pred.copy()

        n_pred = len(coords_pred)

        for threshold in thresholds:
            if threshold not in variogram_params:
                raise KeyError(f"Variogram parameters missing for threshold {threshold}")

            nugget, sill, rng = variogram_params[threshold]
            indicator_train = _indicator_transform(self.values_, threshold)

            if self.search_radius is None:
                # Batched path: every point gets the same neighbor count k,
                # so all n_pred systems solve in one stacked call (see
                # _solve_ok_batch's docstring). This is the default mode.
                k = min(self.max_neighbors, len(self.coords_transformed_))
                _, neighbor_idxs = self.tree_.query(coords_pred_t, k=k)
                neighbor_idxs = np.atleast_2d(neighbor_idxs)
                if neighbor_idxs.shape[0] != n_pred:  # k==1 -> query returns (n_pred,)
                    neighbor_idxs = neighbor_idxs.reshape(n_pred, k)

                neighbor_coords = self.coords_transformed_[neighbor_idxs]  # (n_pred, k, 2)
                try:
                    weights = _solve_ok_batch(
                        neighbor_coords, coords_pred_t, self.cov_model,
                        sill, rng, nugget, self.regularization, self.use_gpu,
                    )
                    prob_pred = np.clip(
                        np.einsum("ij,ij->i", weights, indicator_train[neighbor_idxs]), 0.0, 1.0
                    )
                except Exception:
                    # Rare (e.g. a genuinely singular batch member); fall
                    # back to the robust per-point loop below, which
                    # handles failures one point at a time.
                    prob_pred = self._predict_loop(
                        coords_pred_t, neighbor_idxs, indicator_train,
                        sill, rng, nugget,
                    )
            else:
                prob_pred = np.full(n_pred, np.nan)
                for i, target_t in enumerate(coords_pred_t):
                    neighbor_idxs_i = self.tree_.query_ball_point(target_t, r=self.search_radius)
                    if len(neighbor_idxs_i) == 0:
                        continue
                    if len(neighbor_idxs_i) > self.max_neighbors:
                        dists = np.linalg.norm(
                            self.coords_transformed_[neighbor_idxs_i] - target_t, axis=1
                        )
                        neighbor_idxs_i = np.array(neighbor_idxs_i)[np.argsort(dists)[:self.max_neighbors]]

                    try:
                        weights = _solve_ordinary_kriging_system(
                            self.coords_transformed_[neighbor_idxs_i], target_t,
                            self.cov_model, sill, rng, nugget,
                            self.regularization
                        )
                        estimate = np.clip(np.dot(weights, indicator_train[neighbor_idxs_i]), 0.0, 1.0)
                        prob_pred[i] = estimate
                    except Exception:
                        prob_pred[i] = np.mean(indicator_train[neighbor_idxs_i])

            probabilities[threshold] = prob_pred

        return probabilities

    def _predict_loop(self, coords_pred_t, neighbor_idxs, indicator_train, sill, rng, nugget):
        """Per-point fallback for the k-nearest path (used only if the
        batched solve raises, e.g. a genuinely singular system for one
        point in the batch) -- same per-point robustness as the original
        implementation, just reached less often."""
        n_pred = len(coords_pred_t)
        prob_pred = np.full(n_pred, np.nan)
        for i, target_t in enumerate(coords_pred_t):
            idx = neighbor_idxs[i]
            try:
                weights = _solve_ordinary_kriging_system(
                    self.coords_transformed_[idx], target_t,
                    self.cov_model, sill, rng, nugget, self.regularization,
                )
                prob_pred[i] = np.clip(np.dot(weights, indicator_train[idx]), 0.0, 1.0)
            except Exception:
                prob_pred[i] = np.mean(indicator_train[idx])
        return prob_pred
