"""
Soft (Markov-Bayes) Indicator Kriging.

Combines sparse, exact **hard** indicator data with dense, uncertain **soft**
probability data -- typically calibrated from a cheap, widely-available secondary
variable (a correlated proxy assay, geophysics, remote sensing, expert judgment) --
using the Markov-Bayes approximation of Zhu & Journel (1993).

The appeal of Markov-Bayes is that it avoids fitting a full hard-soft
cross-variogram (data-hungry and often unstable). Instead it assumes the
hard-soft and soft-soft covariances are simple rescalings of the hard indicator's
own covariance model $C_H(h)$ by a single calibration coefficient $B$:

    C_HH(h) = C_H(h)
    C_HS(h) = B * C_H(h)
    C_SS(h) = B^2 * C_H(h)

where $B = \\mathrm{Cov}(I, F) / \\mathrm{Var}(F)$ is estimated from hard/soft pairs
observed at (or near) the same locations -- see :func:`markov_bayes_calibrate`.
The combined hard+soft data are then kriged with an ordinary-kriging system (single
unbiasedness constraint over all weights) to give a posterior probability at every
target location, exactly like :class:`pygstat.indicator_kriging.IndicatorKriging`
but informed by the extra soft evidence.

GPU (CuPy) support
-------------------
Every prediction point is independent of every other, so the default
k-nearest-neighbor mode (``search_radius=None``) batches into *one* stacked
linear-algebra solve over all ``n_pred`` points
(``xp.linalg.solve`` on a ``(n_pred, n_h+n_s+1, n_h+n_s+1)`` array) instead of
looping in Python. ``use_gpu='auto'|True|False`` runs that solve on the GPU
via CuPy when usable, same convention as :class:`pygstat.indicator_kriging.IndicatorKriging`.
The ``search_radius`` mode (variable neighbor counts per point) is not
batchable this way and keeps the original per-point loop (CPU only).

This module also provides :func:`fit_soft_probability_model`, a simple, robust way
to build the soft probability data in the first place: logistic-regression
calibration of a continuous proxy variable against co-located hard indicator data.

Reference
---------
Zhu, H. & Journel, A.G. (1993). Formatting and integrating soft data: stochastic
imaging via the Markov-Bayes algorithm. In *Geostatistics Troia '92*, 1-12.
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy.linalg import LinAlgError, solve
from scipy.spatial import cKDTree

from .utils.backend import get_array_module, as_gpu_array, to_numpy, resolve_cupy_use_gpu

__all__ = [
    "indicator_transform",
    "markov_bayes_calibrate",
    "fit_soft_probability_model",
    "SoftKriging",
]


# ==========================================================
# Covariance models (same (nugget, sill, range) convention as
# pygstat.indicator_kriging / pygstat.core.variogram_models)
# Array-module-agnostic (`xp` = numpy or cupy) so the same formula
# works for a 1-D per-point array and an (n_pred, k, k) batch.
# ==========================================================

def _spherical_covariance(h, sill: float, rng: float, nugget: float = 0.0):
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    c = xp.zeros_like(h)
    mask = h < rng
    hr = h[mask] / rng
    c[mask] = sill * (1.0 - 1.5 * hr + 0.5 * hr ** 3)
    c[h == 0] += nugget
    return c


def _exponential_covariance(h, sill: float, rng: float, nugget: float = 0.0):
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    c = sill * xp.exp(-h / rng)
    c[h == 0] += nugget
    return c


def _gaussian_covariance(h, sill: float, rng: float, nugget: float = 0.0):
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    c = sill * xp.exp(-(h / rng) ** 2)
    c[h == 0] += nugget
    return c


_COV_MODELS = {
    "spherical": _spherical_covariance,
    "exponential": _exponential_covariance,
    "gaussian": _gaussian_covariance,
}


# ==========================================================
# Helpers
# ==========================================================

def indicator_transform(values: np.ndarray, threshold: float, exceed: bool = False) -> np.ndarray:
    """
    Convert continuous data to a binary indicator.

    Parameters
    ----------
    values : array-like
    threshold : float
    exceed : bool, default False
        If False (default), ``I = 1`` where ``values <= threshold`` (matches
        GSLIB's cumulative-distribution convention). If True, ``I = 1`` where
        ``values >= threshold`` (an exceedance probability, e.g. contamination
        above a regulatory limit).
    """
    values = np.asarray(values, dtype=float)
    return (values >= threshold).astype(float) if exceed else (values <= threshold).astype(float)


def markov_bayes_calibrate(
    hard_coords: np.ndarray,
    hard_indicator: np.ndarray,
    soft_coords: np.ndarray,
    soft_prob: np.ndarray,
    max_pair_dist: Optional[float] = None,
) -> Tuple[float, float, int]:
    """
    Estimate the Markov-Bayes calibration coefficient
    :math:`B = \\mathrm{Cov}(I, F) / \\mathrm{Var}(F)` (Zhu & Journel, 1993) from
    hard indicator data and co-located (or nearby) soft probability data.

    Parameters
    ----------
    hard_coords : array-like, shape (n_hard, d)
    hard_indicator : array-like, shape (n_hard,)
        0/1 hard indicator values (see :func:`indicator_transform`).
    soft_coords : array-like, shape (n_soft, d)
    soft_prob : array-like, shape (n_soft,)
        Soft probability values in ``[0, 1]``.
    max_pair_dist : float, optional
        Only use hard/soft pairs within this distance of each other (default:
        use every hard datum's nearest soft neighbor regardless of distance --
        appropriate when hard and soft data share the same locations, as with
        two assays measured on the same samples).

    Returns
    -------
    B : float
        The calibration coefficient. ``B=1`` means the soft data behave exactly
        like additional hard data; ``B=0`` means they carry no information.
    corr : float
        Correlation between the hard indicator and the co-located soft
        probability -- a quick diagnostic of how informative the soft data are.
    n_pairs : int
        Number of hard/soft pairs actually used.
    """
    hard_coords = np.atleast_2d(hard_coords)
    soft_coords = np.atleast_2d(soft_coords)
    hard_indicator = np.asarray(hard_indicator, dtype=float)
    soft_prob = np.asarray(soft_prob, dtype=float)

    tree = cKDTree(soft_coords)
    dist, idx = tree.query(hard_coords)
    mask = dist <= max_pair_dist if max_pair_dist is not None else np.ones(len(hard_coords), dtype=bool)

    I = hard_indicator[mask]
    F = soft_prob[idx[mask]]
    n_pairs = int(mask.sum())
    if n_pairs < 2:
        raise ValueError("Fewer than 2 hard/soft pairs found; cannot calibrate B.")

    var_f = np.var(F)
    if var_f <= 0:
        return 0.0, 0.0, n_pairs

    cov_if = np.cov(I, F)[0, 1]
    var_i = np.var(I)
    corr = cov_if / np.sqrt(var_i * var_f) if var_i > 0 else 0.0
    return float(cov_if / var_f), float(corr), n_pairs


def fit_soft_probability_model(proxy_values, indicator, proxy_values_pred=None, **logreg_kwargs):
    """
    Calibrate soft probabilities :math:`P(I=1 \\mid \\text{proxy})` from a
    continuous secondary/proxy variable via logistic regression -- a simple,
    robust default for the calibration step Zhu & Journel's method requires
    before Markov-Bayes kriging can be applied.

    Parameters
    ----------
    proxy_values : array-like, shape (n,)
        Proxy variable values at locations where the hard indicator is also
        known (used to fit the calibration).
    indicator : array-like, shape (n,)
        Co-located 0/1 hard indicator values.
    proxy_values_pred : array-like, optional
        Proxy values at the (typically much more numerous) locations where
        soft probabilities are needed. If given, also returns the predicted
        probabilities there.
    **logreg_kwargs
        Passed through to ``sklearn.linear_model.LogisticRegression``.

    Returns
    -------
    model : sklearn.linear_model.LogisticRegression
        The fitted calibration model (``model.predict_proba`` accepts any new
        proxy values reshaped to ``(-1, 1)``).
    soft_prob : np.ndarray, optional
        Only returned if `proxy_values_pred` is given: predicted
        :math:`P(I=1)` at those proxy values.
    """
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(proxy_values, dtype=float).reshape(-1, 1)
    y = np.asarray(indicator, dtype=float)
    model = LogisticRegression(**logreg_kwargs)
    model.fit(X, y)

    if proxy_values_pred is None:
        return model
    Xp = np.asarray(proxy_values_pred, dtype=float).reshape(-1, 1)
    return model, model.predict_proba(Xp)[:, 1]


def _solve_markov_bayes_system(
    hard_nb: np.ndarray, soft_nb: np.ndarray, target: np.ndarray,
    cov_model, sill: float, rng: float, nugget: float, B: float, regularization: float,
):
    """Assemble and solve one local ordinary Markov-Bayes kriging system."""
    n_h, n_s = len(hard_nb), len(soft_nb)
    n = n_h + n_s
    coords = np.vstack([hard_nb, soft_nb]) if n_s > 0 else hard_nb

    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    scale = np.ones((n, n))
    scale[:n_h, n_h:] = B
    scale[n_h:, :n_h] = B
    scale[n_h:, n_h:] = B ** 2

    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = cov_model(d, sill, rng, nugget) * scale
    K[:n, n] = 1.0
    K[n, :n] = 1.0
    if regularization > 0:
        K[:n, :n] += np.eye(n) * regularization

    d0 = np.sqrt(((coords - target) ** 2).sum(-1))
    c0 = cov_model(d0, sill, rng, nugget)
    c0[n_h:] *= B
    rhs = np.zeros(n + 1)
    rhs[:n] = c0
    rhs[n] = 1.0

    try:
        w = solve(K, rhs, assume_a="sym")
    except LinAlgError:
        w = np.linalg.lstsq(K, rhs, rcond=None)[0]
    return w[:n], c0, w[n]


def _solve_markov_bayes_batch(
    hard_nb: np.ndarray,   # (n_pred, n_h, d)
    soft_nb: np.ndarray,   # (n_pred, n_s, d) -- n_s may be 0
    targets: np.ndarray,   # (n_pred, d)
    cov_model,
    sill: float,
    rng: float,
    nugget: float,
    B: float,
    regularization: float,
    use_gpu: bool,
):
    """
    Solve ``n_pred`` independent Markov-Bayes ordinary-kriging systems in
    one batched linear-algebra call. Valid only when every point has the
    *same* hard count ``n_h`` and soft count ``n_s`` -- true for k-nearest
    neighbor search.

    Returns
    -------
    weights : ndarray, shape (n_pred, n_h + n_s)
    c_vec : ndarray, shape (n_pred, n_h + n_s)
    mu : ndarray, shape (n_pred,)
        Always NumPy, even if solved on GPU.
    """
    n_pred, n_h, _ = hard_nb.shape
    n_s = 0 if soft_nb.size == 0 else soft_nb.shape[1]
    n = n_h + n_s

    if n_s > 0:
        coords = np.concatenate([hard_nb, soft_nb], axis=1)
    else:
        coords = hard_nb

    if use_gpu:
        coords_d = as_gpu_array(coords)
        tg = as_gpu_array(targets)
    else:
        coords_d, tg = coords, targets
    xp = get_array_module(coords_d)

    diff = coords_d[:, :, None, :] - coords_d[:, None, :, :]
    dists = xp.sqrt(xp.sum(diff ** 2, axis=-1))  # (n_pred, n, n)

    scale = xp.ones((n_pred, n, n))
    if n_s > 0:
        scale[:, :n_h, n_h:] = B
        scale[:, n_h:, :n_h] = B
        scale[:, n_h:, n_h:] = B ** 2

    K = xp.zeros((n_pred, n + 1, n + 1))
    K[:, :n, :n] = cov_model(dists, sill, rng, nugget) * scale
    K[:, :n, n] = 1.0
    K[:, n, :n] = 1.0
    if regularization > 0:
        K[:, :n, :n] += xp.eye(n) * regularization

    d0 = xp.sqrt(xp.sum((coords_d - tg[:, None, :]) ** 2, axis=-1))
    c0 = cov_model(d0, sill, rng, nugget)
    if n_s > 0:
        c0 = xp.concatenate([c0[:, :n_h], c0[:, n_h:] * B], axis=1)

    rhs = xp.ones((n_pred, n + 1, 1))
    rhs[:, :n, 0] = c0
    rhs[:, n, 0] = 1.0

    sol = xp.linalg.solve(K, rhs)[:, :, 0]
    return to_numpy(sol[:, :n]), to_numpy(c0), to_numpy(sol[:, n])


# ==========================================================
# Main class
# ==========================================================

class SoftKriging:
    """
    Ordinary Markov-Bayes soft (indicator) kriging.

    Krige a probability field from sparse hard indicator data plus dense soft
    probability data, using a single hard-indicator covariance model rescaled by
    the Markov-Bayes calibration coefficient ``B`` (see module docstring and
    :func:`markov_bayes_calibrate`).

    Parameters
    ----------
    cov_model : {'spherical', 'exponential', 'gaussian'}, default 'exponential'
        Covariance model for the **hard** indicator (also reused, scaled by
        `B`/`B**2`, for the hard-soft and soft-soft covariances).
    max_neighbors_hard : int, default 20
        Maximum hard data used per local kriging system.
    max_neighbors_soft : int, default 20
        Maximum soft data used per local kriging system.
    search_radius : float, optional
        If given, restrict neighbors (of both kinds) to within this radius;
        otherwise use the `max_neighbors_*` closest regardless of distance.
    regularization : float, default 1e-5
        Added to the covariance matrix diagonal for numerical stability --
        important because dense soft data (e.g. a fine geophysical grid) can
        make the soft-soft block nearly singular.
    use_gpu : bool or 'auto', default='auto'
        Solve the batched k-nearest-neighbor Markov-Bayes systems (see module
        docstring) on the GPU via CuPy when usable; verified the same way
        as `pygstat.core.kriging` (not just "did CuPy import"), with
        automatic CPU fallback. Has no effect in `search_radius` mode,
        which isn't batchable and always runs on CPU.

    Notes
    -----
    Avoid feeding a soft datum that sits at the *exact same location* as a
    hard datum into the same local system: since ``C_HS = B * C_H`` at zero
    distance, that soft row becomes an exact multiple of the hard row,
    making the local matrix rank-deficient (``regularization`` and the
    least-squares fallback keep `predict` from crashing, but the estimate is
    less meaningful there). In practice this just means: build soft coverage
    for locations that *lack* a hard measurement -- at a hard location, use
    the hard value directly.

    Examples
    --------
    >>> sk = SoftKriging(cov_model="exponential")
    >>> sk.fit(hard_coords, hard_indicator, soft_coords, soft_prob)
    >>> prob = sk.predict(coords_pred, sill=0.2, range_=1.5, nugget=0.02)
    """

    def __init__(
        self,
        cov_model: str = "exponential",
        max_neighbors_hard: int = 20,
        max_neighbors_soft: int = 20,
        search_radius: Optional[float] = None,
        regularization: float = 1e-5,
        use_gpu: Union[bool, str] = "auto",
    ):
        if cov_model not in _COV_MODELS:
            raise ValueError(f"cov_model must be one of {list(_COV_MODELS.keys())}")
        self.cov_model_name = cov_model
        self.cov_model = _COV_MODELS[cov_model]
        self.max_neighbors_hard = max_neighbors_hard
        self.max_neighbors_soft = max_neighbors_soft
        self.search_radius = search_radius
        self.regularization = regularization
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)
        self.is_fitted_ = False

    def fit(
        self,
        hard_coords: np.ndarray,
        hard_indicator: np.ndarray,
        soft_coords: np.ndarray,
        soft_prob: np.ndarray,
        B: Optional[float] = None,
        max_pair_dist: Optional[float] = None,
    ) -> "SoftKriging":
        """
        Store training data and (unless `B` is given directly) calibrate the
        Markov-Bayes coefficient with :func:`markov_bayes_calibrate`.
        """
        hard_coords = np.atleast_2d(np.asarray(hard_coords, dtype=float))
        soft_coords = np.atleast_2d(np.asarray(soft_coords, dtype=float))
        hard_indicator = np.asarray(hard_indicator, dtype=float)
        soft_prob = np.asarray(soft_prob, dtype=float)

        if len(hard_coords) != len(hard_indicator):
            raise ValueError("hard_coords and hard_indicator must have the same length")
        if len(soft_coords) != len(soft_prob):
            raise ValueError("soft_coords and soft_prob must have the same length")

        self.hard_coords_ = hard_coords
        self.hard_indicator_ = hard_indicator
        self.soft_coords_ = soft_coords
        self.soft_prob_ = soft_prob
        self.hard_tree_ = cKDTree(hard_coords)
        self.soft_tree_ = cKDTree(soft_coords) if len(soft_coords) else None

        if B is None:
            self.B_, self.calibration_corr_, self.n_calibration_pairs_ = markov_bayes_calibrate(
                hard_coords, hard_indicator, soft_coords, soft_prob, max_pair_dist=max_pair_dist,
            )
        else:
            self.B_, self.calibration_corr_, self.n_calibration_pairs_ = float(B), np.nan, 0

        self.is_fitted_ = True
        return self

    def _neighbors(self, tree, coords, target, k):
        """Return (neighbor_coords, neighbor_idx) -- always a matching pair,
        empty arrays of the right shape when there is nothing to find."""
        n_dims = target.shape[0]
        empty = (np.empty((0, n_dims)), np.empty(0, dtype=int))
        if tree is None or k <= 0 or len(coords) == 0:
            return empty
        if self.search_radius is not None:
            idx = tree.query_ball_point(target, r=self.search_radius)
            if len(idx) == 0:
                return empty
            idx = np.asarray(idx)
            if len(idx) > k:
                d = np.linalg.norm(coords[idx] - target, axis=1)
                idx = idx[np.argsort(d)[:k]]
            return coords[idx], idx
        k = min(k, len(coords))
        _, idx = tree.query(target, k=k)
        idx = np.atleast_1d(idx)
        return coords[idx], idx

    def _knn_indices(self, tree, n_data, coords_pred, k):
        """Batched k-NN indices, shape (n_pred, k)."""
        k = min(int(k), n_data)
        _, idx = tree.query(coords_pred, k=k)
        idx = np.atleast_2d(np.asarray(idx))
        if idx.shape[0] != len(coords_pred):
            idx = idx.reshape(len(coords_pred), k)
        return k, idx

    def predict(
        self,
        coords_pred: np.ndarray,
        sill: float,
        range_: float,
        nugget: float = 0.0,
        return_variance: bool = False,
    ):
        """
        Predict the posterior probability (and optionally kriging variance) at
        `coords_pred`, combining nearby hard and soft data.

        Parameters
        ----------
        coords_pred : array-like, shape (n_pred, d)
        sill, range_, nugget : float
            Parameters of the **hard** indicator covariance model.
        return_variance : bool, default False

        Returns
        -------
        prob : np.ndarray, shape (n_pred,)
            Posterior probability, clipped to ``[0, 1]``.
        variance : np.ndarray, shape (n_pred,), optional
            Only returned if `return_variance=True`.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction.")
        coords_pred = np.atleast_2d(np.asarray(coords_pred, dtype=float))

        n_pred = len(coords_pred)
        c0_sill = sill + nugget

        use_batch = (
            self.search_radius is None
            and len(self.hard_coords_) > 0
            and self.max_neighbors_hard >= 1
            and n_pred > 0
        )
        if use_batch:
            try:
                return self._predict_knn_batch(
                    coords_pred, sill, range_, nugget, c0_sill, return_variance,
                )
            except Exception:
                pass  # fall through to the robust per-point loop

        return self._predict_loop(coords_pred, sill, range_, nugget, c0_sill, return_variance)

    def _predict_knn_batch(self, coords_pred, sill, range_, nugget, c0_sill, return_variance):
        n_pred = len(coords_pred)
        n_h, hard_idx = self._knn_indices(
            self.hard_tree_, len(self.hard_coords_), coords_pred, self.max_neighbors_hard,
        )
        hard_nb = self.hard_coords_[hard_idx]

        use_soft = (
            self.soft_tree_ is not None
            and self.max_neighbors_soft > 0
            and len(self.soft_coords_) > 0
        )
        if use_soft:
            n_s, soft_idx = self._knn_indices(
                self.soft_tree_, len(self.soft_coords_), coords_pred, self.max_neighbors_soft,
            )
            soft_nb = self.soft_coords_[soft_idx]
        else:
            n_s, soft_idx = 0, np.empty((n_pred, 0), dtype=int)
            soft_nb = np.empty((n_pred, 0, coords_pred.shape[1]))

        w, c_vec, mu = _solve_markov_bayes_batch(
            hard_nb, soft_nb, coords_pred, self.cov_model, sill, range_, nugget,
            self.B_, self.regularization, self.use_gpu,
        )
        if n_s > 0:
            data_vals = np.concatenate(
                [self.hard_indicator_[hard_idx], self.soft_prob_[soft_idx]], axis=1,
            )
        else:
            data_vals = self.hard_indicator_[hard_idx]
        prob = np.clip(np.einsum("ij,ij->i", w, data_vals), 0.0, 1.0)
        if not return_variance:
            return prob
        var = np.maximum(c0_sill - np.einsum("ij,ij->i", w, c_vec) - mu, 0.0)
        return prob, var

    def _predict_loop(self, coords_pred, sill, range_, nugget, c0_sill, return_variance):
        n_pred = len(coords_pred)
        prob = np.full(n_pred, np.nan)
        var = np.full(n_pred, np.nan)

        for i, target in enumerate(coords_pred):
            hard_nb, hard_idx = self._neighbors(
                self.hard_tree_, self.hard_coords_, target, self.max_neighbors_hard,
            )
            if len(hard_nb) == 0:
                continue
            if self.soft_tree_ is not None and self.max_neighbors_soft > 0:
                soft_nb, soft_idx = self._neighbors(
                    self.soft_tree_, self.soft_coords_, target, self.max_neighbors_soft,
                )
            else:
                soft_nb, soft_idx = np.empty((0, coords_pred.shape[1])), np.empty(0, dtype=int)

            try:
                w, c_vec, mu = _solve_markov_bayes_system(
                    hard_nb, soft_nb, target, self.cov_model, sill, range_, nugget,
                    self.B_, self.regularization,
                )
                data_vals = np.concatenate([self.hard_indicator_[hard_idx], self.soft_prob_[soft_idx]])
                est = np.dot(w, data_vals)
                prob[i] = np.clip(est, 0.0, 1.0)
                if return_variance:
                    var[i] = max(c0_sill - np.dot(w, c_vec) - mu, 0.0)
            except Exception:
                prob[i] = np.mean(self.hard_indicator_[hard_idx])
                if return_variance:
                    var[i] = c0_sill

        if return_variance:
            return prob, var
        return prob
