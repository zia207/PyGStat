"""
Factorial Kriging Analysis (FKA).

Estimates an individual *structure* of a nested variogram model -- a nugget, a
short-range structure, a long-range structure, or any sum of them -- rather than
the regionalized variable itself (Matheron, 1982; Sandjivy, 1984). If

    Z(x) = m + Z_1(x) + Z_2(x) + ... + Z_S(x)

decomposes into mutually uncorrelated, zero-mean components ("factors") each with
its own covariance C_u(h), and the nested variogram model is their sum,
gamma(h) = sum_u gamma_u(h), then a factor Z_v(x0) can be estimated linearly from
the same data used to krige Z itself:

    Z_v*(x0) = sum_i lambda_i Z(x_i),   subject to sum_i lambda_i = 0

(0, not 1, because a factor's mean is zero -- this filters the unknown mean *and*
every structure but v out of the estimate). Minimizing the estimation variance
gives a kriging system whose **left-hand side is the ordinary kriging matrix built
from the full nested model** (identical for every factor) and whose **right-hand
side uses only the target structure's covariance**:

    [ C   1 ] [lambda_v]   [c_v]
    [ 1'  0 ] [  mu    ] = [ 0 ]

where C_ij = sum_u C_u(x_i - x_j) (as in ordinary kriging) and c_v,i = C_v(x_i-x0).
Plain ordinary kriging of Z itself is the special case "factor = all structures,
constraint = 1" -- see :meth:`FactorialKriging.extract`.

Typical uses: filtering measurement noise out of a map (extract everything but the
nugget), or separating a short-range (local) anomaly from a long-range (regional)
trend that are folded into the same nested variogram -- the classic mining/
geochemistry application this module is tested against below.

GPU (CuPy) support
-------------------
Every prediction point is independent of every other, so the default
k-nearest-neighbor mode (``search_radius=None``) batches into *one* stacked
linear-algebra solve over all ``n_pred`` points
(``xp.linalg.solve`` on a ``(n_pred, k+1, k+1)`` array) instead of looping
in Python. ``use_gpu='auto'|True|False`` runs that solve on the GPU via CuPy
when usable, same convention as :class:`pygstat.indicator_kriging.IndicatorKriging`.
The ``search_radius`` mode (variable neighbor counts per point) is not
batchable this way and keeps the original per-point loop (CPU only).

Reference
---------
Matheron, G. (1982). *Pour une analyse krigeante des donnees regionalisees*.
Note N-732, Centre de Geostatistique, Fontainebleau.
Sandjivy, L. (1984). The factorial kriging analysis of regionalized data: its
application to geochemical prospecting. In *Geostatistics for Natural Resources
Characterization*, Reidel.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import LinAlgError, solve
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

from .utils.backend import as_gpu_array, get_array_module, resolve_cupy_use_gpu, to_numpy

__all__ = ["FactorialKriging", "fit_nested_variogram"]


# ==========================================================
# Covariance models (same (sill, range, nugget) convention as
# pygstat.indicator_kriging / pygstat.soft_kriging)
# ==========================================================

# Array-module-agnostic (`xp` = numpy or cupy) so the same formula works for
# a 1-D per-point array and an (n_pred, k, k) batch.


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

_GAMMA_SHAPES = {
    "spherical": lambda u: np.where(u < 1.0, 1.5 * u - 0.5 * u ** 3, 1.0),
    "exponential": lambda u: 1.0 - np.exp(-u),
    "gaussian": lambda u: 1.0 - np.exp(-u ** 2),
}


def _validate_structures(structures: Sequence[Dict]) -> List[Dict]:
    out = []
    for st in structures:
        model = st["model"]
        if model not in _COV_MODELS:
            raise ValueError(f"Unknown model '{model}'; expected one of {list(_COV_MODELS)}")
        out.append({"model": model, "sill": float(st["sill"]), "range": float(st["range"])})
    return out


# ==========================================================
# Covariance of a chosen factor (or the total model)
# ==========================================================

def _total_covariance(h, nugget: float, structures: List[Dict]):
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    c = xp.zeros_like(h)
    for st in structures:
        c = c + _COV_MODELS[st["model"]](h, st["sill"], st["range"])
    c[h == 0] += nugget
    return c


def _resolve_factor_indices(factor: Union[str, int, Sequence[int]], n_structures: int) -> List[int]:
    """Map a user-facing `factor` spec to a list of structure indices (1..S),
    with 0 reserved for the nugget."""
    if isinstance(factor, str):
        key = factor.lower()
        if key in ("total", "all", "z"):
            return list(range(0, n_structures + 1))
        if key in ("nugget", "noise"):
            return [0]
        if key in ("signal", "denoised"):
            return list(range(1, n_structures + 1))
        raise ValueError("factor string must be 'total', 'nugget', 'signal', or 'denoised'")
    if isinstance(factor, (int, np.integer)):
        indices = [int(factor)]
    else:
        indices = [int(f) for f in factor]
    for idx in indices:
        if not (0 <= idx <= n_structures):
            raise ValueError(f"structure index {idx} out of range [0, {n_structures}]")
    return indices


def _factor_covariance(h, nugget: float, structures: List[Dict], indices: List[int]):
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    c = xp.zeros_like(h)
    if 0 in indices:
        c[h == 0] += nugget
    for idx in indices:
        if idx == 0:
            continue
        st = structures[idx - 1]
        c = c + _COV_MODELS[st["model"]](h, st["sill"], st["range"])
    return c


def _factor_covariance_at_zero(nugget: float, structures: List[Dict], indices: List[int]) -> float:
    c0 = nugget if 0 in indices else 0.0
    for idx in indices:
        if idx != 0:
            c0 += structures[idx - 1]["sill"]
    return c0


# ==========================================================
# Empirical nested-variogram fitting
# ==========================================================

def fit_nested_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    models: Sequence[str] = ("spherical", "spherical"),
    n_lags: int = 15,
    maxlag: Optional[float] = None,
) -> Tuple[float, List[Dict], np.ndarray, np.ndarray]:
    """
    Fit a nugget + nested sum of `models` structures to the experimental
    semivariogram of `values` at `coords`, via least squares.

    Parameters
    ----------
    coords : array-like, shape (n, d)
    values : array-like, shape (n,)
    models : sequence of {'spherical', 'exponential', 'gaussian'}
        One entry per structure, in increasing order of range is typical
        (e.g. `('spherical', 'spherical')` for a short + long range model)
        but not required.
    n_lags : int, default 15
        Number of experimental semivariogram lag bins.
    maxlag : float, optional
        Maximum lag distance to bin (default: half the maximum pairwise distance).

    Returns
    -------
    nugget : float
    structures : list of dict
        `[{"model": ..., "sill": ..., "range": ...}, ...]`, one per entry of `models`.
    lag_h, lag_g : np.ndarray
        The experimental semivariogram points the model was fit to (for plotting).
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    values = np.asarray(values, dtype=float)
    models = list(models)

    d = pdist(coords)
    diffsq = pdist(values[:, None]) ** 2
    maxlag = maxlag if maxlag is not None else d.max() * 0.5
    bins = np.linspace(0.0, maxlag, n_lags + 1)
    lag_h, lag_g = [], []
    for i in range(n_lags):
        m = (d >= bins[i]) & (d < bins[i + 1])
        if m.sum() > 5:
            lag_h.append(d[m].mean())
            lag_g.append(0.5 * diffsq[m].mean())
    lag_h, lag_g = np.array(lag_h), np.array(lag_g)
    if len(lag_h) < len(models) + 2:
        raise ValueError("Not enough populated lag bins to fit this many structures; "
                          "reduce n_lags or the number of structures")

    total_var = float(np.var(values))
    n_struct = len(models)

    def nested_gamma(h, *params):
        nugget = params[0]
        g = np.full_like(h, nugget, dtype=float)
        for k in range(n_struct):
            sill, rng = params[1 + 2 * k], params[2 + 2 * k]
            g = g + sill * _GAMMA_SHAPES[models[k]](h / rng)
        return g

    p0 = [total_var / (n_struct + 1)]
    lo, hi = [0.0], [total_var]
    for k in range(n_struct):
        p0 += [total_var / (n_struct + 1), maxlag * (k + 1) / n_struct]
        lo += [0.0, 1e-6]
        hi += [total_var * 2, maxlag * 3]

    popt, _ = curve_fit(nested_gamma, lag_h, lag_g, p0=p0, bounds=(lo, hi), maxfev=20000)

    nugget = float(popt[0])
    structures = [
        {"model": models[k], "sill": float(popt[1 + 2 * k]), "range": float(popt[2 + 2 * k])}
        for k in range(n_struct)
    ]
    # Report structures ordered by increasing range (so "structure 1" is always
    # the shortest-range one, "structure S" the longest), regardless of fit order.
    structures.sort(key=lambda s: s["range"])
    return nugget, structures, lag_h, lag_g


# ==========================================================
# Batched k-NN factorial-kriging solve
# ==========================================================

def _solve_fk_batch(
    neighbor_coords,  # (n_pred, k, d)
    targets,          # (n_pred, d)
    neighbor_vals,    # (n_pred, k)
    nugget: float,
    structures: List[Dict],
    indices: List[int],
    constraint: float,
    regularization: float,
    use_gpu: bool,
    return_variance: bool,
    c_v0: float,
):
    """
    Solve ``n_pred`` independent factorial-kriging systems in one batched
    linear-algebra call. Valid only when every point has the same neighbor
    count ``k`` (k-nearest-neighbor search).

    Returns
    -------
    estimate : ndarray, shape (n_pred,) -- always NumPy
    variance : ndarray, shape (n_pred,) or None
    """
    n_pred, k, _ = neighbor_coords.shape
    if use_gpu:
        nc = as_gpu_array(neighbor_coords)
        tg = as_gpu_array(targets)
        nv = as_gpu_array(neighbor_vals)
    else:
        nc, tg, nv = neighbor_coords, targets, neighbor_vals
    xp = get_array_module(nc)

    diff = nc[:, :, None, :] - nc[:, None, :, :]
    dists = xp.sqrt(xp.sum(diff ** 2, axis=-1))  # (n_pred, k, k)

    K = xp.zeros((n_pred, k + 1, k + 1))
    K[:, :k, :k] = _total_covariance(dists, nugget, structures)
    K[:, :k, k] = 1.0
    K[:, k, :k] = 1.0
    if regularization > 0:
        K[:, :k, :k] += xp.eye(k) * regularization

    d0 = xp.sqrt(xp.sum((nc - tg[:, None, :]) ** 2, axis=-1))  # (n_pred, k)
    rhs = xp.zeros((n_pred, k + 1, 1))
    rhs[:, :k, 0] = _factor_covariance(d0, nugget, structures, indices)
    rhs[:, k, 0] = constraint

    w = xp.linalg.solve(K, rhs)[:, :, 0]  # (n_pred, k+1)
    est = xp.einsum("ij,ij->i", w[:, :k], nv)
    est = to_numpy(est)
    if not return_variance:
        return est, None
    var = c_v0 - xp.einsum("ij,ij->i", w[:, :k], rhs[:, :k, 0]) - w[:, k]
    var = to_numpy(xp.maximum(var, 0.0))
    return est, var


# ==========================================================
# Main class
# ==========================================================

class FactorialKriging:
    """
    Factorial Kriging Analysis: estimate individual structures of a nested
    variogram model, not just their sum.

    Parameters
    ----------
    nugget : float
        Nugget effect (structure 0).
    structures : list of dict
        Nested structures, e.g. `[{"model": "spherical", "sill": 5.0, "range": 0.5},
        {"model": "spherical", "sill": 3.0, "range": 3.0}]` -- a short-range
        structure (index 1) and a long-range structure (index 2). Together with
        `nugget` these must sum to the (already-fitted) variogram of the data --
        see :func:`fit_nested_variogram`.
    max_neighbors : int, default 30
    search_radius : float, optional
        If given, restrict neighbors to within this radius (falling back to the
        `max_neighbors` closest of those); otherwise always use the
        `max_neighbors` closest data regardless of distance.
    regularization : float, default 1e-8
    use_gpu : bool or 'auto', default='auto'
        Solve the batched k-nearest-neighbor kriging systems (see module
        docstring) on the GPU via CuPy when usable; verified the same way
        as `pygstat.core.kriging` (not just "did CuPy import"), with
        automatic CPU fallback. Has no effect in `search_radius` mode,
        which isn't batchable and always runs on CPU.

    Examples
    --------
    >>> fk = FactorialKriging(nugget=0.5, structures=[
    ...     {"model": "spherical", "sill": 2.0, "range": 0.5},
    ...     {"model": "spherical", "sill": 1.5, "range": 3.0},
    ... ])
    >>> fk.fit(coords, values)
    >>> z_hat = fk.extract(coords_pred, factor="total")       # = ordinary kriging of Z
    >>> short = fk.extract(coords_pred, factor=1)             # short-range factor only
    >>> long_ = fk.extract(coords_pred, factor=2)              # long-range factor only
    >>> denoised = fk.extract(coords_pred, factor="denoised") # Z with the nugget filtered out
    """

    def __init__(
        self,
        nugget: float,
        structures: Sequence[Dict],
        max_neighbors: int = 30,
        search_radius: Optional[float] = None,
        regularization: float = 1e-8,
        use_gpu: Union[bool, str] = "auto",
    ):
        self.nugget = float(nugget)
        self.structures = _validate_structures(structures)
        self.max_neighbors = max_neighbors
        self.search_radius = search_radius
        self.regularization = regularization
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)
        self.is_fitted_ = False

    @property
    def n_structures(self) -> int:
        return len(self.structures)

    def total_sill(self) -> float:
        """C(0): nugget plus every structure's sill."""
        return self.nugget + sum(st["sill"] for st in self.structures)

    def fit(self, coords: np.ndarray, values: np.ndarray) -> "FactorialKriging":
        """Store the conditioning data (coords, values) used by :meth:`extract`."""
        coords = np.atleast_2d(np.asarray(coords, dtype=float))
        values = np.asarray(values, dtype=float)
        if len(coords) != len(values):
            raise ValueError("coords and values must have the same length")
        self.coords_ = coords
        self.values_ = values
        self.tree_ = cKDTree(coords)
        self.is_fitted_ = True
        return self

    def _neighbors(self, target: np.ndarray):
        if self.search_radius is not None:
            idx = self.tree_.query_ball_point(target, r=self.search_radius)
            if len(idx) == 0:
                return np.empty(0, dtype=int)
            idx = np.asarray(idx)
            if len(idx) > self.max_neighbors:
                d = np.linalg.norm(self.coords_[idx] - target, axis=1)
                idx = idx[np.argsort(d)[: self.max_neighbors]]
            return idx
        k = min(self.max_neighbors, len(self.coords_))
        _, idx = self.tree_.query(target, k=k)
        return np.atleast_1d(idx)

    def extract(
        self,
        coords_pred: np.ndarray,
        factor: Union[str, int, Sequence[int]] = "total",
        return_variance: bool = False,
    ):
        """
        Estimate the requested factor at `coords_pred`.

        Parameters
        ----------
        coords_pred : array-like, shape (n_pred, d)
        factor : {'total', 'denoised', 'nugget', 'signal'}, int, or sequence of int
            Which structure(s) to estimate:

            - ``'total'`` -- ordinary kriging of Z itself (constraint sum=1).
            - ``'denoised'`` -- Z with the nugget's noise-spike removed from the
              *target* covariance only (constraint sum=1, so it stays in the
              same units/scale as Z, mean included): a practical noise filter.
            - ``'signal'`` -- the pure, zero-mean structured fluctuation
              (everything but the nugget, constraint sum=0): the strict
              Matheron "factor" sense, orthogonal to the mean.
            - ``'nugget'`` (or ``0``) -- the nugget component (constraint
              sum=0; essentially zero except very near a datum).
            - an int ``k`` in ``1..n_structures`` -- that single structure only
              (constraint sum=0), e.g. the short-range or long-range factor.
            - a sequence of such indices -- their sum.
        return_variance : bool, default False

        Notes
        -----
        ``'total'`` and ``'denoised'`` use the sum=1 constraint (they still
        contain Z's mean); every other ``factor`` uses sum=0 (a genuine
        zero-mean fluctuation, which also filters out the unknown, locally
        re-estimated mean along with every structure not requested).
        Consequently ``extract('nugget') + extract(1) + ... +
        extract(n_structures)`` (all sum=0) does **not** reproduce
        ``extract('total')`` -- it reproduces ``extract('total') - m_OK(x0)``,
        where ``m_OK`` is ordinary kriging's own *implicit, locally-varying*
        estimate of the mean. This is the expected, textbook behavior of
        factorial kriging (Chiles & Delfiner, 2012, Ch. 5): each sum=0 factor
        is separately a best linear unbiased estimator of that factor, not a
        summand of a literal decomposition of the OK estimate itself. Use
        ``'total'`` (exact Z) or ``'denoised'`` (Z, noise-smoothed) whenever
        you want an estimate comparable to Z(x0) itself.

        Returns
        -------
        estimate : np.ndarray, shape (n_pred,)
        variance : np.ndarray, shape (n_pred,), optional
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling extract().")
        coords_pred = np.atleast_2d(np.asarray(coords_pred, dtype=float))

        # 'total' and 'denoised' both preserve Z's (unknown, locally re-estimated)
        # mean -- constraint sum=1, same as ordinary kriging -- because they
        # estimate a quantity that still contains that mean. 'denoised' just
        # additionally drops the nugget from the target covariance, so the
        # nugget's noise-spike is smoothed out while Z's units/scale are kept.
        # Every other factor (a genuine zero-mean fluctuation) uses sum=0.
        is_mean_preserving = isinstance(factor, str) and factor.lower() in ("total", "all", "z", "denoised")
        indices = _resolve_factor_indices(factor, self.n_structures)
        constraint = 1.0 if is_mean_preserving else 0.0
        c_v0 = self.total_sill() if isinstance(factor, str) and factor.lower() in ("total", "all", "z") \
            else _factor_covariance_at_zero(self.nugget, self.structures, indices)

        n_pred = len(coords_pred)

        if self.search_radius is None:
            k = min(self.max_neighbors, len(self.coords_))
            _, neighbor_idxs = self.tree_.query(coords_pred, k=k)
            neighbor_idxs = np.atleast_2d(neighbor_idxs)
            if neighbor_idxs.shape[0] != n_pred:
                neighbor_idxs = neighbor_idxs.reshape(n_pred, k)
            neighbor_coords = self.coords_[neighbor_idxs]
            neighbor_vals = self.values_[neighbor_idxs]
            try:
                est, var = _solve_fk_batch(
                    neighbor_coords, coords_pred, neighbor_vals,
                    self.nugget, self.structures, indices, constraint,
                    self.regularization, self.use_gpu, return_variance, c_v0,
                )
            except Exception:
                est, var = self._extract_loop(
                    coords_pred, indices, constraint, c_v0, return_variance,
                    neighbor_idxs=neighbor_idxs,
                )
        else:
            est, var = self._extract_loop(
                coords_pred, indices, constraint, c_v0, return_variance,
            )

        if return_variance:
            return est, var
        return est

    def _extract_loop(
        self,
        coords_pred,
        indices,
        constraint,
        c_v0,
        return_variance,
        neighbor_idxs=None,
    ):
        """Per-point loop: used for ``search_radius`` and as a fallback if
        the batched k-NN solve raises."""
        n_pred = len(coords_pred)
        est = np.full(n_pred, np.nan)
        var = np.full(n_pred, np.nan) if return_variance else None

        for i, target in enumerate(coords_pred):
            if neighbor_idxs is not None:
                idx = neighbor_idxs[i]
            else:
                idx = self._neighbors(target)
            if len(idx) == 0:
                continue
            nbr_coords = self.coords_[idx]
            nbr_vals = self.values_[idx]
            n = len(idx)

            dmat = np.sqrt(((nbr_coords[:, None, :] - nbr_coords[None, :, :]) ** 2).sum(-1))
            K = np.zeros((n + 1, n + 1))
            K[:n, :n] = _total_covariance(dmat, self.nugget, self.structures)
            K[:n, n] = 1.0
            K[n, :n] = 1.0
            if self.regularization > 0:
                K[:n, :n] += np.eye(n) * self.regularization

            d0 = np.sqrt(((nbr_coords - target) ** 2).sum(-1))
            rhs = np.zeros(n + 1)
            rhs[:n] = _factor_covariance(d0, self.nugget, self.structures, indices)
            rhs[n] = constraint

            try:
                w = solve(K, rhs, assume_a="sym")
            except LinAlgError:
                w = np.linalg.lstsq(K, rhs, rcond=None)[0]

            est[i] = np.dot(w[:n], nbr_vals)
            if return_variance:
                var[i] = max(c_v0 - np.dot(w[:n], rhs[:n]) - w[n], 0.0)

        return est, var
