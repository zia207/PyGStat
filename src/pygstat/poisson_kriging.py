"""
Poisson Kriging for areal (block-support) count and rate data.

Poisson Kriging (Goovaerts, 2006) is used to filter/smooth rates computed
from counts that are aggregated over irregularly shaped and sized areas
(e.g. disease rates by county). A naive semivariogram/kriging of the raw
rates is misleading because a rate estimated from a small population is
much less reliable than the same rate estimated from a large population.
Poisson Kriging corrects for this by adding a population-dependent term
to the diagonal of the kriging covariance matrix, so that areas with a
small population support are effectively down-weighted.

Three variants are implemented, all sharing the same population-corrected
kriging-system idea but differing in how an area is represented:

- :class:`PoissonKriging` -- **centroid-based**. Each area is collapsed to
  a single representative point (e.g. its centroid) with a total
  population. Fast and only needs one ``(x, y, value, population)`` row
  per area -- no point support required.

- :class:`AreaPoissonKriging` -- **area-to-area** (:meth:`~AreaPoissonKriging.predict_area`)
  and **area-to-point** (:meth:`~AreaPoissonKriging.predict_points`). Each
  area is represented by its actual *point support* (e.g. the census
  population points that fall inside it, built with :class:`PointSupport`).
  Block-to-block covariances are the point-support-weighted average
  covariance between every pair of points in the two areas, which respects
  each area's real shape and size instead of collapsing it to one point.
  Area-to-point additionally disaggregates the areal prediction back down
  to each individual point-support location, redistributed by its share
  of the area's total population.

  Area-to-area/-point are slower than the centroid-based variant but are
  more accurate because they account for the actual size and shape of
  each area (Goovaerts, 2006).

Implementation notes
--------------------
This version aligns three details with Goovaerts (2006):

1. **Poisson diagonal term** -- the data-reliability term added to the
   diagonal of the kriging matrix is ``rate_base * m* / n(u_i)`` (Eq. 3/7),
   i.e. the population-weighted mean rate divided by *each area's own*
   population and scaled by the rate base. This is what down-weights
   small-population areas; a flat term applied equally to every diagonal
   does not. ``rate_base`` is auto-detected from the rate magnitude (see
   :func:`_detect_rate_base`) or can be passed explicitly.

2. **Area-to-point estimate** -- :meth:`AreaPoissonKriging.predict_points`
   solves a per-point kriging system with an area-to-point right-hand side
   (Eq. 13-14), giving ``R_hat(u_s) = sum_i lambda_i(u_s) z(v_i)`` rather
   than the areal estimate scaled by a population share. The coherence
   constraint (Eq. 15) then holds as a *population-weighted average* of the
   point estimates.

3. **Area-to-point variance** -- the prediction-support reference is
   ``C(0) = sill`` (point support), not the within-area covariance used by
   area-to-area.

GPU (CuPy) support
------------------
:class:`PoissonKriging` (centroid) prediction points are independent of
each other, so :meth:`PoissonKriging.predict` batches every local
``k × k`` system into one stacked ``xp.linalg.solve`` of shape
``(n_pred, k+1, k+1)`` — NumPy on CPU, CuPy on GPU — the same pattern as
:class:`~pygstat.indicator_kriging.IndicatorKriging` and local
:class:`~pygstat.krigeST.STKriging`. Pass ``use_gpu='auto'|True|False``.
:meth:`PoissonKriging.predict_id` (leave-one-out of a single area) and
:class:`AreaPoissonKriging` keep the original per-target loop (block-to-
block covariances are not a fixed-``k`` stack).

References
----------
Goovaerts, P. (2006). Geostatistical analysis of disease data: accounting
for spatial support and population density in the isopleth mapping of
cancer mortality risk. International Journal of Health Geographics, 5(1).

Authors
-------
1. Zia Ahmed, PhD (zia207@gmail.com)
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .utils.backend import (
    as_gpu_array,
    get_array_module,
    resolve_cupy_use_gpu,
    to_numpy,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pk_diagonal_weight(values, populations):
    """
    Population-weighted mean rate, used as the extra Poisson variance
    term added to the diagonal of the covariance matrix.

    Areas with a small ``population`` get relatively more of this penalty
    relative to their own covariance, effectively borrowing strength from
    their well-supported neighbors.

    Parameters
    ----------
    values : numpy array
        Areal (block) values -- e.g. rates -- of the neighboring areas.

    populations : numpy array
        Population (or other size-of-support) counts of the same areas.

    Returns
    -------
    float
        The population-weighted mean rate of the neighborhood.
    """
    return float(np.sum(values * populations) / np.sum(populations))


def _detect_rate_base(mean_rate):
    """
    Auto-detect the rate base (the ``B`` in ``rate = B * count / population``)
    from the magnitude of the observed rates.

    Poisson Kriging adds a data-reliability term to the diagonal of the
    kriging matrix (Goovaerts, 2006, Eq. 3/7). For a rate defined as a raw
    proportion ``z = d/n`` that term is ``m* / n(u_i)`` where ``m*`` is the
    population-weighted mean rate. When the rate is instead reported *per B
    people* (e.g. per 100,000 person-years, the convention for cancer
    mortality), both the rate and its sampling variance are scaled by ``B``,
    so the correct diagonal term becomes ``B * m* / n(u_i)``. Omitting the
    ``B`` factor makes the term ~``B`` times too small, under-regularises the
    block-kriging system, and lets area-to-area / area-to-point estimates
    diverge.

    The base cannot be recovered exactly from the rates alone, so this
    heuristic assumes the underlying per-capita rate is on the order of
    ``1e-3`` (typical for disease incidence / mortality) and picks the
    standard base ``B`` in ``{1, 100, 1000, 10000, 100000, 1000000}`` for
    which ``mean_rate / B`` is closest to that order of magnitude. This
    returns ``1`` for proportion data, ``100`` for percentages, and
    ``100000`` for per-100,000 rates. Pass an explicit ``rate_base`` to
    override the heuristic when the base is known.

    Parameters
    ----------
    mean_rate : float
        A representative (e.g. population-weighted mean) rate.

    Returns
    -------
    float
        The detected rate base.
    """
    bases = np.array([1, 100, 1000, 10000, 100000, 1000000], dtype=float)
    target = 1e-3
    m = max(float(mean_rate), 1e-12)
    per_capita = m / bases
    err = np.abs(np.log10(per_capita) - np.log10(target))
    return float(bases[int(np.argmin(err))])


def _weighted_semivariance(variogram, coords_a, values_a, coords_b, values_b,
                           exclude_diagonal=False):
    r"""
    Point-support-weighted average semivariance between two point supports
    (Goovaerts, 2008):

    .. math::
        \bar\gamma(A, B) = \frac{\sum_s \sum_{s'} w_s w_{s'}\,
                                  \gamma(u_s, u_{s'})}
                                 {\sum_s \sum_{s'} w_s w_{s'}}

    where ``s`` ranges over ``A``'s points and ``s'`` over ``B``'s.

    Set ``exclude_diagonal=True`` (only meaningful when ``A`` and ``B`` are
    the *same* point support) to compute the within-support semivariance,
    which must exclude zero-distance self-pairs.
    """
    d = cdist(coords_a, coords_b)
    gamma = variogram(d)
    w = np.outer(values_a, values_b)
    if exclude_diagonal:
        gamma = gamma.copy()
        np.fill_diagonal(gamma, 0.0)
        w = w.copy()
        np.fill_diagonal(w, 0.0)
    total_w = w.sum()
    if total_w <= 0:
        return 0.0
    return float((gamma * w).sum() / total_w)


def _within_support_semivariance(variogram, coords, values):
    """Point-support-weighted within-block semivariance (self-pairs excluded)."""
    if len(coords) < 2:
        return 0.0
    return _weighted_semivariance(variogram, coords, values, coords, values,
                                  exclude_diagonal=True)


def _point_to_support_semivariance(variogram, point_xy, coords_b, values_b):
    """Population-weighted semivariance between one point and a point support."""
    d = cdist(point_xy[None, :], coords_b)[0]
    gamma = variogram(d)
    total_w = values_b.sum()
    if total_w <= 0:
        return 0.0
    return float((gamma * values_b).sum() / total_w)


# ---------------------------------------------------------------------------
# Centroid-based Poisson Kriging
# ---------------------------------------------------------------------------

class PoissonKriging:
    """
    Centroid-based Poisson Kriging of areal count/rate data.

    Each area is represented by a single point (its centroid or
    representative point) with a total population. See
    :class:`AreaPoissonKriging` for the area-to-area / area-to-point
    variants, which use each area's actual point support instead.

    Parameters
    ----------
    variogram : pygstat.Variogram
        A fitted semivariogram model of the areal values (typically the
        population-weighted rate, or a point-support-regularized model).
        Must expose ``fitted_params = [nugget, sill, range, ...]`` and be
        callable as ``variogram(h)`` -> semivariance at distance(s) ``h``.
    rate_base : float, optional
        Per-capita multiplier of the rate (auto-detected at :meth:`fit`
        when omitted).
    use_gpu : bool or {'auto'}, default='auto'
        Solve the batched k-nearest-neighbor systems in :meth:`predict`
        on the GPU via CuPy when usable. Same convention as
        :class:`~pygstat.core.kriging.OrdinaryKriging`. Has no effect on
        :meth:`predict_id`.

    Examples
    --------
    >>> from pygstat import Variogram, PoissonKriging
    >>> vg = Variogram(coords, rates, model='exponential').fit()
    >>> pk = PoissonKriging(vg).fit(coords, rates, populations, ids=fips)
    >>> result = pk.predict_id(34017, number_of_neighbors=8)
    >>> print(result)
    {'id': 34017, 'zhat': 133.9, 'sig': 5.96}
    """

    def __init__(self, variogram, rate_base=None, use_gpu="auto"):
        self.variogram = variogram
        self.rate_base = rate_base
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)
        self.coords = None
        self.values = None
        self.populations = None
        self.ids = None
        self._tree = None

    def fit(self, coords, values, populations, ids=None):
        """
        Store the known areas used as Poisson Kriging neighbors.

        Parameters
        ----------
        coords : array-like, shape (n, 2)
            Representative point (e.g. centroid) of each area.

        values : array-like, shape (n,)
            Areal rate (or count) of each area.

        populations : array-like, shape (n,)
            Population (or other support-size measure) of each area.
            Must be strictly positive.

        ids : array-like, shape (n,), optional
            Identifier of each area (e.g. FIPS code). Required to use
            :meth:`predict_id` for leave-one-out style prediction. If not
            provided, positional integer indexes are used.
        """
        coords = np.asarray(coords, dtype=float)
        values = np.asarray(values, dtype=float)
        populations = np.asarray(populations, dtype=float)

        if not (len(coords) == len(values) == len(populations)):
            raise ValueError(
                "coords, values, and populations must have the same length"
            )
        if np.any(populations <= 0):
            raise ValueError("populations must be strictly positive")

        # Auto-detect the rate base from the rate magnitude unless one was
        # supplied explicitly. The base scales the Poisson diagonal term so
        # that small-population areas are correctly down-weighted (Eq. 3).
        if self.rate_base is None:
            mstar = _pk_diagonal_weight(values, populations)
            self.rate_base = _detect_rate_base(mstar)

        self.coords = coords
        self.values = values
        self.populations = populations
        self.ids = np.asarray(ids) if ids is not None else np.arange(len(coords))
        self._tree = cKDTree(self.coords)
        return self

    def _sill(self):
        # fitted_params = [nugget, sill, range] or [nugget, sill, range, extra]
        return self.variogram.fitted_params[0] + self.variogram.fitted_params[1]

    def _neighbors(self, target_coord, exclude_mask, number_of_neighbors, neighbors_range):
        """Pick the closest ``number_of_neighbors`` known areas to ``target_coord``."""
        dists = cdist(target_coord[None, :], self.coords)[0]
        dists = np.where(exclude_mask, np.inf, dists)

        if neighbors_range is None:
            neighbors_range = self.variogram.fitted_params[2]

        in_range = np.where(dists <= neighbors_range)[0]
        if len(in_range) < number_of_neighbors:
            # Not enough neighbors within range -- fall back to closest
            # available areas regardless of range, same behavior as
            # centroid_based_poisson_kriging in pyinterpolate.
            candidates = np.argsort(dists)[:number_of_neighbors]
        else:
            candidates = in_range[np.argsort(dists[in_range])[:number_of_neighbors]]

        if np.any(np.isinf(dists[candidates])):
            raise ValueError(
                "Not enough known areas to satisfy number_of_neighbors "
                f"({number_of_neighbors}); only {(~exclude_mask).sum()} available."
            )
        return candidates, dists[candidates]

    def _solve(self, idx, target_coord, raise_when_negative_prediction, raise_when_negative_error):
        n = len(idx)
        values = self.values[idx]
        pops = self.populations[idx]
        neighbor_coords = self.coords[idx]
        sill = self._sill()

        # Covariances: target <-> each neighbor
        d0 = cdist(neighbor_coords, target_coord[None, :])[:, 0]
        c0 = sill - self.variogram(d0)
        rhs = np.append(c0, 1.0)

        # Covariances: neighbor <-> neighbor
        dmat = cdist(neighbor_coords, neighbor_coords)
        cmat = sill - self.variogram(dmat)

        # Poisson (population) correction on the diagonal (Goovaerts 2006,
        # Eq. 3): each neighbor i gets ``rate_base * m* / n(u_i)`` added to
        # its own variance, where m* is the population-weighted mean rate.
        # Because the term is divided by that area's OWN population, areas
        # with a small population support are penalised more and are thus
        # trusted less. The ``rate_base`` factor keeps the term on the same
        # scale as the (per-``rate_base``) rates -- without it the penalty is
        # ~rate_base times too small and the system is under-regularised.
        mstar = _pk_diagonal_weight(values, pops)
        cmat = cmat + np.diag(self.rate_base * mstar / pops)

        # Kriging system with a Lagrange multiplier for unbiasedness.
        K = np.zeros((n + 1, n + 1))
        K[:n, :n] = cmat
        K[:n, n] = 1.0
        K[n, :n] = 1.0

        try:
            w = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                "Singular Poisson Kriging matrix -- check for duplicate "
                "coordinates or too few distinct neighbors."
            ) from e

        zhat = float(values.dot(w[:-1]))
        if zhat < 0 and raise_when_negative_prediction:
            raise ValueError(
                f"Predicted value is {zhat} and should not be lower than 0. "
                f"Check your semivariogram model, number of neighbors, or "
                f"neighbors_range."
            )

        # Ordinary-kriging-style error variance: sigma^2 = C(0) - w . rhs,
        # where C(0) = sill since the target is represented by a single
        # point (semivariance at distance 0 is 0). This matches the
        # variance formula used elsewhere in pygstat (e.g. OrdinaryKriging,
        # Cokriging) -- NOT just `w . rhs` on its own, which would ignore
        # C(0) entirely and collapse to ~sqrt(sill) regardless of how
        # well-supported the local neighborhood is.
        sigmasq = sill - float(w.dot(rhs))
        if sigmasq < 0:
            if raise_when_negative_error:
                raise ValueError(
                    f"Predicted error variance is {sigmasq} and should not "
                    f"be lower than 0. Check your semivariogram model, "
                    f"number of neighbors, or neighbors_range."
                )
            sigma = np.nan
        else:
            sigma = np.sqrt(sigmasq)

        return zhat, sigma

    def _call_variogram(self, h, xp):
        """Evaluate ``γ(h)`` on the same array module as ``h`` when possible."""
        try:
            g = self.variogram(h)
            return xp.asarray(g, dtype=float)
        except Exception:
            g = np.asarray(self.variogram(to_numpy(h)), dtype=float)
            return xp.asarray(g)

    def _solve_batch(self, idx, X_pred, use_gpu,
                     raise_when_negative_prediction, raise_when_negative_error):
        """
        Solve ``n_pred`` independent centroid Poisson-kriging systems in
        one batched ``xp.linalg.solve`` of shape ``(n_pred, k+1, k+1)``.
        """
        n_pred, k = idx.shape
        neighbor_coords = self.coords[idx]          # (n_pred, k, 2)
        values = self.values[idx]                   # (n_pred, k)
        pops = self.populations[idx]                # (n_pred, k)
        sill = float(self._sill())
        X_pred = np.asarray(X_pred, dtype=float)

        if use_gpu:
            nc = as_gpu_array(neighbor_coords)
            tg = as_gpu_array(X_pred)
            values_xp = as_gpu_array(values)
            pops_xp = as_gpu_array(pops)
        else:
            nc, tg = neighbor_coords, X_pred
            values_xp, pops_xp = values, pops
        xp = get_array_module(nc)

        diff = nc[:, :, None, :] - nc[:, None, :, :]
        dmat = xp.sqrt(xp.sum(diff ** 2, axis=-1))                  # (n_pred, k, k)
        d0 = xp.sqrt(xp.sum((nc - tg[:, None, :]) ** 2, axis=-1))   # (n_pred, k)

        cmat = sill - self._call_variogram(dmat, xp)
        c0 = sill - self._call_variogram(d0, xp)

        mstar = xp.sum(values_xp * pops_xp, axis=1) / xp.sum(pops_xp, axis=1)
        penalty = self.rate_base * mstar[:, None] / pops_xp         # (n_pred, k)
        diag = xp.arange(k)
        cmat[:, diag, diag] = cmat[:, diag, diag] + penalty

        K = xp.zeros((n_pred, k + 1, k + 1))
        K[:, :k, :k] = cmat
        K[:, :k, k] = 1.0
        K[:, k, :k] = 1.0

        rhs = xp.ones((n_pred, k + 1, 1))
        rhs[:, :k, 0] = c0

        try:
            w = xp.linalg.solve(K, rhs)[:, :, 0]                    # (n_pred, k+1)
        except xp.linalg.LinAlgError:
            w = xp.linalg.lstsq(K, rhs, rcond=None)[0][:, :, 0]

        zhat = xp.sum(values_xp * w[:, :k], axis=1)
        rhs_vec = rhs[:, :, 0]
        sigmasq = sill - xp.sum(w * rhs_vec, axis=1)

        zhat = to_numpy(zhat)
        sigmasq = to_numpy(sigmasq)

        if raise_when_negative_prediction and np.any(zhat < 0):
            raise ValueError(
                f"Predicted value is {zhat.min()} and should not be lower than 0. "
                f"Check your semivariogram model, number of neighbors, or "
                f"neighbors_range."
            )
        sig = np.sqrt(np.where(sigmasq >= 0, sigmasq, np.nan))
        if raise_when_negative_error and np.any(sigmasq < 0):
            raise ValueError(
                f"Predicted error variance is {np.nanmin(sigmasq)} and should not "
                f"be lower than 0. Check your semivariogram model, "
                f"number of neighbors, or neighbors_range."
            )
        return zhat, sig

    def predict(self, X_pred, number_of_neighbors=8, neighbors_range=None,
               raise_when_negative_prediction=False, raise_when_negative_error=False):
        """
        Predict the areal rate at new (unobserved) locations.

        Each target uses its ``number_of_neighbors`` nearest known areas
        (k-NN). All ``n_pred`` systems are stacked and solved in one
        batched ``xp.linalg.solve`` — NumPy on CPU, CuPy on GPU when
        ``use_gpu`` is true. ``neighbors_range`` is accepted for API
        compatibility with :meth:`predict_id` but k-NN always returns
        exactly ``k`` neighbors (the same fallback :meth:`_neighbors`
        already used when too few areas fall inside the range).

        Parameters
        ----------
        X_pred : array-like, shape (m, 2)
            Coordinates of the locations to predict.

        number_of_neighbors : int, default=8
            Number of known areas used to build each local kriging system.

        neighbors_range : float, optional
            Unused by the batched k-NN path; kept so call sites shared
            with :meth:`predict_id` do not break.

        raise_when_negative_prediction : bool, default=False
            Raise ``ValueError`` if a prediction is negative (rates should
            not be negative). If ``False``, the raw value is returned.

        raise_when_negative_error : bool, default=False
            Raise ``ValueError`` if the prediction error variance is
            negative. If ``False``, ``sig`` is set to ``NaN``.

        Returns
        -------
        zhat : numpy array, shape (m,)
            Predicted rates.

        sig : numpy array, shape (m,)
            Kriging standard errors.
        """
        if self.coords is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_pred = np.atleast_2d(np.asarray(X_pred, dtype=float))
        k = int(number_of_neighbors)
        n_train = len(self.coords)
        if k < 1:
            raise ValueError("number_of_neighbors must be >= 1")
        if k > n_train:
            raise ValueError(
                "Not enough known areas to satisfy number_of_neighbors "
                f"({k}); only {n_train} available."
            )
        _ = neighbors_range  # API compatibility with predict_id
        _, idx = self._tree.query(X_pred, k=k)
        idx = np.atleast_2d(np.asarray(idx, dtype=int))
        if idx.shape == (1, X_pred.shape[0]) and k == 1:
            idx = idx.T
        if idx.shape[1] != k:
            idx = idx.reshape(len(X_pred), k)
        return self._solve_batch(
            idx, X_pred, self.use_gpu,
            raise_when_negative_prediction, raise_when_negative_error,
        )

    def predict_id(self, unknown_id, number_of_neighbors=8, neighbors_range=None,
                   raise_when_negative_prediction=False, raise_when_negative_error=False):
        """
        Predict the value of one of the *known* areas, excluding it from
        its own neighbor search (leave-one-out). Useful for validating a
        Poisson Kriging model against observed rates.

        Parameters
        ----------
        unknown_id : Hashable
            Id (as passed to :meth:`fit` via ``ids``) of the area to
            "hide" and re-predict from its remaining neighbors.

        number_of_neighbors, neighbors_range, raise_when_negative_prediction,
        raise_when_negative_error : see :meth:`predict`.

        Returns
        -------
        dict
            ``{"id": unknown_id, "zhat": prediction, "sig": std_error}``
        """
        matches = np.where(self.ids == unknown_id)[0]
        if len(matches) == 0:
            raise ValueError(f"id {unknown_id!r} not found in the fitted data")
        target_idx = matches[0]
        target_coord = self.coords[target_idx]

        exclude_mask = np.zeros(len(self.coords), dtype=bool)
        exclude_mask[target_idx] = True

        idx, _ = self._neighbors(target_coord, exclude_mask, number_of_neighbors, neighbors_range)
        zhat, sig = self._solve(
            idx, target_coord, raise_when_negative_prediction, raise_when_negative_error
        )
        return {"id": unknown_id, "zhat": zhat, "sig": sig}


# ---------------------------------------------------------------------------
# Point support (for area-to-area / area-to-point Poisson Kriging)
# ---------------------------------------------------------------------------

class PointSupport:
    """
    Groups point-support locations (e.g. census population points) by the
    area (block/polygon) they fall inside. Used by
    :class:`AreaPoissonKriging` for area-to-area and area-to-point Poisson
    Kriging, which represent each area by its real point support rather
    than a single centroid.

    Parameters
    ----------
    coords : array-like, shape (n, 2)
        Coordinates of every point-support location, across all areas.

    values : array-like, shape (n,)
        Support weight of each point (e.g. population count). Must be
        non-negative.

    block_ids : array-like, shape (n,)
        Id of the area each point belongs to.
    """

    def __init__(self, coords, values, block_ids):
        coords = np.asarray(coords, dtype=float)
        values = np.asarray(values, dtype=float)
        block_ids = np.asarray(block_ids)

        if not (len(coords) == len(values) == len(block_ids)):
            raise ValueError(
                "coords, values, and block_ids must have the same length"
            )
        if np.any(values < 0):
            raise ValueError("point support values must be non-negative")

        self.block_ids = np.unique(block_ids)
        self._index = {}
        for bid in self.block_ids:
            mask = block_ids == bid
            self._index[bid] = (coords[mask], values[mask])

    @classmethod
    def from_geodataframes(cls, points, blocks, block_id_col, point_value_col,
                           predicate="within"):
        """
        Build a :class:`PointSupport` by spatially joining a point layer
        (e.g. census population points) to a polygon layer (e.g. counties).

        Parameters
        ----------
        points : geopandas.GeoDataFrame
            Point-support locations, with a ``point_value_col`` column.

        blocks : geopandas.GeoDataFrame
            Polygons, with a ``block_id_col`` column identifying each area.
            Must share ``points``' CRS.

        block_id_col : str
            Column in ``blocks`` identifying each polygon.

        point_value_col : str
            Column in ``points`` with the support weight (e.g. population).

        predicate : str, default='within'
            Spatial predicate passed to ``geopandas.sjoin``.
        """
        import geopandas as gpd

        joined = gpd.sjoin(
            points[[point_value_col, points.geometry.name]],
            blocks[[block_id_col, blocks.geometry.name]],
            how="inner",
            predicate=predicate,
        )
        if len(joined) == 0:
            raise ValueError(
                "No point fell inside any block -- check that `points` and "
                "`blocks` share the same CRS and geometry columns."
            )
        coords = np.column_stack([joined.geometry.x, joined.geometry.y])
        return cls(coords, joined[point_value_col].to_numpy(), joined[block_id_col].to_numpy())

    def coords(self, block_id):
        """Point-support coordinates of ``block_id``, shape ``(k, 2)``."""
        return self._index[block_id][0]

    def values(self, block_id):
        """Point-support values (e.g. population) of ``block_id``, shape ``(k,)``."""
        return self._index[block_id][1]

    def total(self, block_id):
        """Total support (e.g. population) of ``block_id``."""
        return float(self._index[block_id][1].sum())

    def centroid(self, block_id):
        """Population-weighted centroid of ``block_id``'s point support."""
        coords, values = self._index[block_id]
        if values.sum() <= 0:
            return coords.mean(axis=0)
        return np.average(coords, axis=0, weights=values)

    def __contains__(self, block_id):
        return block_id in self._index

    def __len__(self):
        return len(self.block_ids)


# ---------------------------------------------------------------------------
# Area-to-area / area-to-point Poisson Kriging
# ---------------------------------------------------------------------------

class AreaPoissonKriging:
    """
    Area-to-area and area-to-point Poisson Kriging of block-support
    count/rate data, using each area's actual point support (e.g.
    population points, see :class:`PointSupport`) instead of a single
    representative centroid.

    Because block-to-block covariances are computed from the real,
    point-support-weighted geometry of each area (see
    :func:`_weighted_semivariance`), this variant is slower than
    :class:`PoissonKriging` but accounts for each area's true size and
    shape, and is expected to give a lower cross-validated error
    (Goovaerts, 2006).

    Parameters
    ----------
    variogram : pygstat.Variogram
        A fitted semivariogram model, ideally of the point-support-level
        (regularized) process; the plain areal-value semivariogram is a
        workable approximation and is what the examples below use.

    Examples
    --------
    >>> from pygstat import Variogram, PointSupport, AreaPoissonKriging
    >>> ps = PointSupport.from_geodataframes(
    ...     points_gdf, areas_gdf, block_id_col='FIPS', point_value_col='POP10'
    ... )
    >>> vg = Variogram(centroids, rates, model='exponential').fit()
    >>> apk = AreaPoissonKriging(vg).fit(ps, areas_df.set_index('FIPS')['rate'])
    >>> apk.predict_area(34017, number_of_neighbors=8)
    {'id': 34017, 'zhat': 133.4, 'sig': 6.1}
    >>> apk.predict_points(34017, number_of_neighbors=8).head()
    """

    def __init__(self, variogram, rate_base=None):
        self.variogram = variogram
        self.rate_base = rate_base
        self.point_support = None
        self.block_values = None
        self.ids = None
        self._centroids = None

    def fit(self, point_support, block_values, ids=None):
        """
        Parameters
        ----------
        point_support : PointSupport
            Point support of every known (and to-be-predicted) area.

        block_values : dict, pandas.Series, or array-like
            Known areal rate of each area. If a dict or Series, keyed by
            area id. If a plain array, ``ids`` must be given in the same
            order.

        ids : array-like, optional
            Required when ``block_values`` is a plain array.
        """
        self.point_support = point_support

        if isinstance(block_values, dict):
            values_by_id = dict(block_values)
        elif hasattr(block_values, "to_dict"):
            values_by_id = block_values.to_dict()
        else:
            if ids is None:
                raise ValueError(
                    "`ids` is required when block_values is not a dict/Series"
                )
            values_by_id = dict(zip(ids, block_values))

        # Only areas that actually have point support can be used as
        # neighbors or predicted.
        self.block_values = {
            bid: val for bid, val in values_by_id.items() if bid in point_support
        }
        if not self.block_values:
            raise ValueError(
                "None of the given block_values ids have matching point "
                "support -- check that ids line up with `point_support`."
            )
        self.ids = np.array(list(self.block_values.keys()))
        self._centroids = {bid: point_support.centroid(bid) for bid in self.ids}

        # Auto-detect the rate base from the rate magnitude unless supplied,
        # using each area's total point support as its population weight.
        if self.rate_base is None:
            vals = np.array([self.block_values[b] for b in self.ids], dtype=float)
            tots = np.array([point_support.total(b) for b in self.ids], dtype=float)
            mstar = _pk_diagonal_weight(vals, tots)
            self.rate_base = _detect_rate_base(mstar)
        return self

    def _sill(self):
        return self.variogram.fitted_params[0] + self.variogram.fitted_params[1]

    def _neighbors(self, target_centroid, exclude_id, number_of_neighbors, neighbors_range):
        candidate_ids = self.ids[self.ids != exclude_id]
        if len(candidate_ids) == 0:
            raise ValueError("No known neighboring areas available for prediction.")
        candidate_centroids = np.array([self._centroids[bid] for bid in candidate_ids])
        dists = cdist(target_centroid[None, :], candidate_centroids)[0]

        if neighbors_range is None:
            neighbors_range = self.variogram.fitted_params[2]

        in_range = np.where(dists <= neighbors_range)[0]
        if len(in_range) < number_of_neighbors:
            order = np.argsort(dists)[:number_of_neighbors]
        else:
            order = in_range[np.argsort(dists[in_range])[:number_of_neighbors]]
        return candidate_ids[order]

    def _lhs(self, neighbor_ids):
        """
        Build the (n+1)x(n+1) augmented Poisson Kriging system matrix and
        the neighbors' known values, shared by area-to-area and
        area-to-point prediction.
        """
        n = len(neighbor_ids)
        sill = self._sill()
        cmat = np.empty((n, n))

        support = {bid: (self.point_support.coords(bid), self.point_support.values(bid))
                  for bid in neighbor_ids}

        for i, bi in enumerate(neighbor_ids):
            ci, vi = support[bi]
            for j in range(i, n):
                bj = neighbor_ids[j]
                cj, vj = support[bj]
                gamma = _weighted_semivariance(self.variogram, ci, vi, cj, vj,
                                               exclude_diagonal=(bi == bj))
                cov = sill - gamma
                cmat[i, j] = cov
                cmat[j, i] = cov

        values = np.array([self.block_values[bid] for bid in neighbor_ids])
        totals = np.array([self.point_support.total(bid) for bid in neighbor_ids])

        # Poisson (population) correction on the diagonal (Goovaerts 2006,
        # Eq. 7), same idea as the centroid-based variant: neighbor i gets
        # ``rate_base * m* / n(v_i)`` added to its own variance, where n(v_i)
        # is the area's total point support. Less-populated neighbors are
        # penalised more (trusted less), and the ``rate_base`` factor keeps
        # the term on the scale of the per-``rate_base`` rates.
        mstar = _pk_diagonal_weight(values, totals)
        cmat = cmat + np.diag(self.rate_base * mstar / totals)

        K = np.zeros((n + 1, n + 1))
        K[:n, :n] = cmat
        K[:n, n] = 1.0
        K[n, :n] = 1.0
        return K, values

    def _target_c00(self, target_coords, target_values):
        """Support-adjusted C(target, target) = sill - within-block semivariance."""
        gamma_within = _within_support_semivariance(self.variogram, target_coords, target_values)
        return self._sill() - gamma_within

    def predict_area(self, unknown_id, number_of_neighbors=8, neighbors_range=None,
                     raise_when_negative_prediction=False, raise_when_negative_error=False):
        """
        Area-to-area Poisson Kriging: predict the single areal rate of
        ``unknown_id`` from its neighbors' point supports, excluding
        ``unknown_id`` itself from the neighbor search (leave-one-out if
        it is one of the fitted areas).

        Parameters
        ----------
        unknown_id : Hashable
            Id of the area to predict. Must have point support (i.e. be
            present in the ``point_support`` passed to :meth:`fit`).

        number_of_neighbors : int, default=8
            Number of neighboring areas used to build the kriging system.

        neighbors_range : float, optional
            Maximum centroid-to-centroid distance for neighbor search.
            Defaults to the fitted semivariogram range.

        raise_when_negative_prediction, raise_when_negative_error : bool, default=False
            Raise ``ValueError`` on a negative prediction / error variance
            instead of returning it raw / as ``NaN``.

        Returns
        -------
        dict
            ``{"id": unknown_id, "zhat": prediction, "sig": std_error}``
        """
        if unknown_id not in self.point_support:
            raise ValueError(f"id {unknown_id!r} has no point support")

        target_coords = self.point_support.coords(unknown_id)
        target_values = self.point_support.values(unknown_id)
        target_centroid = self.point_support.centroid(unknown_id)

        neighbor_ids = self._neighbors(target_centroid, unknown_id, number_of_neighbors, neighbors_range)
        K, values = self._lhs(neighbor_ids)

        sill = self._sill()
        rhs = np.empty(len(neighbor_ids) + 1)
        for i, bid in enumerate(neighbor_ids):
            ci, vi = self.point_support.coords(bid), self.point_support.values(bid)
            gamma = _weighted_semivariance(self.variogram, ci, vi, target_coords, target_values)
            rhs[i] = sill - gamma
        rhs[-1] = 1.0

        try:
            w = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                "Singular Poisson Kriging matrix -- check for duplicate "
                "area locations or too few distinct neighbors."
            ) from e

        zhat = float(values.dot(w[:-1]))
        if zhat < 0 and raise_when_negative_prediction:
            raise ValueError(
                f"Predicted value is {zhat} and should not be lower than 0. "
                f"Check your semivariogram model, number of neighbors, or "
                f"neighbors_range."
            )

        c00 = self._target_c00(target_coords, target_values)
        sigmasq = c00 - float(w.dot(rhs))
        if sigmasq < 0:
            if raise_when_negative_error:
                raise ValueError(
                    f"Predicted error variance is {sigmasq} and should not "
                    f"be lower than 0. Check your semivariogram model, "
                    f"number of neighbors, or neighbors_range."
                )
            sigma = np.nan
        else:
            sigma = np.sqrt(sigmasq)

        return {"id": unknown_id, "zhat": zhat, "sig": sigma}

    def predict_points(self, unknown_id, number_of_neighbors=8, neighbors_range=None,
                       raise_when_negative_prediction=False, raise_when_negative_error=False,
                       err_to_nan=True, targets=None):
        """
        Area-to-point Poisson Kriging: estimate the risk at individual
        point locations inside ``unknown_id`` (Goovaerts 2006, Eq. 13-14).

        Each target point ``u_s`` gets its own kriging system: the
        left-hand side is the same block-to-block system as
        :meth:`predict_area` (so it is built once and reused), while the
        right-hand side holds **area-to-point** covariances between each
        neighbouring area ``v_i`` and the point ``u_s``. The estimate is a
        genuine local prediction,

            R_hat(u_s) = sum_i lambda_i(u_s) * z(v_i),

        NOT the areal estimate scaled by a population share. Solving this at
        every node of a grid yields a continuous (isopleth) risk surface
        that reveals within-area detail.

        **Coherence (Eq. 15).** When the target points are the area's own
        support points and the same ``number_of_neighbors`` areas are used
        for all of them, the *population-weighted average* of the point
        estimates reproduces the area-to-area estimate:

            sum_s [ n(u_s) / sum_s' n(u_s') ] * R_hat(u_s) == predict_area(id).

        (Earlier releases returned each point's population *share* of the
        areal estimate, whose plain sum equalled the ATA value; that
        collapsed the within-area variability. This version follows the
        reference ATP formulation and satisfies the coherence constraint as
        a population-weighted average.)

        Parameters
        ----------
        unknown_id, number_of_neighbors, neighbors_range : see :meth:`predict_area`.

        raise_when_negative_prediction, raise_when_negative_error : bool, default=False
            Raise ``ValueError`` on a negative point prediction / error
            variance. Ignored where ``err_to_nan=True`` (the default),
            which sets that point's ``zhat``/``sig`` to ``NaN`` instead.

        err_to_nan : bool, default=True
            When a point's prediction or error variance is negative, set
            it to ``NaN`` rather than raising or silently clipping.

        targets : array-like of shape (m, 2), optional
            Coordinates at which to predict. Defaults to ``unknown_id``'s
            own point-support locations (the case needed for the coherence
            check). Pass a grid of nodes to build an isopleth surface; each
            node inherits ``unknown_id``'s block-kriging system, so the
            containing area determines which neighbours are used.

        Returns
        -------
        pandas.DataFrame
            One row per target point, with columns
            ``["x", "y", "population", "zhat", "sig"]``. ``population`` is
            the point-support weight for the area's own points, or 1.0 for
            user-supplied ``targets``.
        """
        if unknown_id not in self.point_support:
            raise ValueError(f"id {unknown_id!r} has no point support")

        target_coords = self.point_support.coords(unknown_id)
        target_values = self.point_support.values(unknown_id)
        target_centroid = self.point_support.centroid(unknown_id)
        total_unknown = target_values.sum()
        if total_unknown <= 0:
            raise ValueError(f"id {unknown_id!r} has zero total point support")

        # Prediction points: the area's own support points by default, or a
        # user-supplied set of grid nodes (weight 1.0 for those).
        if targets is None:
            pred_coords = target_coords
            pred_weights = target_values
        else:
            pred_coords = np.atleast_2d(np.asarray(targets, dtype=float))
            pred_weights = np.ones(len(pred_coords))

        neighbor_ids = self._neighbors(target_centroid, unknown_id, number_of_neighbors, neighbors_range)
        K, values = self._lhs(neighbor_ids)
        sill = self._sill()
        # Point prediction support: the reference is C(0) = sill (Eq. 13),
        # not the within-area covariance used by area-to-area.
        support = {bid: (self.point_support.coords(bid), self.point_support.values(bid))
                   for bid in neighbor_ids}

        rows = []
        for (px, py), pw in zip(pred_coords, pred_weights):
            rhs = np.empty(len(neighbor_ids) + 1)
            for i, bid in enumerate(neighbor_ids):
                cb, vb = support[bid]
                gamma = _point_to_support_semivariance(self.variogram, np.array([px, py]), cb, vb)
                rhs[i] = sill - gamma
            rhs[-1] = 1.0

            try:
                w = np.linalg.solve(K, rhs)
            except np.linalg.LinAlgError as e:
                raise np.linalg.LinAlgError(
                    "Singular Poisson Kriging matrix -- check for duplicate "
                    "area locations or too few distinct neighbors."
                ) from e

            # Proper ATP estimate: weighted sum of neighbour rates.
            zhat = float(values.dot(w[:-1]))
            if zhat < 0:
                if raise_when_negative_prediction and not err_to_nan:
                    raise ValueError(
                        f"Predicted value is {zhat} and should not be "
                        f"lower than 0. Check your semivariogram model, "
                        f"number of neighbors, or neighbors_range."
                    )
                if err_to_nan:
                    rows.append([px, py, pw, np.nan, np.nan])
                    continue

            sigmasq = sill - float(w.dot(rhs))
            if sigmasq < 0:
                if raise_when_negative_error and not err_to_nan:
                    raise ValueError(
                        f"Predicted error variance is {sigmasq} and should "
                        f"not be lower than 0. Check your semivariogram "
                        f"model, number of neighbors, or neighbors_range."
                    )
                sigma = np.nan if err_to_nan else 0.0
            else:
                sigma = np.sqrt(sigmasq)

            rows.append([px, py, pw, zhat, sigma])

        return pd.DataFrame(rows, columns=["x", "y", "population", "zhat", "sig"])
