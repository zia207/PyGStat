"""
KT3D -- Kriging (SK/OK/KT) of a 3-D grid or arbitrary point set.

A Python port of GSLIB's ``kt3d.for`` (Deutsch & Journel, *GSLIB: Geostatistical
Software Library*, 2nd ed.), converted from the Fortran source shipped in this
repository at ``gslib90sc/kt3d.for`` together with the shared low-level
routines it calls from ``gslib90sc/gslib/`` (``cova3``, ``setrot``, ``sqdist``,
``srchsupr``). The kriging mathematics -- rotated/anisotropic nested-structure
covariance, the search-ellipsoid neighborhood with octant balancing, the
simple/ordinary/kriging-with-a-trend (KT) linear systems including polynomial
drift terms and external drift, and the block-covariance/rescaling
conventions -- are reproduced from that source line for line.

Two implementation details are deliberately *not* literal translations,
because they only affect performance, not results:

- **Super-block search** (``setsupr``/``picksupr``/``srchsupr`` in Fortran)
  is a spatial-indexing accelerator so the original program doesn't scan
  every datum for every grid node. This port scans all data directly with
  vectorized NumPy (see ``_find_neighbors``) and applies the *exact same*
  selection rule afterwards: an anisotropic-ellipsoid radius filter,
  ascending-distance sort, then GSLIB's fixed 8-octant cap (``noct`` per
  octant). Results are identical; only large datasets estimated on large
  grids would see this port run slower.
- **``ktsol``**, GSLIB's hand-rolled Gaussian elimination with partial
  pivoting, is replaced by ``numpy.linalg.solve`` (LAPACK), which solves the
  same linear system to machine precision. Ordinary/simple **point** kriging
  with ``noct=0`` additionally stacks those systems into one batched
  ``xp.linalg.solve`` (NumPy or CuPy via ``use_gpu``), the same pattern as
  :class:`pygstat.indicator_kriging.IndicatorKriging`.

Variogram convention: nested structures use GSLIB's own parametrization
(``type``, ``sill`` = partial sill "cc", ``range`` = "a", with practical
ranges baked into each model's own formula -- e.g. exponential reaches ~95%
of its sill at ``3 * range``), matching ``cova3.for`` exactly. This is the
same convention used by :mod:`pygstat.sgsim` and :mod:`pygstat.sisim`, and is
**not** the ``(range)`` convention used by :mod:`pygstat.core.variogram`.
"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .utils.backend import get_array_module, as_gpu_array, to_numpy, resolve_cupy_use_gpu

__all__ = ["kt3d", "kt3d_grid", "kt3d_cross_validate", "build_grid_3d"]

_EPSLON = 1.0e-6
_PMX = 999.0


# ---------------------------------------------------------------------------
# setrot: anisotropic rotation matrix (ang1,ang2,ang3 in GSLIB convention)
# ---------------------------------------------------------------------------

def _setrot(ang1: float, ang2: float, ang3: float, anis1: float, anis2: float) -> np.ndarray:
    """
    3x3 matrix transforming (dx,dy,dz) into anisotropy-corrected space, so
    that the Euclidean norm of the transformed vector is the GSLIB
    anisotropic distance. Direct port of ``gslib/setrot.for``.

    ang1 : azimuth of the major axis (degrees, clockwise from north)
    ang2 : dip of the major axis (degrees, positive down)
    ang3 : rotation of the minor axis about the major axis (degrees)
    anis1 : minor-horizontal / major range ratio
    anis2 : vertical / major range ratio
    """
    if 0.0 <= ang1 < 270.0:
        alpha = np.radians(90.0 - ang1)
    else:
        alpha = np.radians(450.0 - ang1)
    beta = np.radians(-1.0 * ang2)
    theta = np.radians(ang3)

    sina, sinb, sint = np.sin(alpha), np.sin(beta), np.sin(theta)
    cosa, cosb, cost = np.cos(alpha), np.cos(beta), np.cos(theta)

    afac1 = 1.0 / max(anis1, 1.0e-20)
    afac2 = 1.0 / max(anis2, 1.0e-20)

    return np.array([
        [cosb * cosa, cosb * sina, -sinb],
        [afac1 * (-cost * sina + sint * sinb * cosa),
         afac1 * (cost * cosa + sint * sinb * sina),
         afac1 * (sint * cosb)],
        [afac2 * (sint * sina + cost * sinb * cosa),
         afac2 * (-sint * cosa + cost * sinb * sina),
         afac2 * (cost * cosb)],
    ])


# ---------------------------------------------------------------------------
# Variogram model: nugget + nested structures (cova3.for)
# ---------------------------------------------------------------------------

def _normalize_structure(st: Dict) -> Dict:
    stype = st["type"].lower()
    if stype not in ("spherical", "exponential", "gaussian", "power", "hole", "hole_effect"):
        raise ValueError(
            f"Unknown structure type '{st['type']}'; expected one of "
            "'spherical', 'exponential', 'gaussian', 'power', 'hole'"
        )
    rng = float(st["range"])
    if stype == "power" and not (0.0 <= rng <= 2.0):
        raise ValueError("power variogram exponent ('range') must be in [0, 2]")
    return {
        "type": stype,
        "sill": float(st["sill"]),
        "range": rng,
        "range2": float(st.get("range2", rng)),
        "range3": float(st.get("range3", rng)),
        "angles": tuple(st.get("angles", (0.0, 0.0, 0.0))),
    }


def _prepare_variogram(variogram: Dict):
    """Validate `variogram` and precompute a rotation matrix per structure."""
    c0 = float(variogram["nugget"])
    structures = [_normalize_structure(s) for s in variogram["structures"]]
    if len(structures) == 0:
        raise ValueError("variogram must have at least one structure")
    rotmats = []
    covmax = c0
    for st in structures:
        anis1 = st["range2"] / max(st["range"], _EPSLON)
        anis2 = st["range3"] / max(st["range"], _EPSLON)
        ang1, ang2, ang3 = st["angles"]
        rotmats.append(_setrot(ang1, ang2, ang3, anis1, anis2))
        covmax += _PMX if st["type"] == "power" else st["sill"]
    return c0, structures, rotmats, covmax


def _cova3_from_delta(dx, dy, dz, c0: float, structures: List[Dict],
                       rotmats: List[np.ndarray], covmax: float, xp=np):
    """Nested-structure covariance from separation components of any shape."""
    cov = xp.zeros(dx.shape, dtype=float)
    for st, R in zip(structures, rotmats):
        r00, r01, r02 = float(R[0, 0]), float(R[0, 1]), float(R[0, 2])
        r10, r11, r12 = float(R[1, 0]), float(R[1, 1]), float(R[1, 2])
        r20, r21, r22 = float(R[2, 0]), float(R[2, 1]), float(R[2, 2])
        cx = r00 * dx + r01 * dy + r02 * dz
        cy = r10 * dx + r11 * dy + r12 * dz
        cz = r20 * dx + r21 * dy + r22 * dz
        h = xp.sqrt(cx * cx + cy * cy + cz * cz)
        cc, aa, stype = st["sill"], max(st["range"], _EPSLON), st["type"]
        if stype == "spherical":
            hr = h / aa
            cov = cov + xp.where(hr < 1.0, cc * (1.0 - hr * (1.5 - 0.5 * hr * hr)), 0.0)
        elif stype == "exponential":
            cov = cov + cc * xp.exp(-3.0 * h / aa)
        elif stype == "gaussian":
            cov = cov + cc * xp.exp(-3.0 * (h / aa) ** 2)
        elif stype == "power":
            cov = cov + covmax - cc * xp.power(h, aa)
        else:
            cov = cov + cc * xp.cos(h / aa * xp.pi)
    zero = (xp.abs(dx) < _EPSLON) & (xp.abs(dy) < _EPSLON) & (xp.abs(dz) < _EPSLON)
    return xp.where(zero, covmax, cov)


def _cova3_matrix(P1: np.ndarray, P2: np.ndarray, c0: float, structures: List[Dict],
                   rotmats: List[np.ndarray], covmax: float) -> np.ndarray:
    """
    (n, m) covariance matrix between points `P1` and `P2` for the nested
    variogram model. Port of ``gslib/cova3.for``, vectorized over all pairs.
    """
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)
    dx = P1[:, 0:1] - P2[None, :, 0]
    dy = P1[:, 1:2] - P2[None, :, 1]
    dz = P1[:, 2:3] - P2[None, :, 2]
    return _cova3_from_delta(dx, dy, dz, c0, structures, rotmats, covmax, xp=np)


def _point_to_block_cov(P: np.ndarray, xdb: np.ndarray, c0: float, structures: List[Dict],
                         rotmats: List[np.ndarray], covmax: float) -> np.ndarray:
    """
    Average covariance between each row of `P` (data, local coordinates) and
    the `ndb` block-discretization points `xdb` -- kt3d.for's per-datum
    right-hand-side loop (lines ~1094-1113). For block kriging (`ndb` > 1),
    any (datum, discretization-point) pair that coincides exactly has its
    nugget contribution stripped before averaging, matching kt3d.for's
    support correction; for point kriging (`ndb` == 1) no correction applies
    (cova3's own covmax at zero distance is the correct point-to-point value).
    """
    C = _cova3_matrix(P, xdb, c0, structures, rotmats, covmax)
    if len(xdb) > 1:
        diff = P[:, None, :] - xdb[None, :, :]
        coincide = np.all(np.abs(diff) < _EPSLON, axis=2)
        C = np.where(coincide, C - c0, C)
    return C.mean(axis=1)


# ---------------------------------------------------------------------------
# Search neighborhood: anisotropic ellipsoid + octant balancing (srchsupr.for)
# ---------------------------------------------------------------------------

def _find_neighbors(target: np.ndarray, coords: np.ndarray, search_rot: np.ndarray,
                     radsqd: float, ndmax: int, noct: int):
    """
    Indices (into `coords`) of the data neighboring `target`, closest first,
    honoring the search ellipsoid, `ndmax`, and (if `noct` > 0) GSLIB's fixed
    8-octant cap of `noct` per octant. Direct port of ``gslib/srchsupr.for``
    minus the super-block indexing (see module docstring).
    """
    d = coords - target
    cx = search_rot[0, 0] * d[:, 0] + search_rot[0, 1] * d[:, 1] + search_rot[0, 2] * d[:, 2]
    cy = search_rot[1, 0] * d[:, 0] + search_rot[1, 1] * d[:, 1] + search_rot[1, 2] * d[:, 2]
    cz = search_rot[2, 0] * d[:, 0] + search_rot[2, 1] * d[:, 1] + search_rot[2, 2] * d[:, 2]
    hsqd = cx * cx + cy * cy + cz * cz

    within = np.where(hsqd <= radsqd)[0]
    if within.size == 0:
        return within
    order = within[np.argsort(hsqd[within], kind="stable")]

    if noct <= 0:
        return order[:ndmax]

    dx, dy, dz = d[order, 0], d[order, 1], d[order, 2]
    quadrant = np.full(order.shape, 4, dtype=int)
    front = dz >= 0.0
    quadrant[front & (dx <= 0.0) & (dy > 0.0)] = 1
    quadrant[front & (dx > 0.0) & (dy >= 0.0)] = 2
    quadrant[front & (dx < 0.0) & (dy <= 0.0)] = 3
    quadrant[~front] = 8
    quadrant[~front & (dx <= 0.0) & (dy > 0.0)] = 5
    quadrant[~front & (dx > 0.0) & (dy >= 0.0)] = 6
    quadrant[~front & (dx < 0.0) & (dy <= 0.0)] = 7

    picked = []
    counts = {q: 0 for q in range(1, 9)}
    for i, q in enumerate(quadrant):
        if counts[q] < noct:
            counts[q] += 1
            picked.append(order[i])
            if len(picked) >= 8 * noct:
                break
    return np.array(picked[:ndmax], dtype=int)


def _find_neighbors_knn_batch(pts, coords, search_rot, radsqd, ndmax):
    """
    k-nearest neighbors for every target in one KDTree query, in the
    anisotropic search metric. Neighbors with squared distance > `radsqd`
    are marked invalid. Used by the batched SK/OK path (`noct==0`).
    """
    coords_s = coords @ search_rot.T
    pts_s = pts @ search_rot.T
    k = min(int(ndmax), len(coords))
    tree = cKDTree(coords_s)
    dist, idx = tree.query(pts_s, k=k)
    if k == 1:
        dist = np.asarray(dist)[:, None]
        idx = np.asarray(idx)[:, None]
    valid = np.isfinite(dist) & ((dist * dist) <= radsqd + 1e-12)
    return idx.astype(int), valid


def _estimate_ok_sk_batch(
    P, vra, ktype, skmean, c0, structures, rotmats, covmax, unbias, cbb, use_gpu,
):
    """
    Batched simple (ktype=0) or ordinary (ktype=1) point kriging for a
    stack of local neighborhoods with the *same* neighbor count k.
    `P` is (n_pred, k, 3) in target-local coordinates; `vra` is (n_pred, k).
    """
    n_pred, k, _ = P.shape
    if use_gpu:
        P = as_gpu_array(np.asarray(P, dtype=float))
        vra = as_gpu_array(np.asarray(vra, dtype=float))
    else:
        P = np.asarray(P, dtype=float)
        vra = np.asarray(vra, dtype=float)
    xp = get_array_module(P)

    dx = P[:, :, None, 0] - P[:, None, :, 0]
    dy = P[:, :, None, 1] - P[:, None, :, 1]
    dz = P[:, :, None, 2] - P[:, None, :, 2]
    C = _cova3_from_delta(dx, dy, dz, c0, structures, rotmats, covmax, xp=xp)
    c0_vec = _cova3_from_delta(P[:, :, 0], P[:, :, 1], P[:, :, 2],
                               c0, structures, rotmats, covmax, xp=xp)

    if ktype == 0:
        rhs = c0_vec[:, :, None]
        s = xp.linalg.solve(C, rhs)[:, :, 0]
        est = skmean + xp.sum(s * (vra - skmean), axis=1)
        estv = cbb - xp.sum(s * c0_vec, axis=1)
    else:
        neq = k + 1
        A = xp.zeros((n_pred, neq, neq))
        A[:, :k, :k] = C
        A[:, :k, k] = unbias
        A[:, k, :k] = unbias
        rhs = xp.ones((n_pred, neq, 1))
        rhs[:, :k, 0] = c0_vec
        rhs[:, k, 0] = unbias
        s = xp.linalg.solve(A, rhs)[:, :, 0]
        est = xp.sum(s[:, :k] * vra, axis=1)
        estv = cbb - xp.sum(s * rhs[:, :, 0], axis=1)
    return to_numpy(est), np.maximum(to_numpy(estv), 0.0)


# ---------------------------------------------------------------------------
# Discretization points for block kriging
# ---------------------------------------------------------------------------

def _discretization_points(block_size, discretization):
    xsiz, ysiz, zsiz = block_size
    nxdis, nydis, nzdis = (max(1, int(n)) for n in discretization)
    xdis, ydis, zdis = xsiz / nxdis, ysiz / nydis, zsiz / nzdis

    pts = []
    xloc = -0.5 * (xsiz + xdis)
    for _ in range(nxdis):
        xloc += xdis
        yloc = -0.5 * (ysiz + ydis)
        for _ in range(nydis):
            yloc += ydis
            zloc = -0.5 * (zsiz + zdis)
            for _ in range(nzdis):
                zloc += zdis
                pts.append((xloc + 0.5 * xsiz, yloc + 0.5 * ysiz, zloc + 0.5 * zsiz))
    return np.asarray(pts, dtype=float)


# ---------------------------------------------------------------------------
# One kriging system
# ---------------------------------------------------------------------------

_KTYPE_CODES = {"simple": 0, "ordinary": 1, "locally_varying_mean": 2, "external_drift": 3}


def _estimate_one(xdb, cov_cfg, drift_flags, ktype, skmean, itrend,
                   xa, ya, za, vra, vea, resc, unbias, cbb, bv):
    """Simple (0), ordinary (1), or locally-varying-mean (2) kriging system.
    `ktype='external_drift'` (3) is handled separately by `_estimate_one_ext`
    since its extra unbiasedness row needs the caller's `extest*resce`."""
    na = len(xa)
    c0, structures, rotmats, covmax = cov_cfg
    mdt_base = 0 if ktype in (0, 2) else 1 + int(sum(drift_flags))

    if na >= 1 and na <= mdt_base:
        return np.nan, np.nan

    P = np.column_stack([xa, ya, za])

    if na == 1:
        cb1 = _cova3_matrix(P, P, c0, structures, rotmats, covmax)[0, 0]
        cb = float(_point_to_block_cov(P, xdb, c0, structures, rotmats, covmax)[0])
        if ktype in (0, 2):
            local_mean = skmean
            wt = cb / cb1 if cb1 != 0 else 0.0
            est = wt * vra[0] + (1.0 - wt) * local_mean
            estv = cbb - wt * cb
        else:
            est = vra[0]
            estv = cbb - 2.0 * cb + cb1
        return est, max(estv, 0.0)

    mdt = mdt_base
    neq = na + mdt
    A = np.zeros((neq, neq))
    A[:na, :na] = _cova3_matrix(P, P, c0, structures, rotmats, covmax)

    if neq > na:
        A[:na, na] = unbias
        A[na, :na] = unbias

    cb_i = _point_to_block_cov(P, xdb, c0, structures, rotmats, covmax)
    r = np.zeros(neq)
    r[:na] = cb_i
    if neq > na:
        r[na] = unbias

    # `im` walks the extra (non-data) rows/cols one at a time: `na` itself is
    # the OK unbiasedness constraint (already filled above), so each active
    # drift term takes the *next* index -- matches kt3d.for's `im=im+1` before
    # `a(neq*(im-1)+k)=...` once its 1-indexing is converted to 0-indexed.
    im = na
    for k, flag in enumerate(drift_flags):
        if not flag:
            continue
        im += 1
        col = _drift_column(k, xa, ya, za) * resc
        A[im, :na] = col
        A[:na, im] = col
        r[im] = bv[k] * resc

    rr = r.copy()
    if itrend:
        r[:na] = 0.0

    try:
        s = np.linalg.solve(A, r)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    estv = cbb - float(np.dot(s, rr))
    if ktype == 0:
        est = float(np.dot(s[:na], vra - skmean)) + skmean
    elif ktype == 2:
        est = float(np.dot(s[:na], vra - vea)) + skmean
    else:
        est = float(np.dot(s[:na], vra))
    return est, max(estv, 0.0)


def _drift_column(k, xa, ya, za):
    if k == 0:
        return xa
    if k == 1:
        return ya
    if k == 2:
        return za
    if k == 3:
        return xa * xa
    if k == 4:
        return ya * ya
    if k == 5:
        return za * za
    if k == 6:
        return xa * ya
    if k == 7:
        return xa * za
    return ya * za


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_grid_3d(nx: int, xmn: float, xsiz: float,
                   ny: int = 1, ymn: float = 0.0, ysiz: float = 1.0,
                   nz: int = 1, zmn: float = 0.0, zsiz: float = 1.0) -> np.ndarray:
    """
    Build the (nx*ny*nz, 3) array of grid-node centers for a GSLIB-style
    grid definition, in kt3d's own node order (x fastest, then y, then z).
    """
    ix = np.arange(nx)
    iy = np.arange(ny)
    iz = np.arange(nz)
    gx, gy, gz = np.meshgrid(xmn + ix * xsiz, ymn + iy * ysiz, zmn + iz * zsiz, indexing="ij")
    # kt3d's node order is x-fastest; meshgrid with 'ij' + this transpose matches it
    return np.column_stack([gx.ravel(order="F"), gy.ravel(order="F"), gz.ravel(order="F")])


def kt3d(
    data: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    points,
    variogram: Dict,
    z: Optional[str] = None,
    ktype: str = "ordinary",
    skmean: float = 0.0,
    drift: Sequence[int] = (0,) * 9,
    itrend: bool = False,
    ext_drift: Optional[str] = None,
    ext_drift_points: Optional[np.ndarray] = None,
    search_radius: float = 1.0,
    search_radius2: Optional[float] = None,
    search_radius3: Optional[float] = None,
    search_angles: Sequence[float] = (0.0, 0.0, 0.0),
    ndmin: int = 1,
    ndmax: int = 12,
    noct: int = 0,
    block_size: Optional[Sequence[float]] = None,
    block_discretization: Sequence[int] = (1, 1, 1),
    trim: Sequence[float] = (-1.0e21, 1.0e21),
    use_gpu: Union[bool, str] = "auto",
) -> pd.DataFrame:
    """
    Krige `value` from `data` onto `points` (GSLIB's ``kt3d`` program).

    Parameters
    ----------
    data : DataFrame
        Conditioning data.
    x, y, z : str
        Column names for coordinates in `data`. `z` may be omitted for 2-D
        data (treated as z=0 everywhere).
    value : str
        Column name of the variable to krige.
    points : array-like, shape (n, 2) or (n, 3)
        Locations to estimate. Use :func:`build_grid_3d` for a regular grid.
    variogram : dict
        ``{"nugget": c0, "structures": [ {..}, ... ]}``. Each structure is
        ``{"type": "spherical"|"exponential"|"gaussian"|"power"|"hole",
        "sill": cc, "range": a, "range2": a1, "range3": a2,
        "angles": (ang1, ang2, ang3)}``. `range2`/`range3` (minor-horizontal
        and vertical ranges) and `angles` default to isotropic / unrotated.
        Same parametrization as ``cova3.for`` -- see module docstring.
    ktype : {'ordinary', 'simple', 'locally_varying_mean', 'external_drift'}
        Kriging type (GSLIB's ``ktype`` 1, 0, 2, 3 respectively).
        'ordinary' + any `drift` flag set gives kriging with a polynomial
        trend (KT); 'external_drift' additionally requires `ext_drift` /
        `ext_drift_points`.
    skmean : float
        Stationary mean for `ktype='simple'`.
    drift : sequence of 9 0/1 flags
        Polynomial drift terms to include (only used with `ktype='ordinary'`
        or `'external_drift'`), in order x, y, z, x^2, y^2, z^2, xy, xz, yz.
    itrend : bool
        If True, estimate the drift/trend surface itself rather than the
        variable (GSLIB's ``itrend``).
    ext_drift, ext_drift_points : str, array-like
        Column name of the external-drift variable in `data`, and its value
        at each of `points`. Required for `ktype in ('locally_varying_mean',
        'external_drift')`.
    search_radius, search_radius2, search_radius3 : float
        Search ellipsoid radii (major, minor-horizontal, vertical). The
        latter two default to `search_radius` (isotropic search).
    search_angles : (ang1, ang2, ang3)
        Orientation of the search ellipsoid, same convention as variogram
        `angles`.
    ndmin, ndmax : int
        Minimum/maximum data used per estimate.
    noct : int
        If > 0, cap neighbors to `noct` per octant (GSLIB octant search).
    block_size, block_discretization : sequence of 3
        If given, krige the block average over a `block_size`-sized block
        discretized into `block_discretization` points (GSLIB block
        kriging); otherwise point kriging.
    trim : (tmin, tmax)
        Data outside `[tmin, tmax)` on `value` are dropped, matching GSLIB.
    use_gpu : bool or 'auto', default='auto'
        For ordinary/simple **point** kriging with `noct=0` (no octant cap)
        and no extra drift terms, stack all full-neighborhood systems into
        one `(n_pred, k+1, k+1)` solve (NumPy or CuPy). Other modes keep the
        original per-point loop (CPU). Same `'auto'` / True / False
        convention as :class:`pygstat.core.kriging.OrdinaryKriging`.

    Returns
    -------
    DataFrame with columns x, y, z, estimate, variance (NaN where too few
    neighbors were found or the local system was singular).
    """
    if ktype not in _KTYPE_CODES:
        raise ValueError(f"ktype must be one of {list(_KTYPE_CODES)}")
    kcode = _KTYPE_CODES[ktype]
    drift_flags = [int(bool(f)) for f in drift]
    if len(drift_flags) != 9:
        raise ValueError("drift must have exactly 9 entries")
    if kcode in (0, 2):
        drift_flags = [0] * 9
    if kcode == 3 and ext_drift is None:
        raise ValueError("ktype='external_drift' requires ext_drift")
    if kcode == 2 and ext_drift is None:
        raise ValueError("ktype='locally_varying_mean' requires ext_drift")

    tmin, tmax = trim
    # Pull each column directly from `data` (rather than a `data[[...]]`
    # sub-frame) so nothing breaks when `value` and `ext_drift` name the same
    # column (a valid use of 'locally_varying_mean'/'external_drift').
    zcol = data[z].to_numpy(dtype=float) if z is not None else np.zeros(len(data))
    vcol = data[value].to_numpy(dtype=float)
    keep = (vcol >= tmin) & (vcol < tmax)
    xs = data[x].to_numpy(dtype=float)[keep]
    ys = data[y].to_numpy(dtype=float)[keep]
    zs = zcol[keep]
    vs = vcol[keep]
    ves = data[ext_drift].to_numpy(dtype=float)[keep] if ext_drift is not None else np.ones(len(vs))
    if len(vs) < 1:
        raise ValueError("no data left after trimming")
    coords = np.column_stack([xs, ys, zs])

    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    if kcode in (2, 3):
        if ext_drift_points is None:
            raise ValueError("ext_drift_points is required for this ktype")
        ext_pred = np.asarray(ext_drift_points, dtype=float)
        if len(ext_pred) != len(pts):
            raise ValueError("ext_drift_points must have one value per row of `points`")
    else:
        ext_pred = np.zeros(len(pts))

    c0, structures, rotmats, covmax = _prepare_variogram(variogram)
    cov_cfg = (c0, structures, rotmats, covmax)

    r1 = search_radius if search_radius2 is None else search_radius2
    r2 = search_radius if search_radius3 is None else search_radius3
    if search_radius <= 0:
        raise ValueError("search_radius must be > 0")
    search_rot = _setrot(*search_angles, r1 / search_radius, r2 / search_radius)
    radsqd = search_radius ** 2

    if radsqd < 1.0:
        resc = 1.0 / (2.0 * search_radius / max(covmax, 1.0e-4))
    else:
        resc = 1.0 / ((4.0 * radsqd) / max(covmax, 1.0e-4))

    if block_size is not None:
        xdb_all = _discretization_points(block_size, block_discretization)
    else:
        xdb_all = np.zeros((1, 3))
    ndb = len(xdb_all)

    unbias = _cova3_matrix(xdb_all[:1], xdb_all[:1], c0, structures, rotmats, covmax)[0, 0]
    if ndb > 1:
        Cbb = _cova3_matrix(xdb_all, xdb_all, c0, structures, rotmats, covmax)
        np.fill_diagonal(Cbb, np.diag(Cbb) - c0)
        cbb = float(Cbb.mean())
    else:
        cbb = unbias

    bv = np.zeros(9)
    bv[0] = xdb_all[:, 0].mean()
    bv[1] = xdb_all[:, 1].mean()
    bv[2] = xdb_all[:, 2].mean()
    bv[3] = (xdb_all[:, 0] ** 2).mean()
    bv[4] = (xdb_all[:, 1] ** 2).mean()
    bv[5] = (xdb_all[:, 2] ** 2).mean()
    bv[6] = (xdb_all[:, 0] * xdb_all[:, 1]).mean()
    bv[7] = (xdb_all[:, 0] * xdb_all[:, 2]).mean()
    bv[8] = (xdb_all[:, 1] * xdb_all[:, 2]).mean()
    bv *= resc

    estimates = np.full(len(pts), np.nan)
    variances = np.full(len(pts), np.nan)

    use_gpu = resolve_cupy_use_gpu(use_gpu)
    can_batch = (
        noct <= 0
        and not itrend
        and block_size is None
        and kcode in (0, 1)
        and int(sum(drift_flags)) == 0
        and ndmax >= 2
        and len(pts) > 1
    )

    remaining = np.ones(len(pts), dtype=bool)
    if can_batch:
        idx_knn, valid = _find_neighbors_knn_batch(pts, coords, search_rot, radsqd, ndmax)
        full = valid.all(axis=1)
        if np.any(full):
            idx_f = idx_knn[full]
            P = coords[idx_f] - pts[full, None, :]
            vra = vs[idx_f]
            try:
                est_b, var_b = _estimate_ok_sk_batch(
                    P, vra, kcode, skmean, c0, structures, rotmats, covmax,
                    unbias, cbb, use_gpu,
                )
                estimates[full] = est_b
                variances[full] = var_b
                remaining[full] = False
            except Exception:
                remaining[full] = True

    for i in np.where(remaining)[0]:
        target = pts[i]
        idx = _find_neighbors(target, coords, search_rot, radsqd, ndmax, noct)
        na = len(idx)
        if na == 0 or na < ndmin:
            continue

        xa = coords[idx, 0] - target[0]
        ya = coords[idx, 1] - target[1]
        za = coords[idx, 2] - target[2]
        vra = vs[idx]
        vea = ves[idx]

        extest = ext_pred[i]
        skmean_i = extest if kcode == 2 else skmean
        resce = covmax / max(extest, 1.0e-4) if kcode == 3 else 1.0
        r_ext = extest * resce

        na_eff = na
        mdt_base = 0 if kcode in (0, 2) else 1 + int(sum(drift_flags)) + (1 if kcode == 3 else 0)
        if 1 <= na_eff <= mdt_base:
            continue

        if kcode != 3:
            est, var = _estimate_one(
                xdb_all, cov_cfg, drift_flags, kcode, skmean_i, itrend,
                xa, ya, za, vra, vea, resc, unbias, cbb, bv,
            )
        else:
            # external-drift RHS constant (extest*resce) must reach the solver
            est, var = _estimate_one_ext(
                xdb_all, cov_cfg, drift_flags, itrend,
                xa, ya, za, vra, vea, resc, resce, r_ext, unbias, cbb, bv,
            )
        estimates[i] = est
        variances[i] = var

    return pd.DataFrame({
        "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
        "estimate": estimates, "variance": variances,
    })


def _estimate_one_ext(xdb, cov_cfg, drift_flags, itrend,
                       xa, ya, za, vra, vea, resc, resce, r_ext, unbias, cbb, bv):
    """`_estimate_one` specialized for ktype='external_drift' (ktype=3), whose
    external-drift RHS entry is `extest*resce` rather than `bv[k]*resc`."""
    na = len(xa)
    c0, structures, rotmats, covmax = cov_cfg
    mdt = 1 + int(sum(drift_flags)) + 1
    if na <= mdt:
        return np.nan, np.nan

    P = np.column_stack([xa, ya, za])
    neq = na + mdt
    A = np.zeros((neq, neq))
    A[:na, :na] = _cova3_matrix(P, P, c0, structures, rotmats, covmax)
    A[:na, na] = unbias
    A[na, :na] = unbias

    cb_i = _point_to_block_cov(P, xdb, c0, structures, rotmats, covmax)
    r = np.zeros(neq)
    r[:na] = cb_i
    r[na] = unbias

    im = na
    for k, flag in enumerate(drift_flags):
        if not flag:
            continue
        im += 1
        col = _drift_column(k, xa, ya, za) * resc
        A[im, :na] = col
        A[:na, im] = col
        r[im] = bv[k] * resc

    im += 1
    A[im, :na] = vea * resce
    A[:na, im] = vea * resce
    r[im] = r_ext

    rr = r.copy()
    if itrend:
        r[:na] = 0.0

    try:
        s = np.linalg.solve(A, r)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    estv = cbb - float(np.dot(s, rr))
    est = float(np.dot(s[:na], vra))
    return est, max(estv, 0.0)


def kt3d_grid(
    data: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    variogram: Dict,
    nx: int,
    xmn: float,
    xsiz: float,
    ny: int = 1,
    ymn: float = 0.0,
    ysiz: float = 1.0,
    nz: int = 1,
    zmn: float = 0.0,
    zsiz: float = 1.0,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience wrapper: build a regular grid with :func:`build_grid_3d` and
    krige onto it with :func:`kt3d`. All `kt3d` keyword arguments (`z`,
    `ktype`, `variogram` search settings, etc.) are accepted via `**kwargs`.
    """
    pts = build_grid_3d(nx, xmn, xsiz, ny, ymn, ysiz, nz, zmn, zsiz)
    result = kt3d(data, x, y, value, pts, variogram, **kwargs)
    result.insert(0, "iz", np.repeat(np.arange(nz), nx * ny))
    result.insert(0, "iy", np.tile(np.repeat(np.arange(ny), nx), nz))
    result.insert(0, "ix", np.tile(np.arange(nx), ny * nz))
    return result


def kt3d_cross_validate(
    data: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    variogram: Dict,
    z: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Leave-one-out cross-validation: krige each datum's location from all
    *other* data (GSLIB's ``kt3d`` jackknife-against-self option, koption=1).

    Returns a DataFrame with columns x, y, z, true, estimate, variance,
    error (`estimate - true`).
    """
    zcol = data[z].to_numpy(dtype=float) if z is not None else np.zeros(len(data))
    xcol = data[x].to_numpy(dtype=float)
    ycol = data[y].to_numpy(dtype=float)
    vcol = data[value].to_numpy(dtype=float)

    n = len(data)
    est = np.full(n, np.nan)
    var = np.full(n, np.nan)
    ext_drift = kwargs.get("ext_drift")

    for i in range(n):
        held_out = data.drop(data.index[i])
        target = np.array([[xcol[i], ycol[i], zcol[i]]])
        call_kwargs = dict(kwargs)
        if ext_drift is not None:
            call_kwargs["ext_drift_points"] = [data.iloc[i][ext_drift]]
        result = kt3d(held_out, x, y, value, target, variogram, z=z, **call_kwargs)
        est[i] = result["estimate"].iloc[0]
        var[i] = result["variance"].iloc[0]

    return pd.DataFrame({
        "x": xcol, "y": ycol, "z": zcol, "true": vcol,
        "estimate": est, "variance": var, "error": est - vcol,
    })
