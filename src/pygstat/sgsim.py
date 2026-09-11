"""
Sequential Gaussian Simulation (SGS) -- Python port of GSLIB's sgsim.

`Helper_packages/Gslib90/sgsim.exe` is a compiled 2005 Windows binary with no
Fortran source anywhere in this project, so it could not be literally
"converted." Instead, this module implements the same algorithm (Deutsch &
Journel, *GSLIB: Geostatistical Software Library*) faithfully: for each
realization, visit every grid node along a random path, krige (simple or
ordinary) from the nearby data + already-simulated nodes using an octant
search, draw a Gaussian value from that local kriging estimate/variance, and
add it to the conditioning set before moving to the next node.

The kriging engine (rotation matrix for anisotropy, octant neighbor search,
covariance functions, simple/ordinary kriging linear systems) is adapted from
`GStatSim`'s `gstatsim.py` (https://github.com/GatorGlaciology/GStatSim),
which was given as the starting point for this port. Two capabilities of the
real `sgsim.exe` that `gstatsim.py`'s per-call `skrige_sgs`/`okrige_sgs`
functions don't bundle are added here:

- **Multiple realizations in one call** (`nsim`), each with its own
  independent random path and draws, matching `sgsim.exe`'s `nsim` parameter.
- **An integrated normal-score transform/back-transform** (`itrans`), using
  `pygstat.nscore.nscore_forward`/`nscore_back`, matching `sgsim.exe`'s
  `itrans` parameter -- SGS assumes (multi-)Gaussian data, so the real
  program can transform on the way in and back-transform realizations on the
  way out.

Nodes *within* one realization can't be batched (each depends on every
previously-simulated node on that realization's path), but the `nsim`
realizations are independent of each other. When `use_gpu` is truthy and
`nsim >= 2`, `sgsim` exploits that: `_simulate_batch` runs all realizations
in lockstep (one shared step index, each visiting the step-th node of its
own path) and dispatches all `nsim` local kriging systems at each step to
the GPU as one batched call, instead of `_simulate_one`'s one-node-at-a-time
dispatch. See `_simulate_batch`'s docstring for the padding/solve details.

Variogram convention: `vario` follows GSLIB/GStatSim's practical-range
covariance parametrization, **not** `pygstat.core.variogram_models`' own
(range) parametrization -- they decay differently (e.g. exponential here is
``exp(-3h)``, giving ~95% of sill at the practical range; `core.variogram`'s
exponential is ``1 - exp(-h/range)``, reaching that same ~95% at 3x its
`range`). Keeping GSLIB's own convention here matches the actual `sgsim.exe`
behavior being ported, rather than silently changing its semantics.
"""

import math
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.special import kv, gamma

from .nscore import nscore_forward, nscore_back
from .utils.backend import get_array_module, as_gpu_array, to_numpy, resolve_cupy_use_gpu

__all__ = ["sgsim"]


# ---------------------------------------------------------------------------
# Random number generator helper
# ---------------------------------------------------------------------------

def _get_random_generator(seed):
    """int/None/Generator -> numpy.random.Generator (adapted from gstatsim.py)."""
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, (int, np.integer)):
        return np.random.default_rng(seed=int(seed))
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError("seed should be an integer, a NumPy random Generator, or None")


# ---------------------------------------------------------------------------
# Anisotropy rotation matrix
# ---------------------------------------------------------------------------

def _validate_vario(vario: Sequence) -> None:
    """Validate GSLIB-style vario = [azimuth, nugget, major, minor, sill, vtype, ...]."""
    if len(vario) < 6:
        raise ValueError(
            "vario must be [azimuth, nugget, major_range, minor_range, sill, vtype, ...]"
        )
    nugget, major_range, minor_range, sill = vario[1], vario[2], vario[3], vario[4]
    if major_range <= 0 or minor_range <= 0:
        raise ValueError("major_range and minor_range must be > 0")
    if sill < nugget:
        raise ValueError(
            "sill is total C(0) and must be >= nugget "
            "(pass nugget + partial_sill, not the partial sill alone)"
        )


def _make_rotation_matrix(azimuth: float, major_range: float, minor_range: float) -> np.ndarray:
    """
    2x2 matrix that rotates by `azimuth` (degrees from horizontal, clockwise
    per GSLIB convention) and rescales by 1/major_range, 1/minor_range, so
    that distances in the transformed space of exactly 1 correspond to the
    practical range in the corresponding direction.
    """
    theta = (azimuth / 180.0) * np.pi
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
    scale = np.array([[1.0 / major_range, 0.0], [0.0, 1.0 / minor_range]])
    return rotation @ scale


# ---------------------------------------------------------------------------
# Covariance functions (GSLIB practical-range convention)
# ---------------------------------------------------------------------------
# Array-module-agnostic (numpy or cupy, via get_array_module) except
# 'matern', which needs scipy.special.kv/gamma (no CuPy equivalent) and so
# always evaluates on CPU internally, transferring a GPU input over and
# back transparently -- correct either way, just not itself accelerated
# (same handling as pygstat.core.variogram_models.matern).

def _covar(effective_lag: np.ndarray, sill: float, nugget: float, vtype: str,
           smoothness: Optional[float] = None) -> np.ndarray:
    """
    Covariance at a rotated/rescaled ("effective") lag, where 1.0 = the
    practical range. `vtype` is one of 'exponential', 'gaussian',
    'spherical', 'matern' (case-insensitive).
    """
    xp = get_array_module(effective_lag)
    structured_sill = sill - nugget
    v = vtype.lower()
    if v == "exponential":
        c = structured_sill * xp.exp(-3.0 * effective_lag)
    elif v == "gaussian":
        c = structured_sill * xp.exp(-3.0 * xp.square(effective_lag))
    elif v == "spherical":
        c = xp.zeros_like(effective_lag)
        mask = effective_lag <= 1.0
        h = effective_lag[mask]
        c[mask] = structured_sill * (1.0 - 1.5 * h + 0.5 * h ** 3)
    elif v == "matern":
        if smoothness is None:
            raise ValueError("smoothness must be specified for Matern covariance")
        on_gpu = xp is not np
        lag_cpu = np.asarray(to_numpy(effective_lag), dtype=float)
        scale = 0.45246434 * np.exp(-0.70449189 * smoothness) + 1.7863836
        h = np.array(lag_cpu, dtype=float, copy=True)
        h[h == 0.0] = 1e-8
        c = (structured_sill * 2.0 / gamma(smoothness)
             * np.power(scale * h * np.sqrt(smoothness), smoothness)
             * kv(smoothness, 2.0 * scale * h * np.sqrt(smoothness)))
        c[np.isnan(c)] = structured_sill
        return xp.asarray(c) if on_gpu else c
    else:
        raise ValueError("vtype must be 'exponential', 'gaussian', 'spherical', or 'matern'")
    return c


def _pairwise_effective_lag(A, B, xp):
    """Euclidean distance between every row of A and every row of B (both
    already rotated/rescaled into "effective lag" space)."""
    diff = A[:, None, :] - B[None, :, :]
    return xp.sqrt(xp.sum(diff ** 2, axis=2))


def _covariance_matrix(coords: np.ndarray, vario: Sequence, rotation_matrix: np.ndarray) -> np.ndarray:
    """n x n covariance matrix between all pairs of `coords`."""
    xp = get_array_module(coords)
    nugget, sill, vtype = vario[1], vario[4], vario[5]
    smoothness = vario[6] if vtype.lower() == "matern" else None
    rotated = coords @ rotation_matrix
    effective_lag = _pairwise_effective_lag(rotated, rotated, xp)
    cov = _covar(effective_lag, sill, nugget, vtype, smoothness)
    idx = xp.arange(cov.shape[0])
    cov[idx, idx] = sill
    return cov


def _covariance_array(coords: np.ndarray, target: np.ndarray, vario: Sequence,
                       rotation_matrix: np.ndarray) -> np.ndarray:
    """n-length covariance array between `coords` and one repeated `target` point."""
    xp = get_array_module(coords, target)
    nugget, sill, vtype = vario[1], vario[4], vario[5]
    smoothness = vario[6] if vtype.lower() == "matern" else None
    rotated = coords @ rotation_matrix
    target_rotated = target.reshape(-1, 2) @ rotation_matrix
    effective_lag = xp.sqrt(xp.square(rotated - target_rotated).sum(axis=1))
    return _covar(effective_lag, sill, nugget, vtype, smoothness)


# ---------------------------------------------------------------------------
# Octant neighbor search
# ---------------------------------------------------------------------------

def _octant_search(radius: float, num_points: int, loc: np.ndarray, data: pd.DataFrame) -> np.ndarray:
    """
    Nearest-neighbor search within `radius`, split into 8 angular octants and
    capped at `num_points // 8` per octant, so nearby data are spread around
    the target rather than clustered on one side (GSLIB's standard search).
    Returns an (n, 3) array of [X, Y, Z] for the selected neighbors.
    """
    dx = data["X"].values - loc[0]
    dy = data["Y"].values - loc[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dx, dy)

    d = data.copy()
    d["dist"] = dist
    d["angle"] = angle
    d = d[d["dist"] < radius].sort_values("dist")

    bins = [-math.pi, -3 * math.pi / 4, -math.pi / 2, -math.pi / 4, 0,
            math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi + 1e-3]
    d["oct"] = pd.cut(d["angle"], bins=bins, labels=list(range(8)))

    per_octant = max(1, num_points // 8)
    picked = [d[d["oct"] == i].iloc[:per_octant][["X", "Y", "Z"]].values for i in range(8)]
    near = np.concatenate(picked, axis=0) if picked else np.empty((0, 3))

    if near.shape[0] == 0:
        raise ValueError("No neighbors found within radius; try increasing radius or num_points")
    return near


def _pad_size(num_points: int) -> int:
    """Widest neighbor set `_octant_search(..., num_points, ...)` can ever
    return: 8 octants x `num_points // 8` each. `_simulate_batch` zero-pads
    every per-node local system to this fixed width so systems from
    different realizations (which generally find different neighbor
    counts) can be stacked into one batched array."""
    return 8 * max(1, num_points // 8)


def _gather_padded_neighbors(radius: float, num_points: int, pad_size: int,
                              target: np.ndarray, data: pd.DataFrame):
    """`_octant_search`, zero-padded up to a fixed `pad_size` neighbors.
    Returns `(xy, z_vals, mask)` -- `mask` marks the first `n <= pad_size`
    entries (the real neighbors) as valid, the rest as padding. Shared by
    `sgsim._simulate_batch` and `sisim._simulate_batch`, whose batched local
    systems need every realization's neighbor set at the same fixed width
    to stack into one array."""
    near = _octant_search(radius, num_points, target, data)
    n = len(near)
    xy = np.zeros((pad_size, 2))
    z_vals = np.zeros(pad_size)
    mask = np.zeros(pad_size, dtype=bool)
    xy[:n], z_vals[:n], mask[:n] = near[:, :2], near[:, 2], True
    return xy, z_vals, mask


def _batched_local_systems(xy_batch, target_batch, valid_mask, vario: Sequence,
                            rotation_matrix, ridge_frac: float = 1e-7):
    """Batched covariance matrix `cov` (m, pad, pad) and target-covariance
    vector `c0` (m, pad) for `m` stacked, padded per-node neighbor sets
    (`xy_batch`, `target_batch`, `valid_mask` as built by
    `_gather_padded_neighbors` + `np.stack`), one system per batch row.

    Padding (`valid_mask` False) rows/columns of `cov` are forced to an
    identity block (0 off-diagonal, 1 on it) and the matching `c0` entries
    to 0, so solving always assigns padding entries exactly zero weight
    regardless of their (irrelevant) coordinates -- letting ragged
    neighbor counts share one fixed-size batched solve. A small ridge
    (`ridge_frac * sill`) is added to valid diagonal entries since CuPy has
    no batched `lstsq`; `xp.linalg.solve` needs a non-singular system,
    unlike `_simulate_one`/`_indicator_ccdf`'s per-node `lstsq`. Shared by
    `sgsim._simulate_batch` and `sisim._simulate_batch`.
    """
    nugget, sill, vtype = vario[1], vario[4], vario[5]
    smoothness = vario[6] if vtype.lower() == "matern" else None
    xp = get_array_module(xy_batch)
    ridge = ridge_frac * sill
    pad_size = xy_batch.shape[1]

    rotated = xy_batch @ rotation_matrix                                # (m, pad, 2)
    target_rotated = target_batch @ rotation_matrix                    # (m, 2)
    diff = rotated[:, :, None, :] - rotated[:, None, :, :]
    eff_lag = xp.sqrt(xp.sum(diff ** 2, axis=-1))                       # (m, pad, pad)
    eff_lag_t = xp.sqrt(xp.sum((rotated - target_rotated[:, None, :]) ** 2, axis=-1))  # (m, pad)

    cov = _covar(eff_lag, sill, nugget, vtype, smoothness)
    c0 = _covar(eff_lag_t, sill, nugget, vtype, smoothness)

    outer_valid = valid_mask[:, :, None] & valid_mask[:, None, :]
    cov = xp.where(outer_valid, cov, 0.0)
    idx_pad = xp.arange(pad_size)
    cov[:, idx_pad, idx_pad] = xp.where(valid_mask, sill + ridge, 1.0)
    c0 = xp.where(valid_mask, c0, 0.0)
    return cov, c0


# ---------------------------------------------------------------------------
# One realization
# ---------------------------------------------------------------------------

def _simulate_one(prediction_grid: np.ndarray, df: pd.DataFrame, num_points: int,
                   vario: Sequence, radius: float, kriging_type: str,
                   rng: np.random.Generator, quiet: bool, use_gpu: bool = False) -> np.ndarray:
    azimuth, nugget, major_range, minor_range, sill, vtype = vario[:6]
    rotation_matrix = _make_rotation_matrix(azimuth, major_range, minor_range)
    if use_gpu:
        rotation_matrix = as_gpu_array(rotation_matrix)

    data = df[["X", "Y", "Z"]].copy()
    global_mean = data["Z"].mean()

    path = rng.permutation(len(prediction_grid))
    sim = np.empty(len(prediction_grid))

    iterator = path
    if not quiet:
        try:
            from tqdm import tqdm
            iterator = tqdm(path, position=0, leave=True)
        except ImportError:
            pass

    for z in iterator:
        target = prediction_grid[z]
        coincident = np.where(
            (np.abs(data["X"].values - target[0]) <= 1e-12)
            & (np.abs(data["Y"].values - target[1]) <= 1e-12)
        )[0]
        if coincident.size > 0:
            sim[z] = data["Z"].values[coincident[0]]
            continue

        near = _octant_search(radius, num_points, target, data)
        xy, z_vals = near[:, :2], near[:, 2]
        n = len(near)

        # Each node's local system depends on every previously-simulated
        # node (added to `data` below) -- that dependency chain is exactly
        # what makes this "sequential" and is why nodes can't be batched
        # together the way pygstat.indicator_kriging's independent
        # predictions can. `use_gpu` still genuinely dispatches this
        # node's own covariance-matrix build + solve to CuPy when usable;
        # it mainly pays off for a large `num_points` (bigger local
        # systems) -- for small ones, per-call GPU dispatch overhead can
        # outweigh the benefit.
        if use_gpu:
            xy_d, target_d = as_gpu_array(xy), as_gpu_array(target)
        else:
            xy_d, target_d = xy, target
        xp = get_array_module(xy_d)

        if kriging_type == "simple":
            C = _covariance_matrix(xy_d, vario, rotation_matrix)
            c0 = _covariance_array(xy_d, xp.tile(target_d, n), vario, rotation_matrix)
            weights, *_ = xp.linalg.lstsq(C, c0, rcond=None)
            weights, c0 = to_numpy(weights), to_numpy(c0)
            est = global_mean + np.sum(weights * (z_vals - global_mean))
            var = sill - np.sum(weights * c0)
        elif kriging_type == "ordinary":
            local_mean = z_vals.mean()
            C = xp.zeros((n + 1, n + 1))
            C[:n, :n] = _covariance_matrix(xy_d, vario, rotation_matrix)
            C[n, :n] = 1.0
            C[:n, n] = 1.0
            c0 = xp.zeros(n + 1)
            c0[:n] = _covariance_array(xy_d, xp.tile(target_d, n), vario, rotation_matrix)
            c0[n] = 1.0
            weights, *_ = xp.linalg.lstsq(C, c0, rcond=None)
            weights, c0 = to_numpy(weights), to_numpy(c0)
            est = local_mean + np.sum(weights[:n] * (z_vals - local_mean))
            var = sill - np.sum(weights[:n] * c0[:n]) - weights[n]
        else:
            raise ValueError("kriging_type must be 'simple' or 'ordinary'")

        var = max(float(var), 0.0)
        sim[z] = rng.normal(est, math.sqrt(var))

        # Add the just-simulated node to the conditioning set so later nodes
        # on this realization's path are constrained by it too (the
        # "sequential" in Sequential Gaussian Simulation).
        data.loc[len(data)] = [target[0], target[1], sim[z]]

    return sim


# ---------------------------------------------------------------------------
# All realizations at once, batched across realizations (GPU, nsim >= 2)
# ---------------------------------------------------------------------------

def _simulate_batch(prediction_grid: np.ndarray, df: pd.DataFrame, num_points: int,
                     vario: Sequence, radius: float, kriging_type: str,
                     rngs: List[np.random.Generator], quiet: bool) -> np.ndarray:
    """
    GPU path for `nsim >= 2`. Nodes within one realization still can't be
    batched -- each depends on every previously-simulated node on its own
    path -- but the `nsim` realizations don't depend on each other at all.
    This runs them in lockstep (one shared step index, each realization
    visiting the step-th node of its own random path), so at every step
    there are up to `nsim` independent local kriging systems, which *can*
    be dispatched to the GPU as a single batched covariance build +
    `xp.linalg.solve` instead of `nsim` separate tiny ones. That turns
    `_simulate_one`'s `nsim * len(prediction_grid)` one-node-at-a-time GPU
    dispatches into `len(prediction_grid)` dispatches, each doing `nsim`
    times the work -- the batching GPUs need to amortize kernel-launch and
    host<->device transfer overhead.

    Each realization still gets its own path and its own growing
    conditioning set (both generated the same way `_simulate_one` would,
    including drawing the path permutation before any simulated value, so
    the RNG call order per realization is unchanged), so this produces the
    same simulation each realization would on its own -- interleaving the
    realizations' steps doesn't change what any one realization's RNG
    stream draws, since the streams are independent.

    Two differences from `_simulate_one`, both consequences of batching:

    - Octant search still runs once per realization per step (it's a
      per-realization pandas operation on that realization's own growing
      conditioning set, not something that batches). Only the covariance
      build + linear solve is batched.
    - Neighbor counts vary per node/realization, so each local system is
      zero-padded up to `_pad_size(num_points)` with an identity block for
      the padding rows/columns (forced to contribute exactly zero weight)
      rather than solved at its own ragged size. And because CuPy has no
      batched `lstsq`, padded systems are solved with `xp.linalg.solve`
      plus a small ridge (`1e-7 * sill`) on the diagonal for numerical
      stability -- `_simulate_one`'s `lstsq` is more robust to a singular
      or rank-deficient local system (e.g. duplicate/collinear neighbor
      locations); this path is not, beyond what the ridge covers.
    """
    azimuth, nugget, major_range, minor_range, sill, vtype = vario[:6]
    rotation_matrix = as_gpu_array(_make_rotation_matrix(azimuth, major_range, minor_range))

    nsim = len(rngs)
    grid_size = len(prediction_grid)
    pad_size = _pad_size(num_points)

    base = df[["X", "Y", "Z"]].copy()
    global_mean = base["Z"].mean()
    data_list = [base.copy() for _ in range(nsim)]
    # rng.permutation is each realization's first draw, matching _simulate_one.
    paths = [rng.permutation(grid_size) for rng in rngs]

    sims = np.empty((nsim, grid_size))

    steps = range(grid_size)
    if not quiet:
        try:
            from tqdm import tqdm
            steps = tqdm(steps, position=0, leave=True)
        except ImportError:
            pass

    for step in steps:
        # `paths[i]` is a permutation of grid *node indices*; step just
        # walks its positions. The node each realization actually visits
        # this step -- and where its result belongs in `sims`/`data_list`
        # -- is `paths[i][step]`, not `step` itself.
        node_idxs = [paths[i][step] for i in range(nsim)]
        targets = [prediction_grid[node_idxs[i]] for i in range(nsim)]

        need_xy, need_z, need_mask, need_idx, need_node = [], [], [], [], []
        for i in range(nsim):
            target = targets[i]
            data = data_list[i]
            coincident = np.where(
                (np.abs(data["X"].values - target[0]) <= 1e-12)
                & (np.abs(data["Y"].values - target[1]) <= 1e-12)
            )[0]
            if coincident.size > 0:
                sims[i, node_idxs[i]] = data["Z"].values[coincident[0]]
                continue

            xy, z_vals, mask = _gather_padded_neighbors(radius, num_points, pad_size, target, data)
            need_xy.append(xy)
            need_z.append(z_vals)
            need_mask.append(mask)
            need_idx.append(i)
            need_node.append(node_idxs[i])

        if not need_idx:
            continue

        m = len(need_idx)
        xy_batch = as_gpu_array(np.stack(need_xy))                          # (m, pad, 2)
        z_batch = as_gpu_array(np.stack(need_z))                            # (m, pad)
        valid_mask = as_gpu_array(np.stack(need_mask))                      # (m, pad) bool
        target_batch = as_gpu_array(np.stack([targets[i] for i in need_idx]))  # (m, 2)
        xp = get_array_module(xy_batch)

        cov, c0 = _batched_local_systems(xy_batch, target_batch, valid_mask, vario, rotation_matrix)
        valid_f = valid_mask.astype(cov.dtype)
        n_valid = xp.sum(valid_f, axis=1)

        # xp.linalg.solve(a, b) with a (m, n, n) and b (m, n) does NOT treat
        # b as a batch of m vectors here -- it needs b's *last two* axes as
        # its own (core-dim, rhs-count) pair, so a 2-D b of shape (m, n) is
        # read as one non-batched (m, n) system and mismatches a's batch.
        # Adding a trailing size-1 axis makes b unambiguously (m, n, 1) --
        # m batched systems, one RHS column each -- then squeeze it back off.
        if kriging_type == "simple":
            weights = xp.linalg.solve(cov, c0[..., None])[..., 0]
            est = global_mean + xp.sum(weights * (z_batch - global_mean), axis=1)
            var = sill - xp.sum(weights * c0, axis=1)
        elif kriging_type == "ordinary":
            local_mean = xp.sum(z_batch * valid_f, axis=1) / n_valid
            C = xp.zeros((m, pad_size + 1, pad_size + 1))
            C[:, :pad_size, :pad_size] = cov
            C[:, pad_size, :pad_size] = valid_f
            C[:, :pad_size, pad_size] = valid_f
            c0_full = xp.zeros((m, pad_size + 1))
            c0_full[:, :pad_size] = c0
            c0_full[:, pad_size] = 1.0
            weights_full = xp.linalg.solve(C, c0_full[..., None])[..., 0]
            weights, mu = weights_full[:, :pad_size], weights_full[:, pad_size]
            est = local_mean + xp.sum(weights * (z_batch - local_mean[:, None]), axis=1)
            var = sill - xp.sum(weights * c0, axis=1) - mu
        else:
            raise ValueError("kriging_type must be 'simple' or 'ordinary'")

        var = xp.maximum(var, 0.0)
        est_np, var_np = to_numpy(est), to_numpy(var)

        for k, i in enumerate(need_idx):
            drawn = float(rngs[i].normal(est_np[k], math.sqrt(float(var_np[k]))))
            sims[i, need_node[k]] = drawn
            target = targets[i]
            data_list[i].loc[len(data_list[i])] = [target[0], target[1], drawn]

    return sims


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sgsim(
    prediction_grid: np.ndarray,
    df: pd.DataFrame,
    xx: str,
    yy: str,
    zz: str,
    num_points: int,
    vario: Sequence,
    radius: float,
    kriging_type: str = "ordinary",
    nsim: int = 1,
    itrans: bool = False,
    seed: Optional[Union[int, np.random.Generator]] = None,
    quiet: bool = False,
    use_gpu: Union[bool, str] = False,
) -> np.ndarray:
    """
    Sequential Gaussian Simulation.

    Parameters
    ----------
    prediction_grid : np.ndarray, shape (n, 2)
        x, y coordinates of the grid nodes to simulate.
    df : pandas.DataFrame
        Conditioning data.
    xx, yy, zz : str
        Column names in `df` for x, y coordinates and the variable to simulate.
    num_points : int
        Number of neighbors to search for per node (split across 8 octants).
    vario : sequence
        ``[azimuth, nugget, major_range, minor_range, sill, vtype]``, plus a
        7th element (smoothness) if `vtype='matern'`. ``sill`` is the **total
        C(0) = nugget + partial sill** (GSLIB/GStatSim convention), not the
        partial sill alone. Practical-range covariance -- see module docstring.
    radius : float
        Search radius for neighbors.
    kriging_type : {'ordinary', 'simple'}, default 'ordinary'
        Local kriging system used at each node.
    nsim : int, default 1
        Number of independent realizations to generate.
    itrans : bool, default False
        If True, normal-score-transform `df[zz]` before simulating (each
        realization is simulated in Gaussian space) and back-transform the
        results before returning -- matches `sgsim.exe`'s `itrans` option.
        If the data isn't already (roughly) Gaussian, this should be True.
    seed : int, numpy.random.Generator, or None
        Seeds an independent sub-generator per realization for
        reproducibility; None uses fresh entropy.
    quiet : bool, default False
        If False and `tqdm` is installed, shows a progress bar per realization.
    use_gpu : bool or 'auto', default False
        Dispatch local covariance-matrix build + solve to the GPU via CuPy
        (verified usable the same way as `pygstat.core.kriging`, not just
        import-checked). Unlike `Variogram`/`OrdinaryKriging`/etc., this
        defaults to **False**, not `'auto'`: SGS is inherently *sequential*
        within one realization -- each node depends on every
        previously-simulated node on its path -- so nodes can't be batched
        the way independent predictions can.

        What GPU dispatch actually does depends on `nsim`:

        - `nsim == 1`: one node at a time (`_simulate_one`), same as
          before. For the typical small `num_points` (the local system
          size), per-call GPU dispatch overhead usually outweighs any
          benefit -- pass `True`/`'auto'` only if you use unusually large
          `num_points`.
        - `nsim >= 2`: realizations are independent of each other, so all
          `nsim` local kriging systems at each step are batched into one
          GPU dispatch (`_simulate_batch`) instead of one per node per
          realization -- this is where GPU dispatch is actually likely to
          win, and more so as `nsim` grows. See `_simulate_batch`'s
          docstring for the padding/solve tradeoffs this batching makes.

        `vtype='matern'` always evaluates on CPU regardless (no CuPy Bessel
        functions).

    Returns
    -------
    sims : np.ndarray, shape (nsim, len(prediction_grid))
        One row per realization, in the same order as `prediction_grid`.
    """
    use_gpu = resolve_cupy_use_gpu(use_gpu)
    if nsim < 1:
        raise ValueError("nsim must be >= 1")
    _validate_vario(vario)

    work = df.rename(columns={xx: "X", yy: "Y", zz: "Z"})[["X", "Y", "Z"]].copy()

    if itrans:
        transformed, v_sorted, ns_sorted = nscore_forward(work["Z"].values, tails="linear")
        work["Z"] = transformed

    rng = _get_random_generator(seed)
    child_seeds = rng.integers(0, 2**31 - 1, size=nsim)
    realization_rngs = [_get_random_generator(int(s)) for s in child_seeds]

    if use_gpu and nsim >= 2:
        # Realizations don't depend on each other -- only nodes *within*
        # one realization do -- so batch all nsim local kriging systems at
        # each shared step into one GPU dispatch. See _simulate_batch.
        sims = _simulate_batch(
            prediction_grid, work, num_points, vario, radius,
            kriging_type, realization_rngs, quiet,
        )
    else:
        sims = np.empty((nsim, len(prediction_grid)))
        for i in range(nsim):
            sims[i] = _simulate_one(
                prediction_grid, work, num_points, vario, radius,
                kriging_type, realization_rngs[i], quiet, use_gpu,
            )

    if itrans:
        sims = nscore_back(sims.ravel(), v_sorted, ns_sorted, tails="linear").reshape(sims.shape)

    return sims
