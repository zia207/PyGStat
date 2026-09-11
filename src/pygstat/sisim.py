"""
Sequential Indicator Simulation (SIS) -- Python port of GSLIB's sisim.for.

Ported from the actual Fortran source (`Helper_packages/Gslib90/gslib90sc/`):
`sisim.for` (the SIS driver: `readparm`, `sisim`, `ctable`, `srchnd`, `krige`)
plus the shared GSLIB library routines it calls -- `gslib/ordrel.for` (order
relations correction), `gslib/beyond.for` (interpolate/extrapolate a value
from a discrete CDF), `gslib/powint.for` (power interpolation), and
`gslib/cova3.for` (covariance model -- confirmed to use the same
practical-range convention already implemented in `sgsim.py`, so this module
reuses that file's covariance/rotation/octant-search engine directly).

**Deliberate scope reductions from the full 3D Fortran program** (documented,
not silent): 3D -> 2D (matching the rest of pygstat); the super-block search
and multi-grid search strategies (`sstrat`, `mults`) are pure performance
optimizations of `sisim.for`'s own brute-force search and are replaced here
with the same octant search `sgsim.py` uses -- mathematically equivalent
results, just not sped up for huge grids; the covariance look-up table
(`ctable`) is likewise a performance cache and is replaced with direct
covariance evaluation; only single-structure variograms per threshold are
supported (`sisim.for` allows up to 4 nested structures per threshold);
median IK's weight-reuse approximation (`mik=1`), soft/Markov-Bayes data
(`imbsim`), and the "rescaled global CDF" tail options (`ltail/middle/utail
= 3`, which need an external tabulated-quantiles file) are not implemented.

Everything else -- per-threshold ordinary/simple indicator kriging at each
node using both hard data and previously-simulated nodes, order-relations
correction, and CDF interpolation/tail-extrapolation to draw a value -- is a
faithful, verified-against-source port of `krige`, `ordrel`, and `beyond`.

Like `sgsim.py`, nodes *within* one realization can't be batched (each
depends on every previously-simulated node on that realization's path),
but the `nsim` realizations are independent of each other. When `use_gpu`
is truthy and `nsim >= 2`, `sisim` exploits that the same way `sgsim`
does: `_simulate_batch` runs all realizations in lockstep and, for each
threshold, dispatches all `nsim` local indicator-kriging systems at each
step to the GPU as one batched call, instead of `_indicator_ccdf`'s
one-node-one-threshold-at-a-time dispatch. See `_simulate_batch`'s
docstring for the padding/solve details it shares with `sgsim`'s.
"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .sgsim import (
    _batched_local_systems,
    _covariance_array,
    _covariance_matrix,
    _gather_padded_neighbors,
    _get_random_generator,
    _make_rotation_matrix,
    _octant_search,
    _pad_size,
    _validate_vario,
)
from .utils.backend import get_array_module, as_gpu_array, to_numpy, resolve_cupy_use_gpu

__all__ = ["sisim"]


# ---------------------------------------------------------------------------
# powint.for -- power interpolation
# ---------------------------------------------------------------------------

def _powint(xlow: float, xhigh: float, ylow: float, yhigh: float, xval: float, pow_: float) -> float:
    if (xhigh - xlow) < 1e-20:
        return (yhigh + ylow) / 2.0
    return ylow + (yhigh - ylow) * ((xval - xlow) / (xhigh - xlow)) ** pow_


# ---------------------------------------------------------------------------
# ordrel.for -- order relations correction (continuous variable branch)
# ---------------------------------------------------------------------------

def _ordrel(ccdf: np.ndarray) -> np.ndarray:
    """Correct a local ccdf (one value per ascending threshold) for order
    relation violations: clip to [0,1], then average an up-corrected pass
    (enforce non-decreasing) with a down-corrected pass (enforce
    non-increasing scanned backward -- equivalently non-decreasing too)."""
    ccdf = np.clip(np.asarray(ccdf, dtype=float), 0.0, 1.0)
    up = ccdf.copy()
    for i in range(1, len(up)):
        if up[i] < up[i - 1]:
            up[i] = up[i - 1]
    down = ccdf.copy()
    for i in range(len(down) - 2, -1, -1):
        if down[i] > down[i + 1]:
            down[i] = down[i + 1]
    return 0.5 * (up + down)


# ---------------------------------------------------------------------------
# beyond.for -- draw a Z value from a discrete (corrected) ccdf, given a
# uniform cdfval, with lower/middle/upper tail interpolation models
# ---------------------------------------------------------------------------

def _beyond_zval(
    cdfval: float,
    ccut: np.ndarray,
    ccdf: np.ndarray,
    zmin: float,
    zmax: float,
    ltail: str,
    ltpar: float,
    middle: str,
    mpar: float,
    utail: str,
    utpar: float,
) -> float:
    n = len(ccut)
    if cdfval <= ccdf[0]:
        if ltail == "linear":
            zval = _powint(0.0, ccdf[0], zmin, ccut[0], cdfval, 1.0)
        elif ltail == "power":
            zval = _powint(0.0, ccdf[0], zmin, ccut[0], cdfval, 1.0 / ltpar)
        else:
            raise ValueError("ltail must be 'linear' or 'power'")
    elif cdfval >= ccdf[-1]:
        if utail == "linear":
            zval = _powint(ccdf[-1], 1.0, ccut[-1], zmax, cdfval, 1.0)
        elif utail == "power":
            zval = _powint(ccdf[-1], 1.0, ccut[-1], zmax, cdfval, 1.0 / utpar)
        elif utail == "hyperbolic":
            lam = (ccut[-1] ** utpar) * (1.0 - ccdf[-1])
            denom = max(1.0 - cdfval, 1e-15)
            zval = (lam / denom) ** (1.0 / utpar)
        else:
            raise ValueError("utail must be 'linear', 'power', or 'hyperbolic'")
    else:
        cclow = int(np.searchsorted(ccdf, cdfval, side="right") - 1)
        cclow = min(max(cclow, 0), n - 2)
        cchigh = cclow + 1
        if middle == "linear":
            zval = _powint(ccdf[cclow], ccdf[cchigh], ccut[cclow], ccut[cchigh], cdfval, 1.0)
        elif middle == "power":
            zval = _powint(ccdf[cclow], ccdf[cchigh], ccut[cclow], ccut[cchigh], cdfval, 1.0 / mpar)
        else:
            raise ValueError("middle must be 'linear' or 'power'")

    return min(max(zval, zmin), zmax)


# ---------------------------------------------------------------------------
# krige.for's indicator kriging system, per threshold, at one node
# ---------------------------------------------------------------------------

def _indicator_ccdf(
    target: np.ndarray,
    xy: np.ndarray,
    values: np.ndarray,
    thresholds: np.ndarray,
    global_cdf: np.ndarray,
    vario_by_threshold: Dict[float, Sequence],
    kriging_type: str,
    use_gpu: bool = False,
) -> np.ndarray:
    """One node's per-threshold indicator kriging (see `use_gpu` note on
    `pygstat.sisim.sisim`: dispatches each threshold's covariance-matrix
    build + solve to CuPy when usable, same caveats as `pygstat.sgsim`."""
    n = len(values)
    ccdf = np.empty(len(thresholds))
    if use_gpu:
        xy_d, target_d = as_gpu_array(xy), as_gpu_array(target)
    else:
        xy_d, target_d = xy, target
    xp = get_array_module(xy_d)

    for ic, t in enumerate(thresholds):
        vario = vario_by_threshold[t]
        azimuth, nugget, major_range, minor_range, sill, vtype = vario[:6]
        rotation_matrix = _make_rotation_matrix(azimuth, major_range, minor_range)
        if use_gpu:
            rotation_matrix = as_gpu_array(rotation_matrix)
        indicator = (values <= t).astype(float)

        if kriging_type == "simple":
            C = _covariance_matrix(xy_d, vario, rotation_matrix)
            c0 = _covariance_array(xy_d, xp.tile(target_d, n), vario, rotation_matrix)
            weights, *_ = xp.linalg.lstsq(C, c0, rcond=None)
            weights = to_numpy(weights)
            est = global_cdf[ic] + np.sum(weights * (indicator - global_cdf[ic]))
        elif kriging_type == "ordinary":
            C = xp.zeros((n + 1, n + 1))
            C[:n, :n] = _covariance_matrix(xy_d, vario, rotation_matrix)
            C[n, :n] = 1.0
            C[:n, n] = 1.0
            c0 = xp.zeros(n + 1)
            c0[:n] = _covariance_array(xy_d, xp.tile(target_d, n), vario, rotation_matrix)
            c0[n] = 1.0
            weights, *_ = xp.linalg.lstsq(C, c0, rcond=None)
            weights = to_numpy(weights)
            local_mean = indicator.mean()
            est = local_mean + np.sum(weights[:n] * (indicator - local_mean))
        else:
            raise ValueError("kriging_type must be 'simple' or 'ordinary'")

        ccdf[ic] = est
    return ccdf


# ---------------------------------------------------------------------------
# sisim's main loop: one realization
# ---------------------------------------------------------------------------

def _simulate_one(
    prediction_grid: np.ndarray,
    df: pd.DataFrame,
    thresholds: np.ndarray,
    global_cdf: np.ndarray,
    vario_by_threshold: Dict[float, Sequence],
    num_points: int,
    radius: float,
    kriging_type: str,
    zmin: float,
    zmax: float,
    ltail: str,
    ltpar: float,
    middle: str,
    mpar: float,
    utail: str,
    utpar: float,
    rng: np.random.Generator,
    quiet: bool,
    use_gpu: bool = False,
) -> np.ndarray:
    data = df[["X", "Y", "Z"]].copy()
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
        xy, values = near[:, :2], near[:, 2]

        ccdf = _indicator_ccdf(target, xy, values, thresholds, global_cdf,
                                vario_by_threshold, kriging_type, use_gpu)
        ccdf = _ordrel(ccdf)

        cdfval = rng.uniform()
        zval = _beyond_zval(cdfval, thresholds, ccdf, zmin, zmax,
                             ltail, ltpar, middle, mpar, utail, utpar)
        sim[z] = zval

        # Add the just-simulated node to the conditioning set so later nodes
        # on this path see it too -- the "sequential" in Sequential
        # Indicator Simulation (matches srchnd/krige treating previously
        # simulated nodes exactly like hard data, re-indicatorized per
        # threshold from the simulated continuous value).
        data.loc[len(data)] = [target[0], target[1], zval]

    return sim


# ---------------------------------------------------------------------------
# All realizations at once, batched across realizations (GPU, nsim >= 2)
# ---------------------------------------------------------------------------

def _simulate_batch(
    prediction_grid: np.ndarray,
    df: pd.DataFrame,
    thresholds: np.ndarray,
    global_cdf: np.ndarray,
    vario_by_threshold: Dict[float, Sequence],
    num_points: int,
    radius: float,
    kriging_type: str,
    zmin: float,
    zmax: float,
    ltail: str,
    ltpar: float,
    middle: str,
    mpar: float,
    utail: str,
    utpar: float,
    rngs: List[np.random.Generator],
    quiet: bool,
) -> np.ndarray:
    """
    GPU path for `nsim >= 2`. See `pygstat.sgsim._simulate_batch`'s
    docstring for the general approach this mirrors: realizations don't
    depend on each other (only nodes *within* one realization do), so
    running all `nsim` of them in lockstep -- one shared step index, each
    visiting the step-th node of its own path -- turns `nsim` separate
    per-node GPU dispatches at each step into one batched dispatch.

    SIS adds one wrinkle `sgsim` doesn't have: **each node krige's once per
    threshold**, with a separate local system (its own variogram) per
    threshold (`_indicator_ccdf`'s loop). So this batches realizations
    *within* each threshold separately -- `len(prediction_grid) *
    len(thresholds)` batched covariance-build-+-solve dispatches, instead
    of `_indicator_ccdf`'s `nsim * len(prediction_grid) * len(thresholds)`
    one-node-one-threshold `lstsq` calls. `_batched_local_systems` (shared
    with `sgsim`) does the padding/ridge/solve for each threshold's batch;
    see its docstring for that tradeoff versus `_indicator_ccdf`'s `lstsq`.

    `_ordrel` (order-relations correction) and `_beyond_zval` (tail draw)
    stay scalar CPU, one call per realization per node, same as
    `_simulate_one` -- they're cheap next to the kriging systems and
    (`_beyond_zval` especially, with its branching tail models) not a
    natural fit for batching. Each realization's own RNG only ever draws
    `rng.permutation` (first) then one `rng.uniform()` per node it
    actually simulates, in path order -- unchanged from `_simulate_one`,
    so interleaving realizations here doesn't change what any one
    realization's stream draws.
    """
    nsim = len(rngs)
    grid_size = len(prediction_grid)
    pad_size = _pad_size(num_points)
    n_thresh = len(thresholds)

    rotation_matrices = {}
    for t in thresholds:
        vario = vario_by_threshold[float(t)]
        azimuth, major_range, minor_range = vario[0], vario[2], vario[3]
        rotation_matrices[float(t)] = as_gpu_array(_make_rotation_matrix(azimuth, major_range, minor_range))

    base = df[["X", "Y", "Z"]].copy()
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
        z_batch_raw = as_gpu_array(np.stack(need_z))                        # (m, pad) -- raw Z, re-indicatorized per threshold below
        valid_mask = as_gpu_array(np.stack(need_mask))                      # (m, pad) bool
        target_batch = as_gpu_array(np.stack([targets[i] for i in need_idx]))  # (m, 2)
        xp = get_array_module(xy_batch)
        valid_f = valid_mask.astype(xy_batch.dtype)
        n_valid = xp.sum(valid_f, axis=1)

        ccdf_batch = np.empty((m, n_thresh))
        for ic, t in enumerate(thresholds):
            vario = vario_by_threshold[float(t)]
            cov, c0 = _batched_local_systems(
                xy_batch, target_batch, valid_mask, vario, rotation_matrices[float(t)],
            )
            indicator = xp.where(valid_mask, (z_batch_raw <= t).astype(cov.dtype), 0.0)

            if kriging_type == "simple":
                weights = xp.linalg.solve(cov, c0[..., None])[..., 0]
                est = global_cdf[ic] + xp.sum(weights * (indicator - global_cdf[ic]), axis=1)
            elif kriging_type == "ordinary":
                local_mean = xp.sum(indicator * valid_f, axis=1) / n_valid
                C = xp.zeros((m, pad_size + 1, pad_size + 1))
                C[:, :pad_size, :pad_size] = cov
                C[:, pad_size, :pad_size] = valid_f
                C[:, :pad_size, pad_size] = valid_f
                c0_full = xp.zeros((m, pad_size + 1))
                c0_full[:, :pad_size] = c0
                c0_full[:, pad_size] = 1.0
                weights_full = xp.linalg.solve(C, c0_full[..., None])[..., 0]
                weights = weights_full[:, :pad_size]
                est = local_mean + xp.sum(weights * (indicator - local_mean[:, None]), axis=1)
            else:
                raise ValueError("kriging_type must be 'simple' or 'ordinary'")

            ccdf_batch[:, ic] = to_numpy(est)

        for k, i in enumerate(need_idx):
            ccdf = _ordrel(ccdf_batch[k])
            cdfval = rngs[i].uniform()
            zval = _beyond_zval(cdfval, thresholds, ccdf, zmin, zmax,
                                 ltail, ltpar, middle, mpar, utail, utpar)
            sims[i, need_node[k]] = zval
            target = targets[i]
            data_list[i].loc[len(data_list[i])] = [target[0], target[1], zval]

    return sims


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sisim(
    prediction_grid: np.ndarray,
    df: pd.DataFrame,
    xx: str,
    yy: str,
    zz: str,
    thresholds: Sequence[float],
    global_cdf: Sequence[float],
    vario: Union[Sequence, Dict[float, Sequence]],
    num_points: int,
    radius: float,
    kriging_type: str = "ordinary",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    ltail: str = "linear",
    ltpar: float = 1.0,
    middle: str = "linear",
    mpar: float = 1.0,
    utail: str = "linear",
    utpar: float = 1.0,
    nsim: int = 1,
    seed=None,
    quiet: bool = False,
    use_gpu: Union[bool, str] = False,
) -> np.ndarray:
    """
    Sequential Indicator Simulation, for a continuous variable.

    Parameters
    ----------
    prediction_grid : np.ndarray, shape (n, 2)
        x, y coordinates of the grid nodes to simulate.
    df : pandas.DataFrame
        Conditioning data.
    xx, yy, zz : str
        Column names in `df` for x, y coordinates and the variable.
    thresholds : sequence of float
        Cutoffs, strictly ascending. Indicator convention (matching
        `sisim.for`'s continuous-variable branch and this repo's
        `fit_indicator_variogram`): ``I(Z <= threshold)``.
    global_cdf : sequence of float
        Global ``P(Z <= threshold)`` for each threshold, ascending, in (0, 1).
    vario : sequence, or dict mapping threshold -> sequence
        Per-threshold indicator variogram(s):
        ``[azimuth, nugget, major_range, minor_range, sill, vtype]``
        (GSLIB/GStatSim practical-range covariance convention -- same as
        `sgsim.py`, and confirmed against `cova3.for`). A single sequence is
        reused for every threshold; a dict gives each threshold its own.
    num_points : int
        Number of neighbors to search for per node (split across 8 octants).
    radius : float
        Search radius for neighbors.
    kriging_type : {'ordinary', 'simple'}, default 'ordinary'
        Local indicator kriging system used at each node/threshold.
    zmin, zmax : float, optional
        Data limits for tail extrapolation. Default to the min/max of `df[zz]`.
    ltail : {'linear', 'power'}, default 'linear'
        Lower-tail interpolation model (below the lowest threshold).
    ltpar : float
        Power parameter for `ltail='power'`.
    middle : {'linear', 'power'}, default 'linear'
        Interpolation model between thresholds.
    mpar : float
        Power parameter for `middle='power'`.
    utail : {'linear', 'power', 'hyperbolic'}, default 'linear'
        Upper-tail interpolation model (above the highest threshold).
    utpar : float
        Power parameter for `utail='power'`, or the hyperbolic exponent for
        `utail='hyperbolic'`.
    nsim : int, default 1
        Number of independent realizations to generate.
    seed : int, numpy.random.Generator, or None
        Seeds an independent sub-generator per realization for
        reproducibility; None uses fresh entropy.
    quiet : bool, default False
        If False and `tqdm` is installed, shows a progress bar per realization.
    use_gpu : bool or 'auto', default False
        Dispatch covariance-matrix build + solve to the GPU via CuPy
        (verified usable, not just import-checked -- see
        `pygstat.core.kriging`). Defaults to **False**, not `'auto'`, for
        the same reason as `pygstat.sgsim.sgsim`: SIS is sequential
        *within* one realization -- each node depends on every
        previously-simulated node on its path -- so nodes can't be
        batched. What GPU dispatch does depends on `nsim`:

        - `nsim == 1`: one node (and, within it, one threshold) at a time
          (`_indicator_ccdf`). For typical small `num_points` the per-call
          GPU dispatch overhead usually outweighs the benefit -- pass
          `True`/`'auto'` only for unusually large `num_points`.
        - `nsim >= 2`: realizations are independent of each other, so all
          `nsim` local indicator-kriging systems at each step -- one batch
          per threshold -- are dispatched together (`_simulate_batch`)
          instead of one CuPy call per node per threshold per realization.
          See `_simulate_batch`'s docstring for what that batching costs
          numerically (a small ridge + `solve` in place of `lstsq`).

        `vtype='matern'` always evaluates on CPU regardless.

    Returns
    -------
    sims : np.ndarray, shape (nsim, len(prediction_grid))
        One row per realization, in the same order as `prediction_grid`.
    """
    use_gpu = resolve_cupy_use_gpu(use_gpu)
    thresholds = np.asarray(thresholds, dtype=float)
    global_cdf = np.asarray(global_cdf, dtype=float)
    if len(thresholds) < 2:
        raise ValueError("need at least 2 thresholds")
    if not np.all(np.diff(thresholds) > 0):
        raise ValueError("thresholds must be strictly ascending")
    if len(global_cdf) != len(thresholds):
        raise ValueError("global_cdf must have the same length as thresholds")
    if not np.all(np.diff(global_cdf) > 0):
        raise ValueError("global_cdf must be strictly ascending")
    if np.any(global_cdf <= 0) or np.any(global_cdf >= 1):
        raise ValueError("global_cdf values must be in (0, 1)")
    if nsim < 1:
        raise ValueError("nsim must be >= 1")

    if isinstance(vario, dict):
        vario_by_threshold = {float(t): v for t, v in vario.items()}
        missing = [t for t in thresholds if float(t) not in vario_by_threshold]
        if missing:
            raise ValueError(f"vario dict is missing entries for thresholds: {missing}")
        for v in vario_by_threshold.values():
            _validate_vario(v)
    else:
        _validate_vario(vario)
        vario_by_threshold = {float(t): vario for t in thresholds}

    work = df.rename(columns={xx: "X", yy: "Y", zz: "Z"})[["X", "Y", "Z"]].copy()
    if zmin is None:
        zmin = float(work["Z"].min())
    if zmax is None:
        zmax = float(work["Z"].max())

    rng = _get_random_generator(seed)
    child_seeds = rng.integers(0, 2**31 - 1, size=nsim)
    realization_rngs = [_get_random_generator(int(s)) for s in child_seeds]

    if use_gpu and nsim >= 2:
        # Realizations don't depend on each other -- only nodes *within*
        # one realization do -- so batch all nsim local kriging systems
        # (per threshold) at each shared step into one GPU dispatch. See
        # _simulate_batch.
        sims = _simulate_batch(
            prediction_grid, work, thresholds, global_cdf, vario_by_threshold,
            num_points, radius, kriging_type, zmin, zmax,
            ltail, ltpar, middle, mpar, utail, utpar,
            realization_rngs, quiet,
        )
    else:
        sims = np.empty((nsim, len(prediction_grid)))
        for i in range(nsim):
            sims[i] = _simulate_one(
                prediction_grid, work, thresholds, global_cdf, vario_by_threshold,
                num_points, radius, kriging_type, zmin, zmax,
                ltail, ltpar, middle, mpar, utail, utpar,
                realization_rngs[i], quiet, use_gpu,
            )

    return sims
