"""
Python equivalent of GSLIB nscore.exe.

Normal score transformation: transform data to standard normal (mean 0, var 1)
using empirical CDF, with optional tail extrapolation. Back-transform available
for simulation output.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .common import read_gslib, write_gslib


def nscore_forward(
    values: np.ndarray,
    tails: str = "linear",
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normal score transform (forward): data -> standard normal.

    Parameters
    ----------
    values : np.ndarray
        Raw data (1D); NaNs are skipped for building the transform.
    tails : str
        'linear' = linear extrapolation beyond min/max; 'none' = clip at min/max.
    low, high : float, optional
        Optional bounds for tail extrapolation (default: min and max of data).

    Returns
    -------
    nscore : np.ndarray
        Transformed values (same shape as values).
    values_sorted : np.ndarray
        Sorted unique data values used for the transform.
    nscore_sorted : np.ndarray
        Corresponding normal scores (for back-transform).
    """
    if tails not in ("linear", "none"):
        raise ValueError("tails must be 'linear' or 'none'")
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    v = values[valid]
    if v.size == 0:
        return np.full_like(values, np.nan), np.array([]), np.array([])
    # Mid-rank unique values so tied observations keep their frequency.
    v_sorted = np.sort(v)
    # Build a strictly increasing lookup table from unique values, using
    # average ranks for ties (GSLIB-style).
    uniq, start_idx = np.unique(v_sorted, return_index=True)
    n_all = len(v_sorted)
    from scipy import stats
    ranks = (np.arange(1, n_all + 1) - 0.5) / n_all
    nscore_all = stats.norm.ppf(ranks)
    # Average the normal scores of each tied group
    nscore_sorted = np.array([
        nscore_all[start_idx[i]: (start_idx[i + 1] if i + 1 < len(start_idx) else n_all)].mean()
        for i in range(len(uniq))
    ])
    v_sorted = uniq
    n = len(v_sorted)
    work = values.copy()
    if low is not None:
        work = np.where(valid & (work < low), low, work)
    if high is not None:
        work = np.where(valid & (work > high), high, work)
    out = np.interp(work, v_sorted, nscore_sorted)
    if tails == "linear":
        slope_low = (nscore_sorted[1] - nscore_sorted[0]) / (v_sorted[1] - v_sorted[0]) if n > 1 else 0.0
        slope_high = (nscore_sorted[-1] - nscore_sorted[-2]) / (v_sorted[-1] - v_sorted[-2]) if n > 1 else 0.0
        out = np.where(work < v_sorted[0], nscore_sorted[0] + slope_low * (work - v_sorted[0]), out)
        out = np.where(work > v_sorted[-1], nscore_sorted[-1] + slope_high * (work - v_sorted[-1]), out)
    else:
        out = np.clip(out, nscore_sorted[0], nscore_sorted[-1])
    out = np.where(valid, out, np.nan)
    return out, v_sorted, nscore_sorted


def nscore_back(
    nscore_values: np.ndarray,
    values_sorted: np.ndarray,
    nscore_sorted: np.ndarray,
    tails: str = "linear",
) -> np.ndarray:
    """
    Back-transform normal scores to original units (inverse of nscore_forward).

    Parameters
    ----------
    nscore_values : np.ndarray
        Values in normal score space.
    values_sorted, nscore_sorted : np.ndarray
        Lookup table from nscore_forward.
    tails : str
        'linear' = linear extrapolation beyond the training range, mirroring
        nscore_forward(tails='linear'); 'none' = clip at min/max (plain
        np.interp behaviour). Without this, predictions/simulations that fall
        outside the observed normal-score range (a routine occurrence after
        kriging or simulation) would silently clip instead of extrapolating,
        even though the forward transform explicitly supports extrapolation.

    Returns
    -------
    np.ndarray
        Back-transformed values.
    """
    values_sorted = np.asarray(values_sorted, dtype=float)
    nscore_sorted = np.asarray(nscore_sorted, dtype=float)
    nscore_values = np.asarray(nscore_values, dtype=float)
    if tails not in ("linear", "none"):
        raise ValueError("tails must be 'linear' or 'none'")
    if nscore_sorted.size > 1 and not np.all(np.diff(nscore_sorted) > 0):
        raise ValueError("nscore_sorted must be strictly increasing")
    out = np.interp(nscore_values, nscore_sorted, values_sorted)
    if tails == "linear" and len(nscore_sorted) > 1:
        slope_low = (values_sorted[1] - values_sorted[0]) / (nscore_sorted[1] - nscore_sorted[0])
        slope_high = (values_sorted[-1] - values_sorted[-2]) / (nscore_sorted[-1] - nscore_sorted[-2])
        out = np.where(
            nscore_values < nscore_sorted[0],
            values_sorted[0] + slope_low * (nscore_values - nscore_sorted[0]),
            out,
        )
        out = np.where(
            nscore_values > nscore_sorted[-1],
            values_sorted[-1] + slope_high * (nscore_values - nscore_sorted[-1]),
            out,
        )
    return out


def nscore_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    var_index: int = 0,
    tails: str = "linear",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read GSLIB file, apply normal score transform to one variable, write result.

    Parameters
    ----------
    input_path, output_path : path
        GSLIB Geo-EAS files.
    var_index : int
        Column index to transform (0-based).
    tails : str
        'linear' or 'none'.

    Returns
    -------
    values_sorted, nscore_sorted : np.ndarray
        Lookup table for back-transform.
    """
    title, var_names, data = read_gslib(input_path)
    v = data[:, var_index]
    out_data = data.copy()
    nscore, v_sorted, ns_sorted = nscore_forward(v, tails=tails)
    out_data[:, var_index] = nscore
    write_gslib(output_path, title, var_names, out_data)
    return v_sorted, ns_sorted
