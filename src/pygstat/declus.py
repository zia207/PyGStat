"""
Python equivalent of GSLIB declus.exe.

Declustering: weight samples to correct for preferential clustering
(e.g. in irregular drilling). Weights used in histogram/variogram.
"""

from typing import Optional

import numpy as np


def declus_cell(
    x: np.ndarray,
    y: np.ndarray,
    nx: int = 10,
    ny: Optional[int] = None,
) -> np.ndarray:
    """
    Cell declustering: weight = 1 / (number of points in same cell).

    Parameters
    ----------
    x, y : np.ndarray
        Coordinates.
    nx, ny : int
        Number of cells in x and y (ny default = nx).

    Returns
    -------
    np.ndarray
        Weights (sum = n, for unbiased mean).
    """
    x, y = np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if ny is None:
        ny = nx
    ix = np.clip((x - x.min()) / (x.max() - x.min() + 1e-10) * nx, 0, nx - 1).astype(int)
    iy = np.clip((y - y.min()) / (y.max() - y.min() + 1e-10) * ny, 0, ny - 1).astype(int)
    cell_id = ix * ny + iy
    _, inv, counts = np.unique(cell_id, return_inverse=True, return_counts=True)
    n = len(x)
    weights = (1.0 / counts[inv]) * (n / np.sum(1.0 / counts[inv]))
    return weights
