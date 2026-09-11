"""
Common utilities for Gislib_Python (GSLIB-equivalent tools).

Geo-EAS / GSLIB file format:
  Line 1: title
  Line 2: number of variables
  Next N lines: variable names (one per line)
  Remaining lines: whitespace-separated data (one row per record)
"""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


def read_gslib(
    filepath: Union[str, Path],
) -> Tuple[str, List[str], np.ndarray]:
    """
    Read a GSLIB Geo-EAS file.

    Returns
    -------
    title : str
        First line of file.
    var_names : list of str
        Variable names (one per column).
    data : np.ndarray
        (n_records, n_vars) array.
    """
    path = Path(filepath)
    with open(path) as f:
        title = f.readline().strip()
        nvars = int(f.readline().strip())
        var_names = [f.readline().strip() for _ in range(nvars)]
    data = np.loadtxt(path, skiprows=2 + nvars)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[1] != len(var_names):
        raise ValueError(
            f"GSLIB file has {data.shape[1]} data columns but {len(var_names)} "
            f"variable names"
        )
    return title, var_names, data


def write_gslib(
    filepath: Union[str, Path],
    title: str,
    var_names: List[str],
    data: np.ndarray,
    float_fmt: str = "%.12g",
) -> None:
    """Write a GSLIB Geo-EAS file."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[1] != len(var_names):
        raise ValueError(
            f"data has {data.shape[1]} columns but {len(var_names)} variable names"
        )
    path = Path(filepath)
    with open(path, "w") as f:
        f.write(title + "\n")
        f.write(f"{len(var_names)}\n")
        for name in var_names:
            f.write(name + "\n")
        for row in np.atleast_2d(data):
            f.write(" ".join(float_fmt % x for x in row) + "\n")


def read_gslib_grid(
    filepath: Union[str, Path],
) -> Tuple[str, List[str], np.ndarray]:
    """Alias for read_gslib (same format for point or grid data)."""
    return read_gslib(filepath)
