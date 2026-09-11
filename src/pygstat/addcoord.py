"""
Python equivalent of GSLIB addcoord.exe.

Adds X,Y (and optionally Z) coordinates to a GSLIB grid file using
grid parameters: nx, ny, origin, cell size. Grid order is Fortran-style
(x varies fastest) by default.
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np

from .common import read_gslib, write_gslib

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False


def add_coordinates_to_grid(
    nx: int,
    ny: int,
    xorig: float = 0.0,
    yorig: float = 0.0,
    xsize: float = 1.0,
    ysize: float = 1.0,
    order: str = "F",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X and Y coordinate arrays for a 2D grid (GSLIB addcoord style).

    Parameters
    ----------
    nx, ny : int
        Number of cells in x and y.
    xorig, yorig : float
        Origin (e.g. lower-left corner).
    xsize, ysize : float
        Cell size in x and y.
    order : 'F' or 'C'
        'F' = Fortran/column-major (x varies fastest), 'C' = row-major.

    Returns
    -------
    x, y : np.ndarray
        Each shape (nx*ny,) in grid order.
    """
    if order == "F":
        ix = np.tile(np.arange(nx), ny)
        iy = np.repeat(np.arange(ny), nx)
    else:
        iy = np.tile(np.arange(ny), nx)
        ix = np.repeat(np.arange(nx), ny)
    x = xorig + (ix + 0.5) * xsize
    y = yorig + (iy + 0.5) * ysize
    return x, y


def addcoord(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    nx: int,
    ny: int,
    xorig: float = 0.0,
    yorig: float = 0.0,
    xsize: float = 1.0,
    ysize: float = 1.0,
    order: str = "F",
    coord_names: Tuple[str, str] = ("X", "Y"),
):
    """
    Add X,Y coordinates to a GSLIB grid file (Python equivalent of addcoord.exe).

    Parameters
    ----------
    input_path : path
        Input GSLIB grid file (title, nvars, var names, then data rows).
    output_path : path
        Output file with coordinates added.
    nx, ny : int
        Grid dimensions.
    xorig, yorig : float
        Grid origin (e.g. lower-left).
    xsize, ysize : float
        Cell sizes.
    order : 'F' or 'C'
        Grid order (default 'F' for GSLIB).
    coord_names : (str, str)
        Names for the new coordinate columns.

    Returns
    -------
    pd.DataFrame or np.ndarray
        Result table (with coordinates) if pandas available, else (n, ncols) array.
    """
    title, var_names, data = read_gslib(input_path)
    n_cells = data.shape[0]
    if n_cells != nx * ny:
        raise ValueError(
            f"Grid size mismatch: file has {n_cells} rows, nx*ny = {nx*ny}"
        )
    x, y = add_coordinates_to_grid(
        nx, ny, xorig, yorig, xsize, ysize, order=order
    )
    out_names = [coord_names[0], coord_names[1]] + var_names
    out_data = np.column_stack([x, y, data])
    write_gslib(output_path, title, out_names, out_data)
    if _HAS_PANDAS:
        return pd.DataFrame(out_data, columns=out_names)
    return out_data
