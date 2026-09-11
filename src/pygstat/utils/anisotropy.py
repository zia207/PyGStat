"""
Anisotropy handling for geometric transformation of coordinates.
"""

import numpy as np


def transform_aniso(coords, angle=0, ratio=1.0):
    """
    Apply geometric anisotropy transformation.

    Coordinates are rotated by ``angle`` then the second axis is scaled by
    ``ratio``, so distances computed afterwards are isotropic in the
    transformed space.

    Parameters
    ----------
    coords : array-like, shape (n, 2)
        Input coordinates.
    angle : float, default=0
        Rotation of the major axis in degrees, measured counterclockwise
        from the +x axis.
    ratio : float, default=1.0
        Minor/major axis ratio. Must be > 0. Values ≤ 1 shrink the minor
        axis (the usual geometric-anisotropy convention).

    Returns
    -------
    transformed_coords : ndarray
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n, 2)")
    if ratio <= 0:
        raise ValueError("ratio must be > 0")

    theta = np.radians(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    S = np.diag([1.0, ratio])
    return coords @ R @ S
