"""
Python equivalent of GSLIB trans.exe.

Data transformation: log, Box-Cox, or user-defined. Often used before
kriging/simulation. Back-transform available for both log and Box-Cox.
"""

from typing import Optional

import numpy as np


def transform_log(values: np.ndarray, min_val: Optional[float] = None) -> np.ndarray:
    """Log transform; optional floor to avoid log(0)."""
    v = np.asarray(values, dtype=float)
    if min_val is not None:
        v = np.maximum(v, min_val)
    if np.any(v <= 0):
        raise ValueError(
            "log transform requires strictly positive values; "
            "pass min_val > 0 to floor the input"
        )
    return np.log(v)


def back_transform_log(transformed: np.ndarray) -> np.ndarray:
    """Inverse of log: exp."""
    return np.exp(np.asarray(transformed, dtype=float))


def _boxcox_offset(min_val: Optional[float]) -> float:
    """
    Shift applied before Box-Cox so the input is positive.

    ``min_val is None`` keeps the historical default offset of 1.
    ``min_val=0.0`` is a real bound (not treated as missing).
    """
    if min_val is None:
        return 1.0
    return 1.0 - min_val


def transform_boxcox(values: np.ndarray, lmbda: float, min_val: Optional[float] = None) -> np.ndarray:
    """Box-Cox: (x^lambda - 1) / lambda for lambda != 0, else log(x)."""
    from scipy.special import boxcox
    v = np.asarray(values, dtype=float)
    if min_val is not None:
        v = np.maximum(v, min_val)
    return boxcox(v + _boxcox_offset(min_val), lmbda)


def back_transform_boxcox(transformed: np.ndarray, lmbda: float, min_val: Optional[float] = None) -> np.ndarray:
    """
    Inverse of transform_boxcox. `lmbda` and `min_val` must match the values
    used in the forward call. Note this only undoes the Box-Cox transform and
    the additive shift -- it cannot undo the `min_val` floor clip itself
    (values that were clipped up to `min_val` in the forward pass are not
    recoverable, same limitation as back_transform_log).
    """
    from scipy.special import inv_boxcox
    t = np.asarray(transformed, dtype=float)
    return inv_boxcox(t, lmbda) - _boxcox_offset(min_val)
