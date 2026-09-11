"""
Python equivalent of GSLIB backtr.exe.

Back-transform: e.g. normal scores -> original units (uses lookup table
from nscore). For normal score simulation output.
"""

from typing import Union

import numpy as np

from . import nscore


def backtr(
    transformed_values: np.ndarray,
    values_sorted: np.ndarray,
    transformed_sorted: np.ndarray,
    tails: str = "linear",
) -> np.ndarray:
    """
    Back-transform values using a lookup table (e.g. from nscore_forward).

    Parameters
    ----------
    transformed_values : np.ndarray
        Values in transformed space (e.g. normal scores).
    values_sorted : np.ndarray
        Sorted original values (from nscore_forward).
    transformed_sorted : np.ndarray
        Corresponding transformed values (from nscore_forward).
    tails : str
        'linear' = linear extrapolation beyond the training range (default,
        matches nscore_forward(tails='linear')); 'none' = clip at min/max.

    Returns
    -------
    np.ndarray
        Values in original units.
    """
    return nscore.nscore_back(
        np.asarray(transformed_values, dtype=float),
        np.asarray(values_sorted),
        np.asarray(transformed_sorted),
        tails=tails,
    )
