"""
E-type (expected value) estimation from indicator kriging probabilities.
"""

import numpy as np


def compute_etype_from_probabilities(thresholds, prob_arrays, z_min=0.0, z_max=None):
    """
    Compute E-type estimate from conditional probabilities.

    For a non-negative variable, E[Z] = ∫ P(Z ≥ z) dz. This integrates
    P(Z ≥ zᵢ) between the supplied thresholds by trapezoid rule, then adds:

    * a lower tail from ``z_min`` to the first threshold, assuming P(Z ≥ z_min) = 1
    * an optional upper tail from the last threshold to ``z_max`` where P = 0
      (if ``z_max`` is None, the last interval width is reused)

    Parameters
    ----------
    thresholds : array-like
        Sorted cutoff values [z₀, z₁, ..., zₖ].
    prob_arrays : list of array-like
        P(Z ≥ zᵢ) for each threshold, same shape.
    z_min : float, default 0.0
        Lower support of Z used for the lower tail.
    z_max : float, optional
        Upper support of Z used for the upper tail.

    Returns
    -------
    etype : np.ndarray
        E-type estimate (expected value).
    """
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.size == 0:
        raise ValueError("thresholds must be non-empty")
    if not np.all(thresholds[:-1] <= thresholds[1:]):
        raise ValueError("Thresholds must be sorted in ascending order.")

    prob_arrays = [np.asarray(p, dtype=float) for p in prob_arrays]
    if len(prob_arrays) != len(thresholds):
        raise ValueError("prob_arrays must have one array per threshold")
    shape = prob_arrays[0].shape
    if not all(p.shape == shape for p in prob_arrays):
        raise ValueError("All probability arrays must have the same shape.")

    etype = np.zeros(shape)
    z0 = thresholds[0]
    p0 = prob_arrays[0]
    if z0 > z_min:
        etype += 0.5 * (1.0 + p0) * (z0 - z_min)

    for i in range(len(thresholds) - 1):
        dz = thresholds[i + 1] - thresholds[i]
        avg_prob = 0.5 * (prob_arrays[i] + prob_arrays[i + 1])
        etype += avg_prob * dz

    p_last = prob_arrays[-1]
    zk = thresholds[-1]
    if z_max is not None:
        if z_max < zk:
            raise ValueError("z_max must be >= the last threshold")
        etype += 0.5 * p_last * (z_max - zk)
    elif len(thresholds) > 1:
        etype += 0.5 * p_last * (thresholds[-1] - thresholds[-2])

    return etype
