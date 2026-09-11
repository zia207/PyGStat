"""Empirical indicator variograms for sequential indicator simulation and IK."""

import numpy as np
from scipy.spatial.distance import pdist, squareform

def fit_indicator_variogram(coords, values, threshold, maxlag=None, n_lags=10):
    """
    Fit empirical indicator variogram and return lags, gamma.

    Indicator convention: I(Z <= threshold), i.e. 1 if value <= threshold else 0.
    For P(Z >= t) use IndicatorKriging with the same threshold; it uses I(Z >= t) internally.
    """
    coords = np.asarray(coords)
    values = np.asarray(values)

    if coords.ndim != 2:
        raise ValueError("coords must be a 2D array with shape (n_points, dim)")
    if values.ndim != 1 or values.shape[0] != coords.shape[0]:
        raise ValueError("values must be a 1D array with the same length as coords")

    # Create indicator variable: 1 if value <= threshold, else 0
    indicators = (values <= threshold).astype(float)

    # Pairwise distances between coordinates
    dists = pdist(coords)
    if dists.size == 0:
        # No pairs
        lags = np.array([])
        experimental_gamma = np.array([])
        return lags, experimental_gamma

    # Default maxlag is half the maximum pairwise distance
    if maxlag is None:
        maxlag = dists.max() / 2.0

    # Define lag bins and their centers
    bin_edges = np.linspace(0.0, maxlag, n_lags + 1)
    lags = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Pairwise squared differences of indicators (pdist on 1D array)
    ind_diffs_sq = pdist(indicators[:, None], metric='euclidean') ** 2

    # Compute experimental semivariance for each lag bin: 0.5 * mean squared difference
    experimental_gamma = np.full(n_lags, np.nan)
    for i in range(n_lags):
        mask = (dists >= bin_edges[i]) & (dists < bin_edges[i + 1])
        if np.any(mask):
            experimental_gamma[i] = 0.5 * np.mean(ind_diffs_sq[mask])

    return lags, experimental_gamma