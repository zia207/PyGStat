"""
Empirical and theoretical variogram modeling with optional CuDF/CuPy GPU acceleration.
"""

import numpy as np
from scipy.optimize import minimize

from .variogram_models import MODEL_FUNCS
from ..utils.backend import (
    CUPY_AVAILABLE,
    cupy_gpu_usable,
    resolve_cupy_use_gpu,
    as_gpu_array,
    get_array_module,
    to_numpy,
)
from ..utils.anisotropy import transform_aniso


def _matheron(diffs):
    """Classical Matheron estimator: ½ mean(Δ²)."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return np.nan
    return 0.5 * np.mean(diffs ** 2)


def _cressie(diffs):
    """
    Cressie–Hawkins robust estimator (Cressie & Hawkins, 1980):

    γ̂(h) = ½ [mean √|Δ| ]⁴ / (0.457 + 0.494/N)
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = diffs.size
    if n == 0:
        return np.nan
    term = np.mean(np.sqrt(np.abs(diffs)))
    return 0.5 * (term ** 4) / (0.457 + 0.494 / n)


def _dowd(diffs):
    """
    Dowd's robust estimator (Dowd, 1984):

    γ̂(h) = ½ · 2.198 · median(Δ²)
    """
    diffs = np.asarray(diffs, dtype=float)
    d2 = diffs[np.isfinite(diffs)] ** 2
    if d2.size == 0:
        return np.nan
    return 0.5 * 2.198 * np.median(d2)


ESTIMATORS = {
    "matheron": _matheron,
    "cressie": _cressie,
    "dowd": _dowd,
}


class Variogram:
    VALID_MODELS = ["spherical", "exponential", "gaussian", "matern", "stable", "cubic"]

    def __init__(
        self,
        coords,
        values,
        model="spherical",
        estimator="matheron",
        maxlag=None,
        n_lags=10,
        anisotropy=None,
        use_gpu="auto",
    ):
        if model not in self.VALID_MODELS:
            raise ValueError(f"model must be one of {self.VALID_MODELS}")
        if estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {list(ESTIMATORS)}")

        self.coords = np.asarray(coords, dtype=float)
        self.values = np.asarray(values, dtype=float)
        if self.coords.ndim != 2:
            raise ValueError("coords must be a 2D array of shape (n, dim)")
        if len(self.values) != len(self.coords):
            raise ValueError("coords and values must have the same length")
        if len(self.values) < 2:
            raise ValueError("at least 2 observations are required to compute a variogram")

        self.model = model
        self.estimator = estimator
        self.maxlag = maxlag
        self.n_lags = n_lags
        self.anisotropy = anisotropy

        if use_gpu == "auto":
            # Auto-GPU only when the data is *already* GPU-resident (cuDF/
            # CuPy) -- otherwise 'auto' would force a host->device copy
            # every user didn't ask for. Either way, verify CuPy can
            # actually run a kernel here, not just that it imports (see
            # `cupy_gpu_usable`).
            self.use_gpu = cupy_gpu_usable() and self._is_cudf_or_cupy(coords)
        else:
            self.use_gpu = resolve_cupy_use_gpu(use_gpu)

        if self.anisotropy:
            self.coords = transform_aniso(self.coords, **self.anisotropy)

        self.lags = None
        self.experimental = None
        self.fitted_params = None

    def _is_cudf_or_cupy(self, arr):
        """Check if input is CuDF Series/DataFrame or CuPy array."""
        try:
            import cudf
            if isinstance(arr, (cudf.DataFrame, cudf.Series)):
                return True
        except ImportError:
            pass
        if CUPY_AVAILABLE and hasattr(arr, "__cuda_array_interface__"):
            return True
        return False

    def _compute_empirical_gpu(self):
        """Compute empirical variogram using CuPy (GPU). Uses the shared
        `pygstat.utils.backend` dispatch helpers (rather than a fresh
        `import cupy as cp`) so this stays consistent with, and as easily
        mockable/testable as, the kriging classes' GPU code paths."""
        coords = as_gpu_array(self.coords)
        values = as_gpu_array(self.values)
        xp = get_array_module(coords)

        diff = coords[:, None, :] - coords[None, :, :]
        dists = xp.sqrt(xp.sum(diff ** 2, axis=2))
        val_diff = values[:, None] - values[None, :]

        positive = dists[dists > 0]
        if positive.size == 0:
            raise ValueError("All locations are coincident; cannot estimate a variogram")
        if self.maxlag is None:
            self.maxlag = float(xp.percentile(positive, 50))

        bins = xp.linspace(0, self.maxlag, self.n_lags + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        exp_vario = []
        estimator_func = ESTIMATORS[self.estimator]

        for i in range(self.n_lags):
            mask = (dists > bins[i]) & (dists <= bins[i + 1])
            if xp.any(mask):
                exp_vario.append(estimator_func(to_numpy(val_diff[mask])))
            else:
                exp_vario.append(np.nan)

        self.lags = to_numpy(bin_centers)
        self.experimental = np.array(exp_vario)

    def _compute_empirical_cpu(self):
        """Compute empirical variogram using NumPy (CPU)."""
        from scipy.spatial.distance import pdist, squareform

        dists = squareform(pdist(self.coords))
        diffs = self.values[:, None] - self.values[None, :]

        positive = dists[dists > 0]
        if positive.size == 0:
            raise ValueError("All locations are coincident; cannot estimate a variogram")
        if self.maxlag is None:
            self.maxlag = float(np.percentile(positive, 50))

        bins = np.linspace(0, self.maxlag, self.n_lags + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        exp_vario = []
        estimator_func = ESTIMATORS[self.estimator]

        for i in range(self.n_lags):
            mask = (dists > bins[i]) & (dists <= bins[i + 1])
            if np.any(mask):
                exp_vario.append(estimator_func(diffs[mask]))
            else:
                exp_vario.append(np.nan)

        self.lags = bin_centers
        self.experimental = np.array(exp_vario)

    def fit(self, method="auto"):
        """Fit variogram model."""
        if self.use_gpu:
            self._compute_empirical_gpu()
        else:
            self._compute_empirical_cpu()

        if method == "auto":
            self._fit_model()
        elif method == "manual":
            self.fitted_params = None
        else:
            raise ValueError("method must be 'auto' or 'manual'")
        return self

    def set_params(self, nugget, sill, range_, nu=None, alpha=None):
        """
        Set theoretical model parameters.

        Order is always ``nugget, sill, range_``, plus ``nu`` (Matérn) or
        ``alpha`` (stable). Keyword names are required for the extra params
        so they cannot be swapped by kwargs order.
        """
        params = [float(nugget), float(sill), float(range_)]
        if self.model == "matern":
            if nu is None:
                raise ValueError("matern requires nu")
            params.append(float(nu))
        elif self.model == "stable":
            if alpha is None:
                raise ValueError("stable requires alpha")
            params.append(float(alpha))
        elif nu is not None or alpha is not None:
            raise ValueError(f"{self.model} does not take nu or alpha")
        self.fitted_params = params
        return self

    def _fit_model(self):
        """Fit theoretical model (always on CPU)."""
        sill0 = float(np.nanvar(self.values))
        range0 = self.maxlag / 2
        nugget0 = 0.0

        if self.model == "matern":
            x0 = [nugget0, sill0, range0, 1.5]
            bounds = [(0, None), (0, None), (1e-6, self.maxlag * 3), (0.1, 10)]
        elif self.model == "stable":
            x0 = [nugget0, sill0, range0, 1.0]
            bounds = [(0, None), (0, None), (1e-6, self.maxlag * 3), (0.01, 2.0)]
        else:
            x0 = [nugget0, sill0, range0]
            bounds = [(0, None), (0, None), (1e-6, self.maxlag * 3)]

        def obj(params):
            pred = MODEL_FUNCS[self.model](self.lags, *params)
            obs = self.experimental
            mask = ~np.isnan(obs)
            if not np.any(mask):
                return np.inf
            return np.sum((pred[mask] - obs[mask]) ** 2)

        res = minimize(obj, x0, bounds=bounds, method="L-BFGS-B")
        self.fitted_params = res.x if res.success else x0
        self.fit_success_ = bool(res.success)
        return self

    def __call__(self, h):
        if self.fitted_params is None:
            raise RuntimeError("Variogram is not fitted. Call fit() or set_params() first.")
        return MODEL_FUNCS[self.model](h, *self.fitted_params)
