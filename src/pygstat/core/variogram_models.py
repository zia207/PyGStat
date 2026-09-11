"""
Theoretical variogram models.
All functions take distance `h` and return semivariance γ(h).

Every model except `matern` is written against a generic array module `xp`
(`numpy` or `cupy`, picked via `pygstat.utils.backend.get_array_module`) so
the same formula runs unmodified whether `h` is a NumPy array (CPU) or a
CuPy array already resident on a GPU -- no separate GPU code path to keep
in sync. `matern` needs `scipy.special.kv`/`gamma` (no CuPy equivalent),
so it always evaluates on the CPU, transferring a CuPy input over and back
transparently -- correct either way, just not itself GPU-accelerated.
"""

import numpy as np
from scipy.special import kv, gamma

from ..utils.backend import get_array_module, to_numpy, CUPY_AVAILABLE


def spherical(h, nugget, sill, range_):
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    a = xp.clip(h / range_, 0, 1)
    return nugget + sill * xp.where(h < range_, 1.5 * a - 0.5 * a ** 3, 1.0)


def exponential(h, nugget, sill, range_):
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    return nugget + sill * (1.0 - xp.exp(-h / range_))


def gaussian(h, nugget, sill, range_):
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    return nugget + sill * (1.0 - xp.exp(-(h / range_) ** 2))


def matern(h, nugget, sill, range_, nu=1.5):
    xp = get_array_module(h)
    on_gpu = CUPY_AVAILABLE and xp is not np
    h_cpu = np.asarray(to_numpy(h), dtype=float)
    small = h_cpu < 1e-8
    with np.errstate(divide="ignore", invalid="ignore"):
        val = (2 ** (1 - nu) / gamma(nu)) * (h_cpu / range_) ** nu * kv(nu, h_cpu / range_)
        result = nugget + sill * (1.0 - val)
    # np.where works for scalar (0-d) h; boolean indexing does not.
    result = np.where(small, nugget, result)
    return xp.asarray(result) if on_gpu else result


def stable(h, nugget, sill, range_, alpha=1.0):
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    alpha = np.clip(alpha, 0.01, 2.0)
    return nugget + sill * (1.0 - xp.exp(-(h / range_) ** alpha))


def cubic(h, nugget, sill, range_):
    """
    Cubic variogram (Chilès & Delfiner / gstat).

    γ(h) = nugget + sill * (7u² − 8.75u³ + 3.5u⁵ − 0.75u⁷) for u = h/a < 1,
    and nugget + sill otherwise.  γ(0) = nugget.
    """
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    u = h / range_
    gamma_u = 7.0 * u ** 2 - 8.75 * u ** 3 + 3.5 * u ** 5 - 0.75 * u ** 7
    return nugget + sill * xp.where(h < range_, gamma_u, 1.0)


MODEL_FUNCS = {
    "spherical": spherical,
    "exponential": exponential,
    "gaussian": gaussian,
    "matern": matern,
    "stable": stable,
    "cubic": cubic,
}

# Models that always run on CPU regardless of `use_gpu` (no CuPy-side
# special-function support) -- surfaced so callers can warn instead of
# silently getting a CPU roundtrip inside an otherwise GPU-resident pipeline.
CPU_ONLY_MODELS = frozenset({"matern"})


def covariance_from_variogram(h, nugget, sill, range_, model="spherical", **kwargs):
    """
    Covariance C(h) from variogram: C(h) = C(0) - γ(h), with C(0) = nugget + sill.

    The models define γ(0) = nugget (nugget discontinuity), so C(0) must be
    restored explicitly rather than computed as C0 - γ(0).
    """
    xp = get_array_module(h)
    h = xp.asarray(h, dtype=float)
    gamma_h = MODEL_FUNCS[model](h, nugget, sill, range_, **kwargs)
    c0 = nugget + sill
    return xp.where(h == 0, c0, c0 - gamma_h)
