"""
Backend utilities for CPU/GPU dispatch.
Auto-detects CuPy availability and provides safe array conversion.
"""

import numpy as np

# Try to import CuPy for GPU support
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


_cupy_usable_cache = None


def cupy_gpu_usable():
    """
    True only if CuPy is importable **and** can actually run a kernel on
    the attached GPU. Some older cards (e.g. Kepler/Maxwell, compute
    capability < 5.2-ish) import CuPy fine -- `CUPY_AVAILABLE` alone would
    say yes -- but fail at kernel-compile time (NVRTC) for a modern CUDA
    toolkit, which only shows up the first time an actual op runs. Checked
    once with a trivial real operation and cached, mirroring
    `resolve_torch_device`'s explicit compatibility probe for PyTorch.
    """
    global _cupy_usable_cache
    if _cupy_usable_cache is not None:
        return _cupy_usable_cache
    if not CUPY_AVAILABLE:
        _cupy_usable_cache = False
        return False
    try:
        (cp.array([1.0], dtype=cp.float32) + 1.0).get()
        _cupy_usable_cache = True
    except Exception:
        _cupy_usable_cache = False
    return _cupy_usable_cache


def resolve_cupy_use_gpu(use_gpu):
    """
    Resolve a `use_gpu` constructor argument (bool or `'auto'`) to a
    concrete bool for `Variogram`/`OrdinaryKriging`/`SimpleKriging`/
    `UniversalKriging`, verifying CuPy can actually run on the attached
    GPU (see `cupy_gpu_usable`) rather than just that it imports.
    `'auto'` silently prefers CPU when the GPU isn't usable, the whole
    point of auto-detection; an explicit `use_gpu=True` that can't be
    honored warns instead, since the caller asked for it specifically.
    """
    usable = cupy_gpu_usable()
    if use_gpu == "auto":
        return usable
    if bool(use_gpu) and not usable:
        import warnings
        reason = "CuPy is not installed" if not CUPY_AVAILABLE else "CuPy cannot run a kernel on this GPU/CUDA setup"
        warnings.warn(f"GPU requested (use_gpu=True) but {reason}; falling back to CPU.", stacklevel=3)
        return False
    return bool(use_gpu) and usable


def asarray(arr, use_gpu=False):
    """
    Convert input to array on CPU or GPU.

    Parameters
    ----------
    arr : array-like
    use_gpu : bool or 'auto'
        If 'auto', uses GPU only if available and usable.
    """
    use_gpu = resolve_cupy_use_gpu(use_gpu)
    if use_gpu:
        return cp.asarray(arr)
    return np.asarray(arr)


def to_numpy(arr):
    """Convert CuPy array to NumPy; no-op if already NumPy."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def get_array_module(*arrays):
    """
    Return the `cupy` module if any of `arrays` is a CuPy array, else
    `numpy` -- lets a single formula (variogram models, kriging matrix
    assembly) run unmodified on whichever backend its input already lives
    on, instead of duplicating the code per backend.
    """
    if CUPY_AVAILABLE:
        for a in arrays:
            if isinstance(a, cp.ndarray):
                return cp
    return np


def is_cudf_or_cupy(arr):
    """True if `arr` is a cuDF Series/DataFrame or a CuPy array -- i.e.
    already GPU-resident, so it should be moved with `cupy.asarray`
    (device-to-device) rather than forced through `numpy.asarray` (which
    would silently pull it back to host)."""
    try:
        import cudf
        if isinstance(arr, (cudf.DataFrame, cudf.Series)):
            return True
    except ImportError:
        pass
    return CUPY_AVAILABLE and hasattr(arr, "__cuda_array_interface__")


def as_gpu_array(arr):
    """Move `arr` (NumPy array, list, cuDF Series/DataFrame, or CuPy array)
    onto the GPU as a CuPy array. Requires CuPy; raises ImportError
    otherwise (callers only reach this once `use_gpu` has been resolved)."""
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU-accelerated kriging. Install it with: pip install pygstat[gpu]")
    try:
        import cudf
        if isinstance(arr, (cudf.DataFrame, cudf.Series)):
            return cp.asarray(arr.values)
    except ImportError:
        pass
    return cp.asarray(arr)


def get_linalg_module(xp):
    """`scipy.linalg` for `xp is numpy`, `cupyx.scipy.linalg` for `xp is
    cupy` -- both expose the same `lu_factor`/`lu_solve` API (cupyx's
    mirrors scipy's), used by `pygstat.cokriging`'s classes that cache a
    factorization at `fit()` time and reuse it across `predict()` calls."""
    if xp is np:
        import scipy.linalg
        return scipy.linalg
    import cupyx.scipy.linalg
    return cupyx.scipy.linalg


def resolve_torch_device(device="auto"):
    """
    Pick a usable torch device.

    ``device='auto'`` prefers CUDA only when the installed PyTorch build
    actually has kernels for the attached GPU (older cards such as
    ``sm_50`` otherwise crash or silently produce garbage).
    """
    import torch

    if device != "auto":
        return torch.device(device)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        supported = []
        if hasattr(torch.cuda, "get_arch_list"):
            supported = [
                s.replace("compute_", "sm_") for s in torch.cuda.get_arch_list()
            ]
        if supported and arch not in supported:
            return torch.device("cpu")
        torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


def seed_torch(seed):
    """Seed PyTorch for reproducible module fits."""
    if seed is None:
        return
    import torch

    torch.manual_seed(int(seed))
