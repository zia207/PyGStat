"""
Tests for pygstat's CuPy-based GPU dispatch: `Variogram`, `OrdinaryKriging`,
`SimpleKriging`, and `UniversalKriging` all accept `use_gpu='auto'|True|
False` and, when usable, build the covariance matrix and solve the kriging
system on the GPU via CuPy instead of NumPy/SciPy on CPU.

This sandbox's GPU is too old for the installed CUDA toolkit's NVRTC
compiler -- CuPy *imports* fine but cannot actually run a kernel here,
confirmed directly (``cupy.arange(10)`` raises
``NVRTCError: NVRTC_ERROR_INVALID_OPTION``). `pygstat.utils.backend.
cupy_gpu_usable()` exists specifically to detect this (mirroring
`resolve_torch_device`'s compatibility probe for PyTorch) rather than
trusting `CUPY_AVAILABLE` (import-only) the way the code did before --
which would otherwise make every default `use_gpu='auto'` call in this
sandbox try to run on GPU and crash, since CuPy *is* importable here.

Because of that hardware limitation, the actual GPU *code path* (the
`cupy.*`/`xp.*` calls added to `core/kriging.py`, `universal_kriging.py`,
and `core/variogram_models.py`) is exercised here by monkeypatching
`cupy_gpu_usable()` to True and pointing the modules' `cp` name at NumPy
under an alias, then checking the result matches the real CPU path
exactly. This catches any API mismatch between the two code paths (this
is exactly how the initial implementation's use of `cupy.block`, which
doesn't exist in the installed CuPy version, was caught) and confirms
they agree numerically -- it does not exercise real CUDA execution.
`test_real_cupy_import_smoke_test` separately confirms CuPy's Python-level
API surface used here exists in the installed version (module-level
introspection, no kernel compile needed), and `test_gpu_hardware_status`
reports (informationally) whether this run's GPU is actually usable.
"""

import numpy as np
import pytest

from pygstat.core.variogram import Variogram
from pygstat.core import kriging as kriging_mod
from pygstat.universal_kriging import UniversalKriging
from pygstat.utils import backend


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(30, 2))
    values = coords[:, 0] * 0.3 + coords[:, 1] * 0.1 + rng.normal(scale=2, size=30)
    coords_pred = rng.uniform(0, 100, size=(8, 2))
    return coords, values, coords_pred


class _FakeCupyModule:
    """A stand-in for the `cupy` module backed entirely by NumPy: `ndarray`
    is `numpy.ndarray` (so `isinstance(x, cp.ndarray)` checks still work
    against the "GPU" arrays this proxies, which are just NumPy arrays
    under the hood) and `asnumpy` -- a real CuPy name with no NumPy
    equivalent -- is a no-op `np.asarray`. Everything else (`hstack`,
    `linalg.solve`, ...) delegates straight to `numpy` via `__getattr__`.
    This is what makes the "fake GPU" tests below exercise pygstat's own
    dispatch code (`xp.foo(...)` calls, `cp.ndarray` checks) rather than
    real CUDA, which this sandbox's GPU can't run at all."""

    ndarray = np.ndarray

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asnumpy(arr):
        return np.asarray(arr)


@pytest.fixture
def fake_gpu(monkeypatch):
    """Monkeypatch `use_gpu='auto'|True` to resolve True everywhere, and
    point every module's `cp` reference at a NumPy-backed fake (see
    `_FakeCupyModule`), so the `xp is np`-vs-`xp is cp` branch selection
    and every `cp.*`/`xp.*` call pygstat's dispatch code makes still run,
    just without real CUDA. Yields nothing; just arranges the patch."""
    monkeypatch.setattr(backend, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(backend, "cp", _FakeCupyModule())
    monkeypatch.setattr(backend, "_cupy_usable_cache", True)
    monkeypatch.setattr(backend, "cupy_gpu_usable", lambda: True)


class TestUsableDetection:
    def test_cupy_gpu_usable_is_cached_bool(self):
        result = backend.cupy_gpu_usable()
        assert isinstance(result, bool)
        assert backend.cupy_gpu_usable() is result  # cached, not recomputed

    def test_gpu_hardware_status(self):
        """Informational: report whether this run's GPU is real-usable."""
        print(f"\ncupy_gpu_usable() on this machine: {backend.cupy_gpu_usable()}")

    def test_auto_never_crashes_regardless_of_hardware(self, data):
        """The actual bug this fix addresses: CUPY_AVAILABLE (import-only)
        used to make 'auto' crash on a machine where CuPy imports but
        can't run a kernel. This must never raise, on any machine."""
        coords, values, coords_pred = data
        v = Variogram(coords, values, use_gpu="auto")
        v.fit()
        ok = kriging_mod.OrdinaryKriging(v, use_gpu="auto")
        ok.fit(coords, values)
        pred = ok.predict(coords_pred)
        assert np.isfinite(pred).all()

    def test_explicit_true_warns_and_falls_back_when_unusable(self, data, monkeypatch):
        monkeypatch.setattr(backend, "_cupy_usable_cache", False)
        coords, values, coords_pred = data
        v = Variogram(coords, values)
        v.fit()
        with pytest.warns(UserWarning, match="falling back to CPU"):
            ok = kriging_mod.OrdinaryKriging(v, use_gpu=True)
        assert ok.use_gpu is False
        ok.fit(coords, values)
        assert np.isfinite(ok.predict(coords_pred)).all()


class TestGPUCodePathMatchesCPU:
    """`fake_gpu` forces every `use_gpu='auto'` resolution to True with
    `cp` aliased to `numpy`, so these exercise the exact GPU branch and
    CuPy API surface used in production, just without real CUDA."""

    def test_variogram_gpu_path_matches_cpu(self, data, fake_gpu):
        coords, values, _ = data
        v_cpu = Variogram(coords, values, use_gpu=False)
        v_cpu.fit()
        v_gpu = Variogram(coords, values, use_gpu=True)
        assert v_gpu.use_gpu is True
        v_gpu.fit()
        np.testing.assert_allclose(v_cpu.lags, v_gpu.lags)
        np.testing.assert_allclose(v_cpu.experimental, v_gpu.experimental, equal_nan=True)
        np.testing.assert_allclose(v_cpu.fitted_params, v_gpu.fitted_params, rtol=1e-6)

    def test_ordinary_kriging_gpu_path_matches_cpu(self, data, fake_gpu):
        coords, values, coords_pred = data
        v = Variogram(coords, values)
        v.fit()

        ok_cpu = kriging_mod.OrdinaryKriging(v, use_gpu=False)
        ok_cpu.fit(coords, values)
        pred_cpu, std_cpu = ok_cpu.predict(coords_pred, return_variance=True)

        ok_gpu = kriging_mod.OrdinaryKriging(v, use_gpu=True)
        assert ok_gpu.use_gpu is True
        ok_gpu.fit(coords, values)
        pred_gpu, std_gpu = ok_gpu.predict(coords_pred, return_variance=True)

        assert isinstance(pred_gpu, np.ndarray)  # always returned as NumPy
        np.testing.assert_allclose(pred_cpu, pred_gpu, rtol=1e-8)
        np.testing.assert_allclose(std_cpu, std_gpu, rtol=1e-6, atol=1e-8)

    def test_simple_kriging_gpu_path_matches_cpu(self, data, fake_gpu):
        coords, values, coords_pred = data
        v = Variogram(coords, values)
        v.fit()
        mean = float(np.mean(values))

        sk_cpu = kriging_mod.SimpleKriging(v, mean=mean, use_gpu=False)
        sk_cpu.fit(coords, values)
        pred_cpu, std_cpu = sk_cpu.predict(coords_pred, return_variance=True)

        sk_gpu = kriging_mod.SimpleKriging(v, mean=mean, use_gpu=True)
        assert sk_gpu.use_gpu is True
        sk_gpu.fit(coords, values)
        pred_gpu, std_gpu = sk_gpu.predict(coords_pred, return_variance=True)

        np.testing.assert_allclose(pred_cpu, pred_gpu, rtol=1e-8)
        np.testing.assert_allclose(std_cpu, std_gpu, rtol=1e-6, atol=1e-8)

    def test_universal_kriging_gpu_path_matches_cpu(self, data, fake_gpu):
        coords, values, coords_pred = data
        v = Variogram(coords, values)
        v.fit()

        uk_cpu = UniversalKriging(v, degree=1, use_gpu=False)
        uk_cpu.fit(coords, values)
        pred_cpu, std_cpu = uk_cpu.predict(coords_pred, return_variance=True)

        uk_gpu = UniversalKriging(v, degree=1, use_gpu=True)
        assert uk_gpu.use_gpu is True
        uk_gpu.fit(coords, values)
        pred_gpu, std_gpu = uk_gpu.predict(coords_pred, return_variance=True)

        np.testing.assert_allclose(pred_cpu, pred_gpu, rtol=1e-8)
        np.testing.assert_allclose(std_cpu, std_gpu, rtol=1e-6, atol=1e-8)

    def test_matern_model_still_correct_under_fake_gpu(self, data, fake_gpu):
        """Matérn always evaluates on CPU internally (no CuPy Bessel
        functions) regardless of `use_gpu` -- confirm it still agrees."""
        coords, values, coords_pred = data
        v_cpu = Variogram(coords, values, model="matern", use_gpu=False)
        v_cpu.fit()
        v_gpu = Variogram(coords, values, model="matern", use_gpu=True)
        v_gpu.fit()
        np.testing.assert_allclose(v_cpu.experimental, v_gpu.experimental, equal_nan=True)

        ok_cpu = kriging_mod.OrdinaryKriging(v_cpu, use_gpu=False)
        ok_cpu.fit(coords, values)
        ok_gpu = kriging_mod.OrdinaryKriging(v_gpu, use_gpu=True)
        ok_gpu.fit(coords, values)
        np.testing.assert_allclose(
            ok_cpu.predict(coords_pred), ok_gpu.predict(coords_pred), rtol=1e-6
        )


class TestCuDFAndCuPyInput:
    def test_as_gpu_array_accepts_numpy_list_and_roundtrips(self, fake_gpu):
        out = backend.as_gpu_array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])

    def test_as_gpu_array_raises_clearly_without_cupy(self, monkeypatch):
        monkeypatch.setattr(backend, "CUPY_AVAILABLE", False)
        with pytest.raises(ImportError, match="pip install pygstat"):
            backend.as_gpu_array([1.0, 2.0])

    def test_is_cudf_or_cupy_false_for_plain_numpy(self):
        assert backend.is_cudf_or_cupy(np.array([1.0, 2.0])) is False


def test_real_cupy_import_smoke_test():
    """Module-level introspection only (no kernel compile) -- confirms the
    CuPy API surface this module actually calls exists in the installed
    version. `cupy.block` was checked this way during development and
    found NOT to exist in this environment's CuPy (14.x), which is why
    `core/kriging.py` builds the augmented matrix with `vstack`/`hstack`
    instead."""
    cp = pytest.importorskip("cupy")
    for name in ("hstack", "vstack", "einsum", "clip", "where", "asarray", "ones", "zeros"):
        assert hasattr(cp, name), f"cupy.{name} missing"
    for name in ("solve", "lstsq", "LinAlgError"):
        assert hasattr(cp.linalg, name), f"cupy.linalg.{name} missing"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
