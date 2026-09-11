"""
Spatio-temporal kriging (converted from gstat R krigeST.R).
Supports ordinary/simple kriging, local neighbourhood, and trans-Gaussian.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, Callable
from scipy.spatial.distance import cdist

from . import st_variogram_models as stvm
from .utils.backend import as_gpu_array, get_array_module, resolve_cupy_use_gpu, to_numpy

# Re-export for callers that need variogram_line
variogram_line = stvm.variogram_line


def _st_base(model: Dict[str, Any]) -> str:
    return model["stModel"].split("_")[0]


def _temporal_scale(dt: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    """Scale temporal lags to model unit if temporal_unit is set (e.g. 'secs', 'days')."""
    t_unit = model.get("temporal_unit") or model.get("temporal unit")
    if t_unit is None:
        return dt
    xp = get_array_module(dt)
    dt = xp.asarray(dt, dtype=float)
    scale = {
        "secs": 1.0,
        "mins": 60.0,
        "hours": 3600.0,
        "days": 86400.0,
    }.get(str(t_unit).lower())
    if scale is None:
        return dt
    return dt / scale


def cov_fn_st(
    coords_x: np.ndarray,
    time_x: np.ndarray,
    coords_y: np.ndarray,
    time_y: np.ndarray,
    model: Dict[str, Any],
    separate: bool = False,
    grid_shape_x: Optional[Tuple[int, int]] = None,
    grid_shape_y: Optional[Tuple[int, int]] = None,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Spatio-temporal covariance between two point sets.
    coords_*: (n, 2) or (n, 3) spatial coordinates
    time_*: (n,) time values
    model: StVariogramModel dict (from vgm_st)
    separate: if True and model is separable and grid_shape_* given, return dict with Sm, Tm
    Returns covariance matrix (nx x ny) or dict with 'Sm', 'Tm' for separable grid case.
    """
    coords_x = np.asarray(coords_x)
    time_x = np.asarray(time_x).ravel()
    coords_y = np.asarray(coords_y)
    time_y = np.asarray(time_y).ravel()
    nx, ny = len(time_x), len(time_y)
    if coords_x.shape[0] != nx or coords_y.shape[0] != ny:
        raise ValueError("coords and time length mismatch")

    ds = cdist(coords_x, coords_y)
    dt = np.abs(np.subtract.outer(time_x, time_y))
    dt = _temporal_scale(dt, model)

    base = _st_base(model)
    if base == "separable":
        Sm_val = stvm.variogram_line(model["space"], ds, covariance=True) * model["sill"]
        Tm_val = stvm.variogram_line(model["time"], dt, covariance=True)
        C = Sm_val * Tm_val
        if separate and grid_shape_x is not None and grid_shape_y is not None:
            n_sx, n_tx = grid_shape_x
            n_sy, n_ty = grid_shape_y
            if nx != n_sx * n_tx or ny != n_sy * n_ty:
                return C
            # Unique spatial and time indices
            sp_x = coords_x[::n_tx] if n_tx > 0 else coords_x[:1]
            sp_y = coords_y[::n_ty] if n_ty > 0 else coords_y[:1]
            t_x = time_x[:n_tx]
            t_y = time_y[:n_ty]
            ds_ss = cdist(sp_x, sp_y)
            dt_tt = np.abs(np.subtract.outer(t_x, t_y))
            dt_tt = _temporal_scale(dt_tt, model)
            Sm = stvm.variogram_line(model["space"], ds_ss, covariance=True) * model["sill"]
            Tm = stvm.variogram_line(model["time"], dt_tt, covariance=True)
            return {"Sm": np.asarray(Sm), "Tm": np.asarray(Tm)}
        return C

    if base == "productSumOld":
        vs = stvm.variogram_line(model["space"], ds, covariance=True)
        vt = stvm.variogram_line(model["time"], dt, covariance=True)
        sill_s = stvm._sum_psill(model["space"])
        sill_t = stvm._sum_psill(model["time"])
        k = (sill_s + sill_t - (model["sill"] + model["nugget"])) / (sill_s * sill_t)
        return model["sill"] - (vt + vs - k * vt * vs)

    if base == "productSum":
        vs = stvm.variogram_line(model["space"], ds, covariance=True)
        vt = stvm.variogram_line(model["time"], dt, covariance=True)
        return vt + vs + model["k"] * vt * vs

    if base == "sumMetric":
        Sm = stvm.variogram_line(model["space"], ds, covariance=True)
        Tm = stvm.variogram_line(model["time"], dt, covariance=True)
        h = np.sqrt(ds ** 2 + (model["stAni"] * dt) ** 2)
        Mm = stvm.variogram_line(model["joint"], h, covariance=True)
        return Sm + Tm + Mm

    if base == "simpleSumMetric":
        surf = stvm.cov_surf_simple_sum_metric(
            model, ds.ravel(), dt.ravel()
        )
        return surf["gamma"].reshape(ds.shape)

    if base == "metric":
        h = np.sqrt(ds ** 2 + (model["stAni"] * dt) ** 2)
        return stvm.variogram_line(model["joint"], h, covariance=True)

    raise ValueError(f"Unsupported spatio-temporal model: {base}")


def ch_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b for x when A is symmetric positive definite (Cholesky)."""
    A = np.asarray(A)
    b = np.asarray(b)
    if b.ndim == 1:
        b = b[:, np.newaxis]
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x.ravel() if x.shape[1] == 1 else x


def st_solve(
    A: Dict[str, np.ndarray],
    b: Union[Dict[str, np.ndarray], np.ndarray],
    X: np.ndarray,
) -> np.ndarray:
    """
    Solve (Tm ⊗ Sm) x = [b_columns, X_columns] using Kronecker structure.
    A: dict with 'Sm', 'Tm' (spatial and temporal covariance Cholesky factors or matrices?)
    R: A$Tm %x% A$S, so V = Tm ⊗ Sm. They pass A with Tm, Sm as the covariance matrices,
    then chol(A$Tm), chol(A$Sm). So A here should be dict with 'Sm', 'Tm' the full cov matrices.
    Solves V w = [v0 | X] for w. R code: Tm=chol(A$Tm), Sm=chol(A$Sm), then backsolve.
    """
    Sm = A["Sm"]
    Tm = A["Tm"]
    n_s, n_t = Sm.shape[0], Tm.shape[0]
    n = n_s * n_t

    def my_ch_solve(L, z):
        y = np.linalg.solve(L, z)
        return np.linalg.solve(L.T, y)

    Ls = np.linalg.cholesky(Sm)
    Lt = np.linalg.cholesky(Tm)

    # b is dict with 'S' (n_s x ?) and 'T' (n_t x ?) for separable RHS, or b is (n, ncols)
    if isinstance(b, dict):
        # b$T (n_t x k1), b$S (n_s x k2) -> columns are vec(T column %x% S column)
        b_T = b["T"]
        b_S = b["S"]
        ncol_T, ncol_S = b_T.shape[1], b_S.shape[1]
        ret_list = []
        for jt in range(ncol_T):
            for js in range(ncol_S):
                rhs = np.outer(b_S[:, js], b_T[:, jt]).ravel(order="F")
                Y = np.reshape(rhs, (n_s, n_t), order="F")
                Y = my_ch_solve(Ls, Y)
                Y = my_ch_solve(Lt, Y.T).T
                ret_list.append(Y.ravel(order="F"))
        ret1 = np.column_stack(ret_list)
        # X full: columns as (n_s*n_t,) each
        ret2_list = []
        for j in range(X.shape[1]):
            Y = np.reshape(X[:, j], (n_s, n_t), order="F")
            Y = my_ch_solve(Ls, Y)
            Y = my_ch_solve(Lt, Y.T).T
            ret2_list.append(Y.ravel(order="F"))
        ret2 = np.column_stack(ret2_list)
        return np.hstack([ret1, ret2])
    else:
        b = np.asarray(b)
        if b.ndim == 1:
            b = b[:, np.newaxis]
        ncols = b.shape[1]
        out = np.empty((n, ncols))
        for j in range(ncols):
            Y = np.reshape(b[:, j], (n_s, n_t), order="F")
            Y = my_ch_solve(Ls, Y)
            Y = my_ch_solve(Lt, Y.T).T
            out[:, j] = Y.ravel(order="F")
        if X.size > 0:
            ncol_x = X.shape[1]
            out_x = np.empty((n, ncol_x))
            for j in range(ncol_x):
                Y = np.reshape(X[:, j], (n_s, n_t), order="F")
                Y = my_ch_solve(Ls, Y)
                Y = my_ch_solve(Lt, Y.T).T
                out_x[:, j] = Y.ravel(order="F")
            out = np.hstack([out, out_x])
        return out


def extract_formula(
    y: np.ndarray,
    X: Optional[np.ndarray],
    x0: Optional[np.ndarray],
    n_new: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build design matrices. If X/x0 None, use intercept (OK)."""
    n = len(y)
    if X is None:
        X = np.ones((n, 1))
    if x0 is None:
        x0 = np.ones((n_new, 1))
    X = np.asarray(X)
    x0 = np.asarray(x0)
    if X.shape[0] != n:
        raise ValueError("X rows must match y length")
    if x0.shape[0] != n_new:
        raise ValueError("x0 rows must match number of prediction points")
    return np.asarray(y, dtype=float), X, x0


def krige_st_df(
    coords: np.ndarray,
    time: np.ndarray,
    y: np.ndarray,
    coords_new: np.ndarray,
    time_new: np.ndarray,
    model_list: Union[Dict[str, Any], Callable],
    X: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    beta: Optional[np.ndarray] = None,
    nmax: float = np.inf,
    st_ani: Optional[float] = None,
    compute_var: bool = False,
    full_covariance: bool = False,
    buffer_nmax: float = 2.0,
    progress: bool = True,
    separate: Optional[bool] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    grid_shape_new: Optional[Tuple[int, int]] = None,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """
    Spatio-temporal kriging: compute predictions (and optionally variance) at new locations.
    Returns dict with 'var1.pred' and optionally 'var1.var'.
    """
    coords = np.asarray(coords)
    time = np.asarray(time).ravel()
    y = np.asarray(y).ravel()
    coords_new = np.asarray(coords_new)
    time_new = np.asarray(time_new).ravel()
    n, n_new = len(y), len(time_new)
    if coords.shape[0] != n or coords_new.shape[0] != n_new:
        raise ValueError("coords/time/y and coords_new/time_new length mismatch")

    is_separable_grid = (
        grid_shape is not None
        and grid_shape_new is not None
        and isinstance(model_list, dict)
        and _st_base(model_list) == "separable"
    )
    if separate is None:
        separate = is_separable_grid

    y, X, x0 = extract_formula(y, X, x0, n_new)

    if callable(model_list):
        V = model_list(coords, time, coords, time, **kwargs)
        v0 = model_list(coords, time, coords_new, time_new, **kwargs)
        if compute_var:
            c0 = np.asarray(
                model_list(
                    coords_new[:1], time_new[:1], coords_new[:1], time_new[:1], **kwargs
                )
            ).ravel()[0]
    else:
        V = cov_fn_st(
            coords, time, coords, time, model_list,
            separate=separate and is_separable_grid,
            grid_shape_x=grid_shape,
            grid_shape_y=grid_shape,
        )
        v0 = cov_fn_st(coords, time, coords_new, time_new, model_list)
        if compute_var:
            c0 = cov_fn_st(
                coords_new[:1], time_new[:1],
                coords_new[:1], time_new[:1],
                model_list, separate=False,
            )
            if isinstance(c0, dict):
                c0 = c0["Sm"][0, 0] * c0["Tm"][0, 0]
            else:
                c0 = float(np.asarray(c0).ravel()[0])

    v0_mat = np.asarray(v0) if isinstance(v0, np.ndarray) else v0
    if isinstance(V, dict):
        V = np.kron(V["Tm"], V["Sm"])

    if beta is not None:
        # Simple kriging
        skwts = ch_solve(V, v0_mat)
        if skwts.ndim == 1:
            skwts = skwts[:, np.newaxis]
        ViX = None
        if compute_var:
            var = c0 - np.sum(v0_mat * skwts, axis=0)
    else:
        # Ordinary/universal kriging
        M = np.hstack([v0_mat, X])
        skwts = ch_solve(V, M)
        npts = n_new
        ViX = skwts[:, npts:]
        skwts = skwts[:, :npts]
        # ch_solve ravels its result to 1D when X has a single column (the
        # default intercept-only/ordinary-kriging case) -- keep it 2D here so
        # the (p x p) matrix products below stay well-formed for any p >= 1.
        VinvX = ch_solve(V, X)
        if VinvX.ndim == 1:
            VinvX = VinvX[:, np.newaxis]
        beta = np.linalg.solve(X.T @ VinvX, (ViX.T @ y))
        if compute_var:
            Q = (x0.T - ViX.T @ v0_mat).T
            G = ch_solve(X.T @ VinvX, Q.T)
            var = (
                c0
                - np.sum(v0_mat * skwts, axis=0)
                + np.sum(Q * G.T, axis=1)
            )
            if full_covariance:
                cov_new = cov_fn_st(
                    coords_new, time_new, coords_new, time_new, model_list, separate=False
                )
                std = np.sqrt(np.maximum(var, 0))
                cor = cov_new / (np.outer(std, std) + 1e-30)
                np.fill_diagonal(cor, 1.0)
                var = cor * np.outer(std, std)

    # skwts has shape (n_train, n_new): one column of weights per prediction
    # point, one row per training point. (y - X @ beta) is (n_train,), so the
    # correct contraction is against skwts directly, not its transpose.
    pred = (x0 @ beta + (y - X @ beta) @ skwts).ravel()
    if np.isscalar(pred):
        pred = np.array([pred])

    if compute_var:
        var = np.maximum(np.asarray(var).ravel(), 0.0)
        if full_covariance:
            return {"var1.pred": pred, "var1.var": var}
        return {"var1.pred": pred, "var1.var": var}
    return {"var1.pred": pred}


def krige_st(
    coords: np.ndarray,
    time: np.ndarray,
    y: np.ndarray,
    coords_new: np.ndarray,
    time_new: np.ndarray,
    model_list: Union[Dict[str, Any], Callable],
    X: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    beta: Optional[np.ndarray] = None,
    nmax: float = np.inf,
    st_ani: Optional[float] = None,
    compute_var: bool = False,
    full_covariance: bool = False,
    buffer_nmax: float = 2.0,
    progress: bool = True,
    use_gpu: Union[bool, str] = False,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """
    Spatio-temporal kriging: main entry. If nmax < Inf delegates to local kriging.

    ``use_gpu`` only affects **local** kriging (``nmax`` finite): the k-NN
    systems are stacked into one ``(n_pred, k, k)`` solve. Global kriging
    still uses the original dense ``n_train × n_train`` path on CPU.
    """
    if not np.isinf(nmax):
        return krige_st_local(
            coords=coords,
            time=time,
            y=y,
            coords_new=coords_new,
            time_new=time_new,
            model_list=model_list,
            X=X,
            x0=x0,
            beta=beta,
            nmax=int(nmax),
            st_ani=st_ani,
            compute_var=compute_var,
            buffer_nmax=buffer_nmax,
            progress=progress,
            use_gpu=use_gpu,
            **kwargs,
        )
    return krige_st_df(
        coords=coords,
        time=time,
        y=y,
        coords_new=coords_new,
        time_new=time_new,
        model_list=model_list,
        X=X,
        x0=x0,
        beta=beta,
        nmax=nmax,
        st_ani=st_ani,
        compute_var=compute_var,
        full_covariance=full_covariance,
        buffer_nmax=buffer_nmax,
        progress=progress,
        **kwargs,
    )


def _cov_st_batch(coords_x, time_x, coords_y, time_y, model):
    """
    Batched space-time covariance.

    Parameters
    ----------
    coords_x : ndarray, shape (n_pred, n_x, d)
    time_x : ndarray, shape (n_pred, n_x)
    coords_y : ndarray, shape (n_pred, n_y, d)
    time_y : ndarray, shape (n_pred, n_y)

    Returns
    -------
    C : ndarray, shape (n_pred, n_x, n_y)
    """
    ds = get_array_module(coords_x).sqrt(
        ((coords_x[:, :, None, :] - coords_y[:, None, :, :]) ** 2).sum(-1)
    )
    xp = get_array_module(ds)
    dt = xp.abs(time_x[:, :, None] - time_y[:, None, :])
    dt = _temporal_scale(dt, model)
    base = _st_base(model)
    if base == "metric":
        h = xp.sqrt(ds ** 2 + (model["stAni"] * dt) ** 2)
        return stvm.variogram_line(model["joint"], h, covariance=True)
    if base == "separable":
        Sm = stvm.variogram_line(model["space"], ds, covariance=True) * model["sill"]
        Tm = stvm.variogram_line(model["time"], dt, covariance=True)
        return Sm * Tm
    if base == "productSum":
        vs = stvm.variogram_line(model["space"], ds, covariance=True)
        vt = stvm.variogram_line(model["time"], dt, covariance=True)
        return vt + vs + model["k"] * vt * vs
    if base == "productSumOld":
        vs = stvm.variogram_line(model["space"], ds, covariance=True)
        vt = stvm.variogram_line(model["time"], dt, covariance=True)
        sill_s = stvm._sum_psill(model["space"])
        sill_t = stvm._sum_psill(model["time"])
        k = (sill_s + sill_t - (model["sill"] + model["nugget"])) / (sill_s * sill_t)
        return model["sill"] - (vt + vs - k * vt * vs)
    if base == "sumMetric":
        Sm = stvm.variogram_line(model["space"], ds, covariance=True)
        Tm = stvm.variogram_line(model["time"], dt, covariance=True)
        h = xp.sqrt(ds ** 2 + (model["stAni"] * dt) ** 2)
        Mm = stvm.variogram_line(model["joint"], h, covariance=True)
        return Sm + Tm + Mm
    raise ValueError(f"Unsupported spatio-temporal model for batched local kriging: {base}")


def _c0_st(model) -> float:
    """C(0) at a dummy co-located space-time point."""
    z = np.zeros((1, 2))
    t = np.zeros(1)
    return float(np.asarray(cov_fn_st(z, t, z, t, model, separate=False)).ravel()[0])


def _solve_st_ok_batch(
    neighbor_coords,  # (n_pred, k, d)
    neighbor_times,   # (n_pred, k)
    neighbor_vals,    # (n_pred, k)
    target_coords,    # (n_pred, d)
    target_times,     # (n_pred,)
    model,
    use_gpu: bool,
    compute_var: bool,
):
    """
    Solve ``n_pred`` independent intercept-only ordinary ST-kriging systems
    in one batched linear-algebra call. Same GLS formulas as
    :func:`krige_st_df` (``beta is None``, ``X`` a column of ones).
    """
    n_pred, k, _ = neighbor_coords.shape
    if use_gpu:
        nc = as_gpu_array(neighbor_coords)
        nt = as_gpu_array(neighbor_times)
        nv = as_gpu_array(neighbor_vals)
        tc = as_gpu_array(target_coords)
        tt = as_gpu_array(target_times)
    else:
        nc, nt, nv = neighbor_coords, neighbor_times, neighbor_vals
        tc, tt = target_coords, target_times
    xp = get_array_module(nc)
    V = _cov_st_batch(nc, nt, nc, nt, model)
    v0 = _cov_st_batch(
        nc, nt, tc[:, None, :], tt[:, None], model,
    )[:, :, 0]  # (n_pred, k)

    rhs = xp.stack([v0, xp.ones((n_pred, k), dtype=v0.dtype)], axis=-1)  # (n_pred, k, 2)
    W = xp.linalg.solve(V, rhs)  # (n_pred, k, 2)
    skwts = W[:, :, 0]
    VinvX = W[:, :, 1]
    XtVX = VinvX.sum(axis=1)
    beta = xp.einsum("ij,ij->i", VinvX, nv) / XtVX
    resid = nv - beta[:, None]
    pred = to_numpy(beta + xp.einsum("ij,ij->i", resid, skwts))
    if not compute_var:
        return pred, None
    c0 = _c0_st(model)
    Q = 1.0 - xp.einsum("ij,ij->i", VinvX, v0)
    var = c0 - xp.einsum("ij,ij->i", v0, skwts) + (Q * Q) / XtVX
    return pred, to_numpy(xp.maximum(var, 0.0))


def _intercept_only(X, x0, n, n_new) -> bool:
    if X is None and x0 is None:
        return True
    if X is None:
        X = np.ones((n, 1))
    if x0 is None:
        x0 = np.ones((n_new, X.shape[1]))
    X = np.asarray(X)
    x0 = np.asarray(x0)
    return (
        X.ndim == 2 and X.shape[1] == 1
        and x0.ndim == 2 and x0.shape[1] == 1
        and np.allclose(X, 1.0) and np.allclose(x0, 1.0)
    )


def krige_st_local(
    coords: np.ndarray,
    time: np.ndarray,
    y: np.ndarray,
    coords_new: np.ndarray,
    time_new: np.ndarray,
    model_list: Dict[str, Any],
    X: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    beta: Optional[np.ndarray] = None,
    nmax: int = 50,
    st_ani: Optional[float] = None,
    compute_var: bool = False,
    full_covariance: bool = False,
    buffer_nmax: float = 2.0,
    progress: bool = True,
    use_gpu: Union[bool, str] = False,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Local spatio-temporal kriging using nearest neighbours in (space, time*st_ani).

    When every prediction point has the same neighbour count (k-NN) and the
    trend is intercept-only ordinary kriging, the systems are stacked into
    one ``(n_pred, k, k)`` solve — NumPy on CPU, CuPy on GPU (``use_gpu``).
    """
    if full_covariance:
        raise ValueError("full_covariance cannot be returned for local ST kriging")

    st_ani_val = st_ani
    if st_ani_val is None and model_list.get("stAni") is not None:
        st_ani_val = model_list["stAni"]
        t_unit = model_list.get("temporal_unit") or model_list.get("temporal unit")
        if t_unit:
            scale = {"secs": 1, "mins": 60, "hours": 3600, "days": 86400}.get(
                str(t_unit).lower(), 1
            )
            st_ani_val = st_ani_val / scale
    if st_ani_val is None:
        raise ValueError(
            "st_ani or model stAni required for local spatio-temporal kriging"
        )

    coords = np.asarray(coords)
    time = np.asarray(time).ravel()
    y = np.asarray(y).ravel()
    coords_new = np.asarray(coords_new)
    time_new = np.asarray(time_new).ravel()
    n, n_new = coords.shape[0], coords_new.shape[0]
    dim_geom = coords.shape[1]

    # Combined coordinates: (x, y [, z], time*st_ani)
    if dim_geom == 2:
        df = np.column_stack([coords[:, 0], coords[:, 1], time * st_ani_val])
        query = np.column_stack(
            [coords_new[:, 0], coords_new[:, 1], time_new * st_ani_val]
        )
    else:
        df = np.column_stack(
            [coords[:, 0], coords[:, 1], coords[:, 2], time * st_ani_val]
        )
        query = np.column_stack(
            [
                coords_new[:, 0],
                coords_new[:, 1],
                coords_new[:, 2],
                time_new * st_ani_val,
            ]
        )

    from scipy.spatial import cKDTree
    k_buffer = int(np.ceil(buffer_nmax * nmax))
    tree = cKDTree(df)
    nn = tree.query(query, k=min(k_buffer, n), workers=-1)
    if isinstance(nn, tuple):
        nb = nn[1]
    else:
        nb = nn

    if nb.ndim == 1:
        nb = np.atleast_2d(nb).T
    if X is None:
        X = np.ones((n, 1))
    if x0 is None:
        x0 = np.ones((n_new, X.shape[1]))

    use_gpu_flag = resolve_cupy_use_gpu(use_gpu)
    can_batch = (
        beta is None
        and isinstance(model_list, dict)
        and _intercept_only(X, x0, n, n_new)
        and not callable(model_list)
    )
    if can_batch:
        try:
            nb_idx = np.asarray(nb)
            k_got = nb_idx.shape[1]
            if buffer_nmax > 1 and k_got > nmax:
                nc_buf = coords[nb_idx]
                nt_buf = time[nb_idx]
                tc_buf = coords_new[:, None, :]
                tt_buf = time_new[:, None]
                if use_gpu_flag:
                    nc_buf = as_gpu_array(nc_buf)
                    nt_buf = as_gpu_array(nt_buf)
                    tc_buf = as_gpu_array(tc_buf)
                    tt_buf = as_gpu_array(tt_buf)
                cov_buf = to_numpy(_cov_st_batch(
                    nc_buf, nt_buf, tc_buf, tt_buf, model_list,
                ))[:, :, 0]
                top = np.argsort(-cov_buf, axis=1)[:, :nmax]
                nb_idx = np.take_along_axis(nb_idx, top, axis=1)
            pred_b, var_b = _solve_st_ok_batch(
                coords[nb_idx], time[nb_idx], y[nb_idx],
                coords_new, time_new, model_list,
                use_gpu=use_gpu_flag, compute_var=compute_var,
            )
            out = {"var1.pred": pred_b}
            if compute_var and var_b is not None:
                out["var1.var"] = var_b
            return out
        except Exception:
            pass

    pred = np.full(n_new, np.nan)
    var_out = np.full(n_new, np.nan) if compute_var else None

    iterator = range(n_new)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="krige_st_local")
        except ImportError:
            pass

    for i in iterator:
        ind = nb[i]
        if buffer_nmax > 1 and len(ind) > nmax:
            nghbr_coords = coords[ind]
            nghbr_time = time[ind]
            cov_to_i = cov_fn_st(
                nghbr_coords, nghbr_time,
                coords_new[i : i + 1], time_new[i : i + 1],
                model_list, separate=False,
            )
            cov_to_i = np.asarray(cov_to_i).ravel()
            top = np.argsort(cov_to_i)[::-1][:nmax]
            ind = ind[top]
        nghbr_coords = coords[ind]
        nghbr_time = time[ind]
        nghbr_y = y[ind]
        nghbr_X = X[ind]
        res = krige_st_df(
            nghbr_coords,
            nghbr_time,
            nghbr_y,
            coords_new[i : i + 1],
            time_new[i : i + 1],
            model_list,
            X=nghbr_X,
            x0=x0[i : i + 1],
            beta=beta,
            compute_var=compute_var,
            progress=False,
        )
        pred[i] = res["var1.pred"].flat[0]
        if compute_var and "var1.var" in res:
            var_out[i] = res["var1.var"].flat[0]

    out = {"var1.pred": pred}
    if compute_var and var_out is not None:
        out["var1.var"] = var_out
    return out


# ----- Trans-Gaussian (Box-Cox) -----
def phi_inv(x: np.ndarray, lambda_: float) -> np.ndarray:
    """Forward Box-Cox (gstat's phiInv): (x^λ − 1)/λ, or log(x) if λ=0."""
    x = np.asarray(x)
    if lambda_ == 0:
        return np.log(x)
    return (np.power(x, lambda_) - 1) / lambda_


def phi(x: np.ndarray, lambda_: float) -> np.ndarray:
    """Inverse Box-Cox (gstat's phi): (λx + 1)^{1/λ}, or exp(x) if λ=0."""
    x = np.asarray(x)
    if lambda_ == 0:
        return np.exp(x)
    return np.power(x * lambda_ + 1, 1 / lambda_)


def phi_prime(x: np.ndarray, lambda_: float) -> np.ndarray:
    """First derivative of the inverse Box-Cox (gstat's phi)."""
    x = np.asarray(x)
    if lambda_ == 0:
        return np.exp(x)
    return np.power(x * lambda_ + 1, 1 / lambda_ - 1)


def phi_double(x: np.ndarray, lambda_: float) -> np.ndarray:
    """Second derivative of the inverse Box-Cox (gstat's phi)."""
    x = np.asarray(x)
    if lambda_ == 0:
        return np.exp(x)
    return (
        lambda_
        * (1 / lambda_ - 1)
        * np.power(lambda_ * x + 1, 1 / lambda_ - 2)
    )


def krige_st_tg(
    coords: np.ndarray,
    time: np.ndarray,
    y: np.ndarray,
    coords_new: np.ndarray,
    time_new: np.ndarray,
    model_list: Union[Dict[str, Any], Callable],
    lambda_: float = 0,
    X: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    nmax: float = np.inf,
    st_ani: Optional[float] = None,
    buffer_nmax: float = 2.0,
    progress: bool = True,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """
    Trans-Gaussian spatio-temporal kriging (Box-Cox transform).
    Only formula with intercept (y ~ 1) is supported.
    """
    if not np.isinf(nmax):
        return krige_st_tg_local(
            coords=coords,
            time=time,
            y=y,
            coords_new=coords_new,
            time_new=time_new,
            model_list=model_list,
            lambda_=lambda_,
            nmax=int(nmax),
            st_ani=st_ani,
            buffer_nmax=buffer_nmax,
            progress=progress,
            **kwargs,
        )

    n = len(y)
    n_new = len(time_new)
    if X is not None and np.size(X) > n:
        if np.asarray(X).ndim == 2 and np.asarray(X).shape[1] > 1:
            raise ValueError("Only formula with intercept allowed, e.g. y ~ 1")
    X = np.ones((n, 1))
    x0 = np.ones((n_new, 1))

    value = phi_inv(y, lambda_)
    ok = krige_st(
        coords, time, value, coords_new, time_new, model_list,
        X=X, x0=x0, compute_var=True, progress=progress,
        buffer_nmax=buffer_nmax, st_ani=st_ani, **kwargs,
    )

    separate = (
        isinstance(model_list, dict)
        and _st_base(model_list) == "separable"
        and coords.shape[0] > 1
        and coords_new.shape[0] > 1
    )
    V = cov_fn_st(coords, time, coords, time, model_list, separate=separate)
    if isinstance(V, dict):
        V = np.kron(V["Tm"], V["Sm"])
    Vi = np.linalg.inv(V)
    muhat = np.sum(Vi @ value) / np.sum(Vi)
    v0 = cov_fn_st(coords, time, coords_new, time_new, model_list)
    m = (1 - np.sum(Vi @ v0, axis=0)) / np.sum(Vi)
    pred = ok["var1.pred"]
    var_ok = ok["var1.var"]
    ok["var1TG.pred"] = (
        phi(pred, lambda_)
        + phi_double(np.full_like(pred, muhat), lambda_) * (var_ok / 2 - m)
    )
    ok["var1TG.var"] = np.power(phi_prime(np.full_like(pred, muhat), lambda_), 2) * var_ok
    return ok


def krige_st_tg_local(
    coords: np.ndarray,
    time: np.ndarray,
    y: np.ndarray,
    coords_new: np.ndarray,
    time_new: np.ndarray,
    model_list: Dict[str, Any],
    lambda_: float = 0,
    nmax: int = 50,
    st_ani: Optional[float] = None,
    buffer_nmax: float = 2.0,
    progress: bool = True,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Local trans-Gaussian ST kriging."""
    n_new = len(time_new)
    pred = np.full(n_new, np.nan)
    var_ok = np.full(n_new, np.nan)
    tg_pred = np.full(n_new, np.nan)
    tg_var = np.full(n_new, np.nan)

    st_ani_val = st_ani or model_list.get("stAni")
    if st_ani_val is None:
        raise ValueError("st_ani or model stAni required")
    t_unit = model_list.get("temporal_unit") or model_list.get("temporal unit")
    if t_unit:
        scale = {"secs": 1, "mins": 60, "hours": 3600, "days": 86400}.get(
            str(t_unit).lower(), 1
        )
        st_ani_val = st_ani_val / scale

    dim_geom = coords.shape[1]
    if dim_geom == 2:
        df = np.column_stack([coords[:, 0], coords[:, 1], time * st_ani_val])
        query = np.column_stack(
            [coords_new[:, 0], coords_new[:, 1], time_new * st_ani_val]
        )
    else:
        df = np.column_stack(
            [coords[:, 0], coords[:, 1], coords[:, 2], time * st_ani_val]
        )
        query = np.column_stack(
            [
                coords_new[:, 0],
                coords_new[:, 1],
                coords_new[:, 2],
                time_new * st_ani_val,
            ]
        )

    from scipy.spatial import cKDTree
    k_buffer = int(np.ceil(buffer_nmax * nmax))
    tree = cKDTree(df)
    nn = tree.query(query, k=min(k_buffer, len(y)), workers=-1)[1]
    if nn.ndim == 1:
        nn = np.atleast_2d(nn).T

    iterator = range(n_new)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="krige_st_tg_local")
        except ImportError:
            pass

    for i in iterator:
        ind = nn[i]
        if buffer_nmax > 1 and len(ind) > nmax:
            cov_to_i = cov_fn_st(
                coords[ind], time[ind],
                coords_new[i : i + 1], time_new[i : i + 1],
                model_list, separate=False,
            )
            cov_to_i = np.asarray(cov_to_i).ravel()
            ind = ind[np.argsort(cov_to_i)[::-1][:nmax]]
        res = krige_st_tg(
            coords[ind], time[ind], y[ind],
            coords_new[i : i + 1], time_new[i : i + 1],
            model_list, lambda_=lambda_, nmax=np.inf, progress=False, **kwargs,
        )
        pred[i] = res["var1.pred"].flat[0]
        var_ok[i] = res["var1.var"].flat[0]
        tg_pred[i] = res["var1TG.pred"].flat[0]
        tg_var[i] = res["var1TG.var"].flat[0]

    return {
        "var1.pred": pred,
        "var1.var": var_ok,
        "var1TG.pred": tg_pred,
        "var1TG.var": tg_var,
    }


def vgm_area_st(
    coords_x: np.ndarray,
    time_x: np.ndarray,
    coords_y: np.ndarray,
    time_y: np.ndarray,
    model: Dict[str, Any],
    ndiscr_space: int = 16,
    verbose: bool = False,
    covariance: bool = True,
) -> np.ndarray:
    """
    Area-to-point spatio-temporal covariance (e.g. for block kriging).
    Simplified: treats all locations as points; for true area support
    you would discretize polygons. Here we compute point-to-point covariance.
    """
    return cov_fn_st(coords_x, time_x, coords_y, time_y, model, separate=False)
