"""
Spatio-temporal variogram models (converted from gstat R stVariogramModels.R).
Supports: separable, productSum, productSumOld, sumMetric, simpleSumMetric, metric.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from scipy.spatial.distance import cdist
from .core.variogram_models import MODEL_FUNCS
from .utils.backend import get_array_module, to_numpy

# R model name -> pygstat model name
_MODEL_ALIAS = {
    "Nug": "nugget",
    "Sph": "spherical",
    "Exp": "exponential",
    "Gau": "gaussian",
    "Mat": "matern",
    "Ste": "stable",
    "Cub": "cubic",
}


def _resolve_model(name: str) -> str:
    """Resolve R-style model name to pygstat MODEL_FUNCS key."""
    if name in _MODEL_ALIAS:
        return _MODEL_ALIAS[name]
    if name in MODEL_FUNCS:
        return name
    raise ValueError(f"Unknown variogram model: {name}")


def vgm(
    psill: float,
    model: str,
    range_: float,
    nugget: Optional[float] = None,
    kappa: float = 0.5,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Build a variogram model structure (list of components), compatible with R's vgm.
    Each component is a dict with keys: model, psill, range, kappa (for matern).
    """
    resolved = _resolve_model(model)
    comps: List[Dict[str, Any]] = []
    if nugget is not None and nugget != 0:
        comps.append({"model": "nugget", "psill": float(nugget), "range": 0.0, "kappa": 0.5})
    comps.append({
        "model": resolved,
        "psill": float(psill),
        "range": float(range_),
        "kappa": float(kappa),
        **kwargs,
    })
    return comps


def variogram_line(
    model: Union[List[Dict[str, Any]], Dict[str, Any]],
    dist_vector: np.ndarray,
    covariance: bool = False,
) -> np.ndarray:
    """
    Evaluate variogram (or covariance) for a variogram model at given distances.
    model: list of component dicts (from vgm) or single dict with model, psill, range, kappa.
    dist_vector: array of distances (or 2D array; output shape matches).
    Returns array of gamma (semivariance) or covariance.
    """
    xp = get_array_module(dist_vector)
    dist = xp.asarray(dist_vector, dtype=float)
    scalar = False
    try:
        scalar = int(dist.ndim) == 0
    except Exception:
        scalar = False
    if scalar:
        dist = dist.reshape((1,))

    if isinstance(model, dict):
        components = [model]
    else:
        components = list(model)

    out = xp.zeros_like(dist)
    c0 = 0.0  # C(0) = sum of all psills

    for comp in components:
        m = comp.get("model", "")
        if isinstance(m, str):
            m = str(m).strip()
        if isinstance(m, str) and m in _MODEL_ALIAS:
            m = _MODEL_ALIAS[m]
        psill = float(comp["psill"])
        range_val = float(comp.get("range", comp.get("range_", 0)))
        kappa = float(comp.get("kappa", 0.5))
        c0 += psill

        if m == "nugget" or range_val == 0:
            # Nugget: gamma(h)=0 at 0, psill elsewhere
            out = out + psill * (dist > 1e-10).astype(dist.dtype)
            continue

        py_name = _resolve_model(m) if isinstance(m, str) else m
        if py_name not in MODEL_FUNCS:
            raise ValueError(f"Model {py_name} not in MODEL_FUNCS")
        func = MODEL_FUNCS[py_name]
        # Component variogram: no nugget in this component, sill = psill
        if py_name == "matern":
            gam = func(dist, 0.0, psill, range_val, nu=kappa)
        elif py_name == "stable":
            gam = func(dist, 0.0, psill, range_val, alpha=kappa)
        else:
            gam = func(dist, 0.0, psill, range_val)
        out = out + gam

    if covariance:
        # C(h) = C(0) - gamma(h)
        out = c0 - out
        out = xp.where(dist < 1e-10, xp.asarray(c0, dtype=out.dtype), out)
    if scalar:
        return float(np.asarray(to_numpy(out)).reshape(-1)[0])
    return out


def vgm_st(
    st_model: str,
    space: Optional[Union[List[Dict], Dict]] = None,
    time: Optional[Union[List[Dict], Dict]] = None,
    joint: Optional[Union[List[Dict], Dict]] = None,
    sill: Optional[float] = None,
    k: Optional[float] = None,
    nugget: Optional[float] = None,
    st_ani: Optional[float] = None,
    temporal_unit: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Construct a spatio-temporal variogram model (StVariogramModel).
    st_model: one of "separable", "productSum", "productSumOld", "sumMetric",
              "simpleSumMetric", "metric".
    """
    if not isinstance(st_model, str) or len(st_model) == 0:
        raise ValueError("st_model must be a non-empty string")
    base = st_model.split("_")[0]

    if sill is not None and sill <= 0:
        raise ValueError('"sill" must be positive.')
    if k is not None and k <= 0:
        raise ValueError('"k" must be positive.')
    if nugget is not None and nugget < 0:
        raise ValueError('"nugget" must be non-negative.')
    if st_ani is not None and st_ani <= 0:
        raise ValueError('"stAni" must be positive.')

    if base == "productSum" and sill is not None:
        raise ValueError(
            "The sill argument for the product-sum model has been removed "
            "due to a change in notation. Re-fit your model or use 'productSumOld' instead."
        )

    if base == "separable":
        vgm_model = {"space": space, "time": time, "sill": sill}
    elif base == "productSum":
        vgm_model = {"space": space, "time": time, "k": k}
    elif base == "productSumOld":
        vgm_model = {"space": space, "time": time, "sill": sill, "nugget": nugget}
    elif base == "sumMetric":
        vgm_model = {"space": space, "time": time, "joint": joint, "stAni": st_ani}
    elif base == "simpleSumMetric":
        vgm_model = {
            "space": space,
            "time": time,
            "joint": joint,
            "nugget": nugget,
            "stAni": st_ani,
        }
    elif base == "metric":
        vgm_model = {"joint": joint, "stAni": st_ani}
    else:
        raise ValueError(f"model {base} unknown")

    vgm_model["stModel"] = st_model
    if temporal_unit is not None:
        vgm_model["temporal_unit"] = temporal_unit
    return vgm_model


def variogram_surface(
    model: Dict[str, Any],
    dist_grid: Union[Dict[str, np.ndarray], "np.ndarray"],
    covariance: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute spatio-temporal variogram (or covariance) surface.
    dist_grid: dict with keys 'spacelag' and 'timelag' (arrays of same shape),
               or array-like of shape (n, 2) with columns [spacelag, timelag].
    Returns dict with 'spacelag', 'timelag', 'gamma'.
    """
    if hasattr(dist_grid, "columns") and hasattr(dist_grid, "values"):
        # DataFrame-like
        spacelag = np.asarray(dist_grid["spacelag"])
        timelag = np.asarray(dist_grid["timelag"])
    elif isinstance(dist_grid, dict):
        spacelag = np.asarray(dist_grid["spacelag"])
        timelag = np.asarray(dist_grid["timelag"])
    else:
        arr = np.asarray(dist_grid)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            spacelag = arr[:, 0]
            timelag = arr[:, 1]
        else:
            raise ValueError("dist_grid must have 'spacelag' and 'timelag' or shape (n,2)")
    if spacelag.shape != timelag.shape:
        raise ValueError("spacelag and timelag must have the same shape")

    base = model["stModel"].split("_")[0]
    if covariance:
        dispatcher = {
            "separable": cov_surf_separable,
            "productSum": cov_surf_prod_sum,
            "productSumOld": cov_surf_prod_sum_old,
            "sumMetric": cov_surf_sum_metric,
            "simpleSumMetric": cov_surf_simple_sum_metric,
            "metric": cov_surf_metric,
        }
    else:
        dispatcher = {
            "separable": vgm_separable,
            "productSum": vgm_prod_sum,
            "productSumOld": vgm_prod_sum_old,
            "sumMetric": vgm_sum_metric,
            "simpleSumMetric": vgm_simple_sum_metric,
            "metric": vgm_metric,
        }
    if base not in dispatcher:
        raise ValueError(
            f"Only separable, productSum, sumMetric, simpleSumMetric, metric are implemented; got {base}"
        )
    return dispatcher[base](model, spacelag, timelag)


def _ensure_grid(spacelag: np.ndarray, timelag: np.ndarray) -> tuple:
    """Return (spacelag, timelag) as same-shaped arrays."""
    s = np.asarray(spacelag)
    t = np.asarray(timelag)
    if s.shape != t.shape:
        t = np.broadcast_to(t, s.shape)
    return s, t


# ----- separable: C_s * C_t -----
def vgm_separable(
    model: Dict[str, Any],
    spacelag: np.ndarray,
    timelag: np.ndarray,
) -> Dict[str, np.ndarray]:
    s, t = _ensure_grid(spacelag, timelag)
    vs = variogram_line(model["space"], s, covariance=False)
    vt = variogram_line(model["time"], t, covariance=False)
    gamma = model["sill"] * (vs + vt - vs * vt)
    return {"spacelag": s, "timelag": t, "gamma": gamma}


def cov_surf_separable(
    model: Dict[str, Any],
    spacelag: np.ndarray,
    timelag: np.ndarray,
) -> Dict[str, np.ndarray]:
    s, t = _ensure_grid(spacelag, timelag)
    Sm = variogram_line(model["space"], s, covariance=True) * model["sill"]
    Tm = variogram_line(model["time"], t, covariance=True)
    return {"spacelag": s, "timelag": t, "gamma": Tm * Sm}


# ----- productSum (old) -----
def vgm_prod_sum_old(
    model: Dict[str, Any],
    spacelag: np.ndarray,
    timelag: np.ndarray,
) -> Dict[str, np.ndarray]:
    s, t = _ensure_grid(spacelag, timelag)
    vs = variogram_line(model["space"], s, covariance=False)
    vt = variogram_line(model["time"], t, covariance=False)
    vn = np.full_like(vs, model["nugget"])
    vn[(vs == 0) & (vt == 0)] = 0
    sill_s = _sum_psill(model["space"])
    sill_t = _sum_psill(model["time"])
    k = (sill_s + sill_t - (model["sill"] + model["nugget"])) / (sill_s * sill_t)
    if k <= 0 or k > 1 / max(_last_psill(model["space"]), _last_psill(model["time"])):
        k = 1e6 * abs(k)
    gamma = vs + vt - k * vs * vt + vn
    return {"spacelag": s, "timelag": t, "gamma": np.asarray(gamma).reshape(s.shape)}


def cov_surf_prod_sum_old(
    model: Dict[str, Any],
    spacelag: np.ndarray,
    timelag: np.ndarray,
) -> Dict[str, np.ndarray]:
    s, t = _ensure_grid(spacelag, timelag)
    vs = variogram_line(model["space"], spacelag, covariance=True)
    vt = variogram_line(model["time"], timelag, covariance=True)
    sill_s = _sum_psill(model["space"])
    sill_t = _sum_psill(model["time"])
    k = (sill_s + sill_t - (model["sill"] + model["nugget"])) / (sill_s * sill_t)
    return {"spacelag": s, "timelag": t, "gamma": model["sill"] - (vt + vs - k * vt * vs)}


def _sum_psill(m: Union[List[Dict], Dict]) -> float:
    if isinstance(m, dict):
        return float(m.get("psill", 0))
    return sum(float(c.get("psill", 0)) for c in m)


def _last_psill(m: Union[List[Dict], Dict]) -> float:
    if isinstance(m, dict):
        return float(m.get("psill", 0))
    return float(m[-1].get("psill", 0))


# ----- productSum (new) -----
def vgm_prod_sum(
    model: Dict[str, Any],
    spacelag: np.ndarray,
    timelag: np.ndarray,
) -> Dict[str, np.ndarray]:
    if model.get("sill") is not None:
        return vgm_prod_sum_old(model, spacelag, timelag)
    s, t = _ensure_grid(spacelag, timelag)
    vs = variogram_line(model["space"], s, covariance=False)
    vt = variogram_line(model["time"], t, covariance=False)
    sill_s = _sum_psill(model["space"])
    sill_t = _sum_psill(model["time"])
    k = model["k"]
    gamma = (k * sill_t + 1) * vs + (k * sill_s + 1) * vt - k * vs * vt
    return {"spacelag": s, "timelag": t, "gamma": np.asarray(gamma).reshape(s.shape)}


def cov_surf_prod_sum(
    model: Dict[str, Any],
    spacelag: np.ndarray,
    timelag: np.ndarray,
) -> Dict[str, np.ndarray]:
    s, t = _ensure_grid(spacelag, timelag)
    vs = variogram_line(model["space"], s, covariance=True)
    vt = variogram_line(model["time"], t, covariance=True)
    return {"spacelag": s, "timelag": t, "gamma": vt + vs + model["k"] * vt * vs}


# ----- sumMetric -----
def vgm_sum_metric(
    model: Dict[str, Any],
    spacelag: np.ndarray,
    timelag: np.ndarray,
) -> Dict[str, np.ndarray]:
    s, t = _ensure_grid(spacelag, timelag)
    vs = variogram_line(model["space"], s, covariance=False)
    vt = variogram_line(model["time"], t, covariance=False)
    h = np.sqrt(s ** 2 + (model["stAni"] * np.asarray(t, dtype=float)) ** 2)
    vst = variogram_line(model["joint"], h, covariance=False)
    gamma = vs + vt + vst
    return {"spacelag": s, "timelag": t, "gamma": gamma}


def cov_surf_sum_metric(
    model: Dict[str, Any],
    spacelag: np.ndarray,
    timelag: np.ndarray,
) -> Dict[str, np.ndarray]:
    s, t = _ensure_grid(spacelag, timelag)
    Sm = variogram_line(model["space"], s, covariance=True)
    Tm = variogram_line(model["time"], t, covariance=True)
    h = np.sqrt(s ** 2 + (model["stAni"] * np.asarray(t, dtype=float)) ** 2)
    Mm = variogram_line(model["joint"], h, covariance=True)
    return {"spacelag": s, "timelag": t, "gamma": Sm + Tm + Mm}


# ----- simpleSumMetric -----
def vgm_simple_sum_metric(
    model: Dict[str, Any],
    spacelag: np.ndarray,
    timelag: np.ndarray,
) -> Dict[str, np.ndarray]:
    s, t = _ensure_grid(spacelag, timelag)
    vs = variogram_line(model["space"], s, covariance=False)
    vt = variogram_line(model["time"], t, covariance=False)
    h = np.sqrt(s ** 2 + (model["stAni"] * np.asarray(t, dtype=float)) ** 2)
    vm = variogram_line(model["joint"], h, covariance=False)
    vn = variogram_line(vgm(model["nugget"], "Nug", 0.0), h, covariance=False)
    gamma = vs + vt + vm + vn
    return {"spacelag": s, "timelag": t, "gamma": gamma}


def cov_surf_simple_sum_metric(
    model: Dict[str, Any],
    spacelag: np.ndarray,
    timelag: np.ndarray,
) -> Dict[str, np.ndarray]:
    # Convert to sumMetric and delegate
    joint_comps = _joint_non_nugget(model["joint"])
    model_new = vgm_st(
        "sumMetric",
        space=model["space"],
        time=model["time"],
        joint=vgm(
            joint_comps["psill"],
            joint_comps["model"],
            joint_comps["range"],
            model.get("nugget"),
            kappa=joint_comps.get("kappa", 0.5),
        ),
        st_ani=model["stAni"],
    )
    if model.get("temporal_unit") is not None:
        model_new["temporal_unit"] = model["temporal_unit"]
    return cov_surf_sum_metric(model_new, spacelag, timelag)


def _joint_non_nugget(joint: Union[List[Dict], Dict]) -> Dict:
    """Return the first non-nugget component of joint (for simpleSumMetric -> sumMetric)."""
    if isinstance(joint, dict):
        if (joint.get("model") or "").lower() == "nugget" or joint.get("range", 1) == 0:
            return {"model": "spherical", "psill": 0.0, "range": 1.0, "kappa": 0.5}
        return joint
    for c in joint:
        if (c.get("model") or "").lower() != "nugget" and c.get("range", 0) != 0:
            return c
    return joint[0] if joint else {"model": "spherical", "psill": 0.0, "range": 1.0, "kappa": 0.5}


# ----- metric -----
def vgm_metric(
    model: Dict[str, Any],
    spacelag: np.ndarray,
    timelag: np.ndarray,
) -> Dict[str, np.ndarray]:
    s, t = _ensure_grid(spacelag, timelag)
    h = np.sqrt(s ** 2 + (model["stAni"] * np.asarray(t, dtype=float)) ** 2)
    gamma = variogram_line(model["joint"], h, covariance=False)
    return {"spacelag": s, "timelag": t, "gamma": gamma}


def cov_surf_metric(
    model: Dict[str, Any],
    spacelag: np.ndarray,
    timelag: np.ndarray,
) -> Dict[str, np.ndarray]:
    s, t = _ensure_grid(spacelag, timelag)
    h = np.sqrt(s ** 2 + (model["stAni"] * np.asarray(t, dtype=float)) ** 2)
    gamma = variogram_line(model["joint"], h, covariance=True)
    return {"spacelag": s, "timelag": t, "gamma": gamma}


# ----- Fitting: insert/extract parameters -----
def _st_base(model: Dict[str, Any]) -> str:
    return model["stModel"].split("_")[0]


def extract_par(model: Dict[str, Any]) -> np.ndarray:
    """Extract free parameters from an StVariogramModel as a 1D array."""
    base = _st_base(model)
    space = model.get("space")
    time = model.get("time")
    joint = model.get("joint")

    if base == "separable":
        # range.s, nugget.s, range.t, nugget.t, sill
        return np.array([
            _range_second(space),
            _psill_first(space),
            _range_second(time),
            _psill_first(time),
            model["sill"],
        ])
    if base == "productSumOld":
        return np.array([
            _last_psill(space),
            _last_range(space),
            _last_psill(time),
            _last_range(time),
            model["sill"],
            model["nugget"],
        ])
    if base == "productSum":
        return np.array([
            _psill_second(space),
            _range_second(space),
            _psill_first(space),
            _psill_second(time),
            _range_second(time),
            _psill_first(time),
            model["k"],
        ])
    if base == "sumMetric":
        return np.array([
            _psill_second(space),
            _range_second(space),
            _psill_first(space),
            _psill_second(time),
            _range_second(time),
            _psill_first(time),
            _psill_second(joint),
            _range_second(joint),
            _psill_first(joint),
            model["stAni"],
        ])
    if base == "simpleSumMetric":
        return np.array([
            _last_psill(space),
            _last_range(space),
            _last_psill(time),
            _last_range(time),
            _last_psill(joint),
            _last_range(joint),
            model["nugget"],
            model["stAni"],
        ])
    if base == "metric":
        return np.array([
            _psill_second(joint),
            _range_second(joint),
            _psill_first(joint),
            model["stAni"],
        ])
    raise ValueError(f"extract_par not implemented for {base}")


def _psill_first(m: Any) -> float:
    if isinstance(m, dict):
        return float(m.get("psill", 0))
    return float(m[0].get("psill", 0))


def _psill_second(m: Any) -> float:
    if isinstance(m, dict):
        return float(m.get("psill", 0))
    if len(m) < 2:
        return 0.0
    return float(m[1].get("psill", 0))


def _range_second(m: Any) -> float:
    if isinstance(m, dict):
        return float(m.get("range", m.get("range_", 0)))
    if len(m) < 2:
        return 0.0
    return float(m[1].get("range", m[1].get("range_", 0)))


def _last_range(m: Any) -> float:
    if isinstance(m, dict):
        return float(m.get("range", m.get("range_", 0)))
    return float(m[-1].get("range", m[-1].get("range_", 0)))


def insert_par(par: np.ndarray, model: Dict[str, Any]) -> Dict[str, Any]:
    """Insert parameter vector into StVariogramModel template; return new model."""
    base = _st_base(model)
    par = np.asarray(par).ravel()
    if base == "separable":
        return insert_par_separable(par, model)
    if base == "productSumOld":
        return insert_par_prod_sum_old(par, model)
    if base == "productSum":
        return insert_par_prod_sum(par, model)
    if base == "sumMetric":
        return insert_par_sum_metric(par, model)
    if base == "simpleSumMetric":
        return insert_par_simple_sum_metric(par, model)
    if base == "metric":
        return insert_par_metric(par, model)
    raise ValueError(f"insert_par not implemented for {base}")


def _model_second(m: Any) -> str:
    """Model name of second component (structure) for template."""
    if isinstance(m, dict):
        return m.get("model", "spherical")
    if isinstance(m, list) and len(m) >= 2:
        return m[1].get("model", "spherical")
    return "spherical"


def _kappa_second(m: Any) -> float:
    if isinstance(m, dict):
        return float(m.get("kappa", 0.5))
    if isinstance(m, list) and len(m) >= 2:
        return float(m[1].get("kappa", 0.5))
    return 0.5


def insert_par_separable(par: np.ndarray, model: Dict[str, Any]) -> Dict[str, Any]:
    space = model["space"]
    time = model["time"]
    space_vgm = _vgm_from_components(space, 1 - par[1], par[0], par[1])
    time_vgm = _vgm_from_components(time, 1 - par[3], par[2], par[3])
    return vgm_st("separable", space=space_vgm, time=time_vgm, sill=par[4])


def insert_par_prod_sum_old(par: np.ndarray, model: Dict[str, Any]) -> Dict[str, Any]:
    space = model["space"]
    time = model["time"]
    space_vgm = _vgm_last_structure(space, par[0], par[1])
    time_vgm = _vgm_last_structure(time, par[2], par[3])
    return vgm_st(
        "productSumOld",
        space=space_vgm,
        time=time_vgm,
        sill=par[4],
        nugget=par[5],
    )


def insert_par_prod_sum(par: np.ndarray, model: Dict[str, Any]) -> Dict[str, Any]:
    space = model["space"]
    time = model["time"]
    space_vgm = _vgm_two_components(space, par[2], par[1], par[0])  # nugget.s, range.s, sill.s
    time_vgm = _vgm_two_components(time, par[5], par[4], par[3])
    return vgm_st("productSum", space=space_vgm, time=time_vgm, k=par[6])


def insert_par_sum_metric(par: np.ndarray, model: Dict[str, Any]) -> Dict[str, Any]:
    space = model["space"]
    time = model["time"]
    joint = model["joint"]
    space_vgm = _vgm_two_components(space, par[2], par[1], par[0])
    time_vgm = _vgm_two_components(time, par[5], par[4], par[3])
    joint_vgm = _vgm_two_components(joint, par[8], par[7], par[6])
    return vgm_st(
        "sumMetric",
        space=space_vgm,
        time=time_vgm,
        joint=joint_vgm,
        st_ani=par[9],
    )


def insert_par_simple_sum_metric(par: np.ndarray, model: Dict[str, Any]) -> Dict[str, Any]:
    space = model["space"]
    time = model["time"]
    joint = model["joint"]
    space_vgm = _vgm_last_structure(space, par[0], par[1])
    time_vgm = _vgm_last_structure(time, par[2], par[3])
    joint_vgm = _vgm_last_structure(joint, par[4], par[5])
    return vgm_st(
        "simpleSumMetric",
        space=space_vgm,
        time=time_vgm,
        joint=joint_vgm,
        nugget=par[6],
        st_ani=par[7],
    )


def insert_par_metric(par: np.ndarray, model: Dict[str, Any]) -> Dict[str, Any]:
    joint = model["joint"]
    joint_vgm = _vgm_two_components(joint, par[2], par[1], par[0])
    return vgm_st("metric", joint=joint_vgm, st_ani=par[3])


def _vgm_from_components(
    template: Any,
    psill: float,
    range_: float,
    nugget: float,
) -> List[Dict]:
    model_name = _model_second(template)
    kappa = _kappa_second(template)
    return vgm(psill, model_name, range_, nugget=nugget, kappa=kappa)


def _vgm_last_structure(template: Any, psill: float, range_: float) -> List[Dict]:
    model_name = "spherical"
    kappa = 0.5
    if isinstance(template, list):
        last = template[-1]
        model_name = last.get("model", "spherical")
        kappa = last.get("kappa", 0.5)
    return vgm(psill, model_name, range_, kappa=kappa)


def _vgm_two_components(
    template: Any,
    nugget: float,
    range_: float,
    sill: float,
) -> List[Dict]:
    model_name = "spherical"
    kappa = 0.5
    if isinstance(template, list) and len(template) >= 2:
        model_name = template[1].get("model", "spherical")
        kappa = template[1].get("kappa", 0.5)
    return vgm(sill, model_name, range_, nugget=nugget, kappa=kappa)


def fit_st_variogram(
    object_df: Dict[str, np.ndarray],
    model: Dict[str, Any],
    method: str = "L-BFGS-B",
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    fit_method: int = 6,
    st_ani: Optional[float] = None,
    wles: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Fit a spatio-temporal variogram model to empirical (binned) values.
    object_df: dict with keys at least 'dist' (spatial lag), 'timelag', 'gamma', 'np'.
    """
    from scipy.optimize import minimize

    dist = np.asarray(object_df["dist"]).ravel()
    timelag = np.asarray(object_df["timelag"]).ravel()
    gamma_emp = np.asarray(object_df["gamma"]).ravel()
    np_obs = object_df.get("np", np.ones_like(gamma_emp))
    valid = ~(np.isnan(gamma_emp) | np.isnan(dist) | np.isnan(timelag))
    dist = dist[valid]
    timelag = timelag[valid]
    gamma_emp = gamma_emp[valid]
    np_obs = np_obs[valid]

    if wles is not None:
        fit_method = 1 if wles else 6

    if fit_method == 0:
        surf = variogram_surface(
            model,
            {"spacelag": dist, "timelag": timelag},
            covariance=False,
        )
        mse = np.mean((gamma_emp - surf["gamma"].ravel()) ** 2)
        return {
            **model,
            "optim.output": "no fit",
            "MSE": mse,
        }

    base = _st_base(model)
    cur_st_ani = np.nan if st_ani is None else st_ani
    if (fit_method in (7, 11)) and model.get("stAni") is None and (st_ani is None or np.isnan(st_ani)):
        cur_st_ani = 1.0  # default

    def weighting(obj_dist, obj_timelag, obj_np, gamma_mod, cur_par_st_ani):
        if fit_method == 1:
            return obj_np
        if fit_method == 2:
            return obj_np / (gamma_mod ** 2 + 1e-20)
        if fit_method == 3:
            return obj_np
        if fit_method == 4:
            return obj_np / (gamma_mod ** 2 + 1e-20)
        if fit_method == 5:
            raise NotImplementedError("fit.method = 5 (REML) is not yet implemented")
        if fit_method == 6:
            return np.ones_like(obj_dist)
        if fit_method == 7:
            denom = obj_dist ** 2 + (cur_par_st_ani * obj_timelag) ** 2
            return obj_np / (denom + 1e-20)
        if fit_method == 8:
            d = np.where(obj_dist == 0, np.min(obj_dist[obj_dist > 0], initial=1e10), obj_dist)
            return obj_np / (d ** 2 + 1e-20)
        if fit_method == 9:
            d = np.where(obj_timelag == 0, np.min(obj_timelag[obj_timelag > 0], initial=1e10), obj_timelag)
            return obj_np / (d ** 2 + 1e-20)
        if fit_method == 10:
            return 1.0 / (gamma_mod ** 2 + 1e-20)
        if fit_method == 11:
            denom = obj_dist ** 2 + (cur_par_st_ani * obj_timelag) ** 2
            return 1.0 / (denom + 1e-20)
        if fit_method == 12:
            d = np.where(obj_dist == 0, np.min(obj_dist[obj_dist > 0], initial=1e10), obj_dist)
            return 1.0 / (d ** 2 + 1e-20)
        if fit_method == 13:
            d = np.where(obj_timelag == 0, np.min(obj_timelag[obj_timelag > 0], initial=1e10), obj_timelag)
            return 1.0 / (d ** 2 + 1e-20)
        raise ValueError(f"fit.method = {fit_method} is not implemented")

    def fit_fun(par, trace=False):
        cur_model = insert_par(par, model)
        surf = variogram_surface(
            cur_model,
            {"spacelag": dist, "timelag": timelag},
            covariance=False,
        )
        gamma_mod = surf["gamma"].ravel()
        res_sq = (gamma_emp - gamma_mod) ** 2
        cur_ani = cur_model.get("stAni")
        if cur_ani is None or (isinstance(cur_ani, float) and np.isnan(cur_ani)):
            cur_ani = cur_st_ani
        w = weighting(dist, timelag, np_obs, gamma_mod, cur_ani)
        res_sq = res_sq * w
        if trace:
            print({"par": par, "MSE": np.mean(res_sq)})
        return np.mean(res_sq)

    min_s = np.min(dist[dist > 0]) * 0.05
    min_t = np.min(timelag[timelag > 0]) * 0.05
    pos = np.sqrt(np.finfo(float).eps)
    if lower is None:
        lower = {
            "separable": [min_s, 0, min_t, 0, 0],
            "productSum": [0, min_s, 0, 0, min_t, 0, pos],
            "productSumOld": [0, min_s, 0, min_t, 0, 0],
            "sumMetric": [0, min_s, 0, 0, min_t, 0, 0, pos, 0, pos],
            "simpleSumMetric": [0, min_s, 0, min_t, 0, pos, 0, pos],
            "metric": [0, pos, 0, pos],
        }.get(base)
        if lower is None:
            raise ValueError(f"No lower bounds for {base}")
        lower = np.array(lower, dtype=float)
    x0 = extract_par(model)
    if len(lower) != len(x0):
        raise ValueError(
            f"lower bounds length {len(lower)} does not match {base} "
            f"parameter count {len(x0)}"
        )
    if upper is None:
        upper = np.full(len(x0), np.inf)

    result = minimize(fit_fun, x0, method=method, bounds=list(zip(lower, upper)))
    ret = insert_par(result.x, model)
    ret["optim.output"] = result
    surf_final = variogram_surface(
        ret,
        {"spacelag": dist, "timelag": timelag},
        covariance=False,
    )
    ret["MSE"] = float(np.mean((gamma_emp - surf_final["gamma"].ravel()) ** 2))
    return ret


def extract_par_names(model: Dict[str, Any]) -> List[str]:
    """Return parameter names for the model type (for display)."""
    base = _st_base(model)
    names = {
        "separable": ["range.s", "nugget.s", "range.t", "nugget.t", "sill"],
        "productSumOld": ["sill.s", "range.s", "sill.t", "range.t", "sill", "nugget"],
        "productSum": ["sill.s", "range.s", "nugget.s", "sill.t", "range.t", "nugget.t", "k"],
        "sumMetric": [
            "sill.s", "range.s", "nugget.s", "sill.t", "range.t", "nugget.t",
            "sill.st", "range.st", "nugget.st", "anis",
        ],
        "simpleSumMetric": [
            "sill.s", "range.s", "sill.t", "range.t", "sill.st", "range.st",
            "nugget", "anis",
        ],
        "metric": ["sill", "range", "nugget", "anis"],
    }
    return names.get(base, [])


# ---------------------------------------------------------------------------
# Empirical space-time semivariogram estimation (matrix form)
# ---------------------------------------------------------------------------
def empirical_st_variogram(coords, times, values, n_space_bins=12, max_space_lag=None):
    """
    Matheron-estimator empirical space-time semivariogram of a *complete*
    area x time panel (every area observed at every time stamp).

    Binned on a (spatial-lag bin) x (integer temporal lag) grid: temporal
    lags are used as-is (0, 1, 2, ... time steps) rather than further
    binned, since a regularly-spaced panel only has as many distinct lags
    as there are time stamps and each is already well replicated across
    area pairs and times.

    Parameters
    ----------
    coords : array-like, shape (n_areas, 2)
        One row per area/site, in the SAME order as the columns of the
        (n_areas, n_times) ``values`` matrix.

    times : array-like, shape (n_times,)
        Distinct, sorted time stamps. Must be evenly spaced since temporal
        lag is measured as an index difference converted to time units by
        the median step.

    values : array-like, shape (n_areas, n_times)
        Complete panel of values (no missing cells).

    n_space_bins : int, default=12
        Number of equal-width spatial-lag bins.

    max_space_lag : float, optional
        Upper edge of the spatial binning. Defaults to half the maximum
        pairwise centroid distance (a common rule of thumb: reliable
        semivariogram estimates need many pairs per bin, which thins out
        past half the field's extent).

    Returns
    -------
    dict
        ``{"dist": spacelag, "timelag": timelag, "gamma": gamma, "np": npairs}``,
        one flattened array per (space bin, temporal lag) cell -- directly
        usable as ``object_df`` in :func:`fit_st_variogram`.
    """
    coords = np.asarray(coords, dtype=float)
    times = np.asarray(times, dtype=float)
    Z = np.asarray(values, dtype=float)
    n_areas, n_times = Z.shape
    if coords.shape[0] != n_areas:
        raise ValueError("coords must have one row per area (Z.shape[0])")
    if len(times) != n_times:
        raise ValueError("times must have one entry per column of Z (Z.shape[1])")

    D = cdist(coords, coords)
    if max_space_lag is None:
        nonzero = D[D > 0]
        max_space_lag = 0.5 * float(nonzero.max()) if nonzero.size else 1.0
    edges = np.linspace(0.0, max_space_lag, n_space_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    space_bin = np.digitize(D, edges) - 1  # -1 .. n_space_bins (last bin catches h>max)

    time_step = float(np.median(np.diff(times))) if n_times > 1 else 1.0

    rows_dist, rows_time, rows_gamma, rows_np = [], [], [], []
    for lag in range(n_times):  # temporal lag in index steps: 0, 1, ..., n_times-1
        if lag == 0:
            # same-time, different-area pairs (i < j only, else double counted)
            sq = np.mean(Z ** 2, axis=1)
            cross = (Z @ Z.T) / n_times
            diff2 = np.clip(sq[:, None] + sq[None, :] - 2.0 * cross, 0.0, None)
            iu = np.triu_indices(n_areas, k=1)
            cell_bin = space_bin[iu]
            cell_gamma = 0.5 * diff2[iu]
            n_common = n_times
        else:
            A = Z[:, lag:]              # value at t = a + lag
            B = Z[:, : n_times - lag]   # value at t = a
            n_common = n_times - lag
            sqA = np.mean(A ** 2, axis=1)
            sqB = np.mean(B ** 2, axis=1)
            cross = (A @ B.T) / n_common
            diff2 = np.clip(sqA[:, None] + sqB[None, :] - 2.0 * cross, 0.0, None)
            # every (i, j) cell (including i == j, the pure-temporal lag at
            # zero spatial distance) is a distinct, valid area-time pair.
            cell_bin = space_bin.ravel()
            cell_gamma = 0.5 * diff2.ravel()

        for b in range(n_space_bins):
            m = cell_bin == b
            cnt = int(m.sum())
            if cnt == 0:
                continue
            rows_dist.append(bin_centers[b])
            rows_time.append(lag * time_step)
            rows_gamma.append(float(cell_gamma[m].mean()))
            rows_np.append(cnt * n_common)

    return {
        "dist": np.array(rows_dist),
        "timelag": np.array(rows_time),
        "gamma": np.array(rows_gamma),
        "np": np.array(rows_np, dtype=float),
    }


def fit_metric_st_variogram(emp, model="exponential", n_range_guess=None, st_ani_guess=None):
    """
    Fit a "metric" spatio-temporal variogram model (:func:`vgm_st`) to an
    empirical variogram from :func:`empirical_st_variogram`, using
    reasonable data-driven starting values.

    The "metric" model collapses space and time into one joint distance,
    ``h = sqrt(dist**2 + (stAni * timelag)**2)``, and fits a single
    variogram to ``h`` -- the simplest, most robust space-time model to fit
    from a modest panel, and a good default when there is no strong prior
    on separate spatial vs. temporal structure.
    """
    total_var = float(np.average(emp["gamma"], weights=emp["np"]))
    if n_range_guess is None:
        n_range_guess = float(np.percentile(emp["dist"][emp["dist"] > 0], 60))
    if st_ani_guess is None:
        # meters of "equivalent" space per unit of time, so that a handful
        # of time steps corresponds to roughly one practical range.
        max_lag = emp["timelag"].max()
        st_ani_guess = n_range_guess / max(max_lag, 1.0) * 3.0

    joint0 = vgm(psill=0.7 * total_var, model=model, range_=n_range_guess,
                 nugget=0.1 * total_var)
    model0 = vgm_st("metric", joint=joint0, st_ani=st_ani_guess)
    fitted = fit_st_variogram(emp, model0)
    return fitted


def joint_nugget_sill_range(joint):
    """
    Extract ``(nugget, sill, range)`` from a fitted ``metric`` (or
    ``simpleSumMetric``) model's ``joint`` component list, robust to a
    fitted nugget of exactly 0 -- :func:`vgm` omits the nugget component
    entirely when ``nugget == 0``, collapsing ``joint`` from 2 components
    down to 1, so code that always indexes ``joint[0]``/``joint[1]`` breaks
    on a perfect (zero-nugget) fit.
    """
    if isinstance(joint, dict):
        return 0.0, float(joint["psill"]), float(joint.get("range", joint.get("range_", 0.0)))
    if len(joint) == 1:
        return 0.0, float(joint[0]["psill"]), float(joint[0].get("range", joint[0].get("range_", 0.0)))
    return (
        float(joint[0]["psill"]),
        float(joint[1]["psill"]),
        float(joint[1].get("range", joint[1].get("range_", 0.0))),
    )
