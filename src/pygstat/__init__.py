"""
pygstat: GPU-accelerated geostatistics for Python.
"""

import importlib


def _optional_import(module_path, names, extra_name):
    """
    Import `names` from `module_path`. If the module (or one of its own heavy
    dependencies, e.g. h2o/torch/tensorflow) fails to import, bind each name
    to a stub that raises a clear, actionable ImportError only when actually
    *used* -- instead of breaking `import pygstat` entirely for everyone,
    including users who only need core Variogram/Kriging functionality.
    """
    try:
        mod = importlib.import_module(module_path, __name__)
        return {n: getattr(mod, n) for n in names}
    except Exception as e:
        # Broad on purpose: a broken optional backend can fail with all sorts
        # of exceptions, not just ImportError (e.g. protobuf VersionError from
        # a TensorFlow/protobuf version mismatch, OSError for a missing CUDA
        # shared library, etc.). None of those should break `import pygstat`.
        reason = f"{type(e).__name__}: {e}"

        def _make_stub(n):
            class _MissingOptionalDependency:
                def __init__(self, *a, **kw):
                    raise ImportError(
                        f"pygstat.{n} could not be loaded because an optional "
                        f"dependency is missing or broken in this environment "
                        f"({reason}). Install it with: pip install pygstat[{extra_name}]"
                    )
            _MissingOptionalDependency.__name__ = n
            return _MissingOptionalDependency

        return {n: _make_stub(n) for n in names}


# Core variogram and kriging (required deps only -- always available)
from .regression_kriging import (
    RegressionKriging,
    GAMRegressor,
    ENSEMBLE_REGRESSORS,
    STATISTICAL_REGRESSORS,
    AVAILABLE_REGRESSORS,
    enforce_quantile_monotonicity,
)
from .core.variogram import Variogram
from .core.kriging import OrdinaryKriging, SimpleKriging
from .core.variogram_cloud import VariogramCloud, compute_variogram_cloud, plot_variogram_cloud
from .cokriging import (
    Cokriging,
    CrossVariogram,
    MultivariateCokriging,
    ColocatedCokriging,
    fit_lmc,
    check_lmc_validity,
)
from .universal_kriging import UniversalKriging
from .indicator_variogram import fit_indicator_variogram
from .indicator_kriging import IndicatorKriging
from .disjunctive_kriging import (
    DisjunctiveKriging,
    hermite_polynomials,
    hermite_indicator_coefficients,
)
from .ebk import EmpiricalBayesianKriging
from .etype_estimates import compute_etype_from_probabilities
from .poisson_kriging import PoissonKriging, PointSupport, AreaPoissonKriging
from . import multi_poisson_kriging
from .krige_st import (
    krige_st,
    krige_st_df,
    krige_st_local,
    krige_st_tg,
    krige_st_tg_local,
    cov_fn_st,
    vgm_area_st,
)
from .st_variogram_models import (
    vgm,
    vgm_st,
    variogram_line,
    variogram_surface,
    fit_st_variogram,
    extract_par,
    insert_par,
    extract_par_names,
    empirical_st_variogram,
    fit_metric_st_variogram,
    joint_nugget_sill_range,
)
from .STPoisson_kriging import STPoissonKriging, load_atlantic_panel
from .krigeST import STKriging, load_ca_pm25_panel, make_prediction_grid

# Validation (updated to expose loo_cv and kfold_cv separately)
from .validation import loo_cv, kfold_cv
from .spatial_cv_rk import KFoldCV_RK, LOOCV_RK
from .spatial_cv_pytorch_rk import KFoldCV_PyTorch_RK
from .spatial_cv_cokriging import KFoldCV_Cokriging
from .spatial_cv_tf import KFoldCV_TF_RK
from .spatial_cv_IK import IndicatorKrigingCV

# Optional heavy backends: H2O AutoML, PyTorch, TensorFlow.
# Each is only imported by the module(s) that need it, so a missing/broken
# backend never breaks the rest of pygstat.
globals().update(_optional_import(
    ".regression_kriging_h2o", ["RegressionKrigingH2O", "H2O_MODEL_TYPES"], "h2o"
))
globals().update(_optional_import(
    ".regression_kriging_pytorch_gpu", ["DeepRegressionKriging"], "pytorch"
))
globals().update(_optional_import(
    ".regression_kriging_gnn", ["GNNRegressionKriging", "KCN", "IGNNK"], "pytorch"
))
globals().update(_optional_import(
    ".kriging_STGNN", ["STGNNRegressionKriging", "TGCN", "DCRNN"], "pytorch"
))
globals().update(_optional_import(
    ".kriging_STGTN", ["STGTNRegressionKriging", "STTN", "GMAN"], "pytorch"
))
globals().update(_optional_import(
    ".kriging_CNNLSTM", ["CNNLSTMRegressionKriging", "ConvLSTM"], "pytorch"
))
globals().update(_optional_import(
    ".regression_kriging_tf", ["DeepRegressionKrigingTF"], "tensorflow"
))
globals().update(_optional_import(
    ".spatial_cv_h2o_rk",
    ["KFoldCV_H2O_RK", "KFoldCV_H2O_ResidualRK", "H2OResidualKriging"],
    "h2o",
))
globals().update(_optional_import(
    ".spatial_cv_automl",
    ["KFoldCV_AutoML", "LOOCV_AutoML"],
    "h2o",
))

# I/O helpers
from .io.geopandas_io import from_geodataframe

# GSLIB-equivalent helper utilities (Geo-EAS I/O, normal-score transform,
# log/Box-Cox transform, cell declustering, grid coordinate generation).
from .common import read_gslib, write_gslib, read_gslib_grid
from .nscore import nscore_forward, nscore_back, nscore_file
# `backtr.py`'s public function is also named `backtr`; import it under a
# distinct name so it doesn't collide with the `pygstat.backtr` submodule.
from .backtr import backtr as backtr_transform
from .trans import transform_log, back_transform_log, transform_boxcox, back_transform_boxcox
from .declus import declus_cell
from .addcoord import add_coordinates_to_grid, addcoord
from .sgsim import sgsim
from .sisim import sisim
from .kt3d import kt3d, kt3d_grid, kt3d_cross_validate, build_grid_3d
from .soft_kriging import (
    SoftKriging,
    indicator_transform,
    markov_bayes_calibrate,
    fit_soft_probability_model,
)
from .factorial_kriging import FactorialKriging, fit_nested_variogram

# Version
__version__ = "0.1.1"

# Public API
__all__ = [
    # Variogram & Kriging
    "Variogram",
    "OrdinaryKriging",
    "SimpleKriging",
    "VariogramCloud",
    "compute_variogram_cloud",
    "plot_variogram_cloud",
    "RegressionKriging",
    "GAMRegressor",
    "ENSEMBLE_REGRESSORS",
    "STATISTICAL_REGRESSORS",
    "AVAILABLE_REGRESSORS",
    "enforce_quantile_monotonicity",
    "RegressionKrigingH2O",
    "H2O_MODEL_TYPES",
    "DeepRegressionKriging",
    "GNNRegressionKriging",
    "KCN",
    "IGNNK",
    "STGNNRegressionKriging",
    "TGCN",
    "DCRNN",
    "STGTNRegressionKriging",
    "STTN",
    "GMAN",
    "CNNLSTMRegressionKriging",
    "ConvLSTM",
    "DeepRegressionKrigingTF",
    "UniversalKriging",
    "Cokriging",
    "CrossVariogram",
    "MultivariateCokriging",
    "ColocatedCokriging",
    "fit_lmc",
    "check_lmc_validity",
    "fit_indicator_variogram",
    "IndicatorKriging",
    "DisjunctiveKriging",
    "hermite_polynomials",
    "hermite_indicator_coefficients",
    "EmpiricalBayesianKriging",
    "compute_etype_from_probabilities",
    "PoissonKriging",
    "PointSupport",
    "AreaPoissonKriging",
    "multi_poisson_kriging",
    # Spatio-temporal kriging
    "krige_st",
    "krige_st_df",
    "krige_st_local",
    "krige_st_tg",
    "krige_st_tg_local",
    "cov_fn_st",
    "vgm_area_st",
    # Spatio-temporal variogram models
    "vgm",
    "vgm_st",
    "variogram_line",
    "variogram_surface",
    "fit_st_variogram",
    "extract_par",
    "insert_par",
    "extract_par_names",
    "empirical_st_variogram",
    "fit_metric_st_variogram",
    "joint_nugget_sill_range",
    # Spatio-temporal Poisson kriging
    "STPoissonKriging",
    "load_atlantic_panel",
    # Spatio-temporal (Ordinary) kriging
    "STKriging",
    "load_ca_pm25_panel",
    "make_prediction_grid",
    # Validation
    "loo_cv",
    "kfold_cv",
    "KFoldCV_RK",
    "LOOCV_RK",
    "KFoldCV_H2O_RK",
    "KFoldCV_H2O_ResidualRK",
    "H2OResidualKriging",
    "KFoldCV_AutoML",
    "LOOCV_AutoML",
    "KFoldCV_PyTorch_RK",
    "KFoldCV_Cokriging",
    "KFoldCV_TF_RK",
    "IndicatorKrigingCV",
    # I/O
    "from_geodataframe",
    # GSLIB-equivalent helpers
    "read_gslib",
    "write_gslib",
    "read_gslib_grid",
    "nscore_forward",
    "nscore_back",
    "nscore_file",
    "backtr_transform",
    "transform_log",
    "back_transform_log",
    "transform_boxcox",
    "back_transform_boxcox",
    "declus_cell",
    "add_coordinates_to_grid",
    "addcoord",
    "sgsim",
    "sisim",
    "kt3d",
    "kt3d_grid",
    "kt3d_cross_validate",
    "build_grid_3d",
    "SoftKriging",
    "indicator_transform",
    "markov_bayes_calibrate",
    "fit_soft_probability_model",
    "FactorialKriging",
    "fit_nested_variogram",
]