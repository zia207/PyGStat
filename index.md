---
title: "PyGStat"
description: "GPU-accelerated geostatistics for Python — variograms, kriging, Poisson kriging, and sequential simulation on CPU or GPU."
---

<p align="center">
  <img src="Image/pygstat_logo_light.svg" alt="pygstat" width="180">
</p>

<p align="center">
  <a href="https://pypi.org/project/pygstat/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/pygstat.svg"></a>
  <a href="LICENSE.md"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="https://pypi.org/project/pygstat/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/pygstat"></a>
</p>

**GPU-accelerated geostatistics for Python**, inspired by R's [`gstat`](https://cran.r-project.org/package=gstat) and [GSLIB](http://www.gslib.com/).

Fit variograms, interpolate with kriging, smooth areal rates with Poisson kriging, run sequential simulation, and map uncertainty — on **CPU or GPU**.


## Features

**Variograms**

- Models: spherical, exponential, Gaussian, Matérn, stable, cubic
- Estimators: Matheron, Cressie–Hawkins, Dowd
- Variogram cloud diagnostics
- Geometric anisotropy
- Space–time (metric) variogram models

**Kriging**

- Ordinary and simple kriging
- Universal kriging (polynomial drift)
- Cokriging with a linear model of coregionalization
- Indicator kriging and E-type / probability maps
- Disjunctive kriging (Matheron, 1976): a nonlinear estimator built from a Gaussian anamorphosis (reusing the normal-score transform) + Hermite polynomial expansion, kriging each Hermite order independently (via `SimpleKriging`, including GPU dispatch) under the bi-Gaussian model; produces both a point estimate and exceedance-probability maps at arbitrary cutoffs from the same kriged components
- Empirical Bayesian Kriging (Krivoruchko, 2012): propagates semivariogram-*parameter* uncertainty into the prediction, by simulating and refitting an ensemble of semivariograms (weighted by leave-one-out cross-validation consistency) and combining an `OrdinaryKriging`/`SimpleKriging` model per ensemble member — total predictive variance and probability maps then reflect both spatial-interpolation error and semivariogram-fitting error, unlike plain kriging's variance which only accounts for the former
- Soft (Markov-Bayes) indicator kriging: fuse sparse hard data with a dense, correlated soft-probability proxy (Zhu & Journel, 1993)
- Factorial Kriging Analysis: extract individual (short-range, long-range, nugget) structures of a nested variogram, or a denoised/noise-filtered map (Matheron, 1982)
- Regression kriging with a choice of trend models: linear, ensembles (random forest, gradient boosting, bagging, stacking), statistical (GAM via pyGAM, GLM, Bayesian ridge, quantile regression), H2O (AutoML, Deep Learning, DRF, GLM, GAM, GBM, XGBoost, Uplift DRF, Stacked Ensembles, with optional hyperparameter-grid tuning), PyTorch or TensorFlow MLPs (both with optional Optuna hyperparameter search), or Graph Neural Networks (plain spatial GNN, KCN, IGNNK)
- Spatio-temporal ordinary kriging (`STKriging`)
- Spatio-temporal Graph Neural Network regression kriging: `TGCN` (Zhao et al., 2019) and `DCRNN` (Li et al., 2018), for a fixed sensor network observed repeatedly over time
- Spatial-Temporal Graph Transformer Network regression kriging: `STTN` (Xu et al., 2020) and `GMAN` (Zheng et al., 2020), the attention-based counterpart to `TGCN`/`DCRNN`
- Convolutional LSTM regression kriging (`ConvLSTM`, Shi et al., 2015): a raster-based alternative that rasterizes the sensor network per time step instead of using a graph

**Areal / count data**

- Centroid-based Poisson kriging (Goovaerts, 2006)
- Area-to-area and area-to-point Poisson kriging
- Multivariate Poisson cokriging
- Spatio-temporal Poisson kriging for rate panels

**Simulation & GSLIB helpers**

- Sequential Gaussian Simulation (`sgsim`): GSLIB-style SGS with `nsim` realizations, optional normal-score transform (`itrans`), octant search, and ordinary/simple kriging at each node
- Sequential Indicator Simulation (`sisim`): GSLIB-style SIS for a continuous variable — per-threshold indicator kriging, order-relations correction, and CDF interpolation/tail models (linear, power, hyperbolic)
- 3-D kriging (`kt3d`): SK/OK/kriging-with-a-trend, point or block support, external drift, search-ellipsoid neighborhoods, leave-one-out cross-validation — a direct Python port of GSLIB's `kt3d.for`
- Geo-EAS I/O, normal-score and Box–Cox transforms, cell declustering, grid coordinates

**Validation & backends**

- Leave-one-out and k-fold cross-validation (including spatial CV helpers)
- GPU acceleration via CuPy for `Variogram` (empirical semivariogram), `OrdinaryKriging`, `SimpleKriging`, and `UniversalKriging` (covariance-matrix build + linear solve) — `use_gpu='auto'|True|False`, with automatic, verified (not just import-checked) CPU fallback; accepts NumPy, CuPy, or cuDF input
- The same `use_gpu` convention (default `'auto'`) is wired through `IndicatorKriging` (k-NN mode batches every prediction into one stacked solve), `Cokriging` / `MultivariateCokriging` / `ColocatedCokriging` / `CrossVariogram`, centroid `PoissonKriging` and `STPoissonKriging`, local `STKriging` / `krige_st`, `FactorialKriging`, `SoftKriging`, `DisjunctiveKriging`, `EmpiricalBayesianKriging`, `kt3d`, and residual ordinary kriging inside the regression-kriging backends
- `sgsim`/`sisim` accept `use_gpu` too, but default to `False` (not `'auto'`): nodes *within* one realization are sequential, so they cannot be batched. When `nsim >= 2` and GPU is on, independent realizations are stepped in lockstep and their local kriging systems are batched into one GPU solve per step — that is where GPU dispatch actually helps. For `nsim == 1`, pass `use_gpu=True` only if you use unusually large `num_points`
- Interoperable with GeoPandas, pandas, NumPy, and scikit-learn
- Kriging standard error and uncertainty maps

Heavy optional backends (H2O, PyTorch, TensorFlow) are imported lazily, so `import pygstat` still works if they are not installed.

## Python Package

| | |
|---|---|
| **Name** | `PyGStat` |
| **Version** | 0.1.1 |
| **PyPI** | [pypi.org/project/pygstat](https://pypi.org/project/pygstat/) |
| **Documentation** | [zia207.github.io/PyGStat](https://zia207.github.io/PyGStat/) |
| **Repository** | [github.com/zia207/PyGStat](https://github.com/zia207/PyGStat) |
| **License** | MIT |
| **Python** | ≥ 3.8 |
| **Author** | Zia Ahmed, Upatta Analytic |

See [PyGStat Installation](install.md) for PyPI, GitHub, extras, and GPU setup.

## Project Layout

```
pygstat/
├── src/pygstat/        # Installed package
│   ├── core/           # Variogram models, empirical variograms, OK/SK
│   ├── io/             # GeoDataFrame helpers
│   └── utils/          # CPU/GPU backend and anisotropy
├── Tutorial/           # Workflow notebooks
├── tests/              # pytest suite and notebook tests
├── data/               # Sample datasets (Meuse, Jura, CA PM2.5, county rates, …)
├── examples/data/      # Extra spatial files used by some tutorials
├── scripts/            # Standalone utilities (plotting, Fortran readers, grid prediction)
├── Image/              # Figures used in docs and tutorials
└── pyproject.toml
```

Only `src/pygstat/` is built and published to PyPI. Tutorials, tests, and sample data stay in the repository.

## Quick Start

```python
import pandas as pd
import numpy as np
from pygstat import Variogram, OrdinaryKriging, loo_cv

df = pd.read_csv("data/meuse.csv")
coords = df[["x", "y"]].values
values = np.log(df["zinc"].values)

vg = Variogram(coords, values, model="spherical", estimator="matheron")
vg.fit()

ok = OrdinaryKriging(vg)
ok.fit(coords, values)

grid = pd.read_csv("data/meuse_grid.csv")
pred, se = ok.predict(grid[["x", "y"]].values, return_variance=True)

cv = loo_cv(ok, coords, values)
print(f"LOO RMSE: {cv['rmse']:.3f}, R²: {cv['r2']:.3f}")
```

From a GeoDataFrame:

```python
from pygstat import from_geodataframe, Variogram, OrdinaryKriging

coords, values = from_geodataframe(gdf, value_col="zinc")
vg = Variogram(coords, values, model="spherical")
vg.fit()
ok = OrdinaryKriging(vg).fit(coords, values)
```

Try the same Meuse workflow in the browser with no install: the [Kriging Workbench](kriging-workbench.html). The workbench is a five-tab dashboard (Variable, Exploratory, Variogram, Kriging, Probability) on 155 Meuse soil samples and a 3,103-cell prediction grid.

Continue with [Getting Started](Tutorial/00_getting_started.ipynb) for the full walkthrough.

## Tutorials {#tutorials}

Start with getting started, then pick a method. Each page is an exported notebook.

### Core interpolation

| Tutorial | Topic |
|----------|--------|
| [Getting started](Tutorial/00_getting_started.ipynb) | Variograms, ordinary/simple kriging, cross-validation |
| [Variogram modeling](Tutorial/01_variogram_modeling.ipynb) | Model choice, estimators, anisotropy |
| [Simple kriging](Tutorial/02_simple_kriging.ipynb) | Known-mean interpolation |
| [Ordinary kriging](Tutorial/03_ordinary_kriging.ipynb) | Unknown-mean interpolation |
| [Universal kriging](Tutorial/04_universal_kriging.ipynb) | Polynomial drift |
| [Cokriging](Tutorial/05_cokriging.ipynb) | Linear model of coregionalization |
| [3-D kriging (KT3D)](Tutorial/08_kriging_3d.ipynb) | GSLIB-style 3-D SK/OK/KT |

### Regression kriging

| Tutorial | Topic |
|----------|--------|
| [scikit-learn](Tutorial/06_01_regression_kriging_scikit_learn.ipynb) | Linear, ensembles, GAM, GLM, quantile |
| [H2O AutoML](Tutorial/06_02_regression_kriging_H20.ipynb) | AutoML, GBM, DRF, Deep Learning |
| [PyTorch MLP](Tutorial/06_03_regression_kriging_pytorch.ipynb) | Deep residual kriging (+ Optuna) |
| [TensorFlow / Keras](Tutorial/06_04_regression_krigingDNN_TF.ipynb) | Keras MLP residual kriging |
| [Graph neural nets](Tutorial/06_05_regression_kriging_gnn.ipynb) | GNN, KCN, IGNNK |

### Indicators, structure, and space–time

| Tutorial | Topic |
|----------|--------|
| [Indicator kriging](Tutorial/07_indicator_kriging.ipynb) | Probability and E-type maps |
| [Disjunctive kriging](Tutorial/10_disjunctive_kriging.ipynb) | Hermite-polynomial nonlinear estimator |
| [Empirical Bayesian kriging](Tutorial/11_empirical_bayesian_kriging.ipynb) | Variogram-parameter uncertainty |
| [Soft kriging](Tutorial/09_soft_kriging.ipynb) | Markov-Bayes hard + soft data |
| [Factorial kriging](Tutorial/12_factorial_kriging.ipynb) | Nested structures / denoising |
| [Spatio-temporal OK](Tutorial/13_krigingST.ipynb) | Metric space–time ordinary kriging |
| [ConvLSTM](Tutorial/14_kriging_CNNLSTM.ipynb) | Raster ST residual kriging |
| [STGNN (TGCN / DCRNN)](Tutorial/15_kriging_STGNN.ipynb) | Recurrent graph residual kriging |
| [STGTN (STTN / GMAN)](Tutorial/16_kriging_STGTN.ipynb) | Attention-based graph residual kriging |

### Poisson kriging

| Tutorial | Topic |
|----------|--------|
| [Area-to-area / area-to-point](Tutorial/17_01_poisson_Kriging_ata_atp.ipynb) | Rate smoothing on polygons |
| [Poisson cokriging](Tutorial/17_02_poisson_cokriging.ipynb) | Multivariate areal rates |
| [Spatio-temporal Poisson](Tutorial/17_03_poisson_kriging_ST.ipynb) | Rate panels over time |

### Simulation

| Tutorial | Topic |
|----------|--------|
| [SGSIM](Tutorial/18_sgsim.ipynb) | Sequential Gaussian Simulation |
| [SISIM](Tutorial/19_sisim.ipynb) | Sequential Indicator Simulation |

## Related Projects

- [gstat](https://cran.r-project.org/package=gstat) — the R package this project is inspired by
- [GSLIB](http://www.gslib.com/) — sequential simulation and Geo-EAS conventions
- [GSTools](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/) — another Python geostatistics toolbox
- [PyKrige](https://geostat-framework.readthedocs.io/projects/pykrige/) — kriging in Python

## License

MIT — see [LICENSE.md](LICENSE.md).

<p align="center">
  <br>
  Developed by<br>
  <img src="Image/upatta_logo.png" alt="Upatta Data Analytics logo featuring a head silhouette combining circuit patterns and spatial map imagery" width="200"/>
</p>
