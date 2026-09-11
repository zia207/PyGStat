---
title: "PyGStat Installation"
description: "Install pygstat from PyPI or GitHub, with optional GPU, plotting, and deep-learning extras."
---

Heavy optional backends (H2O, PyTorch, TensorFlow) are imported lazily, so `import pygstat` still works if they are not installed.

## From PyPI

```bash
# Core
pip install pygstat

# Plotting
pip install pygstat[plot]

# GPU (CUDA 11.x extra)
pip install pygstat[gpu]

# GPU (CUDA 12.x) — install CuPy separately
pip install pygstat
pip install cupy-cuda12x

# Deep / AutoML backends
pip install pygstat[h2o]
pip install pygstat[pytorch]
pip install pygstat[tensorflow]
pip install pygstat[deep]          # all three
```

## From GitHub

Clone the repository and install in editable mode:

```bash
git clone https://github.com/zia207/PyGStat.git
cd PyGStat
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
# Optional: install extras
pip install -e ".[plot,test,gpu]"
```

## From source (editable install)

```bash
cd /path/to/PyGStat

python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS

pip install --upgrade pip
pip install -e .
pip install -e ".[plot,test]"      # optional extras
```

Confirm the install:

```bash
python3 -c "import pygstat; print(pygstat.__version__)"
```

## Optional extras

| Extra | Description |
|-------|-------------|
| `gpu` | CuPy (`cupy-cuda11x`) for GPU-accelerated `Variogram`, `OrdinaryKriging`, `SimpleKriging`, and `UniversalKriging` |
| `plot` | Matplotlib and contextily for mapping |
| `test` | pytest and pytest-cov |
| `h2o` | H2O regression kriging (AutoML, Deep Learning, DRF, GLM, GAM, GBM, XGBoost, Uplift DRF, Stacked Ensembles) |
| `pytorch` | Deep, GNN, STGNN, STGTN, and ConvLSTM regression kriging |
| `tensorflow` | Deep regression kriging (TensorFlow/Keras) |
| `gam` | pyGAM, for `RegressionKriging(regressor="gam")` |
| `optuna` | Optuna, for `DeepRegressionKriging(tune_hyperparameters=True)` |
| `deep` | `h2o` + `pytorch` + `tensorflow` |

For CUDA 12.x, install CuPy yourself (`pip install cupy-cuda12x`) rather than using the `gpu` extra.

## GPU setup (optional, Ubuntu + CUDA)

Skip this section for CPU-only use.

```bash
lsb_release -a
nvcc --version
```

### CUDA Toolkit 12.8 on Ubuntu 24.04

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

```bash
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

### CuPy for CUDA 12.x

```bash
pip install cupy-cuda12x
```

### RAPIDS cuDF / cuML (optional)

```bash
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.10.*" "dask-cudf-cu12==25.10.*" "cuml-cu12==25.10.*" \
    "cugraph-cu12==25.10.*" "nx-cugraph-cu12==25.10.*" "cuxfilter-cu12==25.10.*" \
    "cucim-cu12==25.10.*" "pylibraft-cu12==25.10.*" "raft-dask-cu12==25.10.*" \
    "cuvs-cu12==25.10.*"
```

## Using GPU acceleration

`Variogram`, `OrdinaryKriging`, `SimpleKriging`, and `UniversalKriging` all
take a `use_gpu` argument:

```python
from pygstat import Variogram, OrdinaryKriging

v = Variogram(coords, values, model="spherical", use_gpu="auto")  # default
v.fit()  # empirical semivariogram computed on GPU if usable

ok = OrdinaryKriging(v, use_gpu="auto")
ok.fit(coords, values)
pred, std = ok.predict(coords_grid, return_variance=True)  # matrix build + solve on GPU if usable
```

- `use_gpu='auto'` (the default) uses the GPU only when CuPy is installed
  **and** can actually run a kernel on the attached GPU — checked once
  with a real op and cached, not just "did `import cupy` succeed." Older
  cards can import CuPy fine but fail at kernel-compile time for a newer
  CUDA toolkit; `'auto'` silently stays on CPU in that case rather than
  crashing.
- `use_gpu=True` forces GPU and warns (falling back to CPU) if it isn't
  actually usable; `use_gpu=False` always uses CPU (NumPy/SciPy).
- `X`/`X_pred` may be a NumPy array, a CuPy array, or a cuDF
  Series/DataFrame — predictions are always returned as NumPy.
- One exception: the Matérn model always evaluates on CPU internally
  (SciPy's Bessel functions have no CuPy equivalent), transferring data
  over and back transparently — correct either way, just not itself
  GPU-accelerated.

`IndicatorKriging` also takes `use_gpu` (default `'auto'`): its default
k-nearest-neighbor mode predicts every point independently, so all of them
batch into one stacked `xp.linalg.solve` call instead of a Python loop — a
real CPU speedup too, and what makes GPU dispatch worthwhile there.
(`search_radius` mode has variable neighbor counts per point, isn't
batchable the same way, and always runs on CPU.)

`sgsim`/`sisim` accept `use_gpu` as well, but default to **`False`**, not
`'auto'`. Nodes *within* one realization are sequential (each depends on
every previously-simulated node), so they cannot be batched the way
independent kriging predictions can:

- `nsim == 1`: one node at a time. For typical small `num_points`, GPU
  dispatch overhead usually outweighs the benefit — pass `use_gpu=True`
  only if you use unusually large neighborhoods.
- `nsim >= 2`: realizations are independent, so all `nsim` local kriging
  systems at each step are batched into one GPU solve. That is the useful
  GPU path for sequential simulation, and it scales with `nsim`.

`pygstat.cokriging`'s `CrossVariogram`, `Cokriging`, `MultivariateCokriging`,
and `ColocatedCokriging` also take `use_gpu` (default `'auto'`): `Cokriging`/
`MultivariateCokriging` build one big dense covariance matrix and solve it
in one call (`MultivariateCokriging` factors once at `fit()` and reuses it
across `predict()` calls, same as `ColocatedCokriging`'s primary system);
`ColocatedCokriging` additionally batches every prediction point's small
secondary (colocated) system into one stacked solve instead of a Python
loop, the same idea as `IndicatorKriging` above.

## Testing

```bash
pip install -e ".[test]"
pytest
```

Next: [Getting Started](Tutorial/00_getting_started.ipynb) or the [Kriging Workbench](kriging-workbench.html).
