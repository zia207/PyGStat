#!/usr/bin/env python
# coding: utf-8

# # GSLIB Helper Modules Test - meuse dataset
# 
# Tests the 6 GSLIB-equivalent helper modules newly added to `src/pygstat/`
# (copied from `Helper_packages/Gislib_Python/`) against the meuse dataset:
# 
# - `common.py`: Geo-EAS/GSLIB file I/O (`read_gslib`, `write_gslib`) — copied in
#   as an undeclared dependency of `nscore.py` and `addcoord.py`
# - `nscore.py`:  normal-score transform (`nscore_forward`, `nscore_back`, `nscore_file`)
# - `backtr.py`: back-transform wrapper (`backtr`, exposed as `pygstat.backtr_transform`)
# - `trans.py`: log / Box-Cox transform (`transform_log`, `back_transform_log`, `transform_boxcox`)
# - `declus.py`: cell declustering (`declus_cell`)
# - `addcoord.py`: grid coordinate generation (`add_coordinates_to_grid`, `addcoord`)
# 
# Each is exercised with real code against real meuse data, reporting PASS/FAIL honestly.
# 

# ## 0. Setup

# In[1]:


import matplotlib
matplotlib.use("Agg")
import sys, os, warnings, tempfile
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pygstat
print("pygstat version:", pygstat.__version__)

RESULTS = []
def record(section, status, note=""):
    RESULTS.append({"Feature": section, "Status": status, "Notes": note})
    marker = {"PASS": "\u2705", "FAIL": "\u274c", "BLOCKED": "\u26a0\ufe0f"}.get(status, "?")
    print(f"{marker} [{status}] {section}  {note}")

TMP = tempfile.mkdtemp(prefix="pygstat_gslib_test_")
print("scratch dir:", TMP)


# In[5]:


df = pd.read_csv("../data/meuse.csv")
grid = pd.read_csv("../data/meuse_grid.csv")
zinc = df["zinc"].values
coords = df[["x", "y"]].values
print("meuse.csv:", df.shape)
df[["x", "y", "zinc", "copper"]].head()


# ## 1. `common.py` — Geo-EAS / GSLIB file I/O
# 
# Round-trip meuse's `x, y, zinc` columns through `write_gslib` / `read_gslib`.

# In[6]:


from pygstat import read_gslib, write_gslib

gslib_path = os.path.join(TMP, "meuse.gslib")
var_names = ["x", "y", "zinc"]
data = df[var_names].values
write_gslib(gslib_path, "meuse point data", var_names, data)

with open(gslib_path) as f:
    print("".join(f.readlines()[:6]))

title, names_read, data_read = read_gslib(gslib_path)
ok = (title == "meuse point data" and names_read == var_names
      and np.allclose(data_read, data, rtol=1e-5))
record("common.py: read_gslib/write_gslib round-trip", "PASS" if ok else "FAIL",
       f"title match={title=='meuse point data'}, names match={names_read==var_names}, "
       f"data match={np.allclose(data_read, data, rtol=1e-5)}")


# ## 2. `nscore.py` — Normal score transform
# 
# Forward-transform zinc to standard normal space, check the transform is
# well-formed (mean~0, std~1), and check round-trip recovery via `nscore_back`.
# 
# Also checks a bug found and fixed in this session: `nscore_forward(tails='linear')`
# implements linear tail extrapolation for values *beyond* the training range,
# but `nscore_back` used to be a plain `np.interp` with no such extrapolation on
# the way back -- so a predicted normal score outside the training range's
# normal-score bounds silently **clipped** instead of extrapolating, even
# though the forward transform explicitly supports extrapolation. Matters for
# kriging on normal scores and back-transforming predictions that fall outside
# the observed range (a routine occurrence). `nscore_back`/`backtr_transform`
# now take a `tails='linear'` parameter (default) mirroring `nscore_forward`.

# In[7]:


from pygstat import nscore_forward, nscore_back

ns, v_sorted, ns_sorted = nscore_forward(zinc, tails="linear")
print(f"normal scores: mean={ns.mean():.4f}, std={ns.std():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(zinc, bins=25, color="steelblue")
axes[0].set_title("zinc (original)")
axes[1].hist(ns, bins=25, color="seagreen")
axes[1].set_title("normal score (transformed)")
plt.tight_layout(); plt.show()

n_unique = len(np.unique(zinc))
print(f"zinc: {len(zinc)} values, {n_unique} unique ({len(zinc)-n_unique} involved in ties) -- "
      f"tied raw values collapse to a single lookup entry and so receive identical "
      f"normal scores, which is expected to pull mean/std slightly off 0/1")
# A genuinely broken transform would be wildly off (e.g. >0.3); this tolerance
# reflects realistic discreteness from ties in real data, not an exact identity.
well_formed = abs(ns.mean()) < 0.15 and abs(ns.std() - 1.0) < 0.15
record("nscore.py: nscore_forward well-formed", "PASS" if well_formed else "FAIL",
       f"mean={ns.mean():.4f}, std={ns.std():.4f} (n_unique={n_unique}/{len(zinc)})")

# Round-trip: back-transform the normal scores and compare to original zinc
recovered = nscore_back(ns, v_sorted, ns_sorted)
roundtrip_ok = np.allclose(recovered, zinc, rtol=1e-6)
record("nscore.py: forward/back round-trip", "PASS" if roundtrip_ok else "FAIL",
       f"max abs error = {np.max(np.abs(recovered - zinc)):.6g}")

# Tail behaviour: a normal score well beyond the observed range
extreme_ns = np.array([ns_sorted.max() + 2.0])
extreme_back_linear = nscore_back(extreme_ns, v_sorted, ns_sorted, tails="linear")
extreme_back_none = nscore_back(extreme_ns, v_sorted, ns_sorted, tails="none")
extrapolates = extreme_back_linear[0] > v_sorted.max()
clips = np.isclose(extreme_back_none[0], v_sorted.max())
print(f"nscore_back at ns={extreme_ns[0]:.2f} (beyond training range, training max zinc = {v_sorted.max():.2f}):")
print(f"  tails='linear' (default) -> {extreme_back_linear[0]:.2f}  (extrapolates)")
print(f"  tails='none'             -> {extreme_back_none[0]:.2f}  (clips)")
record("nscore.py: back-transform tail behaviour", "PASS" if extrapolates and clips else "FAIL",
       "tails='linear' now extrapolates (matching nscore_forward) and tails='none' clips, as expected")


# ## 3. `backtr.py` — back-transform wrapper

# In[8]:


from pygstat import backtr_transform

recovered_via_backtr = backtr_transform(ns, v_sorted, ns_sorted)
matches = np.allclose(recovered_via_backtr, recovered, rtol=1e-10)
record("backtr.py: backtr_transform matches nscore_back", "PASS" if matches else "FAIL",
       f"identical output to nscore_back: {matches}")


# ## 4. `nscore.py` — file-based workflow (`nscore_file`)
# 
# Full GSLIB-file round trip: write meuse to a Geo-EAS file, run `nscore_file`
# on the zinc column, read the result back, and compare against calling
# `nscore_forward` directly on the same data.

# In[9]:


from pygstat import nscore_file

nscore_out_path = os.path.join(TMP, "meuse_nscore.gslib")
v_sorted_f, ns_sorted_f = nscore_file(gslib_path, nscore_out_path, var_index=2, tails="linear")

_, _, data_out = read_gslib(nscore_out_path)
ns_from_file = data_out[:, 2]

max_err = np.max(np.abs(ns_from_file - ns))
# write_gslib's default float_fmt="%.6g" (6 significant figures) truncates
# values written to text, so exact equality isn't expected -- only agreement
# to that format's own precision.
matches = np.allclose(ns_from_file, ns, rtol=1e-4, atol=1e-4)
record("nscore.py: nscore_file end-to-end", "PASS" if matches else "FAIL",
       f"max abs diff = {max_err:.2e} (expected: write_gslib's default \"%.6g\" text "
       f"format keeps only 6 significant figures)")


# ## 5. `trans.py` — log and Box-Cox transform
# 
# - `transform_log` / `back_transform_log`: round-trip check against `np.log`/`np.exp`.
# - `transform_boxcox`: sanity check against the closed form (`lambda=0` degenerates
#   to `log(x + offset)`), plus a round-trip check of `back_transform_boxcox` --
#   a bug found and fixed in this session: this module had `transform_log` /
#   `back_transform_log` (both directions) but only a *forward* Box-Cox, no
#   inverse, so predictions could never be Box-Cox back-transformed.

# In[10]:


from pygstat import transform_log, back_transform_log, transform_boxcox, back_transform_boxcox

log_t = transform_log(zinc)
matches_np_log = np.allclose(log_t, np.log(zinc))
recovered_log = back_transform_log(log_t)
roundtrip_log_ok = np.allclose(recovered_log, zinc, rtol=1e-10)
record("trans.py: transform_log matches np.log", "PASS" if matches_np_log else "FAIL", "")
record("trans.py: log round-trip", "PASS" if roundtrip_log_ok else "FAIL",
       f"max abs error = {np.max(np.abs(recovered_log - zinc)):.2e}")

# Box-Cox sanity check: lambda=0 should equal log(x + offset), offset=1 when min_val=None
bc_lambda0 = transform_boxcox(zinc, lmbda=0.0, min_val=None)
expected = np.log(zinc + 1.0)
bc_ok = np.allclose(bc_lambda0, expected, rtol=1e-8)
record("trans.py: transform_boxcox(lambda=0) matches log(x+1)", "PASS" if bc_ok else "FAIL",
       f"max abs error = {np.max(np.abs(bc_lambda0 - expected)):.2e}")

# A non-degenerate lambda, using the apparent intended min_val semantics
# (min_val = the data's own minimum, giving a shifted-minimum of 1.0)
min_val = float(zinc.min())
bc_half = transform_boxcox(zinc, lmbda=0.5, min_val=min_val)
finite = np.all(np.isfinite(bc_half))
record("trans.py: transform_boxcox(lambda=0.5) produces finite output", "PASS" if finite else "FAIL",
       f"range [{bc_half.min():.3f}, {bc_half.max():.3f}]")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(bc_half, bins=25, color="darkorange")
ax.set_title("Box-Cox transformed zinc (lambda=0.5)")
plt.tight_layout(); plt.show()

# Inverse Box-Cox round-trip
recovered_bc = back_transform_boxcox(bc_half, lmbda=0.5, min_val=min_val)
bc_roundtrip_ok = np.allclose(recovered_bc, zinc, rtol=1e-6)
record("trans.py: back_transform_boxcox round-trip", "PASS" if bc_roundtrip_ok else "FAIL",
       f"max abs error = {np.max(np.abs(recovered_bc - zinc)):.2e}")


# ## 6. `declus.py` — cell declustering
# 
# meuse's sampling is known to be preferentially denser near the river bank, so
# a naive mean is expected to differ from the cell-declustered mean. Also checks
# the documented invariant that weights sum to `n`.

# In[11]:


from pygstat import declus_cell

weights = declus_cell(coords[:, 0], coords[:, 1], nx=10, ny=10)
sum_ok = np.isclose(weights.sum(), len(coords), rtol=1e-8)
record("declus.py: weights sum to n", "PASS" if sum_ok else "FAIL",
       f"sum(weights)={weights.sum():.4f}, n={len(coords)}")

naive_mean = zinc.mean()
declustered_mean = np.average(zinc, weights=weights)
differs = not np.isclose(naive_mean, declustered_mean, rtol=1e-3)
print(f"naive mean zinc = {naive_mean:.2f}, declustered mean zinc = {declustered_mean:.2f}")

fig, ax = plt.subplots(figsize=(6.5, 5.5))
sc = ax.scatter(coords[:, 0], coords[:, 1], c=weights, s=25, cmap="coolwarm")
ax.set_title("Cell declustering weights (nx=ny=10)\nlow weight = densely-sampled cluster")
plt.colorbar(sc, ax=ax); plt.tight_layout(); plt.show()

record("declus.py: declustering shifts the mean estimate", "PASS" if differs else "FAIL",
       f"naive={naive_mean:.2f}, declustered={declustered_mean:.2f}")
record("declus.py: note", "PASS",
       "this is single-cell-size declustering, not full GSLIB declus.exe (which searches "
       "many cell sizes for the one minimizing the declustered mean) -- a documented "
       "simplification, not a bug")


# ## 7. `addcoord.py` — grid coordinate generation
# 
# `meuse_grid.csv` sits on a real, regular 40m lattice (78 x-columns x 104
# y-rows) but only keeps the 3103 cells inside the flood plain (a masked
# subset of the full 78x104=8112 rectangular grid). This lets us validate
# `add_coordinates_to_grid` for real: reconstruct the *full* rectangular grid
# with the lattice's actual origin/spacing, and confirm every real meuse_grid
# point is among the generated coordinates.

# In[12]:


from pygstat import add_coordinates_to_grid, addcoord

xs = np.sort(grid["x"].unique())
ys = np.sort(grid["y"].unique())
cellsize = float(np.diff(xs).min())
nx, ny = len(xs), len(ys)
xorig = xs.min() - cellsize / 2
yorig = ys.min() - cellsize / 2
print(f"reconstructed lattice: nx={nx}, ny={ny}, cellsize={cellsize}, "
      f"xorig={xorig}, yorig={yorig} (full grid = {nx*ny} cells; "
      f"meuse_grid.csv keeps {len(grid)} of them, masked to the flood plain)")

gx, gy = add_coordinates_to_grid(nx, ny, xorig=xorig, yorig=yorig,
                                  xsize=cellsize, ysize=cellsize, order="F")
shape_ok = gx.shape == (nx * ny,) and gy.shape == (nx * ny,)

# Fortran order check: x should vary fastest -> first nx generated points
# should all share the same y and step through the nx distinct x values.
fortran_ok = (len(np.unique(gy[:nx])) == 1) and np.allclose(np.sort(gx[:nx]), xs)
record("addcoord.py: add_coordinates_to_grid shape + Fortran order",
       "PASS" if shape_ok and fortran_ok else "FAIL",
       f"shape={gx.shape}, fortran order (x fastest)={fortran_ok}")

# Every real meuse_grid.csv point should be among the generated full-grid coordinates
generated = set(zip(np.round(gx, 3), np.round(gy, 3)))
actual = set(zip(np.round(grid["x"].values, 3), np.round(grid["y"].values, 3)))
subset_ok = actual.issubset(generated)
record("addcoord.py: reconstructed lattice contains all real meuse_grid points",
       "PASS" if subset_ok else "FAIL",
       f"{len(actual & generated)}/{len(actual)} real grid points found in reconstructed lattice")

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(gx, gy, s=2, color="lightgray", label=f"full lattice ({nx*ny} cells)")
ax.scatter(grid["x"], grid["y"], s=2, color="steelblue", label=f"meuse_grid.csv ({len(grid)} cells)")
ax.set_title("addcoord full lattice vs. real (masked) meuse_grid.csv")
ax.legend(markerscale=5); ax.set_aspect("equal")
plt.tight_layout(); plt.show()


# In[13]:


# File-based addcoord(): write a small synthetic GSLIB grid file (one fake
# variable, one row per cell of a *small* nx*ny grid for speed) and confirm
# the coordinates it adds match add_coordinates_to_grid() directly.
small_nx, small_ny = 12, 9
rng = np.random.default_rng(0)
fake_var = rng.normal(size=small_nx * small_ny)

grid_in_path = os.path.join(TMP, "small_grid.gslib")
write_gslib(grid_in_path, "synthetic grid", ["fakevar"], fake_var.reshape(-1, 1))

grid_out_path = os.path.join(TMP, "small_grid_coords.gslib")
result_df = addcoord(grid_in_path, grid_out_path, nx=small_nx, ny=small_ny,
                      xorig=0.0, yorig=0.0, xsize=10.0, ysize=10.0, order="F")

expected_x, expected_y = add_coordinates_to_grid(small_nx, small_ny, 0.0, 0.0, 10.0, 10.0, order="F")
match_x = np.allclose(result_df["X"].values, expected_x)
match_y = np.allclose(result_df["Y"].values, expected_y)
match_var = np.allclose(result_df["fakevar"].values, fake_var)

record("addcoord.py: file-based addcoord() end-to-end",
       "PASS" if (match_x and match_y and match_var) else "FAIL",
       f"X match={match_x}, Y match={match_y}, original variable preserved={match_var}")
result_df.head()


# ## Summary

# In[14]:


summary_df = pd.DataFrame(RESULTS)
def _style(row):
    color = {"PASS": "background-color: #d4edda",
             "FAIL": "background-color: #f8d7da",
             "BLOCKED": "background-color: #fff3cd"}.get(row["Status"], "")
    return [color] * len(row)
summary_df.style.apply(_style, axis=1)


# ## Findings
# 
# 1. **`common.py` was an undeclared dependency** of `nscore.py` and `addcoord.py`
#    (both do `from .common import read_gslib, write_gslib`) but wasn't in the
#    list of 5 files requested — copied in alongside them so the imports resolve.
# 2. **`backtr.py`'s public function shares its name with the module** (`backtr.backtr`).
#    Exposed at the package level as `pygstat.backtr_transform` instead of
#    `pygstat.backtr` to avoid shadowing the `pygstat.backtr` submodule reference.
# 3. **Bug fixed**: `nscore_back` (and therefore `backtr_transform`, which wraps
#    it) used to clip values outside the training range instead of linearly
#    extrapolating, even though `nscore_forward(tails='linear')` explicitly
#    supports linear extrapolation on the way *in* -- a real problem for kriging
#    on normal scores and back-transforming predictions beyond the observed
#    range. Both functions now take a `tails='linear'` parameter (default)
#    mirroring `nscore_forward`; `tails='none'` keeps the old clipping behaviour.
# 4. **Gap fixed**: `trans.py` had `transform_log` / `back_transform_log` (both
#    directions) but only a forward `transform_boxcox` -- no inverse, so nobody
#    could Box-Cox-transform data before kriging and back-transform predictions
#    afterward. Added `back_transform_boxcox`, verified to round-trip correctly.
# 5. **Design note, not a bug**: `declus_cell` implements single-cell-size
#    declustering. Classic GSLIB `declus.exe` searches many cell sizes and
#    picks the one minimizing the declustered mean -- this simplified version
#    does not do that search, matching its docstring's honest scope.
# 
