"""
Test for pygstat.kt3d against the classic GSLIB cluster.dat example.

This reproduces the standard GSLIB KT3D tutorial workflow (Deutsch &
Journel, GSLIB 2nd ed.): normal-score transform the (heavily skewed)
"Primary" variable from ``cluster.dat`` (140 clustered samples over a
50x50 domain), krige the transformed variable with an isotropic spherical
variogram (nugget=0.1, sill=0.9, range=10) by ordinary kriging (radius=20,
ndmin=1, ndmax=8) onto a 50x50 point grid, and back-transform. A separate
leave-one-out cross-validation sanity-checks the fit, and additional cases
exercise simple kriging, block kriging, and kriging-with-a-trend (KT).
"""

import numpy as np
import pandas as pd
import pytest

from pygstat.common import read_gslib
from pygstat.nscore import nscore_forward, nscore_back
from pygstat.kt3d import kt3d, kt3d_grid, kt3d_cross_validate, build_grid_3d

DATA_PATH = "data/GSLIB_data/cluster.dat"

# Standard nugget/sill/range used throughout the classic GSLIB KT3D tutorial
# for the normal-score transform of cluster.dat's "Primary" variable.
VARIOGRAM = {
    "nugget": 0.1,
    "structures": [
        {"type": "spherical", "sill": 0.9, "range": 10.0},
    ],
}


def _load_cluster():
    title, var_names, data = read_gslib(DATA_PATH)
    df = pd.DataFrame(data, columns=var_names)
    ns, v_sorted, ns_sorted = nscore_forward(df["Primary"].to_numpy(), tails="linear")
    df["NS_Primary"] = ns
    return df, v_sorted, ns_sorted


def test_read_cluster_data():
    df, _, _ = _load_cluster()
    assert len(df) == 140
    assert "Primary" in df.columns
    # normal-score transform should give ~unit variance
    assert abs(df["NS_Primary"].std() - 1.0) < 0.05


def test_build_grid_3d_shape_and_order():
    pts = build_grid_3d(nx=3, xmn=0.5, xsiz=1.0, ny=2, ymn=0.5, ysiz=1.0)
    assert pts.shape == (6, 3)
    # x varies fastest
    np.testing.assert_allclose(pts[:3, 0], [0.5, 1.5, 2.5])
    np.testing.assert_allclose(pts[:3, 1], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(pts[3:, 1], [1.5, 1.5, 1.5])


def test_kt3d_grid_ordinary_kriging():
    df, v_sorted, ns_sorted = _load_cluster()
    result = kt3d_grid(
        df, x="Xlocation", y="Ylocation", value="NS_Primary",
        variogram=VARIOGRAM,
        nx=50, xmn=0.5, xsiz=1.0,
        ny=50, ymn=0.5, ysiz=1.0,
        ktype="ordinary",
        search_radius=20.0, ndmin=1, ndmax=8,
    )
    assert len(result) == 2500
    n_estimated = result["estimate"].notna().sum()
    # With radius=20 over a 50x50 domain and 140 data, essentially every
    # node should get an estimate.
    assert n_estimated > 2400

    valid = result.dropna(subset=["estimate"])
    assert valid["variance"].min() >= -1e-8
    # OK variance should stay close to (and not wildly exceed) the a priori
    # sill of the normal-score variogram (1.0).
    assert valid["variance"].max() <= 1.2
    # Normal-score estimates should stay well within a plausible range.
    assert valid["estimate"].between(-4.0, 4.0).all()

    back = nscore_back(valid["estimate"].to_numpy(), v_sorted, ns_sorted, tails="linear")
    assert np.isfinite(back).all()
    assert (back > 0).all()  # Primary is a strictly positive assay value


def test_kt3d_simple_kriging_far_point_is_nan():
    """Far from all data (outside every search radius), no kriging system
    can be built -- the estimate should be NaN rather than a bogus value."""
    df, _, _ = _load_cluster()
    far_point = np.array([[-500.0, -500.0, 0.0]])
    sk = kt3d(
        df, x="Xlocation", y="Ylocation", value="NS_Primary", points=far_point,
        variogram=VARIOGRAM, ktype="simple", skmean=0.0,
        search_radius=20.0, ndmin=1, ndmax=8,
    )
    assert np.isnan(sk["estimate"].iloc[0])


def test_kt3d_simple_kriging_matches_ordinary_near_data():
    """Simple and ordinary kriging should agree closely in data-dense areas
    when skmean is set to the (near-zero) normal-score mean."""
    df, _, _ = _load_cluster()
    pts = np.array([[25.0, 25.0, 0.0]])
    ok = kt3d(df, x="Xlocation", y="Ylocation", value="NS_Primary", points=pts,
              variogram=VARIOGRAM, ktype="ordinary",
              search_radius=20.0, ndmin=1, ndmax=8)
    sk = kt3d(df, x="Xlocation", y="Ylocation", value="NS_Primary", points=pts,
              variogram=VARIOGRAM, ktype="simple", skmean=0.0,
              search_radius=20.0, ndmin=1, ndmax=8)
    assert np.isfinite(ok["estimate"].iloc[0])
    assert np.isfinite(sk["estimate"].iloc[0])
    assert abs(ok["estimate"].iloc[0] - sk["estimate"].iloc[0]) < 0.3
    # SK variance (no unbiasedness Lagrange penalty) should be <= OK variance
    assert sk["variance"].iloc[0] <= ok["variance"].iloc[0] + 1e-8


def test_kt3d_cross_validate_reasonable_fit():
    df, _, _ = _load_cluster()
    cv = kt3d_cross_validate(
        df, x="Xlocation", y="Ylocation", value="NS_Primary",
        variogram=VARIOGRAM, ktype="ordinary",
        search_radius=20.0, ndmin=1, ndmax=8,
    )
    valid = cv.dropna(subset=["estimate"])
    assert len(valid) > 130  # nearly all 140 data get a LOO estimate

    corr = np.corrcoef(valid["true"], valid["estimate"])[0, 1]
    mae = valid["error"].abs().mean()
    print(f"\nKT3D LOO cross-validation on cluster.dat (normal-score space): "
          f"n={len(valid)}, corr(true,est)={corr:.3f}, MAE={mae:.4f}")

    assert corr > 0.4
    assert mae < 1.0  # normal-score units (unit variance)


def test_kt3d_block_kriging_smooths_relative_to_point_kriging():
    df, _, _ = _load_cluster()
    pt = np.array([[25.0, 25.0, 0.0]])
    point_result = kt3d(df, x="Xlocation", y="Ylocation", value="NS_Primary", points=pt,
                         variogram=VARIOGRAM, ktype="ordinary",
                         search_radius=20.0, ndmin=1, ndmax=8)
    block_result = kt3d(df, x="Xlocation", y="Ylocation", value="NS_Primary", points=pt,
                         variogram=VARIOGRAM, ktype="ordinary",
                         search_radius=20.0, ndmin=1, ndmax=8,
                         block_size=(5.0, 5.0, 1.0), block_discretization=(4, 4, 1))
    assert np.isfinite(point_result["estimate"].iloc[0])
    assert np.isfinite(block_result["estimate"].iloc[0])
    # Block kriging variance should be smaller than point kriging variance
    # (averaging over the block reduces estimation variance).
    assert block_result["variance"].iloc[0] < point_result["variance"].iloc[0]


def test_kt3d_universal_kriging_with_linear_drift():
    df, _, _ = _load_cluster()
    pts = build_grid_3d(nx=10, xmn=2.5, xsiz=5.0, ny=10, ymn=2.5, ysiz=5.0)
    result = kt3d(
        df, x="Xlocation", y="Ylocation", value="NS_Primary", points=pts,
        variogram=VARIOGRAM, ktype="ordinary", drift=(1, 1, 0, 0, 0, 0, 0, 0, 0),
        search_radius=25.0, ndmin=4, ndmax=16,
    )
    valid = result.dropna(subset=["estimate"])
    assert len(valid) > 90  # nearly all 100 grid nodes should be estimated
    assert np.isfinite(valid["estimate"]).all()
    assert (valid["variance"] >= -1e-8).all()


def test_kt3d_locally_varying_mean_and_external_drift():
    """Smoke-test ktype='locally_varying_mean' (GSLIB ktype=2) and
    'external_drift' (GSLIB ktype=3) using cluster.dat's "Secondary" as the
    external-drift variable (nearest-neighbor-extrapolated onto the grid,
    which is enough for a functional check, not a real drift analysis)."""
    df, _, _ = _load_cluster()
    pts = build_grid_3d(nx=10, xmn=2.5, xsiz=5.0, ny=10, ymn=2.5, ysiz=5.0)

    from scipy.spatial import cKDTree
    tree = cKDTree(df[["Xlocation", "Ylocation"]].to_numpy())
    _, nn = tree.query(pts[:, :2])
    ext_pts = df["Secondary"].to_numpy()[nn]

    lvm = kt3d(df, x="Xlocation", y="Ylocation", value="Secondary", points=pts,
               variogram=VARIOGRAM, ktype="locally_varying_mean", ext_drift="Secondary",
               ext_drift_points=ext_pts, search_radius=25.0, ndmin=1, ndmax=16)
    assert lvm["estimate"].notna().all()
    assert (lvm["variance"] >= -1e-8).all()

    ked = kt3d(df, x="Xlocation", y="Ylocation", value="NS_Primary", points=pts,
               variogram=VARIOGRAM, ktype="external_drift", ext_drift="Secondary",
               ext_drift_points=ext_pts, search_radius=25.0, ndmin=4, ndmax=16)
    assert ked["estimate"].notna().all()
    assert (ked["variance"] >= -1e-8).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
