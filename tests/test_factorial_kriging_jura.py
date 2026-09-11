"""
Test for pygstat.factorial_kriging against the Jura geochemical data set
(data/jura_data.csv): fit a nested (nugget + short-range + long-range) variogram
to Cd, then use FactorialKriging to extract the short-range and long-range
structures separately and check that they behave the way the theory predicts.
"""

import numpy as np
import pandas as pd
import pytest

from pygstat.factorial_kriging import FactorialKriging, fit_nested_variogram

DATA_PATH = "data/jura_data.csv"


def _load_jura():
    df = pd.read_csv(DATA_PATH, index_col=0)
    coords = df[["Xloc", "Yloc"]].to_numpy()
    cd = df["Cd"].to_numpy()
    return df, coords, cd


def _fit_cd_model(coords, cd):
    nugget, structures, lag_h, lag_g = fit_nested_variogram(
        coords, cd, models=("spherical", "spherical"), n_lags=15, maxlag=3.0
    )
    return nugget, structures, lag_h, lag_g


def _grid(coords, n=40):
    gx = np.linspace(coords[:, 0].min(), coords[:, 0].max(), n)
    gy = np.linspace(coords[:, 1].min(), coords[:, 1].max(), n)
    GX, GY = np.meshgrid(gx, gy)
    return GX, GY, np.column_stack([GX.ravel(), GY.ravel()])


def _lag_autocorr(grid_2d, k):
    """Average row-wise and column-wise correlation between cells `k` apart."""
    a, b = grid_2d[:-k, :].ravel(), grid_2d[k:, :].ravel()
    c1 = np.corrcoef(a, b)[0, 1]
    a2, b2 = grid_2d[:, :-k].ravel(), grid_2d[:, k:].ravel()
    c2 = np.corrcoef(a2, b2)[0, 1]
    return (c1 + c2) / 2.0


def test_fit_nested_variogram_finds_two_ordered_structures():
    _, coords, cd = _load_jura()
    nugget, structures, lag_h, lag_g = _fit_cd_model(coords, cd)

    assert nugget > 0
    assert len(structures) == 2
    assert structures[0]["range"] < structures[1]["range"]  # sorted short -> long
    for st in structures:
        assert st["sill"] > 0 and st["range"] > 0

    total_sill = nugget + sum(s["sill"] for s in structures)
    # the fitted model's total sill should be in the right ballpark of the
    # data's own variance (this data set's variogram keeps trending upward at
    # long lags -- a regional drift outside this bounded two-structure model's
    # scope -- so we allow a generous but bounded tolerance).
    assert 0.5 * cd.var() < total_sill < 1.5 * cd.var()


def test_factorial_kriging_total_is_an_exact_interpolator():
    """extract(factor='total') is ordinary kriging of Cd itself, which must
    exactly reproduce the data at the data locations."""
    _, coords, cd = _load_jura()
    nugget, structures, *_ = _fit_cd_model(coords, cd)
    fk = FactorialKriging(nugget=nugget, structures=structures, max_neighbors=30)
    fk.fit(coords, cd)

    z_hat = fk.extract(coords, factor="total")
    np.testing.assert_allclose(z_hat, cd, atol=1e-5)


def test_factor_sums_are_linear_in_the_covariance():
    """extract([1,2]) (one solve, combined RHS) must exactly equal
    extract(1) + extract(2) (two solves, same neighbors, same sum=0
    constraint) -- a direct consequence of the kriging system being linear
    in its right-hand side."""
    _, coords, cd = _load_jura()
    nugget, structures, *_ = _fit_cd_model(coords, cd)
    fk = FactorialKriging(nugget=nugget, structures=structures, max_neighbors=30)
    fk.fit(coords, cd)

    targets = coords[:30]
    combined = fk.extract(targets, factor=[1, 2])
    separate = fk.extract(targets, factor=1) + fk.extract(targets, factor=2)
    np.testing.assert_allclose(combined, separate, atol=1e-9)

    signal = fk.extract(targets, factor="signal")
    np.testing.assert_allclose(signal, combined, atol=1e-9)


def test_short_range_factor_decorrelates_faster_than_long_range_factor():
    """The core claim of factorial kriging: the extracted short-range
    structure should lose spatial correlation over a much shorter distance
    than the extracted long-range structure."""
    _, coords, cd = _load_jura()
    nugget, structures, *_ = _fit_cd_model(coords, cd)
    fk = FactorialKriging(nugget=nugget, structures=structures, max_neighbors=30)
    fk.fit(coords, cd)

    GX, GY, grid_pts = _grid(coords, n=40)
    short_grid = fk.extract(grid_pts, factor=1).reshape(GX.shape)
    long_grid = fk.extract(grid_pts, factor=2).reshape(GX.shape)

    # A lag comfortably *inside* the short structure's own range: close enough
    # to the origin that the long-range factor is still highly autocorrelated
    # there, far enough out that the short-range factor has mostly decayed.
    # (At a lag *at* the short range itself the theoretical correlation is
    # ~0 for a spherical model and the finite-sample estimate gets noisy --
    # this is the well-conditioned regime, confirmed stable across several
    # grid resolutions during development.)
    cell_size = (coords[:, 0].max() - coords[:, 0].min()) / 39
    short_range = structures[0]["range"]
    k = max(1, int(round(0.6 * short_range / cell_size)))

    corr_short = _lag_autocorr(short_grid, k)
    corr_long = _lag_autocorr(long_grid, k)
    print(f"\nAt lag {k * cell_size:.2f} km (~0.6x the short range of {short_range:.2f} km): "
          f"corr(short)={corr_short:.3f}  corr(long)={corr_long:.3f}")

    assert corr_long > corr_short + 0.15


def test_nugget_factor_is_near_zero_away_from_data():
    _, coords, cd = _load_jura()
    nugget, structures, *_ = _fit_cd_model(coords, cd)
    fk = FactorialKriging(nugget=nugget, structures=structures, max_neighbors=30)
    fk.fit(coords, cd)

    # Points offset from every datum by more than a hair -- the nugget factor
    # has zero covariance with the data there, so its kriged value should be ~0.
    far_points = coords + np.array([0.05, 0.05])
    nugget_hat = fk.extract(far_points, factor="nugget")
    assert np.nanmax(np.abs(nugget_hat)) < 0.05 * nugget


def test_denoised_stays_in_original_units_while_signal_is_zero_mean():
    """'denoised' (sum=1) keeps Z's mean and units, unlike 'signal' (sum=0),
    which is a genuine zero-mean fluctuation -- see FactorialKriging.extract's
    docstring. Confusing the two was a real bug caught while building the
    Tutorial 09 notebook (a 'denoised Cd' plot came out centered on zero)."""
    _, coords, cd = _load_jura()
    nugget, structures, *_ = _fit_cd_model(coords, cd)
    fk = FactorialKriging(nugget=nugget, structures=structures, max_neighbors=30)
    fk.fit(coords, cd)

    denoised = fk.extract(coords, factor="denoised")
    signal = fk.extract(coords, factor="signal")

    assert abs(denoised.mean() - cd.mean()) < 0.05 * cd.mean()
    assert (denoised > 0).all()  # Cd is a concentration; a proper filter stays positive
    assert abs(signal.mean()) < 0.05 * cd.mean()
    # denoised should have less noise-scale scatter than the raw exact-interpolation
    # values, but far more spread than the zero-mean signal alone (different scale).
    assert denoised.std() < cd.std()


def test_extract_returns_valid_variance():
    _, coords, cd = _load_jura()
    nugget, structures, *_ = _fit_cd_model(coords, cd)
    fk = FactorialKriging(nugget=nugget, structures=structures, max_neighbors=30)
    fk.fit(coords, cd)

    _, _, grid_pts = _grid(coords, n=15)
    for factor in ("total", "signal", 1, 2):
        est, var = fk.extract(grid_pts, factor=factor, return_variance=True)
        valid = ~np.isnan(est)
        assert valid.sum() > 0
        assert np.nanmin(var[valid]) >= -1e-8


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
