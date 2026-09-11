"""
Tests for pygstat.PoissonKriging against the Atlantic-region county cancer
rate dataset (COUNTY_ATLANTIC.shp / data_atlantic_1998_2012.csv), following
the same leave-one-out evaluation pattern as the pyinterpolate
"Poisson Kriging - Centroid based approach" tutorial.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pygstat import Variogram, PoissonKriging, load_atlantic_panel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def atlantic():
    panel = load_atlantic_panel(DATA_DIR)
    df = panel.loc[panel["year"] == 2005, ["FIPS", "x", "y", "rate", "pop"]].copy()
    df = df.rename(columns={"rate": "lbc_rate"})
    coords = df[["x", "y"]].to_numpy(dtype=float)
    rates = df["lbc_rate"].to_numpy(dtype=float)
    pops = df["pop"].to_numpy(dtype=float)
    ids = df["FIPS"].to_numpy()
    return df, coords, rates, pops, ids


@pytest.fixture(scope="module")
def fitted_variogram(atlantic):
    _, coords, rates, _, _ = atlantic
    vg = Variogram(coords, rates, model="exponential", n_lags=15)
    vg.fit()
    return vg


def test_variogram_fits(fitted_variogram):
    assert fitted_variogram.fitted_params is not None
    nugget, sill, rng = fitted_variogram.fitted_params
    assert nugget >= 0
    assert sill > 0
    assert rng > 0


def test_fit_requires_matching_lengths(fitted_variogram, atlantic):
    _, coords, rates, pops, ids = atlantic
    pk = PoissonKriging(fitted_variogram)
    with pytest.raises(ValueError):
        pk.fit(coords, rates[:-1], pops, ids=ids)


def test_fit_requires_positive_population(fitted_variogram, atlantic):
    _, coords, rates, pops, ids = atlantic
    bad_pops = pops.copy()
    bad_pops[0] = 0
    pk = PoissonKriging(fitted_variogram)
    with pytest.raises(ValueError):
        pk.fit(coords, rates, bad_pops, ids=ids)


def test_predict_id_leave_one_out(fitted_variogram, atlantic):
    df, coords, rates, pops, ids = atlantic
    pk = PoissonKriging(fitted_variogram).fit(coords, rates, pops, ids=ids)

    rng = np.random.default_rng(0)
    sample_ids = rng.choice(ids, size=20, replace=False)

    results = [
        pk.predict_id(
            fips,
            number_of_neighbors=8,
            neighbors_range=fitted_variogram.fitted_params[2] * 2,
        )
        for fips in sample_ids
    ]
    results = pd.DataFrame(results)

    # Every held-out area must have gotten a finite prediction and error.
    assert np.isfinite(results["zhat"]).all()
    assert np.isfinite(results["sig"]).all()
    assert (results["sig"] >= 0).all()

    actual = df.set_index("FIPS").loc[results["id"], "lbc_rate"].to_numpy()
    rmse = np.sqrt(np.mean((actual - results["zhat"].to_numpy()) ** 2))

    # Sanity bound: LOO predictions should track observed rates reasonably
    # well (not be wildly off), well under the raw rate's own std dev.
    assert rmse < rates.std()


def test_predict_new_location(fitted_variogram, atlantic):
    _, coords, rates, pops, ids = atlantic
    pk = PoissonKriging(fitted_variogram).fit(coords, rates, pops, ids=ids)

    # Midpoint between the first two known areas -- not one of the fitted
    # points, so no self-match/leave-one-out logic is involved here.
    new_point = coords[:2].mean(axis=0, keepdims=True)
    zhat, sig = pk.predict(new_point, number_of_neighbors=8)

    assert zhat.shape == (1,)
    assert sig.shape == (1,)
    assert np.isfinite(zhat[0])


def test_predict_id_unknown_raises(fitted_variogram, atlantic):
    _, coords, rates, pops, ids = atlantic
    pk = PoissonKriging(fitted_variogram).fit(coords, rates, pops, ids=ids)
    with pytest.raises(ValueError):
        pk.predict_id(-1)
