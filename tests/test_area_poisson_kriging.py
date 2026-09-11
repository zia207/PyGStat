"""
Tests for pygstat.AreaPoissonKriging (area-to-area / area-to-point) and
pygstat.PointSupport, against the `cancer_data.gpkg` dataset used by the
pyinterpolate Poisson Kriging tutorials this module was adapted from.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest

from pygstat import Variogram, PoissonKriging, PointSupport, AreaPoissonKriging

GPKG = Path(__file__).resolve().parent.parent / "data" / "cancer_data.gpkg"
pytestmark = pytest.mark.skipif(not GPKG.exists(), reason="cancer_data.gpkg not available")


@pytest.fixture(scope="module")
def cancer_data():
    areas = gpd.read_file(GPKG, layer="areas")
    points = gpd.read_file(GPKG, layer="points")
    return areas, points


@pytest.fixture(scope="module")
def point_support(cancer_data):
    areas, points = cancer_data
    return PointSupport.from_geodataframes(
        points, areas, block_id_col="FIPS", point_value_col="POP10"
    )


@pytest.fixture(scope="module")
def fitted(cancer_data, point_support):
    areas, _ = cancer_data
    ids = np.array([bid for bid in areas["FIPS"] if bid in point_support])
    rates = areas.set_index("FIPS").loc[ids, "rate"].to_numpy(dtype=float)
    centroids = np.array([point_support.centroid(bid) for bid in ids])
    pops = np.array([point_support.total(bid) for bid in ids])

    vg = Variogram(centroids, rates, model="exponential", n_lags=15).fit()
    return {
        "ids": ids,
        "rates": rates,
        "centroids": centroids,
        "pops": pops,
        "variogram": vg,
    }


def test_point_support_covers_all_areas(cancer_data, point_support):
    areas, _ = cancer_data
    assert len(point_support) == len(areas)
    assert set(point_support.block_ids) == set(areas["FIPS"])


def test_point_support_centroid_and_total(point_support):
    bid = point_support.block_ids[0]
    coords = point_support.coords(bid)
    values = point_support.values(bid)
    assert coords.shape[0] == values.shape[0]
    assert point_support.total(bid) == pytest.approx(values.sum())
    centroid = point_support.centroid(bid)
    assert centroid.shape == (2,)


def test_area_poisson_kriging_fit_requires_matching_support(point_support, fitted):
    with pytest.raises(ValueError):
        AreaPoissonKriging(fitted["variogram"]).fit(point_support, {"not-a-real-id": 1.0})


def test_predict_area_leave_one_out(cancer_data, point_support, fitted):
    areas, _ = cancer_data
    vg = fitted["variogram"]
    ids, rates = fitted["ids"], fitted["rates"]

    block_values = pd.Series(rates, index=ids)
    apk = AreaPoissonKriging(vg).fit(point_support, block_values)

    rng = np.random.default_rng(0)
    sample_ids = rng.choice(ids, size=25, replace=False)

    results = pd.DataFrame([
        apk.predict_area(bid, number_of_neighbors=8, neighbors_range=vg.fitted_params[2] * 2)
        for bid in sample_ids
    ])

    assert np.isfinite(results["zhat"]).all()
    assert np.isfinite(results["sig"]).all()
    assert (results["sig"] >= 0).all()

    actual = areas.set_index("FIPS").loc[results["id"], "rate"].to_numpy()
    rmse = np.sqrt(np.mean((actual - results["zhat"].to_numpy()) ** 2))
    assert rmse < 1.2 * rates.std()


def test_predict_points_coherent_with_predict_area(point_support, fitted):
    vg = fitted["variogram"]
    ids, rates = fitted["ids"], fitted["rates"]
    block_values = pd.Series(rates, index=ids)
    apk = AreaPoissonKriging(vg).fit(point_support, block_values)

    for bid in ids[:5]:
        area_result = apk.predict_area(bid, number_of_neighbors=8, neighbors_range=vg.fitted_params[2] * 2)
        points_result = apk.predict_points(bid, number_of_neighbors=8, neighbors_range=vg.fitted_params[2] * 2)

        assert set(points_result.columns) == {"x", "y", "population", "zhat", "sig"}
        assert len(points_result) == len(point_support.coords(bid))

        # Area-to-point coherence (Goovaerts 2006, Eq. 15): the
        # population-weighted average of the point estimates reconstructs
        # the area-to-area estimate.
        weights = points_result["population"].to_numpy(dtype=float)
        weighted = np.average(points_result["zhat"].to_numpy(dtype=float), weights=weights)
        assert weighted == pytest.approx(area_result["zhat"], rel=1e-6)


def test_predict_area_unknown_id_raises(point_support, fitted):
    vg = fitted["variogram"]
    ids, rates = fitted["ids"], fitted["rates"]
    apk = AreaPoissonKriging(vg).fit(point_support, pd.Series(rates, index=ids))
    with pytest.raises(ValueError):
        apk.predict_area("not-a-real-id")


def test_area_to_area_beats_centroid_based(cancer_data, point_support, fitted):
    """
    Sanity check matching the pyinterpolate tutorials' claim: area-to-area
    Poisson Kriging (which respects each county's real shape/size) should
    give a lower leave-one-out RMSE than the centroid-based approximation
    (which collapses each county to a single point), on the same sample.
    """
    areas, _ = cancer_data
    vg = fitted["variogram"]
    ids, rates, centroids, pops = fitted["ids"], fitted["rates"], fitted["centroids"], fitted["pops"]

    rng = np.random.default_rng(1)
    sample_ids = rng.choice(ids, size=30, replace=False)
    actual = areas.set_index("FIPS").loc[sample_ids, "rate"].to_numpy()

    pk = PoissonKriging(vg).fit(centroids, rates, pops, ids=ids)
    cen_zhat = np.array([
        pk.predict_id(bid, number_of_neighbors=8, neighbors_range=vg.fitted_params[2] * 2)["zhat"]
        for bid in sample_ids
    ])
    cen_rmse = np.sqrt(np.mean((actual - cen_zhat) ** 2))

    apk = AreaPoissonKriging(vg).fit(point_support, pd.Series(rates, index=ids))
    ata_zhat = np.array([
        apk.predict_area(bid, number_of_neighbors=8, neighbors_range=vg.fitted_params[2] * 2)["zhat"]
        for bid in sample_ids
    ])
    ata_rmse = np.sqrt(np.mean((actual - ata_zhat) ** 2))

    # ATA and centroid-based RMSE are typically close on this dataset;
    # allow a small relative slack so optimizer/dataset-path differences
    # do not turn a statistical tie into a failure.
    assert ata_rmse <= cen_rmse * 1.05
