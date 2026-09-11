from pathlib import Path

import pytest
import pandas as pd
import numpy as np
from pygstat import Variogram, OrdinaryKriging
from pygstat.validation import loo_cv

MEUSE_CSV = Path(__file__).resolve().parent.parent / "data" / "meuse.csv"

@pytest.fixture
def meuse():
    # Local sample data (was previously fetched from a now-dead external URL,
    # which made this fixture -- and every test using it -- fail outright).
    df = pd.read_csv(MEUSE_CSV)
    return df[['x','y']].values, np.log(df['zinc'].values)

def test_variogram_fit(meuse):
    coords, vals = meuse
    vg = Variogram(coords, vals, model='spherical')
    vg.fit()
    assert len(vg.fitted_params) == 3

def test_kriging_predict(meuse):
    coords, vals = meuse
    vg = Variogram(coords, vals, model='exponential')
    vg.fit()
    ok = OrdinaryKriging(vg)
    ok.fit(coords, vals)
    pred, se = ok.predict(coords[:3], return_variance=True)
    assert pred.shape == (3,)
    assert se.shape == (3,)

def test_cv(meuse):
    coords, vals = meuse
    vg = Variogram(coords, vals)
    vg.fit()
    ok = OrdinaryKriging(vg)
    scores = loo_cv(ok, coords, vals)
    assert 0 <= scores['r2'] <= 1