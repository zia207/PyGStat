"""
Test for pygstat.soft_kriging (Markov-Bayes soft/indicator kriging) against the
Jura geochemical data set (data/jura_data.csv).

Scenario: only a sparse 20% subset of the 359 samples has an expensive, exact
Cd assay ("hard" data); the cheap, densely-available Zn assay (measured at all
359 locations) is calibrated into a "soft" exceedance probability via logistic
regression and combined with the sparse hard data through Markov-Bayes kriging
(Zhu & Journel, 1993). We validate against the 80% of samples held out from the
hard set.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from pygstat.soft_kriging import (
    SoftKriging,
    fit_soft_probability_model,
    indicator_transform,
    markov_bayes_calibrate,
)

DATA_PATH = "data/jura_data.csv"
THRESHOLD = 1.0  # mg/kg Cd exceedance cutoff, consistent with Tutorial 04


def _load_jura_split(test_size=0.80, random_state=42):
    df = pd.read_csv(DATA_PATH, index_col=0)
    coords = df[["Xloc", "Yloc"]].to_numpy()
    indicator = indicator_transform(df["Cd"].to_numpy(), THRESHOLD, exceed=True)
    hard_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_size, random_state=random_state, stratify=indicator
    )
    return df, coords, indicator, hard_idx, test_idx


def _fit_exponential_indicator_variogram(coords, indicator, n_lags=10):
    d = pdist(coords)
    diffsq = pdist(indicator[:, None]) ** 2
    maxlag = d.max() * 0.5
    bins = np.linspace(0, maxlag, n_lags + 1)
    lag_h, lag_g = [], []
    for i in range(n_lags):
        m = (d >= bins[i]) & (d < bins[i + 1])
        if m.sum() > 5:
            lag_h.append(d[m].mean())
            lag_g.append(0.5 * diffsq[m].mean())
    lag_h, lag_g = np.array(lag_h), np.array(lag_g)

    def expo(h, nugget, sill, rng):
        return nugget + sill * (1.0 - np.exp(-h / rng))

    popt, _ = curve_fit(expo, lag_h, lag_g, p0=[0.05, indicator.var(), maxlag / 3],
                         bounds=(0, [1.0, 1.0, maxlag * 3]))
    return tuple(popt)  # nugget, sill, range


def test_read_jura_data():
    df, *_ = _load_jura_split()
    assert len(df) == 359
    assert {"Xloc", "Yloc", "Cd", "Zn"}.issubset(df.columns)


def test_indicator_transform_directions():
    x = np.array([0.5, 1.0, 1.5, 2.0])
    assert list(indicator_transform(x, 1.0, exceed=True)) == [0.0, 1.0, 1.0, 1.0]
    assert list(indicator_transform(x, 1.0, exceed=False)) == [1.0, 1.0, 0.0, 0.0]


def test_markov_bayes_calibrate_recovers_known_correlation():
    df, coords, indicator, hard_idx, test_idx = _load_jura_split()
    hard_coords, hard_ind = coords[hard_idx], indicator[hard_idx]
    zn_hard = df["Zn"].to_numpy()[hard_idx]

    _, soft_prob_hard = fit_soft_probability_model(zn_hard, hard_ind, proxy_values_pred=zn_hard)
    B, corr, n_pairs = markov_bayes_calibrate(hard_coords, hard_ind, hard_coords, soft_prob_hard)

    assert n_pairs == len(hard_idx)
    assert 0.3 < corr < 1.0  # Zn is a genuinely informative (not perfect) proxy for Cd here
    assert B > 0


def test_soft_kriging_predict_returns_valid_probabilities():
    df, coords, indicator, hard_idx, test_idx = _load_jura_split()
    hard_coords, hard_ind = coords[hard_idx], indicator[hard_idx]
    test_coords = coords[test_idx]
    zn_hard = df["Zn"].to_numpy()[hard_idx]
    zn_test = df["Zn"].to_numpy()[test_idx]

    model, soft_prob_hard = fit_soft_probability_model(zn_hard, hard_ind, proxy_values_pred=zn_hard)
    soft_prob_test = model.predict_proba(zn_test.reshape(-1, 1))[:, 1]
    B, _, _ = markov_bayes_calibrate(hard_coords, hard_ind, hard_coords, soft_prob_hard)
    nugget, sill, rng = _fit_exponential_indicator_variogram(hard_coords, hard_ind)

    sk = SoftKriging(cov_model="exponential", max_neighbors_hard=15, max_neighbors_soft=15)
    sk.fit(hard_coords, hard_ind, test_coords, soft_prob_test, B=B)
    prob, var = sk.predict(test_coords, sill=sill, range_=rng, nugget=nugget, return_variance=True)

    valid = ~np.isnan(prob)
    assert valid.sum() > 0.95 * len(test_idx)
    assert np.nanmin(prob) >= 0.0 and np.nanmax(prob) <= 1.0
    assert np.nanmin(var[valid]) >= -1e-8


def test_soft_kriging_beats_hard_only_indicator_kriging():
    """The core claim this module exists to demonstrate: with only 20% of the
    samples assayed for Cd, folding in a densely-available correlated proxy
    (Zn) via Markov-Bayes kriging should score well above plain indicator
    kriging from the sparse hard data alone."""
    df, coords, indicator, hard_idx, test_idx = _load_jura_split()
    hard_coords, hard_ind = coords[hard_idx], indicator[hard_idx]
    test_coords, test_ind = coords[test_idx], indicator[test_idx]
    zn_hard = df["Zn"].to_numpy()[hard_idx]
    zn_test = df["Zn"].to_numpy()[test_idx]

    model, soft_prob_hard = fit_soft_probability_model(zn_hard, hard_ind, proxy_values_pred=zn_hard)
    soft_prob_test = model.predict_proba(zn_test.reshape(-1, 1))[:, 1]
    B, _, _ = markov_bayes_calibrate(hard_coords, hard_ind, hard_coords, soft_prob_hard)
    nugget, sill, rng = _fit_exponential_indicator_variogram(hard_coords, hard_ind)

    sk_hard = SoftKriging(cov_model="exponential", max_neighbors_hard=25, max_neighbors_soft=0)
    sk_hard.fit(hard_coords, hard_ind, test_coords, soft_prob_test, B=0.0)
    prob_hard = sk_hard.predict(test_coords, sill=sill, range_=rng, nugget=nugget)

    sk_soft = SoftKriging(cov_model="exponential", max_neighbors_hard=15, max_neighbors_soft=15)
    sk_soft.fit(hard_coords, hard_ind, test_coords, soft_prob_test, B=B)
    prob_soft = sk_soft.predict(test_coords, sill=sill, range_=rng, nugget=nugget)

    v_hard = ~np.isnan(prob_hard)
    v_soft = ~np.isnan(prob_soft)
    auc_hard = roc_auc_score(test_ind[v_hard], prob_hard[v_hard])
    auc_soft = roc_auc_score(test_ind[v_soft], prob_soft[v_soft])
    brier_hard = brier_score_loss(test_ind[v_hard], prob_hard[v_hard])
    brier_soft = brier_score_loss(test_ind[v_soft], prob_soft[v_soft])

    print(f"\nJura Cd>{THRESHOLD} exceedance, n_hard={len(hard_idx)}, n_test={len(test_idx)}")
    print(f"  hard-only indicator kriging : AUC={auc_hard:.3f}  Brier={brier_hard:.4f}")
    print(f"  Markov-Bayes soft kriging   : AUC={auc_soft:.3f}  Brier={brier_soft:.4f}")

    assert auc_soft > auc_hard
    assert brier_soft < brier_hard
    assert auc_soft > 0.75  # a clearly-informative result on held-out data


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
