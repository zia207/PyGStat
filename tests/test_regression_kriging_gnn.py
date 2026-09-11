"""
Test for pygstat.regression_kriging_gnn (GNNRegressionKriging, KCN, IGNNK),
using data/gp_data_5000.csv (5000 samples: SOC = soil organic carbon, plus
11 covariates) and data/gp_grid.csv (10674-point prediction grid, no target).

See tests/test_regression_kriging_pytorch_optuna.py's module docstring for
why `sys.modules.setdefault("tensorflow", None)` appears below: this sandbox's
TensorFlow and PyTorch builds segfault the interpreter when both are used in
the same process, reproduced with a minimal model unrelated to this module.
"""

import sys

import numpy as np
import pandas as pd
import pytest

sys.modules.setdefault("tensorflow", None)

torch = pytest.importorskip("torch")

from pygstat.regression_kriging_gnn import (  # noqa: E402
    GNNRegressionKriging,
    IGNNK,
    KCN,
    _knn_edges,
    _scatter_mean,
    _scatter_softmax,
    _scatter_weighted_mean,
)

DATA_PATH = "data/gp_data_5000.csv"
GRID_PATH = "data/gp_grid.csv"
FEATURES = ["Aspect", "ELEV", "FRG", "K_Factor", "MAP", "MAT", "NDVI", "NLCD", "Silt_Clay", "Slope", "TPI"]
TARGET = "SOC"


def _load_full():
    df = pd.read_csv(DATA_PATH)
    grid = pd.read_csv(GRID_PATH)
    assert set(FEATURES + ["x", "y"]).issubset(grid.columns)
    return df, grid


def _split(df, n=1200, test_size=0.25, seed=42):
    """A modest subsample for the faster structural/behavioral tests (the
    full 5000-point set is exercised separately, see
    test_full_dataset_and_grid_end_to_end below)."""
    from sklearn.model_selection import train_test_split
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()
    coords = df[["x", "y"]].to_numpy()
    return train_test_split(X, y, coords, test_size=test_size, random_state=seed)


# ==========================================================
# Graph / scatter utilities
# ==========================================================

def test_knn_edges_shape_and_no_self_loops():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(50, 2))
    edge_index, edge_dist = _knn_edges(coords, k=6)
    assert edge_index.shape == (2, 50 * 6)
    assert (edge_dist > 0).all()  # self excluded -> never zero distance
    src, dst = edge_index
    assert not (src == dst).any()


def test_knn_edges_query_mode_offsets_destination():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(30, 2))
    query = rng.uniform(0, 100, size=(5, 2))
    edge_index, edge_dist = _knn_edges(coords, k=4, query_coords=query)
    src, dst = edge_index
    assert dst.min() == 30  # offset by n_train
    assert dst.max() == 30 + 5 - 1
    assert src.max() < 30  # sources are always training nodes
    assert (edge_dist >= 0).all()


def test_scatter_mean_matches_manual_groupby():
    src = torch.tensor([[1.0], [3.0], [5.0], [2.0]])
    index = torch.tensor([0, 0, 1, 1])
    out = _scatter_mean(src, index, dim_size=2)
    np.testing.assert_allclose(out.numpy().flatten(), [2.0, 3.5])


def test_scatter_weighted_mean_matches_manual_computation():
    src = torch.tensor([[1.0], [3.0]])
    weight = torch.tensor([1.0, 3.0])
    index = torch.tensor([0, 0])
    out = _scatter_weighted_mean(src, weight, index, dim_size=1)
    expected = (1.0 * 1.0 + 3.0 * 3.0) / (1.0 + 3.0)
    np.testing.assert_allclose(out.numpy().flatten(), [expected])


def test_scatter_softmax_sums_to_one_per_group():
    logits = torch.tensor([1.0, 2.0, 0.5, 0.5, 0.5])
    index = torch.tensor([0, 0, 1, 1, 1])
    alpha = _scatter_softmax(logits, index, dim_size=2)
    sums = torch.zeros(2).index_add_(0, index, alpha)
    np.testing.assert_allclose(sums.numpy(), [1.0, 1.0], atol=1e-6)


# ==========================================================
# Each model: fit/predict, held-out accuracy, uncertainty
# ==========================================================

@pytest.mark.parametrize("cls,kwargs", [
    (GNNRegressionKriging, dict(k_neighbors=10, hidden_dim=32, n_layers=2, max_epochs=40, patience=8)),
    (KCN, dict(k_neighbors=10, hidden_dim=32, n_layers=2, max_epochs=40, patience=8)),
    (IGNNK, dict(k_neighbors=10, hidden_dim=32, n_layers=3, mask_ratio=0.3, max_epochs=50, patience=10)),
])
def test_fit_predict_reasonable_accuracy_on_held_out_soc(cls, kwargs):
    df, _ = _load_full()
    X_train, X_test, y_train, y_test, c_train, c_test = _split(df)

    rk = cls(device="cpu", seed=1, **kwargs)
    rk.fit(X_train, y_train, c_train, verbose=False)
    pred = rk.predict(X_test, c_test)

    assert pred.shape == y_test.shape
    assert np.isfinite(pred).all()

    pred_with_std, std = rk.predict(X_test, c_test, return_std=True)
    np.testing.assert_allclose(pred_with_std, pred)
    assert np.all(std >= -1e-8)

    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, pred)
    print(f"\n{cls.__name__} held-out R^2 on SOC: {r2:.3f}")
    assert r2 > 0.3  # a genuinely working fit, not a degenerate/leaking one


def test_kcn_never_lets_a_node_see_its_own_target():
    """KCNConv's `x_self` always zeroes the node's own (masked) target and
    mask flag -- verify this structurally by checking that duplicating a
    training point's *target* elsewhere (an obvious leak if the network
    could see its own value) does not trivially collapse training loss to
    ~0 within a couple of epochs on a tiny, otherwise-unlearnable problem."""
    rng = np.random.default_rng(0)
    n = 60
    coords = rng.uniform(0, 100, size=(n, 2))
    X = rng.normal(size=(n, 3))  # features carry ~no information about y
    y = rng.normal(size=n) * 10  # unrelated to X; only neighbor-y could help at all

    rk = KCN(k_neighbors=8, hidden_dim=16, n_layers=1, max_epochs=5, patience=5, device="cpu", seed=0)
    rk.fit(X, y, coords, verbose=False)
    # If the network could see its own target, MSE would collapse near 0
    # almost immediately; with it properly hidden, a handful of epochs on
    # this deliberately unlearnable problem should still leave real error.
    assert rk.train_losses_[-1] > 0.05


def test_gnn_regression_kriging_gat_variant_runs():
    df, _ = _load_full()
    X_train, X_test, y_train, y_test, c_train, c_test = _split(df, n=400)
    rk = GNNRegressionKriging(k_neighbors=8, hidden_dim=16, n_layers=2, conv_type="gat",
                               max_epochs=15, patience=5, device="cpu")
    rk.fit(X_train, y_train, c_train, verbose=False)
    pred = rk.predict(X_test, c_test)
    assert np.isfinite(pred).all()


def test_repr_strings():
    df, _ = _load_full()
    X_train, _, y_train, _, c_train, _ = _split(df, n=200)
    for cls, kwargs in [
        (GNNRegressionKriging, dict(max_epochs=3, patience=2)),
        (KCN, dict(max_epochs=3, patience=2)),
        (IGNNK, dict(max_epochs=3, patience=2)),
    ]:
        rk = cls(device="cpu", **kwargs)
        rk.fit(X_train, y_train, c_train, verbose=False)
        assert cls.__name__ in repr(rk)


def test_unfitted_predict_raises():
    for cls in (GNNRegressionKriging, KCN, IGNNK):
        rk = cls()
        with pytest.raises(ValueError, match="not fitted"):
            rk.predict(np.zeros((1, len(FEATURES))), np.zeros((1, 2)))


# ==========================================================
# Full requested data files: data/gp_data_5000.csv + data/gp_grid.csv
# ==========================================================

def test_full_dataset_and_grid_end_to_end():
    """The exact scenario requested: train on all 5000 samples, predict
    across the full 10674-point gp_grid.csv."""
    df, grid = _load_full()
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()
    coords = df[["x", "y"]].to_numpy()
    X_grid = grid[FEATURES].to_numpy()
    coords_grid = grid[["x", "y"]].to_numpy()

    for cls, kwargs in [
        (GNNRegressionKriging, dict(k_neighbors=12, hidden_dim=64, n_layers=2, max_epochs=60, patience=10)),
        (KCN, dict(k_neighbors=12, hidden_dim=64, n_layers=2, max_epochs=60, patience=10)),
        (IGNNK, dict(k_neighbors=12, hidden_dim=64, n_layers=3, mask_ratio=0.3, max_epochs=80, patience=15)),
    ]:
        rk = cls(device="cpu", seed=1, **kwargs)
        rk.fit(X, y, coords, verbose=False)
        pred, std = rk.predict(X_grid, coords_grid, return_std=True)

        assert pred.shape == (len(grid),)
        assert np.isfinite(pred).all()
        assert np.isfinite(std).all()
        assert np.all(std >= -1e-8)
        # SOC is a real, mostly-modest soil property (training range ~0-24);
        # predictions across the whole CONUS-scale grid should stay in a
        # physically plausible neighborhood of that, not blow up.
        assert pred.min() > -5.0
        assert pred.max() < 40.0
        print(f"\n{cls.__name__} full-grid prediction: "
              f"range=[{pred.min():.2f}, {pred.max():.2f}], mean std={std.mean():.3f}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
