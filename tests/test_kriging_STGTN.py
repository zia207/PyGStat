"""
Test for pygstat.kriging_STGTN (STGTNRegressionKriging, STTN, GMAN), using
data/CA_pm25_2025.csv (163 California PM2.5 monitoring stations, daily 2025
readings, aggregated here to monthly station means via
pygstat.krigeST.load_ca_pm25_panel) and data/CA_pm25_grid_predictions.csv
(the classical space-time Ordinary Kriging reference surface for the same
data, produced by scripts/predict_ca_pm25_grid.py via pygstat.krigeST) --
the same two files and reference target used to validate the recurrent
spatio-temporal GNNs in tests/test_kriging_STGNN.py, for a direct
apples-to-apples comparison between the two attention-based architectures
here and those GRU-based ones there.

Real-data numbers observed while developing this test (seed=0, 25 held-out
stations, matching pygstat.krigeST's own validation exactly): classical
STKriging gets RMSE=2.243 (R^2=0.577); STTN gets RMSE=2.100 (R^2=0.632);
GMAN gets RMSE=2.435 (R^2=0.505) -- both competitive with the classical
baseline and with kriging_STGNN.py's TGCN/DCRNN (RMSE 2.03/2.34). Full-grid
predictions correlate with the classical reference at r=0.72-0.92 per month
(mean ~0.85) for both architectures.

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

from pygstat.kriging_STGTN import (  # noqa: E402
    GMAN,
    GatedFusionBlock,
    GraphMultiHeadAttention,
    STGTNRegressionKriging,
    SpatialTemporalBlock,
    STTN,
    _knn_edges,
    _scatter_softmax,
    _sinusoidal_positional_encoding,
)
from pygstat.krigeST import load_ca_pm25_panel  # noqa: E402

CSV_PATH = "data/CA_pm25_2025.csv"
GRID_PATH = "data/CA_pm25_grid_predictions.csv"


def _load_panel():
    panel = load_ca_pm25_panel(CSV_PATH)
    wide = panel.pivot(index="site_id", columns="month", values="pm25").sort_index()
    coords = (
        panel[["site_id", "x", "y"]].drop_duplicates("site_id").set_index("site_id")
        .loc[wide.index][["x", "y"]].to_numpy()
    )
    return coords, wide.to_numpy()  # coords: (163,2); values: (163,12), NaN allowed


def _subsample(coords, values, n=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(coords), size=min(n, len(coords)), replace=False)
    return coords[idx], values[idx]


# ==========================================================
# Graph / attention utilities
# ==========================================================

class TestAttentionUtilities:
    def test_knn_edges_training_graph_shape(self):
        coords = np.random.rand(20, 2) * 100
        edge_index, edge_dist = _knn_edges(coords, k=5)
        assert edge_index.shape == (2, 100)
        assert edge_dist.shape == (100,)
        assert not (edge_index[0] == edge_index[1]).any()

    def test_knn_edges_inductive_extension_no_new_to_new(self):
        coords = np.random.rand(15, 2) * 100
        query = np.random.rand(6, 2) * 100
        edge_index, _ = _knn_edges(coords, k=4, query_coords=query)
        src, dst = edge_index
        n_train = len(coords)
        assert (dst >= n_train).all()
        assert (src < n_train).all()

    def test_scatter_softmax_sums_to_one_per_group(self):
        logits = torch.randn(12)
        index = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3])
        alpha = _scatter_softmax(logits, index, dim_size=4)
        sums = torch.zeros(4).index_add_(0, index, alpha)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_positional_encoding_shape_and_bounds(self):
        pe = _sinusoidal_positional_encoding(max_len=20, dim=8)
        assert pe.shape == (20, 8)
        assert (pe >= -1).all() and (pe <= 1).all()

    def test_graph_multihead_attention_output_shape(self):
        coords = np.random.rand(10, 2) * 10
        edge_index, _ = _knn_edges(coords, k=3)
        attn = GraphMultiHeadAttention(dim=8, n_heads=2)
        x = torch.randn(10, 8)
        out = attn(x, edge_index, n_nodes=10)
        assert out.shape == (10, 8)
        assert torch.isfinite(out).all()

    def test_graph_multihead_attention_rejects_indivisible_dim(self):
        with pytest.raises(ValueError, match="divisible"):
            GraphMultiHeadAttention(dim=7, n_heads=2)


class TestSpatioTemporalBlocks:
    @pytest.mark.parametrize("Block", [SpatialTemporalBlock, GatedFusionBlock])
    def test_block_forward_shape(self, Block):
        coords = np.random.rand(8, 2) * 10
        edge_index, _ = _knn_edges(coords, k=3)
        block = Block(dim=8, n_heads=2, dropout=0.0)
        h = torch.randn(5, 8, 8)  # (T=5, n_nodes=8, dim=8)
        out = block(h, edge_index, n_nodes=8)
        assert out.shape == h.shape
        assert torch.isfinite(out).all()


# ==========================================================
# Fit / predict behavior (small subsample -- fast)
# ==========================================================

class TestFitPredict:
    @pytest.mark.parametrize("cls", [STTN, GMAN])
    def test_fit_predict_shapes_and_finiteness(self, cls):
        coords, values = _load_panel()
        coords, values = _subsample(coords, values, n=40)
        model = cls(k_neighbors=6, hidden_dim=8, n_heads=2, n_layers=2,
                     max_epochs=15, patience=5, device="cpu", seed=0)
        model.fit(coords, values, verbose=False)
        assert model.is_fitted_

        coords_pred = np.random.default_rng(1).uniform(coords.min(0), coords.max(0), size=(5, 2))
        pred = model.predict(coords_pred)
        assert pred.shape == (5, values.shape[1])
        assert np.isfinite(pred).all()

        pred2, std = model.predict(coords_pred, return_std=True)
        assert pred2.shape == std.shape == pred.shape
        assert (std >= 0).all()
        np.testing.assert_allclose(pred2, pred)  # deterministic in eval mode

    def test_general_engine_accepts_block_type_kwarg(self):
        coords, values = _load_panel()
        coords, values = _subsample(coords, values, n=30)
        model = STGTNRegressionKriging(block_type="gman", k_neighbors=5, hidden_dim=8, n_heads=2,
                                        max_epochs=10, patience=5, device="cpu", seed=0)
        model.fit(coords, values, verbose=False)
        assert "gman" in repr(model)
        pred = model.predict(coords[:3])
        assert pred.shape == (3, values.shape[1])

    def test_invalid_block_type_raises(self):
        with pytest.raises(ValueError, match="block_type"):
            STGTNRegressionKriging(block_type="not_a_real_block")

    def test_hidden_dim_not_divisible_by_heads_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            STGTNRegressionKriging(hidden_dim=10, n_heads=3)

    def test_predict_before_fit_raises(self):
        model = STTN()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.zeros((2, 2)))

    def test_handles_missing_station_months(self):
        """21/1956 station-months in the real panel are genuinely missing
        (NaN) -- confirm fit/predict tolerate that without special-casing
        by the caller."""
        coords, values = _load_panel()
        assert np.isnan(values).sum() > 0  # sanity: the real data does have gaps
        coords, values = _subsample(coords, values, n=50)
        model = STTN(k_neighbors=6, hidden_dim=8, n_heads=2, max_epochs=15, patience=5, device="cpu", seed=0)
        model.fit(coords, values, verbose=False)
        pred = model.predict(coords[:5])
        assert np.isfinite(pred).all()

    def test_krige_residuals_false_skips_kriging_step(self):
        coords, values = _load_panel()
        coords, values = _subsample(coords, values, n=30)
        model = STTN(k_neighbors=5, hidden_dim=8, n_heads=2, max_epochs=10, patience=5,
                     device="cpu", seed=0, krige_residuals=False)
        model.fit(coords, values, verbose=False)
        assert model.krige_by_step_ == {}
        pred, std = model.predict(coords[:3], return_std=True)
        assert np.all(std == 0)  # trend-only, no residual-kriging contribution


# ==========================================================
# Real-data validation: held-out stations
# ==========================================================

class TestHeldOutStationValidation:
    """Mirrors pygstat.krigeST's own held-out-station validation exactly
    (same seed, same n_test_sites), for a direct, apples-to-apples
    comparison against the classical space-time Ordinary Kriging baseline
    and against kriging_STGNN.py's recurrent architectures."""

    @pytest.mark.parametrize("cls,name", [(STTN, "STTN"), (GMAN, "GMAN")])
    def test_beats_naive_mean_baseline_on_held_out_stations(self, cls, name):
        coords, values = _load_panel()
        rng = np.random.default_rng(0)
        counts = (~np.isnan(values)).sum(axis=1)
        complete = np.flatnonzero(counts == 12)
        test_idx = rng.choice(complete, size=25, replace=False)
        train_mask = np.ones(len(values), dtype=bool)
        train_mask[test_idx] = False

        model = cls(k_neighbors=8, hidden_dim=32, n_heads=4, n_layers=2,
                     max_epochs=300, patience=30, device="cpu", seed=0)
        model.fit(coords[train_mask], values[train_mask], verbose=False)
        pred = model.predict(coords[test_idx])

        obs_mask = ~np.isnan(values[test_idx])
        resid = values[test_idx][obs_mask] - pred[obs_mask]
        rmse = np.sqrt(np.mean(resid ** 2))
        train_mean = np.nanmean(values[train_mask])
        baseline_rmse = np.sqrt(np.mean((values[test_idx][obs_mask] - train_mean) ** 2))

        print(f"\n{name} held-out-station RMSE={rmse:.3f} vs naive-mean baseline={baseline_rmse:.3f}")
        assert rmse < 0.85 * baseline_rmse  # a real, comfortable margin (observed ~0.6x-0.7x)


# ==========================================================
# Real-data validation: full grid vs. the classical reference
# ==========================================================

class TestFullDatasetAndGridEndToEnd:
    @pytest.mark.parametrize("cls,name", [(STTN, "STTN"), (GMAN, "GMAN")])
    def test_grid_predictions_correlate_with_classical_stkriging(self, cls, name):
        coords, values = _load_panel()
        grid = pd.read_csv(GRID_PATH)
        grid_xy = grid[grid["month"] == 0][["x", "y"]].to_numpy()

        model = cls(k_neighbors=8, hidden_dim=32, n_heads=4, n_layers=2,
                     max_epochs=300, patience=30, device="cpu", seed=0)
        model.fit(coords, values, verbose=False)
        pred = model.predict(grid_xy)
        assert pred.shape == (len(grid_xy), 12)
        assert np.isfinite(pred).all()

        corrs = []
        for month in range(12):
            ref = grid.loc[grid["month"] == month, "pm25_pred"].to_numpy()
            corrs.append(np.corrcoef(ref, pred[:, month])[0, 1])
        mean_corr = float(np.mean(corrs))
        print(f"\n{name} vs classical STKriging reference: per-month corr = "
              f"{[f'{c:.2f}' for c in corrs]}, mean={mean_corr:.3f}")
        assert min(corrs) > 0.4
        assert mean_corr > 0.6

        assert pred.mean() > 0
        assert pred.max() < 200  # well above the observed data's max with headroom


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
