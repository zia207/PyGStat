"""
Test for pygstat.kriging_STGNN (STGNNRegressionKriging, TGCN,
DCRNN), using data/CA_pm25_2025.csv (163 California PM2.5 monitoring
stations, daily 2025 readings, aggregated here to monthly station means via
pygstat.krigeST.load_ca_pm25_panel) and data/CA_pm25_grid_predictions.csv
(the classical space-time Ordinary Kriging reference surface for the same
data, produced by scripts/predict_ca_pm25_grid.py via pygstat.krigeST).

Real-data numbers observed while developing this test (seed=0, 25 held-out
stations, matching pygstat.krigeST's own validation exactly): classical
STKriging gets RMSE=2.243 (R^2=0.577); TGCN gets RMSE=2.032 (R^2=0.655,
*better* than the classical baseline here); DCRNN gets RMSE=2.339 (R^2=0.543,
competitive). Full-grid predictions correlate with the classical reference
at r=0.66-0.99 per month (mean ~0.82-0.86) -- strong agreement between a
from-scratch deep spatio-temporal model and an independently-implemented
classical geostatistical one on the same real network.

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

from pygstat.kriging_STGNN import (  # noqa: E402
    DCRNN,
    STGNNRegressionKriging,
    TGCN,
    _DiffusionConv,
    _GraphConv,
    _knn_edges,
    _scatter_mean,
    DCRNNCell,
    TGCNCell,
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
# Graph / layer utilities
# ==========================================================

class TestGraphUtilities:
    def test_knn_edges_training_graph_shape(self):
        coords = np.random.rand(20, 2) * 100
        edge_index, edge_dist = _knn_edges(coords, k=5)
        assert edge_index.shape == (2, 100)
        assert edge_dist.shape == (100,)
        assert (edge_dist >= 0).all()
        # no self-loops
        assert not (edge_index[0] == edge_index[1]).any()

    def test_knn_edges_inductive_extension_no_new_to_new(self):
        coords = np.random.rand(15, 2) * 100
        query = np.random.rand(6, 2) * 100
        edge_index, _ = _knn_edges(coords, k=4, query_coords=query)
        src, dst = edge_index
        n_train = len(coords)
        assert (dst >= n_train).all()   # destinations are all query nodes
        assert (src < n_train).all()    # sources are all training nodes -- no new-to-new edges

    def test_scatter_mean_basic(self):
        src = torch.tensor([[1.0], [3.0], [5.0]])
        index = torch.tensor([0, 0, 1])
        out = _scatter_mean(src, index, dim_size=2)
        assert torch.allclose(out, torch.tensor([[2.0], [5.0]]))

    def test_graph_conv_output_shape(self):
        coords = np.random.rand(10, 2) * 10
        edge_index, _ = _knn_edges(coords, k=3)
        conv = _GraphConv(in_dim=4, out_dim=6)
        x = torch.randn(10, 4)
        out = conv(x, edge_index, n_nodes=10)
        assert out.shape == (10, 6)

    def test_diffusion_conv_output_shape_and_hops(self):
        coords = np.random.rand(10, 2) * 10
        edge_index, _ = _knn_edges(coords, k=3)
        edge_index_rev = edge_index.flip(0)
        conv = _DiffusionConv(in_dim=4, out_dim=6, k_hops=3)
        assert len(conv.lin_f) == 3 and len(conv.lin_b) == 3
        x = torch.randn(10, 4)
        out = conv(x, edge_index, edge_index_rev, n_nodes=10)
        assert out.shape == (10, 6)


class TestRecurrentCells:
    def test_tgcn_cell_forward_shape(self):
        coords = np.random.rand(8, 2) * 10
        edge_index, _ = _knn_edges(coords, k=3)
        cell = TGCNCell(in_dim=2, hidden_dim=5)
        x, h = torch.randn(8, 2), torch.zeros(8, 5)
        h_new = cell(x, h, edge_index, n_nodes=8)
        assert h_new.shape == (8, 5)
        assert torch.isfinite(h_new).all()

    def test_dcrnn_cell_forward_shape(self):
        coords = np.random.rand(8, 2) * 10
        edge_index, _ = _knn_edges(coords, k=3)
        edge_index_rev = edge_index.flip(0)
        cell = DCRNNCell(in_dim=2, hidden_dim=5, k_hops=2)
        x, h = torch.randn(8, 2), torch.zeros(8, 5)
        h_new = cell(x, h, edge_index, edge_index_rev, n_nodes=8)
        assert h_new.shape == (8, 5)
        assert torch.isfinite(h_new).all()


# ==========================================================
# Fit / predict behavior (small subsample -- fast)
# ==========================================================

class TestFitPredict:
    @pytest.mark.parametrize("cls", [TGCN, DCRNN])
    def test_fit_predict_shapes_and_finiteness(self, cls):
        coords, values = _load_panel()
        coords, values = _subsample(coords, values, n=40)
        model = cls(k_neighbors=6, hidden_dim=8, max_epochs=15, patience=5, device="cpu", seed=0)
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

    def test_general_engine_accepts_cell_type_kwarg(self):
        coords, values = _load_panel()
        coords, values = _subsample(coords, values, n=30)
        model = STGNNRegressionKriging(cell_type="dcrnn", k_neighbors=5, hidden_dim=8,
                                        max_epochs=10, patience=5, device="cpu", seed=0)
        model.fit(coords, values, verbose=False)
        assert "dcrnn" in repr(model)
        pred = model.predict(coords[:3])
        assert pred.shape == (3, values.shape[1])

    def test_invalid_cell_type_raises(self):
        with pytest.raises(ValueError, match="cell_type"):
            STGNNRegressionKriging(cell_type="not_a_real_cell")

    def test_predict_before_fit_raises(self):
        model = TGCN()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.zeros((2, 2)))

    def test_handles_missing_station_months(self):
        """21/1956 station-months in the real panel are genuinely missing
        (NaN) -- confirm fit/predict tolerate that without special-casing
        by the caller."""
        coords, values = _load_panel()
        assert np.isnan(values).sum() > 0  # sanity: the real data does have gaps
        coords, values = _subsample(coords, values, n=50)
        model = TGCN(k_neighbors=6, hidden_dim=8, max_epochs=15, patience=5, device="cpu", seed=0)
        model.fit(coords, values, verbose=False)
        pred = model.predict(coords[:5])
        assert np.isfinite(pred).all()

    def test_krige_residuals_false_skips_kriging_step(self):
        coords, values = _load_panel()
        coords, values = _subsample(coords, values, n=30)
        model = TGCN(k_neighbors=5, hidden_dim=8, max_epochs=10, patience=5,
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
    (same seed, same n_test_sites) for a direct, apples-to-apples comparison
    against the classical space-time Ordinary Kriging baseline."""

    @pytest.mark.parametrize("cls,name", [(TGCN, "TGCN"), (DCRNN, "DCRNN")])
    def test_beats_naive_mean_baseline_on_held_out_stations(self, cls, name):
        coords, values = _load_panel()
        rng = np.random.default_rng(0)
        counts = (~np.isnan(values)).sum(axis=1)
        complete = np.flatnonzero(counts == 12)
        test_idx = rng.choice(complete, size=25, replace=False)
        train_mask = np.ones(len(values), dtype=bool)
        train_mask[test_idx] = False

        model = cls(k_neighbors=8, hidden_dim=32, max_epochs=300, patience=30, device="cpu", seed=0)
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
    @pytest.mark.parametrize("cls,name", [(TGCN, "TGCN"), (DCRNN, "DCRNN")])
    def test_grid_predictions_correlate_with_classical_stkriging(self, cls, name):
        coords, values = _load_panel()
        grid = pd.read_csv(GRID_PATH)
        grid_xy = grid[grid["month"] == 0][["x", "y"]].to_numpy()

        model = cls(k_neighbors=8, hidden_dim=32, max_epochs=300, patience=30, device="cpu", seed=0)
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

        # Physically sane range (a real air-quality network, not degenerate output)
        assert pred.mean() > 0
        assert pred.max() < 200  # well above the observed data's max (~50 monthly mean) with headroom


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
