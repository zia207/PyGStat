"""
Test for pygstat.kriging_CNNLSTM (CNNLSTMRegressionKriging, ConvLSTM), using
data/CA_pm25_2025.csv (163 California PM2.5 monitoring stations, daily 2025
readings, aggregated here to monthly station means via
pygstat.krigeST.load_ca_pm25_panel) and data/CA_pm25_grid_predictions.csv
(the classical space-time Ordinary Kriging reference surface for the same
data, produced by scripts/predict_ca_pm25_grid.py via pygstat.krigeST) --
the same two files and reference target used to validate the graph-based
spatio-temporal models in tests/test_kriging_STGNN.py and
tests/test_kriging_STGTN.py, for a direct comparison between this
raster/CNN-based architecture and those graph-based ones.

Real-data numbers observed while developing this test (seed=0, 25 held-out
stations, matching pygstat.krigeST's own validation exactly, grid_size=32):
classical STKriging gets RMSE=2.243 (R^2=0.577); ConvLSTM gets RMSE as low
as 2.03 (R^2=0.655) -- competitive with (in some runs, matching almost
exactly) TGCN's 2.03/0.655 from kriging_STGNN.py; exact numbers vary a
little run to run, since neither this nor the sibling spatio-temporal
modules fix PyTorch's global weight-init RNG. Full-grid predictions
correlate with the classical reference at roughly r=0.6-0.9 per month
(mean ~0.74-0.77) -- somewhat below the graph-based models' ~0.82-0.88, a
real and expected consequence of first rasterizing the sparse 163-station
network onto a coarse 32-cell-per-axis image before the CNN-LSTM ever
sees it.

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

from pygstat.kriging_CNNLSTM import (  # noqa: E402
    CNNLSTMRegressionKriging,
    ConvLSTM,
    ConvLSTMCell,
    _cell_index,
    _make_grid_shape,
    _norm_grid_coords,
    _rasterize,
    _sample_raster_at_points,
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
# Rasterization utilities
# ==========================================================

class TestRasterUtilities:
    def test_make_grid_shape_preserves_aspect_ratio(self):
        # California-like: much taller (N-S) than wide (E-W)
        bbox = (0.0, 100.0, 0.0, 300.0)
        H, W = _make_grid_shape(bbox, grid_size=30)
        assert H == 30
        assert W == 10  # 100/300 * 30

    def test_make_grid_shape_square_bbox(self):
        H, W = _make_grid_shape((0.0, 50.0, 0.0, 50.0), grid_size=16)
        assert H == W == 16

    def test_cell_index_within_bounds(self):
        coords = np.random.rand(30, 2) * 100
        bbox = (0.0, 100.0, 0.0, 100.0)
        idx = _cell_index(coords, bbox, H=10, W=10)
        assert idx.min() >= 0
        assert idx.max() < 100

    def test_norm_grid_coords_extremes_map_near_unit_box(self):
        bbox = (0.0, 100.0, 0.0, 100.0)
        coords = np.array([[0.0, 0.0], [100.0, 100.0], [50.0, 50.0]])
        g = _norm_grid_coords(coords, bbox, H=20, W=20)
        assert np.allclose(g[0], [-1.0, -1.0], atol=1e-6)
        assert np.allclose(g[1], [1.0, 1.0], atol=1e-6)
        assert np.allclose(g[2], [0.0, 0.0], atol=1e-6)

    def test_rasterize_shape_and_masking(self):
        coords = np.array([[10.0, 10.0], [90.0, 90.0], [10.0, 12.0]])  # first & third share a cell
        bbox = (0.0, 100.0, 0.0, 100.0)
        H, W = 10, 10
        cell_idx = _cell_index(coords, bbox, H, W)
        values_scaled = np.array([[1.0, np.nan], [2.0, 2.0], [3.0, 3.0]])
        mask = ~np.isnan(values_scaled)
        values_scaled = np.nan_to_num(values_scaled)
        img = _rasterize(values_scaled, mask, cell_idx, H, W)
        assert img.shape == (2, 2, H, W)
        # station 0 and 2 share a cell at t=0: averaged value = (1+3)/2 = 2
        r, c = divmod(cell_idx[0], W)
        assert np.isclose(img[0, 0, r, c], 2.0)
        assert img[0, 1, r, c] == 1.0  # mask channel: observed

    def test_sample_raster_at_points_recovers_pixel_values_at_centers(self):
        H, W = 6, 6
        raster = torch.arange(H * W, dtype=torch.float32).reshape(1, 1, H, W)
        raster = raster.expand(3, -1, -1, -1).contiguous()  # (T=3,1,H,W), same image every t
        # sample exactly at pixel (2,3)'s center
        bbox = (0.0, W - 1.0, 0.0, H - 1.0)
        coords = np.array([[3.0, 2.0]])  # (x=col=3, y=row=2)
        norm_grid = torch.tensor(_norm_grid_coords(coords, bbox, H, W), dtype=torch.float32)
        sampled = _sample_raster_at_points(raster, norm_grid)
        assert sampled.shape == (3, 1)
        expected = 2 * W + 3
        assert torch.allclose(sampled[0], torch.tensor([float(expected)]), atol=1e-4)


class TestConvLSTMCell:
    def test_forward_shape(self):
        cell = ConvLSTMCell(in_channels=2, hidden_channels=5, kernel_size=3)
        x = torch.randn(1, 2, 8, 8)
        h = torch.zeros(1, 5, 8, 8)
        c = torch.zeros(1, 5, 8, 8)
        h_new, c_new = cell(x, h, c)
        assert h_new.shape == c_new.shape == (1, 5, 8, 8)
        assert torch.isfinite(h_new).all() and torch.isfinite(c_new).all()


# ==========================================================
# Fit / predict behavior (small subsample -- fast)
# ==========================================================

class TestFitPredict:
    def test_fit_predict_shapes_and_finiteness(self):
        coords, values = _load_panel()
        coords, values = _subsample(coords, values, n=40)
        model = ConvLSTM(grid_size=10, hidden_channels=8, max_epochs=15, patience=5, device="cpu", seed=0)
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

    def test_general_engine_and_alias_are_equivalent(self):
        """ConvLSTM is a plain subclass of CNNLSTMRegressionKriging with no
        overrides -- confirm they really do build the same architecture.
        `seed=` only controls the (node, time) masking pattern, not
        PyTorch's global weight-init RNG (neither this module nor its
        siblings seed that), so `torch.manual_seed` is set explicitly here
        around each construction to make the two runs comparable."""
        coords, values = _load_panel()
        coords, values = _subsample(coords, values, n=30)
        torch.manual_seed(0)
        m1 = CNNLSTMRegressionKriging(grid_size=8, hidden_channels=6, max_epochs=8, patience=4, device="cpu", seed=0)
        m1.fit(coords, values, verbose=False)
        torch.manual_seed(0)
        m2 = ConvLSTM(grid_size=8, hidden_channels=6, max_epochs=8, patience=4, device="cpu", seed=0)
        m2.fit(coords, values, verbose=False)
        pred1 = m1.predict(coords[:3])
        pred2 = m2.predict(coords[:3])
        np.testing.assert_allclose(pred1, pred2)

    def test_predict_before_fit_raises(self):
        model = ConvLSTM()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.zeros((2, 2)))

    def test_handles_missing_station_months(self):
        """21/1956 station-months in the real panel are genuinely missing
        (NaN) -- confirm fit/predict tolerate that without special-casing
        by the caller."""
        coords, values = _load_panel()
        assert np.isnan(values).sum() > 0
        coords, values = _subsample(coords, values, n=50)
        model = ConvLSTM(grid_size=10, hidden_channels=8, max_epochs=15, patience=5, device="cpu", seed=0)
        model.fit(coords, values, verbose=False)
        pred = model.predict(coords[:5])
        assert np.isfinite(pred).all()

    def test_krige_residuals_false_skips_kriging_step(self):
        coords, values = _load_panel()
        coords, values = _subsample(coords, values, n=30)
        model = ConvLSTM(grid_size=8, hidden_channels=6, max_epochs=10, patience=5,
                          device="cpu", seed=0, krige_residuals=False)
        model.fit(coords, values, verbose=False)
        assert model.krige_by_step_ == {}
        pred, std = model.predict(coords[:3], return_std=True)
        assert np.all(std == 0)

    def test_non_square_aspect_ratio_preserved_from_data(self):
        """A tall, narrow point cloud (like California) should produce a
        tall, narrow raster, not a square one."""
        rng = np.random.default_rng(0)
        coords = np.column_stack([rng.uniform(0, 50, 60), rng.uniform(0, 300, 60)])
        values = rng.normal(size=(60, 4))
        model = ConvLSTM(grid_size=30, hidden_channels=4, max_epochs=3, patience=2, device="cpu", seed=0)
        model.fit(coords, values, verbose=False)
        assert model.H_ == 30
        assert model.W_ < model.H_


# ==========================================================
# Real-data validation: held-out stations
# ==========================================================

class TestHeldOutStationValidation:
    """Mirrors pygstat.krigeST's own held-out-station validation exactly
    (same seed, same n_test_sites), for a direct, apples-to-apples
    comparison against the classical space-time Ordinary Kriging baseline
    and against the graph-based architectures in the sibling modules."""

    def test_beats_naive_mean_baseline_on_held_out_stations(self):
        coords, values = _load_panel()
        rng = np.random.default_rng(0)
        counts = (~np.isnan(values)).sum(axis=1)
        complete = np.flatnonzero(counts == 12)
        test_idx = rng.choice(complete, size=25, replace=False)
        train_mask = np.ones(len(values), dtype=bool)
        train_mask[test_idx] = False

        model = ConvLSTM(grid_size=32, hidden_channels=16, n_layers=2,
                          max_epochs=300, patience=30, device="cpu", seed=0)
        model.fit(coords[train_mask], values[train_mask], verbose=False)
        pred = model.predict(coords[test_idx])

        obs_mask = ~np.isnan(values[test_idx])
        resid = values[test_idx][obs_mask] - pred[obs_mask]
        rmse = np.sqrt(np.mean(resid ** 2))
        train_mean = np.nanmean(values[train_mask])
        baseline_rmse = np.sqrt(np.mean((values[test_idx][obs_mask] - train_mean) ** 2))

        print(f"\nConvLSTM held-out-station RMSE={rmse:.3f} vs naive-mean baseline={baseline_rmse:.3f}")
        assert rmse < 0.8 * baseline_rmse  # a real, comfortable margin (observed ~0.59x)


# ==========================================================
# Real-data validation: full grid vs. the classical reference
# ==========================================================

class TestFullDatasetAndGridEndToEnd:
    def test_grid_predictions_correlate_with_classical_stkriging(self):
        coords, values = _load_panel()
        grid = pd.read_csv(GRID_PATH)
        grid_xy = grid[grid["month"] == 0][["x", "y"]].to_numpy()

        model = ConvLSTM(grid_size=32, hidden_channels=16, n_layers=2,
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
        print(f"\nConvLSTM vs classical STKriging reference: per-month corr = "
              f"{[f'{c:.2f}' for c in corrs]}, mean={mean_corr:.3f}")
        assert min(corrs) > 0.4
        assert mean_corr > 0.55

        assert pred.mean() > 0
        assert pred.max() < 200


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
