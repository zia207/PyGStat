# src/pygstat/kriging_CNNLSTM.py
"""
Convolutional LSTM Regression Kriging.

A raster-based counterpart to :mod:`pygstat.kriging_STGNN` (recurrent
graph neural networks) and :mod:`pygstat.kriging_STGTN` (graph
Transformers): instead of message-passing on a k-nearest-neighbor graph,
the sensor network's observations at each time step are **rasterized**
onto a small, regular image covering the study area, and a
:class:`ConvLSTM` (Shi, Chen, Wang, Yeung, Wong & Woo, 2015,
"Convolutional LSTM Network: A Machine Learning Approach for Precipitation
Nowcasting", https://arxiv.org/abs/1506.04214) processes that image
sequence -- 2-D convolutions replace the fully-connected transforms of a
plain LSTM in every gate, so each pixel's next state is a *local, shared*
function of its own and its immediate neighbors' current state, exactly
the translational-invariance assumption a CNN encodes.

Because the raster is a regular grid but a station's true location rarely
sits exactly on a pixel center, this module reads predictions back off the
network's output image with **bilinear interpolation**
(`torch.nn.functional.grid_sample`) at each query point's exact
coordinates, rather than snapping to the nearest pixel -- so training and
prediction both work at genuine point precision despite the coarse raster
representation used internally. As in every other
``regression_kriging_*``/``kriging_*`` module in this package, the
network's residuals are then kriged spatially (independently per time
step) and added back.

Training follows the same IGNNK-style (Wu, Cui, Nie & Wang, 2021) random
masking used in :mod:`pygstat.kriging_STGNN`/:mod:`pygstat.kriging_STGTN`:
each epoch, a random subset of (station, time step) observations is
excluded from the rasterized input, and the network must reconstruct each
hidden station's *own* value (via bilinear sampling at its exact
coordinate, not just its containing pixel's blocky average) from the
surrounding image -- with a validation subset of stations always fully
excluded, so the model generalizes to genuinely new, never-seen-during-
training locations.

Tested against ``data/CA_pm25_2025.csv`` (163 California PM2.5 monitoring
stations, daily 2025 readings aggregated to monthly means) and validated
against ``data/CA_pm25_grid_predictions.csv``, the classical space-time
Ordinary Kriging reference surface produced by
``scripts/predict_ca_pm25_grid.py`` -- see
``tests/test_kriging_CNNLSTM.py``.

Authors
-------
Zia Ahmed, PhD (zia207@gmail.com)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from .core.kriging import OrdinaryKriging
from .core.variogram import Variogram

__all__ = ["CNNLSTMRegressionKriging", "ConvLSTM"]


# ==========================================================
# Device
# ==========================================================

from .utils.backend import resolve_torch_device, seed_torch


def _resolve_device(device):
    return resolve_torch_device(device)


# ==========================================================
# Rasterization: irregular points <-> regular image, at pixel and at
# continuous (bilinear-sampled) precision
# ==========================================================

def _make_grid_shape(bbox, grid_size):
    """Pick ``(H, W)`` so the longer bounding-box axis gets `grid_size`
    cells and the shorter one is scaled by the aspect ratio -- keeps
    non-square study areas (e.g. California's elongated N-S extent) from
    being squashed into a square raster."""
    xmin, xmax, ymin, ymax = bbox
    span_x, span_y = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    if span_x >= span_y:
        W = grid_size
        H = max(2, round(grid_size * span_y / span_x))
    else:
        H = grid_size
        W = max(2, round(grid_size * span_x / span_y))
    return H, W


def _continuous_pixel_coords(coords, bbox, H, W):
    """``coords`` (n,2) -> continuous ``(row, col)`` pixel coordinates in
    ``[0, H-1] x [0, W-1]`` (``align_corners=True`` convention: pixel index
    0 and pixel index N-1 are the extreme cell *centers*)."""
    xmin, xmax, ymin, ymax = bbox
    col = (coords[:, 0] - xmin) / max(xmax - xmin, 1e-9) * (W - 1)
    row = (coords[:, 1] - ymin) / max(ymax - ymin, 1e-9) * (H - 1)
    return row, col


def _cell_index(coords, bbox, H, W):
    """Nearest-pixel integer index (``row * W + col``) for rasterizing
    point observations onto the regular grid."""
    row, col = _continuous_pixel_coords(coords, bbox, H, W)
    row = np.clip(np.round(row).astype(int), 0, H - 1)
    col = np.clip(np.round(col).astype(int), 0, W - 1)
    return row * W + col


def _norm_grid_coords(coords, bbox, H, W):
    """Continuous pixel coordinates -> ``[-1, 1]`` grid_sample coordinates
    (``align_corners=True``), returned as ``(n, 2)`` with column 0 = x
    (width axis) and column 1 = y (height axis), matching
    `torch.nn.functional.grid_sample`'s ``grid[..., 0]``/``grid[..., 1]``
    convention."""
    row, col = _continuous_pixel_coords(coords, bbox, H, W)
    gx = 2.0 * col / max(W - 1, 1) - 1.0
    gy = 2.0 * row / max(H - 1, 1) - 1.0
    return np.stack([gx, gy], axis=-1)


def _rasterize(values_scaled, mask, cell_idx, H, W):
    """``values_scaled``, ``mask``: ``(n_stations, T)`` -> ``(T, 2, H, W)``.
    Stations sharing a pixel are mean-aggregated; channel 0 is the masked
    mean value, channel 1 is the observed-fraction mask (1 if >=1 station
    known that pixel/step, else 0)."""
    n_stations, T = values_scaled.shape
    HW = H * W
    imgs = np.zeros((T, 2, HW), dtype=np.float32)
    for t in range(T):
        known = mask[:, t]
        if not known.any():
            continue
        idx = cell_idx[known]
        v = values_scaled[known, t]
        cell_sum = np.zeros(HW, dtype=np.float64)
        cell_cnt = np.zeros(HW, dtype=np.float64)
        np.add.at(cell_sum, idx, v)
        np.add.at(cell_cnt, idx, 1.0)
        has = cell_cnt > 0
        imgs[t, 0, has] = (cell_sum[has] / cell_cnt[has]).astype(np.float32)
        imgs[t, 1, has] = 1.0
    return imgs.reshape(T, 2, H, W)


def _sample_raster_at_points(raster_t1hw, norm_grid):
    """`raster_t1hw`: ``(T, 1, H, W)`` tensor. `norm_grid`: ``(n, 2)``
    tensor of grid_sample-ready coordinates. Returns ``(T, n)``:
    every time step's raster bilinearly sampled at every query point."""
    T = raster_t1hw.shape[0]
    n = norm_grid.shape[0]
    grid = norm_grid.view(1, n, 1, 2).expand(T, -1, -1, -1)
    sampled = F.grid_sample(raster_t1hw, grid, mode="bilinear", align_corners=True, padding_mode="border")
    return sampled.reshape(T, n)


# ==========================================================
# ConvLSTM (Shi et al., 2015)
# ==========================================================

class ConvLSTMCell(nn.Module):
    """One Conv-LSTM cell: input-to-state and state-to-state transforms
    for all four gates (input, forget, output, candidate) are a *single*
    2-D convolution over the channel-concatenated ``[x_t, h_{t-1}]``,
    split into four chunks -- the standard, efficient formulation."""

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class _ConvLSTMNet(nn.Module):
    """Stacks `n_layers` :class:`ConvLSTMCell`, steps them over a
    ``(T, in_channels, H, W)`` image sequence (batch size 1 -- one
    space-time scene), and applies a shared 1x1-conv head to every time
    step's top-layer hidden state."""

    def __init__(self, in_channels, hidden_channels, n_layers=2, kernel_size=3, dropout=0.0):
        super().__init__()
        chans = [in_channels] + [hidden_channels] * n_layers
        self.cells = nn.ModuleList(
            [ConvLSTMCell(chans[l], chans[l + 1], kernel_size) for l in range(n_layers)]
        )
        self.dropout = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x_seq):
        """`x_seq`: ``(T, in_channels, H, W)``. Returns ``(T, 1, H, W)``."""
        T, _, H, W = x_seq.shape
        device = x_seq.device
        h = [torch.zeros(1, cell.hidden_channels, H, W, device=device) for cell in self.cells]
        c = [torch.zeros_like(hh) for hh in h]
        outs = []
        for t in range(T):
            inp = x_seq[t].unsqueeze(0)  # (1, C, H, W)
            for l, cell in enumerate(self.cells):
                h[l], c[l] = cell(inp, h[l], c[l])
                inp = self.dropout(h[l])
            outs.append(self.head(inp))  # (1, 1, H, W)
        return torch.cat(outs, dim=0)  # (T, 1, H, W)


# ==========================================================
# Per-time-step residual kriging (identical recipe to the sibling modules)
# ==========================================================

def _fit_residual_kriging_by_step(coords, observed_mask, residuals, variogram_model, variogram_kwargs, min_points=6):
    vg_kwargs = {"model": variogram_model, "estimator": "matheron"}
    vg_kwargs.update(variogram_kwargs or {})
    n_steps = residuals.shape[1]
    krige_by_step = {}
    for t in range(n_steps):
        obs_t = observed_mask[:, t]
        if obs_t.sum() < min_points:
            continue
        try:
            variogram = Variogram(coords[obs_t], residuals[obs_t, t], **vg_kwargs)
            variogram.fit()
            krige_by_step[t] = OrdinaryKriging(variogram).fit(coords[obs_t], residuals[obs_t, t])
        except Exception:
            continue
    return krige_by_step


def _combine_with_kriging_by_step(krige_by_step, trend_pred, coords_pred, return_std):
    n_pred, n_steps = trend_pred.shape
    out = trend_pred.copy()
    std = np.zeros_like(trend_pred)
    for t in range(n_steps):
        krige = krige_by_step.get(t)
        if krige is None:
            continue
        if return_std:
            resid_pred, resid_std = krige.predict(coords_pred, return_variance=True)
            out[:, t] += resid_pred
            std[:, t] = resid_std
        else:
            out[:, t] += krige.predict(coords_pred)
    return (out, std) if return_std else out


# ==========================================================
# Public engine
# ==========================================================

class CNNLSTMRegressionKriging:
    """
    Convolutional-LSTM trend model + per-time-step kriging of residuals,
    for a fixed set of monitoring locations observed repeatedly over time.

    Parameters
    ----------
    grid_size : int, default=32
        Number of raster cells along the *longer* axis of the training
        coordinates' bounding box (the shorter axis is scaled by the
        aspect ratio, see :func:`_make_grid_shape`); the whole space-time
        scene is one ``(T, 2, H, W)`` image sequence.
    hidden_channels : int, default=16
    n_layers : int, default=2
    kernel_size : int, default=3
    mask_ratio : float, default=0.3
        Fraction of the (non-validation) pool's *observed* (station, time)
        cells excluded from the rasterized input -- and used as that
        epoch's reconstruction targets -- at each training step.
    dropout, learning_rate, weight_decay : float
    max_epochs, patience : int
        Training budget and early-stopping patience (on a held-out station
        fraction, `val_fraction` in :meth:`fit`).
    device : {'auto', 'cpu', 'cuda', ...}, default='auto'
    krige_residuals : bool, default=True
    variogram_model : str, default='spherical'
    variogram_kwargs : dict, optional
    seed : int, default=42

    Examples
    --------
    >>> cl = CNNLSTMRegressionKriging(grid_size=32, hidden_channels=16)
    >>> cl.fit(coords, values)              # values: (n_stations, n_months), NaN allowed
    >>> pred = cl.predict(coords_grid)      # -> (n_grid, n_months)
    """

    def __init__(
        self,
        grid_size=32,
        hidden_channels=16,
        n_layers=2,
        kernel_size=3,
        mask_ratio=0.3,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_epochs=300,
        patience=30,
        device="auto",
        krige_residuals=True,
        variogram_model="spherical",
        variogram_kwargs=None,
        seed=42,
    ):
        self.grid_size = grid_size
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.mask_ratio = mask_ratio
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.krige_residuals = krige_residuals
        self.variogram_model = variogram_model
        self.variogram_kwargs = variogram_kwargs or {}
        self.seed = seed
        self.is_fitted_ = False

    def _raster_tensor(self, values_scaled, mask):
        img = _rasterize(values_scaled, mask, self.cell_idx_, self.H_, self.W_)
        return torch.tensor(img, dtype=torch.float32, device=self.device_)

    def fit(self, coords, values, val_fraction=0.15, n_residual_folds=5, verbose=True):
        """
        Parameters
        ----------
        coords : array-like, shape (n_nodes, 2)
        values : array-like, shape (n_nodes, n_steps)
            NaN marks a genuinely missing (node, time) observation.
        """
        coords = np.asarray(coords, dtype=float)
        values = np.asarray(values, dtype=float)
        n_nodes, n_steps = values.shape
        if len(coords) != n_nodes:
            raise ValueError("coords and values must have the same number of rows")

        seed_torch(self.seed)
        self.device_ = _resolve_device(self.device)
        observed_mask = ~np.isnan(values)
        if not observed_mask.any():
            raise ValueError("values has no observed entries at all")

        margin = 0.05
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        pad_x, pad_y = max((xmax - xmin) * margin, 1e-6), max((ymax - ymin) * margin, 1e-6)
        self.bbox_ = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)
        self.H_, self.W_ = _make_grid_shape(self.bbox_, self.grid_size)
        self.cell_idx_ = _cell_index(coords, self.bbox_, self.H_, self.W_)
        norm_grid_np = _norm_grid_coords(coords, self.bbox_, self.H_, self.W_)
        norm_grid = torch.tensor(norm_grid_np, dtype=torch.float32, device=self.device_)

        self.scaler_ = StandardScaler()
        self.scaler_.fit(values[observed_mask].reshape(-1, 1))
        values_scaled = np.where(
            observed_mask, (values - self.scaler_.mean_[0]) / self.scaler_.scale_[0], 0.0
        )

        rng = np.random.default_rng(self.seed)
        eligible = np.flatnonzero(observed_mask.any(axis=1))
        perm = rng.permutation(eligible)
        n_val = max(1, int(val_fraction * len(eligible)))
        val_idx = perm[:n_val]

        pool_mask = observed_mask.copy()
        pool_mask[val_idx, :] = False
        pool_cells = np.argwhere(pool_mask)  # (n_pool_cells, 2) as (node, step)
        n_mask_per_epoch = max(1, int(self.mask_ratio * len(pool_cells)))

        self.model_ = _ConvLSTMNet(
            in_channels=2, hidden_channels=self.hidden_channels, n_layers=self.n_layers,
            kernel_size=self.kernel_size, dropout=self.dropout,
        ).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        best_val, patience_ctr = float("inf"), 0
        best_state = self.model_.state_dict()
        self.train_losses_, self.val_losses_ = [], []

        val_only_mask = observed_mask.copy()
        val_only_mask[val_idx, :] = False
        val_target_nodes, val_target_steps = np.where(observed_mask[val_idx, :])
        val_target_nodes = val_idx[val_target_nodes]
        val_target_vals = torch.tensor(
            values_scaled[val_target_nodes, val_target_steps], dtype=torch.float32, device=self.device_
        )
        val_target_nodes_t = torch.tensor(val_target_nodes, device=self.device_)
        val_target_steps_t = torch.tensor(val_target_steps, device=self.device_)

        for epoch in range(self.max_epochs):
            self.model_.train()
            chosen = pool_cells[rng.choice(len(pool_cells), size=n_mask_per_epoch, replace=False)]
            train_mask = pool_mask.copy()
            train_mask[val_idx, :] = False
            train_mask[chosen[:, 0], chosen[:, 1]] = False

            optimizer.zero_grad()
            x_img = self._raster_tensor(values_scaled, train_mask)
            out = self.model_(x_img)  # (T, 1, H, W)
            sampled = _sample_raster_at_points(out, norm_grid)  # (T, n_nodes)
            target_nodes = torch.tensor(chosen[:, 0], device=self.device_)
            target_steps = torch.tensor(chosen[:, 1], device=self.device_)
            target_vals = torch.tensor(values_scaled[chosen[:, 0], chosen[:, 1]], dtype=torch.float32, device=self.device_)
            loss = criterion(sampled[target_steps, target_nodes], target_vals)
            loss.backward()
            optimizer.step()
            self.train_losses_.append(loss.item())

            self.model_.eval()
            with torch.no_grad():
                x_img_val = self._raster_tensor(values_scaled, val_only_mask)
                out_val = self.model_(x_img_val)
                sampled_val = _sample_raster_at_points(out_val, norm_grid)
                val_loss = criterion(sampled_val[val_target_steps_t, val_target_nodes_t], val_target_vals).item()
            self.val_losses_.append(val_loss)

            if val_loss < best_val - 1e-6:
                best_val, patience_ctr = val_loss, 0
                best_state = self.model_.state_dict()
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break
            if verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs} - Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

        self.model_.load_state_dict(best_state)

        # Honest out-of-fold reconstructions for every node (same rationale
        # as IGNNK / kriging_STGNN.py / kriging_STGTN.py): excluding a
        # fold's stations from the raster entirely, then bilinearly
        # sampling the output *at their exact coordinates*, avoids both
        # "copy your own answer" leakage and the coarse-raster ambiguity
        # of stations sharing a pixel with each other.
        self.model_.eval()
        n_folds = max(1, min(n_residual_folds, n_nodes))
        fold_id = rng.permutation(n_nodes) % n_folds
        fitted_scaled = np.zeros((n_nodes, n_steps), dtype=np.float32)
        with torch.no_grad():
            for f in range(n_folds):
                mask = observed_mask.copy()
                mask[fold_id == f, :] = False
                x_img = self._raster_tensor(values_scaled, mask)
                out = self.model_(x_img)
                sampled = _sample_raster_at_points(out, norm_grid).cpu().numpy().T  # (n_nodes, T)
                fitted_scaled[fold_id == f] = sampled[fold_id == f]
        fitted = fitted_scaled * self.scaler_.scale_[0] + self.scaler_.mean_[0]
        residuals = np.where(observed_mask, values - fitted, np.nan)

        self.coords_, self.values_, self.observed_mask_ = coords, values, observed_mask
        self.n_steps_ = n_steps
        if self.krige_residuals:
            self.krige_by_step_ = _fit_residual_kriging_by_step(
                coords, observed_mask, residuals, self.variogram_model, self.variogram_kwargs
            )
        else:
            self.krige_by_step_ = {}
        self.is_fitted_ = True
        return self

    def predict(self, coords_pred, return_std=False):
        """
        Predict the full ``(n_pred, n_steps)`` sequence at new locations.
        The trained network's output raster (built from every training
        station's full, unmasked history) is bilinearly sampled at each
        query point's exact coordinate -- so query points need not fall on
        the training raster's coarse pixel grid, and any number of them
        can be scored in one call with no interaction between them.
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet.")
        coords_pred = np.asarray(coords_pred, dtype=float)

        values_scaled_full = np.where(
            self.observed_mask_,
            (self.values_ - self.scaler_.mean_[0]) / self.scaler_.scale_[0],
            0.0,
        )
        self.model_.eval()
        with torch.no_grad():
            x_img = self._raster_tensor(values_scaled_full, self.observed_mask_)
            out = self.model_(x_img)  # (T, 1, H, W)
            norm_grid_pred = torch.tensor(
                _norm_grid_coords(coords_pred, self.bbox_, self.H_, self.W_),
                dtype=torch.float32, device=self.device_,
            )
            trend_scaled = _sample_raster_at_points(out, norm_grid_pred).cpu().numpy().T  # (n_pred, T)
        trend_pred = trend_scaled * self.scaler_.scale_[0] + self.scaler_.mean_[0]

        if not self.krige_residuals or not self.krige_by_step_:
            if return_std:
                return trend_pred, np.zeros_like(trend_pred)
            return trend_pred
        return _combine_with_kriging_by_step(self.krige_by_step_, trend_pred, coords_pred, return_std)

    def __repr__(self):
        status = "fitted" if self.is_fitted_ else "not fitted"
        return f"{self.__class__.__name__}(grid_size={self.grid_size}, {status})"


class ConvLSTM(CNNLSTMRegressionKriging):
    """
    Convolutional LSTM (Shi, Chen, Wang, Yeung, Wong & Woo, 2015,
    https://arxiv.org/abs/1506.04214) + kriging of residuals. A plain
    alias of :class:`CNNLSTMRegressionKriging` (there is only one
    architecture in this module -- unlike :mod:`pygstat.kriging_STGNN`'s
    TGCN/DCRNN or :mod:`pygstat.kriging_STGTN`'s STTN/GMAN pairs -- named
    to match this module's literature reference directly).

    Examples
    --------
    >>> cl = ConvLSTM(grid_size=32, hidden_channels=16)
    >>> cl.fit(coords, values)
    >>> pred = cl.predict(coords_grid)
    """
    pass
