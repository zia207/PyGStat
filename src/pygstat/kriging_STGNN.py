# src/pygstat/kriging_STGNN.py
"""
Spatio-Temporal Graph Neural Network Regression Kriging.

A deep-learning counterpart to :mod:`pygstat.krigeST` (classical space-time
Ordinary Kriging): nodes are fixed monitoring locations observed repeatedly
over time (e.g. an air-quality sensor network); a graph neural network
combines **spatial** message passing over a k-nearest-neighbor graph with a
**temporal** recurrence across time steps, then, as in every other
``regression_kriging_*`` module in this package, the network's residuals are
kriged spatially (independently per time step) and added back.

Two published spatio-temporal GNN cell architectures are provided, both
built from scratch in pure PyTorch (no `torch_geometric`/`torch_scatter`
dependency -- k-NN graphs via `scipy.spatial.cKDTree`, neighbor aggregation
via `Tensor.index_add_`, following the same convention as
:mod:`pygstat.regression_kriging_gnn`):

- **`TGCN`** -- Temporal Graph Convolutional Network (Zhao, Song, Zhang, Liu,
  Wang & Li, 2019, https://arxiv.org/abs/1811.05320). A GRU whose gates each
  replace their usual linear layer with a 1-hop spatial graph convolution
  (self-transform + neighbor mean-aggregation, GraphSAGE-style), so the
  hidden state at every time step is informed by both a node's own recent
  history *and* its neighbors' current state.

- **`DCRNN`** -- Diffusion Convolutional Recurrent Neural Network (Li, Yi,
  Shahabi & Liu, 2018, https://arxiv.org/abs/1707.01926). Same GRU skeleton,
  but each gate's transform is a *K*-hop **bidirectional diffusion
  convolution**: :math:`\\sum_{k=0}^K \\big(W^f_k P_f^k x + W^b_k P_b^k x\\big)`,
  where :math:`P_f = D^{-1}A` and :math:`P_b = D^{-1}A^\\top` are the
  forward/backward row-stochastic random-walk transition matrices of the
  k-NN graph (each application of :math:`P` is exactly a mean-aggregation
  step, i.e. `index_add_`-based scatter-mean over the graph's edges). The
  extra hop depth and direction-awareness is DCRNN's key architectural
  difference from T-GCN's single-hop, undirected-style aggregation.

Both share the general-purpose :class:`STGNNRegressionKriging` engine
(``cell_type='tgcn'|'dcrnn'``); `TGCN`/`DCRNN` are thin convenience
subclasses pinning that choice with each paper's typical defaults.

Every node's per-time-step input is ``[mask * value, mask]`` -- exactly
IGNNK's masking idea (Wu, Cui, Nie & Wang, 2021), but here applied to its
*original*, genuinely spatiotemporal setting rather than the static
adaptation used in :class:`pygstat.regression_kriging_gnn.IGNNK`. Training
repeatedly hides a random fraction of (node, time step) cells -- among a
pool of nodes with a validation subset always fully hidden -- and
reconstructs them from the rest of the space-time graph, so the model
generalizes to genuinely new, never-seen-during-training locations (the
"inductive" property IGNNK's name refers to) at every observed time step.

Tested against ``data/CA_pm25_2025.csv`` (163 California PM2.5 monitoring
stations, daily 2025 readings aggregated to monthly means) and validated
against ``data/CA_pm25_grid_predictions.csv``, the classical space-time
Ordinary Kriging reference surface produced by
``scripts/predict_ca_pm25_grid.py`` -- see
``tests/test_kriging_STGNN.py``.

Authors
-------
Zia Ahmed, PhD (zia207@gmail.com)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

from .core.kriging import OrdinaryKriging
from .core.variogram import Variogram

__all__ = ["STGNNRegressionKriging", "TGCN", "DCRNN"]


# ==========================================================
# Device / graph construction (mirrors regression_kriging_gnn.py)
# ==========================================================

from .utils.backend import resolve_torch_device, seed_torch


def _resolve_device(device):
    return resolve_torch_device(device)


def _knn_edges(coords, k, query_coords=None):
    """
    Directed k-nearest-neighbor edges as ``(edge_index, edge_dist)``, with
    ``edge_index`` shape ``(2, E)`` (row 0 = source/neighbor, row 1 =
    destination/query).

    If `query_coords` is None, builds each of `coords`' own k nearest
    *other* neighbors (self excluded) -- the training graph. Otherwise,
    builds each of `query_coords`' k nearest neighbors *within* `coords`;
    destination indices are offset by ``len(coords)``, so this can be
    concatenated onto the training graph to inductively attach new nodes
    without ever adding new-to-new or new-to-train edges (no leakage
    between simultaneously-scored query locations).
    """
    k = min(k, len(coords) - (1 if query_coords is None else 0))
    tree = cKDTree(coords)
    if query_coords is None:
        dist, idx = tree.query(coords, k=k + 1)
        dist, idx = np.atleast_2d(dist)[:, 1:], np.atleast_2d(idx)[:, 1:]
        dst = np.repeat(np.arange(len(coords)), k)
        src = idx.ravel()
        edge_dist = dist.ravel()
    else:
        dist, idx = tree.query(query_coords, k=k)
        dist = np.atleast_2d(dist).reshape(len(query_coords), -1)
        idx = np.atleast_2d(idx).reshape(len(query_coords), -1)
        n_train = len(coords)
        dst = np.repeat(np.arange(len(query_coords)) + n_train, idx.shape[1])
        src = idx.ravel()
        edge_dist = dist.ravel()
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_dist = torch.tensor(edge_dist, dtype=torch.float32)
    return edge_index, edge_dist


def _scatter_mean(src, index, dim_size):
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    count.index_add_(0, index, torch.ones(len(index), device=src.device, dtype=src.dtype))
    return out / count.clamp(min=1).unsqueeze(1)


# ==========================================================
# Spatial graph convolution operators
# ==========================================================

class _GraphConv(nn.Module):
    """1-hop spatial graph convolution: separate linear transforms for a
    node's own features and its mean-aggregated neighbor features, summed
    (GraphSAGE-mean style -- the same aggregation used throughout
    :mod:`pygstat.regression_kriging_gnn`). This is T-GCN's spatial half."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, n_nodes):
        src, dst = edge_index
        agg = _scatter_mean(x[src], dst, n_nodes)
        return self.lin_self(x) + self.lin_neigh(agg)


class _DiffusionConv(nn.Module):
    """
    K-hop bidirectional diffusion convolution (Li, Yi, Shahabi & Liu, 2018,
    Eq. 2-3): ``W_0 x + sum_{k=1}^K (W^f_k P_f^k x + W^b_k P_b^k x)``.

    `edge_index` supplies the forward direction (P_f, a row-stochastic
    random walk -- exactly what repeated `_scatter_mean` computes, since it
    divides by each destination's in-degree); `edge_index_rev` is the same
    edges with source/destination swapped, giving the backward/reverse walk
    P_b. Each hop re-aggregates the *previous* hop's output along the same
    edges, which is how matrix powers P^k are built via repeated
    application, and a separate learnable linear map is used per hop order
    (a polynomial filter, as in the paper) rather than one shared weight.
    """

    def __init__(self, in_dim, out_dim, k_hops=2):
        super().__init__()
        self.k_hops = k_hops
        self.lin0 = nn.Linear(in_dim, out_dim)
        self.lin_f = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(k_hops)])
        self.lin_b = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(k_hops)])

    def forward(self, x, edge_index, edge_index_rev, n_nodes):
        out = self.lin0(x)
        src_f, dst_f = edge_index
        src_b, dst_b = edge_index_rev
        xf, xb = x, x
        for k in range(self.k_hops):
            xf = _scatter_mean(xf[src_f], dst_f, n_nodes)
            xb = _scatter_mean(xb[src_b], dst_b, n_nodes)
            out = out + self.lin_f[k](xf) + self.lin_b[k](xb)
        return out


# ==========================================================
# Recurrent cells
# ==========================================================

class TGCNCell(nn.Module):
    """T-GCN recurrent cell (Zhao et al., 2019): a GRU whose update/reset/
    candidate transforms are each a :class:`_GraphConv` over
    ``[x_t, h_{t-1}]`` instead of a plain linear layer."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv_z = _GraphConv(in_dim + hidden_dim, hidden_dim)
        self.conv_r = _GraphConv(in_dim + hidden_dim, hidden_dim)
        self.conv_h = _GraphConv(in_dim + hidden_dim, hidden_dim)

    def forward(self, x, h, edge_index, n_nodes):
        zin = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self.conv_z(zin, edge_index, n_nodes))
        r = torch.sigmoid(self.conv_r(zin, edge_index, n_nodes))
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=-1), edge_index, n_nodes))
        return z * h + (1 - z) * h_tilde


class DCRNNCell(nn.Module):
    """DCRNN recurrent cell (Li et al., 2018): same GRU skeleton as
    :class:`TGCNCell`, but each gate is a :class:`_DiffusionConv`."""

    def __init__(self, in_dim, hidden_dim, k_hops=2):
        super().__init__()
        self.conv_z = _DiffusionConv(in_dim + hidden_dim, hidden_dim, k_hops)
        self.conv_r = _DiffusionConv(in_dim + hidden_dim, hidden_dim, k_hops)
        self.conv_h = _DiffusionConv(in_dim + hidden_dim, hidden_dim, k_hops)

    def forward(self, x, h, edge_index, edge_index_rev, n_nodes):
        zin = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self.conv_z(zin, edge_index, edge_index_rev, n_nodes))
        r = torch.sigmoid(self.conv_r(zin, edge_index, edge_index_rev, n_nodes))
        h_tilde = torch.tanh(
            self.conv_h(torch.cat([x, r * h], dim=-1), edge_index, edge_index_rev, n_nodes)
        )
        return z * h + (1 - z) * h_tilde


class _STGNNNet(nn.Module):
    """Stacks `n_layers` recurrent cells (T-GCN or DCRNN) and steps them
    over a ``(T, n_nodes, in_dim)`` input sequence, applying a shared linear
    head to every time step's top-layer hidden state."""

    def __init__(self, in_dim, hidden_dim, cell_type="tgcn", n_layers=1, k_hops=2, dropout=0.1):
        super().__init__()
        self.cell_type = cell_type
        self.hidden_dim = hidden_dim
        if cell_type == "tgcn":
            self.cells = nn.ModuleList(
                [TGCNCell(in_dim if l == 0 else hidden_dim, hidden_dim) for l in range(n_layers)]
            )
        elif cell_type == "dcrnn":
            self.cells = nn.ModuleList(
                [DCRNNCell(in_dim if l == 0 else hidden_dim, hidden_dim, k_hops) for l in range(n_layers)]
            )
        else:
            raise ValueError(f"cell_type must be 'tgcn' or 'dcrnn', got {cell_type!r}")
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq, edge_index, edge_index_rev, n_nodes):
        """`x_seq`: ``(T, n_nodes, in_dim)``. Returns ``(T, n_nodes)``."""
        device = x_seq.device
        h = [torch.zeros(n_nodes, self.hidden_dim, device=device) for _ in self.cells]
        outs = []
        for t in range(x_seq.shape[0]):
            inp = x_seq[t]
            for l, cell in enumerate(self.cells):
                if self.cell_type == "tgcn":
                    h[l] = cell(inp, h[l], edge_index, n_nodes)
                else:
                    h[l] = cell(inp, h[l], edge_index, edge_index_rev, n_nodes)
                inp = self.dropout(h[l])
            outs.append(self.head(inp).squeeze(-1))
        return torch.stack(outs, dim=0)


# ==========================================================
# Per-time-step residual kriging
# ==========================================================

def _fit_residual_kriging_by_step(coords, observed_mask, residuals, variogram_model, variogram_kwargs, min_points=6):
    """Fit one spatial :class:`OrdinaryKriging` per time step, using only
    that step's genuinely-observed rows. Steps with fewer than `min_points`
    observations are skipped (trend-only fallback at predict time)."""
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
            continue  # degenerate step (e.g. near-constant residuals) -- trend-only fallback
    return krige_by_step


def _combine_with_kriging_by_step(krige_by_step, trend_pred, coords_pred, return_std):
    """`trend_pred`: ``(n_pred, n_steps)``. Adds each step's kriged
    residual where available; falls back to trend-only otherwise."""
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

class STGNNRegressionKriging:
    """
    Spatio-temporal graph neural network trend model + per-time-step
    kriging of residuals, for a fixed set of monitoring locations observed
    repeatedly over time.

    Parameters
    ----------
    cell_type : {'tgcn', 'dcrnn'}, default='tgcn'
        Recurrent cell architecture -- see :class:`TGCN`/:class:`DCRNN` for
        convenience subclasses pinning this choice.
    k_neighbors : int, default=8
        Spatial k-nearest-neighbor graph size.
    hidden_dim : int, default=32
    n_layers : int, default=1
        Number of stacked recurrent layers.
    diffusion_k : int, default=2
        Diffusion hop count, `dcrnn` only (Li et al. use K=2-3).
    mask_ratio : float, default=0.3
        Fraction of the (non-validation) pool's *observed* (node, time)
        cells masked -- and used as that epoch's reconstruction targets --
        at each training step.
    dropout, learning_rate, weight_decay : float
    max_epochs, patience : int
        Training budget and early-stopping patience (on a held-out node
        fraction, `val_fraction` in :meth:`fit`).
    device : {'auto', 'cpu', 'cuda', ...}, default='auto'
    krige_residuals : bool, default=True
    variogram_model : str, default='spherical'
    variogram_kwargs : dict, optional
    seed : int, default=42

    Examples
    --------
    >>> stgnn = STGNNRegressionKriging(cell_type="tgcn", k_neighbors=8)
    >>> stgnn.fit(coords, values)              # values: (n_stations, n_months), NaN allowed
    >>> pred = stgnn.predict(coords_grid)      # -> (n_grid, n_months)
    """

    def __init__(
        self,
        cell_type="tgcn",
        k_neighbors=8,
        hidden_dim=32,
        n_layers=1,
        diffusion_k=2,
        mask_ratio=0.3,
        dropout=0.1,
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
        if cell_type not in ("tgcn", "dcrnn"):
            raise ValueError(f"cell_type must be 'tgcn' or 'dcrnn', got {cell_type!r}")
        self.cell_type = cell_type
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.diffusion_k = diffusion_k
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

    def _make_x_seq(self, values_scaled, mask):
        """`values_scaled`, `mask`: ``(n_nodes, n_steps)`` -> ``(n_steps, n_nodes, 2)``."""
        v = torch.tensor((values_scaled * mask).T, dtype=torch.float32, device=self.device_)
        m = torch.tensor(mask.T, dtype=torch.float32, device=self.device_)
        return torch.stack([v, m], dim=-1)

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

        self.scaler_ = StandardScaler()
        self.scaler_.fit(values[observed_mask].reshape(-1, 1))
        values_scaled = np.where(
            observed_mask,
            (values - self.scaler_.mean_[0]) / self.scaler_.scale_[0],
            0.0,
        )

        self.edge_index_, _ = _knn_edges(coords, self.k_neighbors)
        self.edge_index_rev_ = self.edge_index_.flip(0)
        edge_index = self.edge_index_.to(self.device_)
        edge_index_rev = self.edge_index_rev_.to(self.device_)

        rng = np.random.default_rng(self.seed)
        eligible = np.flatnonzero(observed_mask.any(axis=1))
        perm = rng.permutation(eligible)
        n_val = max(1, int(val_fraction * len(eligible)))
        val_idx = perm[:n_val]
        pool_idx = np.setdiff1d(np.arange(n_nodes), val_idx)  # everything else, incl. ineligible rows
        val_idx_t = torch.tensor(val_idx, device=self.device_)

        pool_mask = observed_mask.copy()
        pool_mask[val_idx, :] = False
        pool_cells = np.argwhere(pool_mask)  # (n_pool_cells, 2) as (node, step)
        n_mask_per_epoch = max(1, int(self.mask_ratio * len(pool_cells)))

        self.model_ = _STGNNNet(
            in_dim=2, hidden_dim=self.hidden_dim, cell_type=self.cell_type,
            n_layers=self.n_layers, k_hops=self.diffusion_k, dropout=self.dropout,
        ).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        best_val, patience_ctr = float("inf"), 0
        best_state = self.model_.state_dict()
        self.train_losses_, self.val_losses_ = [], []

        val_only_mask = observed_mask.copy()
        val_only_mask[val_idx, :] = False  # validation nodes fully hidden; everyone else fully known
        val_target_nodes, val_target_steps = np.where(observed_mask[val_idx, :])
        val_target_nodes = val_idx[val_target_nodes]

        for epoch in range(self.max_epochs):
            self.model_.train()
            chosen = pool_cells[rng.choice(len(pool_cells), size=n_mask_per_epoch, replace=False)]
            train_mask = pool_mask.copy()
            train_mask[val_idx, :] = False  # keep validation nodes hidden too, for realism
            train_mask[chosen[:, 0], chosen[:, 1]] = False

            optimizer.zero_grad()
            x_seq = self._make_x_seq(values_scaled, train_mask)
            out = self.model_(x_seq, edge_index, edge_index_rev, n_nodes)  # (T, n_nodes)
            target_nodes = torch.tensor(chosen[:, 0], device=self.device_)
            target_steps = torch.tensor(chosen[:, 1], device=self.device_)
            target_vals = torch.tensor(values_scaled[chosen[:, 0], chosen[:, 1]], dtype=torch.float32, device=self.device_)
            loss = criterion(out[target_steps, target_nodes], target_vals)
            loss.backward()
            optimizer.step()
            self.train_losses_.append(loss.item())

            self.model_.eval()
            with torch.no_grad():
                x_seq_val = self._make_x_seq(values_scaled, val_only_mask)
                out_val = self.model_(x_seq_val, edge_index, edge_index_rev, n_nodes)
                vt_nodes = torch.tensor(val_target_nodes, device=self.device_)
                vt_steps = torch.tensor(val_target_steps, device=self.device_)
                vt_vals = torch.tensor(values_scaled[val_target_nodes, val_target_steps], dtype=torch.float32, device=self.device_)
                val_loss = criterion(out_val[vt_steps, vt_nodes], vt_vals).item()
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

        # Honest out-of-fold reconstructions for every node (see IGNNK's
        # identical rationale in regression_kriging_gnn.py): with every node
        # fully known, the network could just echo its own input value back,
        # since that value is literally part of the input when unmasked.
        self.model_.eval()
        n_folds = max(1, min(n_residual_folds, n_nodes))
        fold_id = rng.permutation(n_nodes) % n_folds
        fitted_scaled = np.zeros((n_nodes, n_steps), dtype=np.float32)
        with torch.no_grad():
            for f in range(n_folds):
                mask = observed_mask.copy()
                mask[fold_id == f, :] = False
                x_seq = self._make_x_seq(values_scaled, mask)
                out = self.model_(x_seq, edge_index, edge_index_rev, n_nodes).cpu().numpy().T  # (n_nodes, T)
                fitted_scaled[fold_id == f] = out[fold_id == f]
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
        New locations attach inductively (k-NN edges into the training
        graph only, never to each other), so predictions for many query
        points can be made in one call with no leakage between them.
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet.")
        coords_pred = np.asarray(coords_pred, dtype=float)
        n_train = len(self.coords_)
        n_pred = len(coords_pred)

        new_edge_index, _ = _knn_edges(self.coords_, self.k_neighbors, query_coords=coords_pred)
        edge_index = torch.cat([self.edge_index_, new_edge_index], dim=1).to(self.device_)
        edge_index_rev = edge_index.flip(0)

        n_all = n_train + n_pred
        values_scaled_all = np.zeros((n_all, self.n_steps_), dtype=np.float64)
        values_scaled_all[:n_train] = np.where(
            self.observed_mask_,
            (self.values_ - self.scaler_.mean_[0]) / self.scaler_.scale_[0],
            0.0,
        )
        mask_all = np.zeros((n_all, self.n_steps_), dtype=bool)
        mask_all[:n_train] = self.observed_mask_  # training rows: known where truly observed

        self.model_.eval()
        with torch.no_grad():
            x_seq = self._make_x_seq(values_scaled_all, mask_all)
            out = self.model_(x_seq, edge_index, edge_index_rev, n_all).cpu().numpy().T  # (n_all, T)
        trend_scaled = out[n_train:]
        trend_pred = trend_scaled * self.scaler_.scale_[0] + self.scaler_.mean_[0]

        if not self.krige_residuals or not self.krige_by_step_:
            if return_std:
                return trend_pred, np.zeros_like(trend_pred)
            return trend_pred
        return _combine_with_kriging_by_step(self.krige_by_step_, trend_pred, coords_pred, return_std)

    def __repr__(self):
        status = "fitted" if self.is_fitted_ else "not fitted"
        return f"{self.__class__.__name__}(cell_type={self.cell_type!r}, k_neighbors={self.k_neighbors}, {status})"


class TGCN(STGNNRegressionKriging):
    """
    Temporal Graph Convolutional Network (Zhao, Song, Zhang, Liu, Wang &
    Li, 2019, https://arxiv.org/abs/1811.05320) + kriging of residuals.
    Convenience subclass of :class:`STGNNRegressionKriging` pinning
    ``cell_type='tgcn'``. See the module docstring for the architecture.

    Examples
    --------
    >>> tgcn = TGCN(k_neighbors=8, hidden_dim=32)
    >>> tgcn.fit(coords, values)
    >>> pred = tgcn.predict(coords_grid)
    """

    def __init__(self, **kwargs):
        kwargs.pop("cell_type", None)
        super().__init__(cell_type="tgcn", **kwargs)


class DCRNN(STGNNRegressionKriging):
    """
    Diffusion Convolutional Recurrent Neural Network (Li, Yi, Shahabi &
    Liu, 2018, https://arxiv.org/abs/1707.01926) + kriging of residuals.
    Convenience subclass of :class:`STGNNRegressionKriging` pinning
    ``cell_type='dcrnn'``. See the module docstring for the architecture.

    Examples
    --------
    >>> dcrnn = DCRNN(k_neighbors=8, hidden_dim=32, diffusion_k=2)
    >>> dcrnn.fit(coords, values)
    >>> pred = dcrnn.predict(coords_grid)
    """

    def __init__(self, **kwargs):
        kwargs.pop("cell_type", None)
        super().__init__(cell_type="dcrnn", **kwargs)
