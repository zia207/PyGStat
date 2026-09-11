# src/pygstat/kriging_STGTN.py
"""
Spatial-Temporal Graph Transformer Network Regression Kriging.

The attention-based counterpart to :mod:`pygstat.kriging_STGNN`'s
recurrent (GRU-style) spatio-temporal GNNs: nodes are fixed monitoring
locations observed repeatedly over time; instead of stepping a recurrent
cell through time, a **spatial attention** layer lets every node attend to
its k-nearest-neighbor graph at each time step, and a **temporal
attention** layer lets every node attend across its own time steps -- the
two are combined per published architecture, then residuals are kriged
spatially per time step and added back, as in every other
``regression_kriging_*``/``kriging_*`` module in this package.

Two published spatio-temporal graph Transformer architectures are
provided, built in pure PyTorch (no `torch_geometric`; k-NN graphs via
`scipy.spatial.cKDTree`, spatial attention via `Tensor.index_add_`-based
scatter-softmax, following the same convention as
:mod:`pygstat.regression_kriging_gnn` and :mod:`pygstat.kriging_STGNN`;
temporal attention uses `torch.nn.MultiheadAttention`, since that piece is
a stock, unmodified Transformer building block):

- **`STTN`** -- Spatial-Temporal Transformer Network (Xu, Dai, Liu, Gao,
  Lin, Qi & Xiong, 2020, https://arxiv.org/abs/2001.02908). Alternates a
  **spatial Transformer block** (graph-masked multi-head self-attention
  over a node's k-NN neighbors, at each time step) with a **temporal
  Transformer block** (multi-head self-attention across time, per node),
  each a standard pre/post-norm residual + feed-forward Transformer block.

- **`GMAN`** -- Graph Multi-Attention Network (Zheng, Fan, Wang & Qi, 2020,
  AAAI, https://arxiv.org/abs/1911.08415). Computes spatial and temporal
  attention **in parallel** from the same input at every layer, then
  combines them with a **learned gate**
  (:math:`h = g \\odot h_{\\text{spatial}} + (1-g) \\odot h_{\\text{temporal}}`,
  :math:`g = \\sigma(W[h_{\\text{spatial}}; h_{\\text{temporal}}])`) rather
  than applying them one after another -- GMAN's key architectural
  difference from STTN's sequential alternation.

Both share the general-purpose :class:`STGTNRegressionKriging` engine
(``block_type='sttn'|'gman'``); `STTN`/`GMAN` are thin convenience
subclasses pinning that choice.

Every node's per-time-step input is ``[mask * value, mask]`` plus a
sinusoidal positional encoding of the time index (Vaswani et al., 2017) --
attention has no inherent sense of order, unlike the recurrent cells in
:mod:`pygstat.kriging_STGNN`, so position must be injected explicitly.
Training follows the same IGNNK-style (Wu, Cui, Nie & Wang, 2021) random
(node, time step) masking as :mod:`pygstat.kriging_STGNN`, with validation
nodes always fully hidden, so the model generalizes to genuinely new,
never-seen-during-training locations at prediction time.

Tested against ``data/CA_pm25_2025.csv`` (163 California PM2.5 monitoring
stations, daily 2025 readings aggregated to monthly means) and validated
against ``data/CA_pm25_grid_predictions.csv``, the classical space-time
Ordinary Kriging reference surface produced by
``scripts/predict_ca_pm25_grid.py`` -- see
``tests/test_kriging_STGTN.py``.

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

__all__ = ["STGTNRegressionKriging", "STTN", "GMAN"]


# ==========================================================
# Device / graph construction (mirrors kriging_STGNN.py)
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


def _scatter_softmax(logits, index, dim_size):
    """Numerically-stable softmax of `logits`, grouped by `index` (one
    softmax per destination node, over its incoming edges)."""
    max_per_group = torch.full((dim_size,), float("-inf"), device=logits.device, dtype=logits.dtype)
    max_per_group = max_per_group.scatter_reduce(0, index, logits, reduce="amax", include_self=True)
    max_per_group = torch.nan_to_num(max_per_group, neginf=0.0)
    exp = (logits - max_per_group[index]).exp()
    denom = torch.zeros(dim_size, device=logits.device, dtype=logits.dtype)
    denom.index_add_(0, index, exp)
    return exp / denom.clamp(min=1e-12)[index]


def _sinusoidal_positional_encoding(max_len, dim):
    """Standard fixed sinusoidal positional encoding (Vaswani et al.,
    2017), shape ``(max_len, dim)``."""
    pe = torch.zeros(max_len, dim)
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-np.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


# ==========================================================
# Attention layers
# ==========================================================

class GraphMultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention restricted to a node's k-NN
    neighbors -- a "graph Transformer" attention layer (Dwivedi & Bresson,
    2020, https://arxiv.org/abs/2012.09699): for each destination node,
    attention weights over its incoming edges are softmax-normalized *per
    head* via scatter-softmax, then its neighbors' Value vectors are
    aggregated by those weights. This is the multi-head, learned-Query/Key
    generalization of the single learned-scalar attention used by
    `GATConv` in :mod:`pygstat.regression_kriging_gnn`.
    """

    def __init__(self, dim, n_heads=4):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, edge_index, n_nodes):
        """`x`: ``(n_nodes, dim)``. Returns ``(n_nodes, dim)``."""
        src, dst = edge_index
        H, Dh = self.n_heads, self.head_dim
        Q = self.q(x).view(n_nodes, H, Dh)
        K = self.k(x).view(n_nodes, H, Dh)
        V = self.v(x).view(n_nodes, H, Dh)
        scores = (Q[dst] * K[src]).sum(-1) / (Dh ** 0.5)  # (E, H)
        alpha = torch.stack(
            [_scatter_softmax(scores[:, h], dst, n_nodes) for h in range(H)], dim=1
        )  # (E, H)
        messages = V[src] * alpha.unsqueeze(-1)  # (E, H, Dh)
        agg = torch.zeros(n_nodes, H, Dh, device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, messages)
        return self.out(agg.reshape(n_nodes, H * Dh))


class _FeedForward(nn.Module):
    def __init__(self, dim, ff_dim=None):
        super().__init__()
        ff_dim = ff_dim or dim * 2
        self.net = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))

    def forward(self, x):
        return self.net(x)


# ==========================================================
# Spatio-temporal Transformer blocks
# ==========================================================

class SpatialTemporalBlock(nn.Module):
    """
    One STTN-style layer (Xu et al., 2020): a spatial Transformer block
    (graph attention, applied independently at each time step) followed by
    a temporal Transformer block (self-attention across time, applied
    independently per node) -- each a standard pre-residual, post-norm
    Transformer sub-layer with its own feed-forward network.
    """

    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.spatial_attn = GraphMultiHeadAttention(dim, n_heads)
        self.spatial_norm1 = nn.LayerNorm(dim)
        self.spatial_ff = _FeedForward(dim)
        self.spatial_norm2 = nn.LayerNorm(dim)

        self.temporal_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.temporal_norm1 = nn.LayerNorm(dim)
        self.temporal_ff = _FeedForward(dim)
        self.temporal_norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, n_nodes):
        """`h`: ``(T, n_nodes, dim)``. Returns ``(T, n_nodes, dim)``."""
        T = h.shape[0]
        hs = torch.stack([self.spatial_attn(h[t], edge_index, n_nodes) for t in range(T)], dim=0)
        h = self.spatial_norm1(h + self.dropout(hs))
        h = self.spatial_norm2(h + self.dropout(self.spatial_ff(h)))

        h_bt = h.transpose(0, 1)  # (n_nodes, T, dim): nodes as batch, T as sequence
        ht, _ = self.temporal_attn(h_bt, h_bt, h_bt, need_weights=False)
        h_bt = self.temporal_norm1(h_bt + self.dropout(ht))
        h_bt = self.temporal_norm2(h_bt + self.dropout(self.temporal_ff(h_bt)))
        return h_bt.transpose(0, 1)


class GatedFusionBlock(nn.Module):
    """
    One GMAN-style layer (Zheng et al., 2020): spatial attention and
    temporal attention are computed **in parallel** from the same input,
    then combined by a learned gate
    ``g = sigmoid(W[h_spatial; h_temporal])``,
    ``h = g * h_spatial + (1-g) * h_temporal`` -- rather than one after the
    other, as in :class:`SpatialTemporalBlock`.
    """

    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.spatial_attn = GraphMultiHeadAttention(dim, n_heads)
        self.temporal_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(2 * dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, n_nodes):
        """`h`: ``(T, n_nodes, dim)``. Returns ``(T, n_nodes, dim)``."""
        T = h.shape[0]
        hs = torch.stack([self.spatial_attn(h[t], edge_index, n_nodes) for t in range(T)], dim=0)

        h_bt = h.transpose(0, 1)
        ht, _ = self.temporal_attn(h_bt, h_bt, h_bt, need_weights=False)
        ht = ht.transpose(0, 1)  # back to (T, n_nodes, dim)

        g = torch.sigmoid(self.gate(torch.cat([hs, ht], dim=-1)))
        fused = g * hs + (1 - g) * ht
        h = self.norm1(h + self.dropout(fused))
        h = self.norm2(h + self.dropout(self.ff(h)))
        return h


class _STGTNNet(nn.Module):
    """Input embedding + sinusoidal positional encoding, `n_layers` stacked
    spatio-temporal Transformer blocks (STTN- or GMAN-style), and a shared
    linear head applied to every (node, time step)."""

    def __init__(self, in_dim, hidden_dim, block_type="sttn", n_layers=2, n_heads=4, dropout=0.1, max_len=64):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        Block = SpatialTemporalBlock if block_type == "sttn" else GatedFusionBlock
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.register_buffer("pos_enc", _sinusoidal_positional_encoding(max_len, hidden_dim))
        self.blocks = nn.ModuleList([Block(hidden_dim, n_heads, dropout) for _ in range(n_layers)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq, edge_index, n_nodes):
        """`x_seq`: ``(T, n_nodes, in_dim)``. Returns ``(T, n_nodes)``."""
        T = x_seq.shape[0]
        h = self.embed(x_seq) + self.pos_enc[:T].unsqueeze(1)
        for block in self.blocks:
            h = block(h, edge_index, n_nodes)
        return self.head(h).squeeze(-1)


# ==========================================================
# Per-time-step residual kriging (identical recipe to kriging_STGNN.py)
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
            continue
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

class STGTNRegressionKriging:
    """
    Spatial-temporal graph Transformer trend model + per-time-step kriging
    of residuals, for a fixed set of monitoring locations observed
    repeatedly over time.

    Parameters
    ----------
    block_type : {'sttn', 'gman'}, default='sttn'
        Spatio-temporal Transformer block -- see :class:`STTN`/:class:`GMAN`
        for convenience subclasses pinning this choice.
    k_neighbors : int, default=8
        Spatial k-nearest-neighbor graph size.
    hidden_dim : int, default=32
        Must be divisible by `n_heads`.
    n_layers : int, default=2
    n_heads : int, default=4
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
    >>> stgtn = STGTNRegressionKriging(block_type="sttn", k_neighbors=8)
    >>> stgtn.fit(coords, values)              # values: (n_stations, n_months), NaN allowed
    >>> pred = stgtn.predict(coords_grid)      # -> (n_grid, n_months)
    """

    def __init__(
        self,
        block_type="sttn",
        k_neighbors=8,
        hidden_dim=32,
        n_layers=2,
        n_heads=4,
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
        if block_type not in ("sttn", "gman"):
            raise ValueError(f"block_type must be 'sttn' or 'gman', got {block_type!r}")
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        self.block_type = block_type
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
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
        if n_steps > 64:
            raise ValueError("n_steps > 64 needs a larger positional-encoding max_len than this module's default")

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
        edge_index = self.edge_index_.to(self.device_)

        rng = np.random.default_rng(self.seed)
        eligible = np.flatnonzero(observed_mask.any(axis=1))
        perm = rng.permutation(eligible)
        n_val = max(1, int(val_fraction * len(eligible)))
        val_idx = perm[:n_val]

        pool_mask = observed_mask.copy()
        pool_mask[val_idx, :] = False
        pool_cells = np.argwhere(pool_mask)  # (n_pool_cells, 2) as (node, step)
        n_mask_per_epoch = max(1, int(self.mask_ratio * len(pool_cells)))

        self.model_ = _STGTNNet(
            in_dim=2, hidden_dim=self.hidden_dim, block_type=self.block_type,
            n_layers=self.n_layers, n_heads=self.n_heads, dropout=self.dropout,
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
            train_mask[val_idx, :] = False
            train_mask[chosen[:, 0], chosen[:, 1]] = False

            optimizer.zero_grad()
            x_seq = self._make_x_seq(values_scaled, train_mask)
            out = self.model_(x_seq, edge_index, n_nodes)  # (T, n_nodes)
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
                out_val = self.model_(x_seq_val, edge_index, n_nodes)
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

        # Honest out-of-fold reconstructions for every node (same rationale
        # as IGNNK / kriging_STGNN.py: with every node fully known, the
        # network could just echo its own input value back).
        self.model_.eval()
        n_folds = max(1, min(n_residual_folds, n_nodes))
        fold_id = rng.permutation(n_nodes) % n_folds
        fitted_scaled = np.zeros((n_nodes, n_steps), dtype=np.float32)
        with torch.no_grad():
            for f in range(n_folds):
                mask = observed_mask.copy()
                mask[fold_id == f, :] = False
                x_seq = self._make_x_seq(values_scaled, mask)
                out = self.model_(x_seq, edge_index, n_nodes).cpu().numpy().T  # (n_nodes, T)
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

        n_all = n_train + n_pred
        values_scaled_all = np.zeros((n_all, self.n_steps_), dtype=np.float64)
        values_scaled_all[:n_train] = np.where(
            self.observed_mask_,
            (self.values_ - self.scaler_.mean_[0]) / self.scaler_.scale_[0],
            0.0,
        )
        mask_all = np.zeros((n_all, self.n_steps_), dtype=bool)
        mask_all[:n_train] = self.observed_mask_

        self.model_.eval()
        with torch.no_grad():
            x_seq = self._make_x_seq(values_scaled_all, mask_all)
            out = self.model_(x_seq, edge_index, n_all).cpu().numpy().T  # (n_all, T)
        trend_scaled = out[n_train:]
        trend_pred = trend_scaled * self.scaler_.scale_[0] + self.scaler_.mean_[0]

        if not self.krige_residuals or not self.krige_by_step_:
            if return_std:
                return trend_pred, np.zeros_like(trend_pred)
            return trend_pred
        return _combine_with_kriging_by_step(self.krige_by_step_, trend_pred, coords_pred, return_std)

    def __repr__(self):
        status = "fitted" if self.is_fitted_ else "not fitted"
        return f"{self.__class__.__name__}(block_type={self.block_type!r}, k_neighbors={self.k_neighbors}, {status})"


class STTN(STGTNRegressionKriging):
    """
    Spatial-Temporal Transformer Network (Xu, Dai, Liu, Gao, Lin, Qi &
    Xiong, 2020, https://arxiv.org/abs/2001.02908) + kriging of residuals.
    Convenience subclass of :class:`STGTNRegressionKriging` pinning
    ``block_type='sttn'`` (sequential spatial-then-temporal attention).
    See the module docstring for the architecture.

    Examples
    --------
    >>> sttn = STTN(k_neighbors=8, hidden_dim=32, n_heads=4)
    >>> sttn.fit(coords, values)
    >>> pred = sttn.predict(coords_grid)
    """

    def __init__(self, **kwargs):
        kwargs.pop("block_type", None)
        super().__init__(block_type="sttn", **kwargs)


class GMAN(STGTNRegressionKriging):
    """
    Graph Multi-Attention Network (Zheng, Fan, Wang & Qi, 2020, AAAI,
    https://arxiv.org/abs/1911.08415) + kriging of residuals. Convenience
    subclass of :class:`STGTNRegressionKriging` pinning
    ``block_type='gman'`` (parallel spatial + temporal attention, combined
    by a learned gate). See the module docstring for the architecture.

    Examples
    --------
    >>> gman = GMAN(k_neighbors=8, hidden_dim=32, n_heads=4)
    >>> gman.fit(coords, values)
    >>> pred = gman.predict(coords_grid)
    """

    def __init__(self, **kwargs):
        kwargs.pop("block_type", None)
        super().__init__(block_type="gman", **kwargs)
