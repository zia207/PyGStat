# src/pygstat/regression_kriging_gnn.py
"""
Graph Neural Network Regression Kriging.

Three GNN-based trend models, each following pygstat's standard regression
-kriging recipe (fit a machine-learned trend, krige its residuals with
:class:`pygstat.core.kriging.OrdinaryKriging`, add the two back together):

- **`GNNRegressionKriging`** -- a plain spatial GNN: a k-nearest-neighbor
  graph of the training coordinates, covariates only as node features,
  mean-aggregation (GraphSAGE-style) or attention (GAT-style) message
  passing. The GNN analogue of `pygstat.regression_kriging_pytorch_gpu`'s MLP.

- **`KCN`** -- Kriging Convolutional Networks (Appleby, Liu & Liu, 2020,
  https://arxiv.org/abs/1907.05003). Each node's *own* target value is
  always hidden from itself; its neighbors' *known* target values are
  aggregated through a learnable-lengthscale Gaussian-kernel-weighted graph
  convolution -- explicitly combining a geostatistical distance kernel (the
  "kriging" in the name) with a learned representation.

- **`IGNNK`** -- Inductive Graph Neural Networks for Kriging (Wu, Cui, Nie &
  Wang, 2021, https://arxiv.org/abs/2006.07527), adapted here from its
  original spatiotemporal-sensor setting to static regression kriging.
  Every node carries ``[covariates, mask * value, mask]``; training
  repeatedly masks a *random* fraction of nodes each epoch and reconstructs
  them from their (partially also-masked) neighbors, exposing the model to
  many missing-data configurations so it generalizes to genuinely new,
  never-seen-during-training locations at prediction time -- the
  "inductive" property the name refers to.

All three are pure PyTorch (no `torch_geometric`/`torch_scatter` dependency):
graphs are built with `scipy.spatial.cKDTree`, and neighbor aggregation uses
`Tensor.index_add_`.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

from .core.kriging import OrdinaryKriging
from .core.variogram import Variogram
from .utils.backend import resolve_cupy_use_gpu, resolve_torch_device, seed_torch

__all__ = ["GNNRegressionKriging", "KCN", "IGNNK"]


# ==========================================================
# Device / graph construction
# ==========================================================


def _resolve_device(device):
    return resolve_torch_device(device)


def _knn_edges(coords, k, query_coords=None):
    """
    Directed k-nearest-neighbor edges as ``(edge_index, edge_dist)``, with
    ``edge_index`` shape ``(2, E)`` (row 0 = source/neighbor, row 1 =
    destination/query) and ``edge_dist`` shape ``(E,)``.

    If `query_coords` is None, builds each of `coords`' own k nearest
    *other* neighbors (self excluded) -- the training graph. Otherwise,
    builds each of `query_coords`' k nearest neighbors *within* `coords`;
    destination node indices are offset by ``len(coords)``, so this can be
    directly concatenated onto the training graph to inductively attach new
    nodes without ever adding new-to-new or new-to-train edges.
    """
    k = min(k, len(coords) - (1 if query_coords is None else 0))
    tree = cKDTree(coords)
    if query_coords is None:
        dist, idx = tree.query(coords, k=k + 1)
        dist, idx = np.atleast_2d(dist)[:, 1:], np.atleast_2d(idx)[:, 1:]  # drop self (col 0, dist 0)
        dst = np.repeat(np.arange(len(coords)), k)
        src = idx.ravel()
        edge_dist = dist.ravel()
    else:
        dist, idx = tree.query(query_coords, k=k)
        dist, idx = np.atleast_2d(dist).reshape(len(query_coords), -1), np.atleast_2d(idx).reshape(len(query_coords), -1)
        n_train = len(coords)
        dst = np.repeat(np.arange(len(query_coords)) + n_train, idx.shape[1])
        src = idx.ravel()
        edge_dist = dist.ravel()
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_dist = torch.tensor(edge_dist, dtype=torch.float32)
    return edge_index, edge_dist


# ==========================================================
# Sparse neighbor aggregation (pure PyTorch: index_add_ only)
# ==========================================================

def _scatter_mean(src, index, dim_size):
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    count.index_add_(0, index, torch.ones(len(index), device=src.device, dtype=src.dtype))
    return out / count.clamp(min=1).unsqueeze(1)


def _scatter_weighted_mean(src, weight, index, dim_size, eps=1e-8):
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src * weight.unsqueeze(1))
    denom = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    denom.index_add_(0, index, weight)
    return out / denom.clamp(min=eps).unsqueeze(1)


def _scatter_softmax(logits, index, dim_size):
    """Numerically-stable softmax of `logits`, grouped by `index` (e.g. one
    softmax per destination node, over its incoming edges)."""
    max_per_group = torch.full((dim_size,), float("-inf"), device=logits.device, dtype=logits.dtype)
    max_per_group = max_per_group.scatter_reduce(0, index, logits, reduce="amax", include_self=True)
    max_per_group = torch.nan_to_num(max_per_group, neginf=0.0)
    exp = (logits - max_per_group[index]).exp()
    denom = torch.zeros(dim_size, device=logits.device, dtype=logits.dtype)
    denom.index_add_(0, index, exp)
    return exp / denom.clamp(min=1e-12)[index]


# ==========================================================
# Message-passing layers
# ==========================================================

class MeanConv(nn.Module):
    """GraphSAGE-mean-style layer: separate linear transforms for a node's
    own features and its (optionally edge-weighted) mean-aggregated
    neighbor features, summed. `x_msg` lets a node's outgoing "message"
    differ from the features used for its own self-transform (needed by
    :class:`KCNConv`'s hidden layers when reused generically; unused --
    `x_msg=None` -- by the plain GNN and IGNNK, where both are identical)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, n_nodes, edge_weight=None, x_msg=None):
        src, dst = edge_index
        source_features = x if x_msg is None else x_msg
        messages = source_features[src]
        if edge_weight is None:
            agg = _scatter_mean(messages, dst, n_nodes)
        else:
            agg = _scatter_weighted_mean(messages, edge_weight, dst, n_nodes)
        return self.lin_self(x) + self.lin_neigh(agg)


class GATConv(nn.Module):
    """Single-head graph attention layer (Velickovic et al., 2018): edge
    weights are a learned function of the endpoint features (softmax
    -normalized per destination node), rather than fixed/uniform."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(2 * out_dim, 1)

    def forward(self, x, edge_index, n_nodes, edge_weight=None, x_msg=None):
        src, dst = edge_index
        h = self.lin(x)
        h_msg = h if x_msg is None else self.lin(x_msg)
        e = F.leaky_relu(self.attn(torch.cat([h[dst], h_msg[src]], dim=-1)).squeeze(-1), 0.2)
        alpha = _scatter_softmax(e, dst, n_nodes)
        messages = h_msg[src] * alpha.unsqueeze(-1)
        out = torch.zeros_like(h)
        out.index_add_(0, dst, messages)
        return out


class KCNConv(nn.Module):
    """
    The Kriging Convolutional layer (Appleby, Liu & Liu, 2020): aggregates
    neighbors' features through a **learnable-lengthscale Gaussian kernel of
    spatial distance** -- a classical geostatistical covariance shape --
    rather than a uniform or purely-learned attention weight, while a
    separate linear transform handles the node's own features. Critically,
    `x_self` and `x_msg` are passed in *already differing*: `x_self` never
    contains the node's own true target value (it is the query), while
    `x_msg` does for neighbors where it is actually known -- this is what
    prevents the trivial "just copy your own answer" leakage a naive
    self-inclusive convolution would allow.
    """

    def __init__(self, in_dim, out_dim, init_lengthscale=1.0, learnable_lengthscale=True):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        log_ls = torch.log(torch.tensor(float(max(init_lengthscale, 1e-6))))
        if learnable_lengthscale:
            self.log_lengthscale = nn.Parameter(log_ls)
        else:
            self.register_buffer("log_lengthscale", log_ls)

    def forward(self, x_self, x_msg, edge_index, edge_dist, n_nodes):
        src, dst = edge_index
        lengthscale = self.log_lengthscale.exp().clamp(min=1e-6)
        weight = torch.exp(-0.5 * (edge_dist / lengthscale) ** 2)
        messages = x_msg[src]
        agg = _scatter_weighted_mean(messages, weight, dst, n_nodes)
        return F.relu(self.lin_self(x_self) + self.lin_neigh(agg))


# ==========================================================
# Networks
# ==========================================================

class _SpatialGNN(nn.Module):
    """Stack of covariate-only message-passing layers + a linear head."""

    def __init__(self, in_dim, hidden_dim, n_layers, conv_type="mean", dropout=0.1):
        super().__init__()
        Conv = MeanConv if conv_type == "mean" else GATConv
        dims = [in_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList([Conv(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        n_nodes = x.shape[0]
        h = x
        for conv in self.convs:
            h = self.dropout(F.relu(conv(h, edge_index, n_nodes)))
        return self.head(h).squeeze(-1)


class _KCNNet(nn.Module):
    """KCNConv (leave-self-out, distance-kernel-weighted) for layer 1,
    plain MeanConv on the resulting hidden states for any further layers."""

    def __init__(self, in_dim, hidden_dim, n_layers, init_lengthscale, learnable_lengthscale=True, dropout=0.1):
        super().__init__()
        self.kcn_layer = KCNConv(in_dim, hidden_dim, init_lengthscale, learnable_lengthscale)
        self.convs = nn.ModuleList([MeanConv(hidden_dim, hidden_dim) for _ in range(max(0, n_layers - 1))])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_self, x_msg, edge_index, edge_dist):
        n_nodes = x_self.shape[0]
        h = self.dropout(self.kcn_layer(x_self, x_msg, edge_index, edge_dist, n_nodes))
        for conv in self.convs:
            h = self.dropout(F.relu(conv(h, edge_index, n_nodes)))
        return self.head(h).squeeze(-1)


class _IGNNKNet(nn.Module):
    """Stack of message-passing layers over ``[covariates, mask*value,
    mask]`` node features -- self and message are the *same* (consistently
    masked) vector; it is the *masking pattern itself* (see :class:`IGNNK`'s
    training loop), not the architecture, that prevents leakage here."""

    def __init__(self, in_dim, hidden_dim, n_layers, conv_type="mean", dropout=0.1):
        super().__init__()
        Conv = MeanConv if conv_type == "mean" else GATConv
        dims = [in_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList([Conv(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        n_nodes = x.shape[0]
        h = x
        for conv in self.convs:
            h = self.dropout(F.relu(conv(h, edge_index, n_nodes)))
        return self.head(h).squeeze(-1)


# ==========================================================
# Shared kriging-of-residuals helpers
# ==========================================================

def _fit_residual_kriging(coords, residuals, variogram_model, variogram_kwargs, use_gpu="auto"):
    vg_kwargs = {"model": variogram_model, "estimator": "matheron"}
    vg_kwargs.update(variogram_kwargs or {})
    vg_kwargs["use_gpu"] = resolve_cupy_use_gpu(use_gpu)
    variogram = Variogram(coords, residuals, **vg_kwargs)
    variogram.fit()
    krige = OrdinaryKriging(variogram, use_gpu=vg_kwargs["use_gpu"]).fit(coords, residuals)
    return variogram, krige


def _combine_with_kriging(krige, trend_pred, coords_pred, return_std):
    if return_std:
        resid_pred, resid_std = krige.predict(coords_pred, return_variance=True)
        return trend_pred + resid_pred, resid_std
    return trend_pred + krige.predict(coords_pred)


# ==========================================================
# 1. Plain spatial GNN regression kriging
# ==========================================================

class GNNRegressionKriging:
    """
    Plain spatial Graph Neural Network trend model + kriging of residuals.

    Builds a k-nearest-neighbor spatial graph over the training coordinates,
    trains a small message-passing GNN on the covariates to predict the
    target, then krige the residuals -- pygstat's standard regression
    -kriging recipe, with a GNN in place of an MLP/ensemble. See :class:`KCN`
    and :class:`IGNNK` for two architectures where the *target values*
    themselves (not just covariates) also propagate through the graph.

    Parameters
    ----------
    k_neighbors : int, default=10
        Number of spatial nearest neighbors per node.
    hidden_dim : int, default=64
    n_layers : int, default=2
    conv_type : {'mean', 'gat'}, default='mean'
        'mean' -- GraphSAGE-style mean aggregation. 'gat' -- learned
        attention weights per edge (Velickovic et al., 2018).
    dropout : float, default=0.1
    learning_rate, weight_decay : float
    max_epochs, patience : int
        Training budget and early-stopping patience (on a held-out node
        fraction, `val_fraction` in :meth:`fit`).
    device : {'auto', 'cpu', 'cuda', ...}, default='auto'
        Device for the PyTorch GNN trend (training and inference).
    use_gpu : bool or 'auto', default='auto'
        Passed through to residual `Variogram` fitting and to
        `OrdinaryKriging` of the residuals (CuPy). Independent of `device`.
    variogram_model : str, default='spherical'
    variogram_kwargs : dict, optional

    Examples
    --------
    >>> rk = GNNRegressionKriging(k_neighbors=12, hidden_dim=64)
    >>> rk.fit(X_train, y_train, coords_train)
    >>> y_pred = rk.predict(X_grid, coords_grid)
    """

    def __init__(
        self,
        k_neighbors=10,
        hidden_dim=64,
        n_layers=2,
        conv_type="mean",
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_epochs=300,
        patience=20,
        device="auto",
        use_gpu="auto",
        variogram_model="spherical",
        variogram_kwargs=None,
        seed=42,
    ):
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.conv_type = conv_type
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)
        self.variogram_model = variogram_model
        self.variogram_kwargs = variogram_kwargs or {}
        self.seed = seed
        self.is_fitted_ = False

    def fit(self, X, y, coords, val_fraction=0.15, verbose=True):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        coords = np.asarray(coords, dtype=float)
        n = len(X)

        seed_torch(self.seed)
        self.device_ = _resolve_device(self.device)
        self.scaler_X_ = StandardScaler()
        X_scaled = self.scaler_X_.fit_transform(X)
        self.scaler_y_ = StandardScaler()
        y_scaled = self.scaler_y_.fit_transform(y.reshape(-1, 1)).flatten()

        self.edge_index_, self.edge_dist_ = _knn_edges(coords, self.k_neighbors)
        edge_index = self.edge_index_.to(self.device_)

        x_t = torch.tensor(X_scaled, dtype=torch.float32, device=self.device_)
        y_t = torch.tensor(y_scaled, dtype=torch.float32, device=self.device_)

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_val = max(1, int(val_fraction * n))
        val_idx = torch.tensor(perm[:n_val], device=self.device_)
        train_idx = torch.tensor(perm[n_val:], device=self.device_)

        self.model_ = _SpatialGNN(X.shape[1], self.hidden_dim, self.n_layers, self.conv_type, self.dropout).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        best_val, patience_ctr = float("inf"), 0
        best_state = self.model_.state_dict()
        self.train_losses_, self.val_losses_ = [], []

        for epoch in range(self.max_epochs):
            self.model_.train()
            optimizer.zero_grad()
            out = self.model_(x_t, edge_index)
            loss = criterion(out[train_idx], y_t[train_idx])
            loss.backward()
            optimizer.step()
            self.train_losses_.append(loss.item())

            self.model_.eval()
            with torch.no_grad():
                out_val = self.model_(x_t, edge_index)
                val_loss = criterion(out_val[val_idx], y_t[val_idx]).item()
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

        self.model_.eval()
        with torch.no_grad():
            y_pred_scaled = self.model_(x_t, edge_index).cpu().numpy()
        y_pred = self.scaler_y_.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        residuals = y - y_pred

        self.coords_, self.X_, self.y_ = coords, X, y
        self.variogram_, self.krige_ = _fit_residual_kriging(
            coords, residuals, self.variogram_model, self.variogram_kwargs, self.use_gpu
        )
        self.is_fitted_ = True
        return self

    def predict(self, X_pred, coords_pred, return_std=False):
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet.")
        X_pred = np.asarray(X_pred, dtype=float)
        coords_pred = np.asarray(coords_pred, dtype=float)
        n_train = len(self.X_)

        new_edge_index, _ = _knn_edges(self.coords_, self.k_neighbors, query_coords=coords_pred)
        edge_index = torch.cat([self.edge_index_, new_edge_index], dim=1).to(self.device_)

        X_all = np.vstack([self.scaler_X_.transform(self.X_), self.scaler_X_.transform(X_pred)])
        x_t = torch.tensor(X_all, dtype=torch.float32, device=self.device_)

        self.model_.eval()
        with torch.no_grad():
            out = self.model_(x_t, edge_index).cpu().numpy()
        reg_pred = self.scaler_y_.inverse_transform(out[n_train:].reshape(-1, 1)).flatten()
        return _combine_with_kriging(self.krige_, reg_pred, coords_pred, return_std)

    def __repr__(self):
        return f"GNNRegressionKriging(conv_type='{self.conv_type}', k_neighbors={self.k_neighbors}, variogram_model='{self.variogram_model}')"


# ==========================================================
# 2. KCN -- Kriging Convolutional Networks
# ==========================================================

class KCN:
    """
    Kriging Convolutional Networks (Appleby, Liu & Liu, 2020) + kriging of
    residuals. See the module docstring and :class:`KCNConv` for the
    architecture; in short, every node's own target value is always hidden
    from itself, while its neighbors' known target values are aggregated
    through a learnable-lengthscale Gaussian-kernel-weighted convolution.

    Parameters
    ----------
    k_neighbors : int, default=10
    hidden_dim : int, default=64
    n_layers : int, default=2
        Total layers: 1 `KCNConv` (leave-self-out, kernel-weighted) plus
        `n_layers - 1` plain `MeanConv` layers on the resulting embeddings.
    init_lengthscale : float, optional
        Initial Gaussian-kernel lengthscale (same units as `coords`).
        Defaults to the median distance to the k-th nearest neighbor across
        the training set (a standard "median heuristic").
    learnable_lengthscale : bool, default=True
        If True, the lengthscale is a trainable parameter (initialized at
        `init_lengthscale`); if False, it stays fixed.
    dropout, learning_rate, weight_decay, max_epochs, patience, device,
    use_gpu, variogram_model, variogram_kwargs : as in :class:`GNNRegressionKriging`.

    Examples
    --------
    >>> rk = KCN(k_neighbors=12, hidden_dim=64)
    >>> rk.fit(X_train, y_train, coords_train)
    >>> y_pred = rk.predict(X_grid, coords_grid)
    """

    def __init__(
        self,
        k_neighbors=10,
        hidden_dim=64,
        n_layers=2,
        init_lengthscale=None,
        learnable_lengthscale=True,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_epochs=300,
        patience=20,
        device="auto",
        use_gpu="auto",
        variogram_model="spherical",
        variogram_kwargs=None,
        seed=42,
    ):
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.init_lengthscale = init_lengthscale
        self.learnable_lengthscale = learnable_lengthscale
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)
        self.variogram_model = variogram_model
        self.variogram_kwargs = variogram_kwargs or {}
        self.seed = seed
        self.is_fitted_ = False

    @staticmethod
    def _make_features(X_scaled, y_scaled, mask):
        """`x_self` always hides the node's own value; `x_msg` reveals it
        wherever `mask` says it is actually known."""
        y_masked = (y_scaled * mask).astype(np.float32)
        x_self = np.concatenate([X_scaled, np.zeros_like(y_masked)[:, None], np.zeros_like(mask)[:, None]], axis=1)
        x_msg = np.concatenate([X_scaled, y_masked[:, None], mask[:, None].astype(np.float32)], axis=1)
        return x_self, x_msg

    def fit(self, X, y, coords, val_fraction=0.15, verbose=True):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        coords = np.asarray(coords, dtype=float)
        n = len(X)

        seed_torch(self.seed)
        self.device_ = _resolve_device(self.device)
        self.scaler_X_ = StandardScaler()
        X_scaled = self.scaler_X_.fit_transform(X)
        self.scaler_y_ = StandardScaler()
        y_scaled = self.scaler_y_.fit_transform(y.reshape(-1, 1)).flatten()

        self.edge_index_, self.edge_dist_ = _knn_edges(coords, self.k_neighbors)
        edge_index = self.edge_index_.to(self.device_)
        edge_dist = self.edge_dist_.to(self.device_)

        if self.init_lengthscale is None:
            self.init_lengthscale_ = float(np.median(self.edge_dist_.numpy()))
        else:
            self.init_lengthscale_ = float(self.init_lengthscale)

        # Every training node acts as a known message source (mask=1) at all
        # times -- the leave-self-out property comes from KCNConv's x_self
        # always masking its own value, not from any epoch-random masking.
        mask_all_known = np.ones(n, dtype=np.float32)
        x_self, x_msg = self._make_features(X_scaled, y_scaled, mask_all_known)
        x_self_t = torch.tensor(x_self, dtype=torch.float32, device=self.device_)
        x_msg_t = torch.tensor(x_msg, dtype=torch.float32, device=self.device_)
        y_t = torch.tensor(y_scaled, dtype=torch.float32, device=self.device_)

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_val = max(1, int(val_fraction * n))
        val_idx = torch.tensor(perm[:n_val], device=self.device_)
        train_idx = torch.tensor(perm[n_val:], device=self.device_)

        in_dim = X.shape[1] + 2
        self.model_ = _KCNNet(
            in_dim, self.hidden_dim, self.n_layers, self.init_lengthscale_, self.learnable_lengthscale, self.dropout
        ).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        best_val, patience_ctr = float("inf"), 0
        best_state = self.model_.state_dict()
        self.train_losses_, self.val_losses_ = [], []

        for epoch in range(self.max_epochs):
            self.model_.train()
            optimizer.zero_grad()
            out = self.model_(x_self_t, x_msg_t, edge_index, edge_dist)
            loss = criterion(out[train_idx], y_t[train_idx])
            loss.backward()
            optimizer.step()
            self.train_losses_.append(loss.item())

            self.model_.eval()
            with torch.no_grad():
                out_val = self.model_(x_self_t, x_msg_t, edge_index, edge_dist)
                val_loss = criterion(out_val[val_idx], y_t[val_idx]).item()
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

        # A node's x_self *always* hides its own value (by construction, not
        # by masking convention), so a single forward pass already gives
        # honest, leakage-free trend predictions for every training node.
        self.model_.eval()
        with torch.no_grad():
            y_pred_scaled = self.model_(x_self_t, x_msg_t, edge_index, edge_dist).cpu().numpy()
        y_pred = self.scaler_y_.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        residuals = y - y_pred

        self.coords_, self.X_, self.y_ = coords, X, y
        self.variogram_, self.krige_ = _fit_residual_kriging(
            coords, residuals, self.variogram_model, self.variogram_kwargs, self.use_gpu
        )
        self.is_fitted_ = True
        return self

    def predict(self, X_pred, coords_pred, return_std=False):
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet.")
        X_pred = np.asarray(X_pred, dtype=float)
        coords_pred = np.asarray(coords_pred, dtype=float)
        n_train = len(self.X_)

        new_edge_index, new_edge_dist = _knn_edges(self.coords_, self.k_neighbors, query_coords=coords_pred)
        edge_index = torch.cat([self.edge_index_, new_edge_index], dim=1).to(self.device_)
        edge_dist = torch.cat([self.edge_dist_, new_edge_dist]).to(self.device_)

        X_train_scaled = self.scaler_X_.transform(self.X_)
        X_pred_scaled = self.scaler_X_.transform(X_pred)
        X_all = np.vstack([X_train_scaled, X_pred_scaled])
        y_train_scaled = self.scaler_y_.transform(self.y_.reshape(-1, 1)).flatten()

        n_all = len(X_all)
        y_all = np.zeros(n_all, dtype=np.float32)
        y_all[:n_train] = y_train_scaled
        mask_all = np.zeros(n_all, dtype=np.float32)
        mask_all[:n_train] = 1.0  # training nodes are always-known message sources; new nodes never send messages

        x_self, x_msg = self._make_features(X_all, y_all, mask_all)
        x_self_t = torch.tensor(x_self, dtype=torch.float32, device=self.device_)
        x_msg_t = torch.tensor(x_msg, dtype=torch.float32, device=self.device_)

        self.model_.eval()
        with torch.no_grad():
            out = self.model_(x_self_t, x_msg_t, edge_index, edge_dist).cpu().numpy()
        reg_pred = self.scaler_y_.inverse_transform(out[n_train:].reshape(-1, 1)).flatten()
        return _combine_with_kriging(self.krige_, reg_pred, coords_pred, return_std)

    def __repr__(self):
        ls = f"{self.init_lengthscale_:.3g}" if hasattr(self, "init_lengthscale_") else "?"
        return f"KCN(k_neighbors={self.k_neighbors}, lengthscale~{ls}, variogram_model='{self.variogram_model}')"


# ==========================================================
# 3. IGNNK -- Inductive Graph Neural Networks for Kriging
# ==========================================================

class IGNNK:
    """
    Inductive Graph Neural Network for Kriging (Wu, Cui, Nie & Wang, 2021),
    adapted from its original spatiotemporal-sensor-network setting to
    static regression kriging, + kriging of residuals.

    Every node's input is ``[covariates, mask * value, mask]``. Training
    repeatedly masks a *random* fraction of the (non-validation) nodes each
    epoch and trains the network to reconstruct their value from the graph
    -- exposing it to many different missing-data patterns during training
    so it generalizes to genuinely new, previously-unseen locations at
    prediction time (masked identically there: `mask=0`).

    Parameters
    ----------
    k_neighbors, hidden_dim : as in :class:`GNNRegressionKriging`.
    n_layers : int, default=3
        IGNNK typically uses more layers than a plain covariate GNN, since
        useful "known value" information may be several hops away.
    conv_type : {'mean', 'gat'}, default='mean'
    mask_ratio : float, default=0.3
        Fraction of the (non-validation) training nodes masked -- and used
        as that epoch's reconstruction targets -- at each training step.
    dropout, learning_rate, weight_decay, max_epochs, patience, device,
    use_gpu, variogram_model, variogram_kwargs : as in :class:`GNNRegressionKriging`.

    Examples
    --------
    >>> rk = IGNNK(k_neighbors=12, hidden_dim=64, mask_ratio=0.3)
    >>> rk.fit(X_train, y_train, coords_train)
    >>> y_pred = rk.predict(X_grid, coords_grid)
    """

    def __init__(
        self,
        k_neighbors=10,
        hidden_dim=64,
        n_layers=3,
        conv_type="mean",
        mask_ratio=0.3,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_epochs=300,
        patience=30,
        device="auto",
        use_gpu="auto",
        variogram_model="spherical",
        variogram_kwargs=None,
        seed=42,
    ):
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.conv_type = conv_type
        self.mask_ratio = mask_ratio
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.use_gpu = resolve_cupy_use_gpu(use_gpu)
        self.variogram_model = variogram_model
        self.variogram_kwargs = variogram_kwargs or {}
        self.seed = seed
        self.is_fitted_ = False

    def fit(self, X, y, coords, val_fraction=0.15, n_residual_folds=5, verbose=True):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        coords = np.asarray(coords, dtype=float)
        n = len(X)

        seed_torch(self.seed)
        self.device_ = _resolve_device(self.device)
        self.scaler_X_ = StandardScaler()
        X_scaled = self.scaler_X_.fit_transform(X)
        self.scaler_y_ = StandardScaler()
        y_scaled = self.scaler_y_.fit_transform(y.reshape(-1, 1)).flatten()

        self.edge_index_, self.edge_dist_ = _knn_edges(coords, self.k_neighbors)
        edge_index = self.edge_index_.to(self.device_)

        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=self.device_)
        y_t = torch.tensor(y_scaled, dtype=torch.float32, device=self.device_)

        def make_x(mask_np):
            mask_t = torch.tensor(mask_np, dtype=torch.float32, device=self.device_)
            return torch.cat([X_t, (y_t * mask_t).unsqueeze(1), mask_t.unsqueeze(1)], dim=1)

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_val = max(1, int(val_fraction * n))
        val_idx = perm[:n_val]
        pool_idx = perm[n_val:]  # eligible for random per-epoch masking
        val_idx_t = torch.tensor(val_idx, device=self.device_)
        n_mask_per_epoch = max(1, int(self.mask_ratio * len(pool_idx)))

        in_dim = X.shape[1] + 2
        self.model_ = _IGNNKNet(in_dim, self.hidden_dim, self.n_layers, self.conv_type, self.dropout).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        best_val, patience_ctr = float("inf"), 0
        best_state = self.model_.state_dict()
        self.train_losses_, self.val_losses_ = [], []

        for epoch in range(self.max_epochs):
            self.model_.train()
            epoch_mask_idx = rng.choice(pool_idx, size=n_mask_per_epoch, replace=False)
            mask_np = np.ones(n, dtype=np.float32)
            mask_np[val_idx] = 0.0        # validation nodes are always hidden, even during training steps
            mask_np[epoch_mask_idx] = 0.0  # this step's random reconstruction targets

            optimizer.zero_grad()
            out = self.model_(make_x(mask_np), edge_index)
            target_idx = torch.tensor(epoch_mask_idx, device=self.device_)
            loss = criterion(out[target_idx], y_t[target_idx])
            loss.backward()
            optimizer.step()
            self.train_losses_.append(loss.item())

            # Validation mirrors real prediction: everything except the
            # held-out validation nodes is fully known.
            self.model_.eval()
            with torch.no_grad():
                val_mask_np = np.ones(n, dtype=np.float32)
                val_mask_np[val_idx] = 0.0
                out_val = self.model_(make_x(val_mask_np), edge_index)
                val_loss = criterion(out_val[val_idx_t], y_t[val_idx_t]).item()
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

        # Honest (self-never-observed) trend predictions for every training
        # node via k-fold masked reconstruction: with every node fully known
        # (mask=1 everywhere), the network would just copy each node's own
        # input value straight through -- a degenerate, uninformative
        # "residual" of ~0 -- since IGNNK's own value is literally part of
        # its input feature when unmasked. A handful of full-graph forward
        # passes (one per fold, each masking only that fold) instead gives
        # every node an out-of-fold estimate, exactly like the random
        # masking used during training, at a fraction of the cost of doing
        # this individually (leave-one-out) for every node.
        self.model_.eval()
        n_folds = max(1, min(n_residual_folds, n))
        fold_id = rng.permutation(n) % n_folds
        y_pred_scaled = np.zeros(n, dtype=np.float32)
        with torch.no_grad():
            for f in range(n_folds):
                mask_np = np.ones(n, dtype=np.float32)
                mask_np[fold_id == f] = 0.0
                out = self.model_(make_x(mask_np), edge_index).cpu().numpy()
                y_pred_scaled[fold_id == f] = out[fold_id == f]
        y_pred = self.scaler_y_.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        residuals = y - y_pred

        self.coords_, self.X_, self.y_ = coords, X, y
        self.variogram_, self.krige_ = _fit_residual_kriging(
            coords, residuals, self.variogram_model, self.variogram_kwargs, self.use_gpu
        )
        self.is_fitted_ = True
        return self

    def predict(self, X_pred, coords_pred, return_std=False):
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet.")
        X_pred = np.asarray(X_pred, dtype=float)
        coords_pred = np.asarray(coords_pred, dtype=float)
        n_train = len(self.X_)

        new_edge_index, _ = _knn_edges(self.coords_, self.k_neighbors, query_coords=coords_pred)
        edge_index = torch.cat([self.edge_index_, new_edge_index], dim=1).to(self.device_)

        X_train_scaled = self.scaler_X_.transform(self.X_)
        X_pred_scaled = self.scaler_X_.transform(X_pred)
        X_all = np.vstack([X_train_scaled, X_pred_scaled])
        y_train_scaled = self.scaler_y_.transform(self.y_.reshape(-1, 1)).flatten()

        n_all = len(X_all)
        y_all = np.zeros(n_all, dtype=np.float32)
        y_all[:n_train] = y_train_scaled
        mask_all = np.zeros(n_all, dtype=np.float32)
        mask_all[:n_train] = 1.0  # training nodes fully known; new/query nodes fully unknown

        x_in = np.concatenate([X_all, y_all[:, None], mask_all[:, None]], axis=1)
        x_t = torch.tensor(x_in, dtype=torch.float32, device=self.device_)

        self.model_.eval()
        with torch.no_grad():
            out = self.model_(x_t, edge_index).cpu().numpy()
        reg_pred = self.scaler_y_.inverse_transform(out[n_train:].reshape(-1, 1)).flatten()
        return _combine_with_kriging(self.krige_, reg_pred, coords_pred, return_std)

    def __repr__(self):
        return f"IGNNK(k_neighbors={self.k_neighbors}, mask_ratio={self.mask_ratio}, variogram_model='{self.variogram_model}')"
