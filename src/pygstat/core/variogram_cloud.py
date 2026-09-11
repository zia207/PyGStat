"""
Variogram cloud: pairwise distances and semivariances, with optional interactive plotting.

Port of gstat's plot.variogramCloud (identify/digitize point pairs on a cloud plot).
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def compute_variogram_cloud(coords, values, maxlag=None):
    """
    Compute the variogram cloud: all pairwise distances and semivariances.

    Parameters
    ----------
    coords : array-like, shape (n, d)
        Spatial coordinates.
    values : array-like, shape (n,)
        Variable values at each location.
    maxlag : float or None
        If given, only pairs with distance <= maxlag are returned.

    Returns
    -------
    dist : ndarray
        Pairwise distances (upper triangle, no diagonal).
    gamma : ndarray
        Semivariance 0.5 * (z_i - z_j)^2 for each pair.
    left : ndarray
        Index of first point in each pair (0-based).
    right : ndarray
        Index of second point in each pair (0-based).
    """
    coords = np.asarray(coords)
    values = np.asarray(values).ravel()
    n = len(values)
    if n != len(coords):
        raise ValueError("coords and values must have same length")

    dists = squareform(pdist(coords))
    i_upper, j_upper = np.triu_indices(n, k=1)
    dist = dists[i_upper, j_upper].astype(float)
    diff = values[i_upper] - values[j_upper]
    gamma = 0.5 * (diff ** 2)

    left = i_upper
    right = j_upper

    if maxlag is not None:
        mask = dist <= maxlag
        dist = dist[mask]
        gamma = gamma[mask]
        left = left[mask]
        right = right[mask]

    return dist, gamma, left, right


class VariogramCloud:
    """
    Container for variogram cloud data (pairwise dist, gamma, and point-pair indices).

    Can be created from coordinates and values via compute_variogram_cloud, or from
    arrays (dist, gamma, left, right) directly. Used by plot_variogram_cloud.
    """

    def __init__(self, dist, gamma, left=None, right=None):
        dist = np.asarray(dist)
        gamma = np.asarray(gamma)
        if len(dist) != len(gamma):
            raise ValueError("dist and gamma must have same length")
        self.dist = dist
        self.gamma = gamma
        n = len(dist)
        if left is None:
            left = np.arange(n)  # placeholder
        if right is None:
            right = np.arange(n)
        self.left = np.asarray(left)
        self.right = np.asarray(right)
        self._sel = None
        self._text = None
        self._poly = None
        self._ppairs = None

    @classmethod
    def from_coords(cls, coords, values, maxlag=None):
        """Build cloud from coordinates and values."""
        dist, gamma, left, right = compute_variogram_cloud(coords, values, maxlag=maxlag)
        return cls(dist=dist, gamma=gamma, left=left, right=right)

    @property
    def point_pairs(self):
        """Return selected point pairs as (left, right) array; None if none selected."""
        return self._ppairs

    def _point_pairs_array(self):
        """(N, 2) array of (left, right) for all pairs; 1-based for display like R."""
        return np.column_stack([self.left + 1, self.right + 1])


def _point_in_polygon(points_xy, poly_xy):
    """Return boolean mask True for points inside polygon. poly_xy is (N, 2)."""
    from matplotlib.path import Path
    path = Path(poly_xy)
    return path.contains_points(points_xy)


def plot_variogram_cloud(
    cloud,
    ax=None,
    identify=False,
    digitize=False,
    xlim=None,
    ylim=None,
    xlab="distance",
    ylab="semivariance",
    keep=False,
    **kwargs
):
    """
    Plot variogram cloud (distance vs semivariance) and optionally identify or digitize point pairs.

    Port of gstat's plot.variogramCloud.

    Parameters
    ----------
    cloud : VariogramCloud or tuple (dist, gamma [, left, right])
        Cloud data. If tuple, (dist, gamma) or (dist, gamma, left, right).
    ax : matplotlib axes, optional
        Axes to plot on; if None, current axes or new figure.
    identify : bool, default False
        If True, run interactive mode: user clicks points to label them; further clicks
        (or key) end selection. Returns the selected point pairs (left, right indices, 1-based).
    digitize : bool, default False
        If True, user clicks to define a polygon; point pairs inside the polygon are
        selected. Right-click or close figure to finish. Returns selected point pairs.
    xlim : tuple (xmin, xmax), optional
    ylim : tuple (ymin, ymax), optional
    xlab : str
        X-axis label.
    ylab : str
        Y-axis label.
    keep : bool, default False
        If True and (identify or digitize), store selection in the cloud object and return
        the cloud; subsequent plots of the same cloud will show labels or polygon.
    **kwargs
        Passed to ax.scatter (non-interactive) or ax.plot (interactive).

    Returns
    -------
    If identify or digitize is True and keep is False:
        point_pairs : ndarray (N, 2)
            Selected (left, right) point indices (1-based).
    If identify or digitize is True and keep is True:
        cloud : VariogramCloud
            Same cloud with attributes set for future plotting.
    If identify and digitize are False:
        ax : matplotlib axes
    """
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plot_variogram_cloud; install with pip install pygstat[plot]")

    if isinstance(cloud, (list, tuple)):
        if len(cloud) == 2:
            cloud = VariogramCloud(cloud[0], cloud[1])
        else:
            cloud = VariogramCloud(cloud[0], cloud[1], left=cloud[2], right=cloud[3])
    if not isinstance(cloud, VariogramCloud):
        raise TypeError("cloud must be VariogramCloud or (dist, gamma [, left, right])")

    dist = cloud.dist
    gamma = cloud.gamma
    ppairs = cloud._point_pairs_array()  # (N, 2) 1-based

    if xlim is None:
        xlim = (0, float(np.nanmax(dist)))
    if ylim is None:
        ylim = (0, float(np.nanmax(gamma)))

    if ax is None:
        ax = plt.gca()

    # Re-plot with stored selection/labels/polygon (no interactive mode)
    if not identify and not digitize:
        sel = getattr(cloud, "_sel", None)
        lab = getattr(cloud, "_text", None)
        poly = getattr(cloud, "_poly", None)
        if sel is not None and lab is not None:
            ax.scatter(dist, gamma, **{**dict(s=10, c="k", alpha=0.6), **kwargs})
            for i, (xi, yi, li) in enumerate(zip(dist[sel], gamma[sel], lab)):
                pos = getattr(cloud, "_sel_pos", None)
                pos_i = pos[i] if pos is not None and i < len(pos) else 4
                ax.annotate(str(li), (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=8)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            return ax
        if poly is not None:
            ax.scatter(dist, gamma, **{**dict(s=10, c="k", alpha=0.6), **kwargs})
            ax.plot(poly[:, 0], poly[:, 1], "r-", linewidth=2)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            return ax
        # Default: scatter only
        ax.scatter(dist, gamma, **{**dict(s=10, c="k", alpha=0.6), **kwargs})
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        return ax

    # Interactive: identify or digitize
    ax.scatter(dist, gamma, **{**dict(s=10, c="k", alpha=0.6), **kwargs})
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if identify:
        print("Click points to identify (key press or close figure to stop).")
        pts = ax.scatter(dist, gamma, s=10, c="k", alpha=0.6, picker=5)
        selected_indices = []
        selected_labels = []
        selected_pos = []
        result_holder = []

        def on_pick(event):
            if event.artist != pts:
                return
            ind = event.ind
            for i in ind:
                if i not in selected_indices:
                    selected_indices.append(i)
                    selected_labels.append(f"{ppairs[i, 0]},{ppairs[i, 1]}")
                    selected_pos.append(4)

        def on_key(event):
            plt.disconnect(cid_key)
            plt.disconnect(cid_pick)
            ret_ppairs = ppairs[selected_indices] if selected_indices else np.empty((0, 2))
            if keep:
                cloud._sel = np.array(selected_indices)
                cloud._text = selected_labels
                cloud._sel_pos = selected_pos
                cloud._ppairs = ret_ppairs
                result_holder.append(cloud)
            else:
                result_holder.append(ret_ppairs)
            plt.close()

        cid_pick = plt.gcf().canvas.mpl_connect("pick_event", on_pick)
        cid_key = plt.gcf().canvas.mpl_connect("key_press_event", on_key)
        plt.show(block=True)
        if result_holder:
            return result_holder[0]
        if keep:
            cloud._sel = np.array(selected_indices)
            cloud._text = selected_labels
            cloud._sel_pos = selected_pos
            cloud._ppairs = ppairs[selected_indices] if selected_indices else None
            return cloud
        return ppairs[selected_indices] if selected_indices else np.empty((0, 2))

    if digitize:
        print("Click to define polygon (right-click or key to close and finish).")
        poly_xy = []
        result_holder = []

        def on_click(event):
            if event.inaxes != ax or event.button not in (1, 3):
                return
            if event.button == 1:
                poly_xy.append([event.xdata, event.ydata])
                if len(poly_xy) > 1:
                    ax.plot([poly_xy[-2][0], poly_xy[-1][0]], [poly_xy[-2][1], poly_xy[-1][1]], "r-")
                ax.figure.canvas.draw_idle()
            else:
                if len(poly_xy) >= 3:
                    poly_xy.append(poly_xy[0])
                    ax.plot([poly_xy[-2][0], poly_xy[-1][0]], [poly_xy[-2][1], poly_xy[-1][1]], "r-")
                    _finish_digitize()

        def on_key_dig(event):
            if len(poly_xy) >= 3:
                poly_xy.append(poly_xy[0])
                if len(poly_xy) > 1:
                    ax.plot([poly_xy[-2][0], poly_xy[-1][0]], [poly_xy[-2][1], poly_xy[-1][1]], "r-")
                _finish_digitize()

        def _finish_digitize():
            plt.disconnect(cid_click)
            plt.disconnect(cid_key_dig)
            poly_arr = np.array(poly_xy)
            inside = _point_in_polygon(np.column_stack([dist, gamma]), poly_arr)
            ret_ppairs = ppairs[inside]
            if keep:
                cloud._poly = poly_arr
                cloud._ppairs = ret_ppairs
                result_holder.append(cloud)
            else:
                result_holder.append(ret_ppairs)
            plt.close()

        cid_click = plt.gcf().canvas.mpl_connect("button_press_event", on_click)
        cid_key_dig = plt.gcf().canvas.mpl_connect("key_press_event", on_key_dig)
        plt.show(block=True)
        if result_holder:
            return result_holder[0]
        if len(poly_xy) >= 3:
            poly_arr = np.array(poly_xy)
            inside = _point_in_polygon(np.column_stack([dist, gamma]), poly_arr)
            if keep:
                cloud._poly = poly_arr
                cloud._ppairs = ppairs[inside]
                return cloud
            return ppairs[inside]
        return np.empty((0, 2))

    return ax
