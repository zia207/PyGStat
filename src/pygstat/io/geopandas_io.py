"""GeoPandas helpers for extracting coordinates and values."""

import numpy as np


def from_geodataframe(gdf, value_col, coord_cols=None):
    """
    Extract coordinates and a value column from a GeoDataFrame.

    Point geometries use ``.x`` / ``.y``. Non-point geometries (polygons,
    multipoints, ...) are reduced to their centroids.
    """
    if coord_cols is None:
        geom = gdf.geometry
        geom_types = getattr(geom, "geom_type", None)
        if geom_types is not None and not geom_types.eq("Point").all():
            geom = geom.centroid
        coords = np.column_stack([geom.x, geom.y])
    else:
        coords = gdf[list(coord_cols)].values
    return np.asarray(coords, dtype=float), np.asarray(gdf[value_col].values)
