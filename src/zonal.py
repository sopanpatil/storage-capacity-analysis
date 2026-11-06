"""Catchment averages of gridded benchmark data over the Caravan polygons.

Uses exactextract, which weights each cell by the fraction of it inside the polygon. This matters for
coarse grids (e.g. 0.5 deg rooting depth) against 50 km2 catchments. On geographic (lon/lat) grids,
cells are also weighted by their spherical area, so the mean is area-weighted at any latitude.
"""
import rasterio
from exactextract import exact_extract


def catchment_means(raster_path, shapes, band=1):
    """Area-weighted mean of one raster band over each polygon, ignoring nodata.

    `shapes`: GeoDataFrame with `gauge_id`. Returns a DataFrame indexed by gauge_id with `mean` and
    `valid_frac`, the fraction of the polygon's area covered by valid (non-nodata) cells (to ~0.5% on
    lon/lat grids, where exactextract's spherical cell areas meet ellipsoidal polygon areas).
    """
    with rasterio.open(raster_path) as r:
        crs, geographic = r.crs, r.crs.is_geographic
        nodata = r.nodata
    shp = shapes[["gauge_id", "geometry"]].to_crs(crs)
    w = "(coverage_weight=area_spherical_m2)" if geographic else ""
    ops = [f"mean{w}", f"count{w}"]
    out = exact_extract(str(raster_path), shp, ops, include_cols=["gauge_id"], output="pandas",
                        strategy="feature-sequential")
    out.columns = ["gauge_id", "mean", "count"]
    out = out.set_index("gauge_id")
    if geographic:
        # with area weights, count is the valid area in m2
        area = shapes.set_index("gauge_id").to_crs("ESRI:54009").area.loc[out.index]
        out["valid_frac"] = out.pop("count") / area.values
    else:
        with rasterio.open(raster_path) as r:
            cell = abs(r.res[0] * r.res[1])
        out["valid_frac"] = out.pop("count") * cell / shp.set_index("gauge_id").area.loc[out.index].values
    out.attrs["nodata"] = nodata
    return out
