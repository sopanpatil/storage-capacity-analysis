"""Config loading and access to the Caravan v1.6 + GRDC-Caravan v0.6 files."""
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=ROOT / "config.yaml"):
    """Read config.yaml, then any overrides in config.local.yaml (the machine's own paths; not tracked)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    local = Path(path).with_name("config.local.yaml")
    if local.exists():
        with open(local) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = p if p.is_absolute() else ROOT / p
    return cfg


def source_of(gauge_id):
    return gauge_id.split("_", 1)[0]


def timeseries_path(cfg, gauge_id):
    src = source_of(gauge_id)
    base = cfg["paths"]["grdc"] if src == "GRDC" else cfg["paths"]["caravan"]
    return base / "timeseries" / "netcdf" / src.lower() / f"{gauge_id}.nc"


def attribute_files(cfg, kind):
    """All attribute CSVs of one kind ('caravan', 'other', 'hydroatlas'), both datasets."""
    files = sorted((cfg["paths"]["caravan"] / "attributes").glob(f"*/attributes_{kind}_*.csv"))
    files.append(cfg["paths"]["grdc"] / "attributes" / "grdc" / f"attributes_{kind}_grdc.csv")
    return files


# HydroATLAS fields used for screening and model setup. dor_pc_pva is percent x 10.
HYDROATLAS_FIELDS = ["ele_mt_sav", "gla_pc_sse", "dor_pc_pva", "for_pc_sse"]


def load_attributes(cfg):
    """One row per catchment: 'other' + 'caravan' attributes, selected HydroATLAS fields, plus GRDC
    nat_id/quality/comment."""
    def read(kind):
        return pd.concat([pd.read_csv(f, dtype={"gauge_id": str}) for f in attribute_files(cfg, kind)],
                         ignore_index=True)
    att = read("other").merge(read("caravan"), on="gauge_id", validate="1:1")
    grdc = pd.read_csv(cfg["paths"]["grdc"] / "attributes" / "grdc" / "attributes_additional_grdc.csv",
                       dtype={"gauge_id": str, "nat_id": str})
    att = att.merge(grdc[["gauge_id", "nat_id", "quality", "comment"]].rename(
        columns={"quality": "grdc_quality", "comment": "grdc_comment"}), on="gauge_id", how="left")
    hydroatlas = read("hydroatlas")[["gauge_id"] + HYDROATLAS_FIELDS]
    att = att.merge(hydroatlas, on="gauge_id", validate="1:1")
    att.insert(1, "source", att.gauge_id.map(source_of))
    return att


def load_timeseries(cfg, gauge_id, variables=None):
    """Daily time series as a DataFrame indexed by date (all variables if none given).

    Reads with netCDF4 directly; ~5x faster than xarray for these small files.
    """
    with netCDF4.Dataset(timeseries_path(cfg, gauge_id)) as ds:
        units = ds["date"].units
        assert units.startswith("days since "), units
        index = pd.to_datetime(units[len("days since "):]) + pd.to_timedelta(ds["date"][:], unit="D")
        variables = variables or [v for v in ds.variables if v != "date"]
        data = {v: np.ma.filled(ds[v][:].astype("float64"), np.nan) for v in variables}
    return pd.DataFrame(data, index=pd.DatetimeIndex(index, name="date"))


def load_forcing(cfg, gauge_id, variables, source=None):
    """load_timeseries with precipitation, PET and mean temperature optionally replaced from another
    forcing source (config `forcing_sources`, CAMELS-GB v2 only); Tmin/Tmax are shifted by the change in
    mean temperature. Days the source does not cover become NaN. Column names stay Caravan's."""
    f = cfg["forcing"]
    if not source:
        return load_timeseries(cfg, gauge_id, variables)
    src = cfg["forcing_sources"][source]
    if source_of(gauge_id) != "camelsgb":
        raise ValueError(f"{source} forcing exists only for camelsgb catchments, not {gauge_id}")
    ts = load_timeseries(cfg, gauge_id, sorted(set(variables) | {f["tmean"]}))
    path, = (ROOT / src["dir"]).glob(f"camels_gb_v2_hydromet_daily_timeseries_{gauge_id.split('_', 1)[1]}_*.csv")
    gb = pd.read_csv(path, index_col="date", parse_dates=True).reindex(ts.index)
    dt = gb[src["tmean"]] - ts[f["tmean"]]
    for k in ("tmin", "tmax"):
        ts[f[k]] = ts[f[k]] + dt if f[k] in ts else None
    for k in ("precip", "pet", "tmean"):
        ts[f[k]] = gb[src[k]]
    return ts[variables]


def load_shapes(cfg, gauge_ids):
    """Catchment polygons (EPSG:4326) for the given gauge IDs, in the given order."""
    import geopandas as gpd
    gauge_ids = list(gauge_ids)
    parts = []
    for src in sorted({source_of(g) for g in gauge_ids}):
        base = cfg["paths"]["grdc"] if src == "GRDC" else cfg["paths"]["caravan"]
        shp = base / "shapefiles" / src.lower() / f"{src.lower()}_basin_shapes.shp"
        g = gpd.read_file(shp)
        if g.crs is None:  # one source ships without a .prj; all Caravan shapes are lon/lat
            g = g.set_crs("EPSG:4326")
        parts.append(g[g.gauge_id.isin(gauge_ids)])
    shapes = pd.concat(parts, ignore_index=True).set_index("gauge_id")
    # 251 screened polygons (camelsaus, camelscl, camelsbr) have self-touching rings; exactextract can
    # crash on them. make_valid fixes them without changing any area (checked 2026-10-09)
    import shapely
    bad = ~shapes.geometry.is_valid
    shapes.loc[bad, "geometry"] = shapely.make_valid(shapes.geometry[bad].values)
    missing = set(gauge_ids) - set(shapes.index)
    if missing:
        raise KeyError(f"no shape for {sorted(missing)[:5]} ({len(missing)} missing)")
    return shapes.loc[gauge_ids].reset_index()
