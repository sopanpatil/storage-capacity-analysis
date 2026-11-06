"""Static capacity benchmarks per catchment.

- soil_sr_mm: soil-derived capacity, effective rooting depth (Yang et al. 2016) x SoilGrids water-holding
  capacity (from src/benchmark_soilgrids.py), integrated layer by layer down to the rooting depth;
  below 2 m the 100-200 cm layer's capacity per mm is extended downward (Zhao et al. 2025; Stocker et
  al. 2023). Uses the catchment-mean rooting depth: the 0.5 deg rooting-depth grid is coarser than all
  but the largest catchments, so a per-pixel product would add little.
  soil_sr_excluded marks North American gauges north of 60 deg N, where Yang et al. call the deep values
  unreasonable; the value is kept but not used.
- rooting_depth_m: catchment mean of the Yang et al. map.
- zhao_sr_mm: GRACE/GRACE-FO Sr (Zhao et al. 2025), a regional (~300 km) value.
- stocker_scwdx80_mm: S_CWDX80 (Stocker et al. 2023).
Each with the fraction of the catchment covered by valid cells.

Needs src/prepare_benchmarks.py and src/benchmark_soilgrids.py first.
Writes data/benchmarks/static_<list name>.csv.

Usage: python src/benchmark_static.py [--list data/catalog/pilot_catchments.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_soilgrids import LAYERS
from caravan_io import ROOT, load_config, load_shapes
from zonal import catchment_means

MAPS = {"rooting_depth_m": "yang2016/rooting_depth_m.tif",
        "zhao_sr_mm": "zhao2025/sr_grace_mm.tif",
        "stocker_scwdx80_mm": "stocker2023/s_cwdx80_mm.tif"}


def soil_capacity(whc, depth_m):
    """Water-holding capacity (mm) from the surface down to depth_m, for one catchment.

    `whc`: mm per standard layer (keys as LAYERS). Below 2 m, the 100-200 cm layer's capacity per mm
    of depth continues downward."""
    depth = depth_m * 1000.0
    top, total = 0.0, 0.0
    for layer, thick in LAYERS.items():
        total += whc[layer] * np.clip(depth - top, 0.0, thick) / thick
        top += thick
    if depth > top:
        total += whc["100-200cm"] / LAYERS["100-200cm"] * (depth - top)
    return total


def load_catalog_coords(ids):
    c = pd.read_csv(ROOT / "data" / "catalog" / "catchments.csv", usecols=["gauge_id", "gauge_lat", "gauge_lon"])
    return c.set_index("gauge_id").loc[list(ids)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/pilot_catchments.csv")
    a = ap.parse_args()
    cfg = load_config()
    name = Path(a.list).stem
    ids = pd.read_csv(ROOT / a.list, dtype={"gauge_id": str}).gauge_id
    shapes = load_shapes(cfg, ids)
    out = pd.DataFrame(index=pd.Index(ids, name="gauge_id"))
    for col, rel in MAPS.items():
        m = catchment_means(cfg["paths"]["benchmarks"] / rel, shapes)
        out[col] = m["mean"]
        out[col.rsplit("_", 1)[0] + "_valid_frac"] = m["valid_frac"]
    whc = pd.read_csv(ROOT / "data" / "benchmarks" / f"soilgrids_whc_{name}.csv", index_col="gauge_id")
    out["whc_0-200cm_mm"] = whc["whc_0-200cm_mm"]
    out["soil_sr_mm"] = [
        soil_capacity({k: whc.at[g, f"whc_{k}_mm"] for k in LAYERS}, out.at[g, "rooting_depth_m"])
        if np.isfinite(out.at[g, "rooting_depth_m"]) else np.nan for g in out.index]
    cat = load_catalog_coords(ids)
    out["soil_sr_excluded"] = (cat.gauge_lon < -50) & (cat.gauge_lat >= 60)
    f = ROOT / "data" / "benchmarks" / f"static_{name}.csv"
    out.round(3).to_csv(f)
    print(f"{len(out)} catchments -> {f}")
    print(out.describe().round(2).T)


if __name__ == "__main__":
    main()
