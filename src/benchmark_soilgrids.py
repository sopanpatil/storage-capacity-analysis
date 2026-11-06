"""Catchment water-holding capacity per soil layer from SoilGrids 2.0 water retention (1 km).

Input for the soil-derived benchmark (rooting depth x WHC): water-holding capacity
theta(33 kPa) - theta(1500 kPa) (Turek et al. 2023; SoilGrids layers wv0033 and wv1500, units of
1e-3 cm3/cm3), multiplied by layer thickness, giving mm of water per standard layer (0-5, 5-15, 15-30,
30-60, 60-100, 100-200 cm). Combined with rooting depth (Yang et al. 2016) in a later step.

Rasters: <benchmarks>/soilgrids_1000m/, from
https://files.isric.org/soilgrids/latest/data_aggregated/1000m/{wv0033,wv1500}/.

Writes data/benchmarks/soilgrids_whc_<list name>.csv: one row per catchment with whc_<layer>_mm,
whc_0-200cm_mm, and the smallest valid-area fraction over the 12 rasters.

Usage: python src/benchmark_soilgrids.py [--list data/catalog/pilot_catchments.csv]
"""
import argparse
from pathlib import Path

import pandas as pd

from caravan_io import ROOT, load_config, load_shapes
from zonal import catchment_means

LAYERS = {"0-5cm": 50, "5-15cm": 100, "15-30cm": 150, "30-60cm": 300, "60-100cm": 400,
          "100-200cm": 1000}  # thickness, mm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/pilot_catchments.csv")
    a = ap.parse_args()
    cfg = load_config()
    ids = pd.read_csv(ROOT / a.list, dtype={"gauge_id": str}).gauge_id
    shapes = load_shapes(cfg, ids)
    src = cfg["paths"]["benchmarks"] / "soilgrids_1000m"
    out = pd.DataFrame(index=pd.Index(ids, name="gauge_id"))
    valid = []
    for layer, thick in LAYERS.items():
        theta = {}
        for v in ["wv0033", "wv1500"]:
            m = catchment_means(src / f"{v}_{layer}_mean_1000.tif", shapes)
            theta[v] = m["mean"] / 1000.0  # cm3/cm3
            valid.append(m["valid_frac"])
        out[f"whc_{layer}_mm"] = (theta["wv0033"] - theta["wv1500"]) * thick
    out["whc_0-200cm_mm"] = out[[f"whc_{k}_mm" for k in LAYERS]].sum(axis=1)
    out["min_valid_frac"] = pd.concat(valid, axis=1).min(axis=1)
    dest = ROOT / "data" / "benchmarks"
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"soilgrids_whc_{Path(a.list).stem}.csv"
    out.round(3).to_csv(f)
    print(f"{len(out)} catchments -> {f}")
    print(out[["whc_0-200cm_mm", "min_valid_frac"]].describe().round(3))


if __name__ == "__main__":
    main()
