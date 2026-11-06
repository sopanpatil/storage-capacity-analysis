"""Catchment-mean SSEBop v5 monthly actual ET, 2003-2020.

Reads the monthly zips from src/download_ssebop.sh through GDAL's /vsizip/ (int16, mm/month, nodata
-32768, EPSG:4326 at ~0.0097 deg, 60S-80N) and averages over each catchment with src/zonal.py.

Writes <benchmarks>/ssebop_v5_catchments/<list name>/<year>.csv: gauge_id, month (YYYY-MM), et_mm,
valid_frac. The pilot takes ~5.5 min; the full sample takes ~11 h on one core
as one task per year (slurm/ssebop_full.sbatch, --year from the array index).

Usage: python src/benchmark_ssebop.py [--list data/catalog/pilot_catchments.csv] [--year 2003]
"""
import argparse
from pathlib import Path

import pandas as pd

from caravan_io import ROOT, load_config, load_shapes
from zonal import catchment_means


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/pilot_catchments.csv")
    ap.add_argument("--year", type=int, help="one year only (default: 2003-2020)")
    a = ap.parse_args()
    cfg = load_config()
    ids = pd.read_csv(ROOT / a.list, dtype={"gauge_id": str}).gauge_id
    shapes = load_shapes(cfg, ids)
    src = cfg["paths"]["benchmarks"] / "ssebop_v5_monthly"
    dest = cfg["paths"]["benchmarks"] / "ssebop_v5_catchments" / Path(a.list).stem
    dest.mkdir(parents=True, exist_ok=True)
    for year in [a.year] if a.year else range(2003, 2021):
        write_year(year, shapes, src, dest)


def write_year(year, shapes, src, dest):
    parts = []
    for month in pd.period_range(f"{year}-01", f"{year}-12", freq="M"):
        tag = month.strftime("m%Y%m")
        zf = src / f"{tag}.zip"
        if not zf.exists():
            raise FileNotFoundError(f"{zf} missing; run src/download_ssebop.sh first")
        m = catchment_means(f"/vsizip/{zf}/{tag}_modisSSEBopETv5_actual_mm.tif", shapes)
        parts.append(m.rename(columns={"mean": "et_mm"}).reset_index().assign(month=str(month)))
    out = pd.concat(parts, ignore_index=True)[["gauge_id", "month", "et_mm", "valid_frac"]]
    f = dest / f"{year}.csv"
    out.round(3).to_csv(f, index=False)
    print(f"{out.gauge_id.nunique()} catchments x {out.month.nunique()} months -> {f}", flush=True)


if __name__ == "__main__":
    main()
