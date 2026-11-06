"""Catchment-mean MOD16 ET and ERA5-Land evaporation from Google Earth Engine, 2003-2020.

For the climate-derived benchmark:
- MOD16A2GF (C6.1, gap-filled) 8-day ET: mean over valid pixels (fill values 32761-32767 = water,
  urban, barren, ice etc. masked), scale factor 0.1 -> mm per composite period, plus the valid-pixel
  fraction. Composites start on day 1, 9, ... of each year; the last one runs to 31 Dec (5 or 6 days).
- ERA5-Land daily total evaporation (DAILY_AGGR, m of water, ECMWF sign: evaporation negative)
  -> mm/d, positive upward; and daily snowfall (m water equivalent) -> mm/d, for the snow-aware
  sensitivity variant (rain/snow split from the reanalysis, as Stocker et al. 2023 did with WFDEI). These are UTC days, whereas Caravan's forcing uses local-time days; only
  the within-month shape is used (daily/monthly ratios), so the offset is minor.
Means are computed in Earth Engine (pixels weighted by their fraction inside the polygon) at native
resolution: 463 m for MOD16, 0.1 deg for ERA5-Land. Polygons are simplified to 50 m (well below both
pixel sizes) to keep requests small.

Results go to <benchmarks>/gee/<list name>/{mod16,era5land}/ as one CSV per year (MOD16) or month
(ERA5-Land); existing parts are skipped, so reruns resume. Uses getInfo, fine for the pilot; the full
sample will need table exports.

Usage: python src/gee_extract.py [--list data/catalog/pilot_catchments.csv] [--years 2003 2020]
"""
import argparse
import time
from pathlib import Path

import ee
import pandas as pd

from caravan_io import ROOT, load_config, load_shapes

MOD16_SCALE_M = 463.3127
ERA5_SCALE_M = 11132.0


def catchments_fc(shapes):
    feats = []
    for gid, geom in zip(shapes.gauge_id, shapes.geometry.simplify(0.0005, preserve_topology=True)):
        feats.append(ee.Feature(ee.Geometry(geom.__geo_interface__), {"gauge_id": gid}))
    return ee.FeatureCollection(feats)


def mod16_year(fc, coll, year):
    def per_image(img):
        # Earth Engine already masks the fill values; unmask so the valid fraction counts them
        et = img.select("ET").unmask(32767)
        valid = et.lte(32700)
        stack = ee.Image.cat(et.updateMask(valid).multiply(0.1).rename("et_mm"), valid.rename("valid_frac"))
        start = img.date().format("YYYY-MM-dd")
        return stack.reduceRegions(fc, ee.Reducer.mean(), scale=MOD16_SCALE_M).map(
            lambda f: f.set("start", start).setGeometry(None))
    images = ee.ImageCollection(coll).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    return images.map(per_image).flatten()


def era5_month(fc, coll, year, month):
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthBegin(1)
    return era5_range(fc, coll, str(start.date()), str(end.date()))


def era5_range(fc, coll, start, end):
    """Daily ERA5-Land E over [start, end) for every catchment in fc."""
    def per_image(img):
        e = ee.Image.cat(img.select("total_evaporation_sum").multiply(-1000.0).rename("e_mm"),
                         img.select("snowfall_sum").multiply(1000.0).rename("snowfall_mm"))
        day = img.date().format("YYYY-MM-dd")
        return e.reduceRegions(fc, ee.Reducer.mean(), scale=ERA5_SCALE_M).map(
            lambda f: f.set("date", day).setGeometry(None))
    images = ee.ImageCollection(coll).filterDate(start, end)
    return images.map(per_image).flatten()


def fetch(fc, columns, dest, retries=4):
    """getInfo with retries; writes a CSV with the given columns."""
    for attempt in range(retries):
        try:
            feats = fc.getInfo()["features"]
            break
        except ee.EEException as e:
            if attempt == retries - 1:
                raise
            print(f"  retry after: {str(e)[:120]}")
            time.sleep(10 * (attempt + 1))
    df = pd.DataFrame([f["properties"] for f in feats]).reindex(columns=columns)
    tmp = dest.with_suffix(".part")
    df.to_csv(tmp, index=False)
    tmp.rename(dest)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/pilot_catchments.csv")
    ap.add_argument("--years", type=int, nargs=2, default=[2003, 2020])
    a = ap.parse_args()
    cfg = load_config()
    gee = cfg["earth_engine"]
    ee.Initialize(project=gee["project"])
    ids = pd.read_csv(ROOT / a.list, dtype={"gauge_id": str}).gauge_id
    fc = catchments_fc(load_shapes(cfg, ids))
    out = cfg["paths"]["benchmarks"] / "gee" / Path(a.list).stem
    for sub in ("mod16", "era5land"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for year in range(a.years[0], a.years[1] + 1):
        f = out / "mod16" / f"{year}.csv"
        if not f.exists():
            df = fetch(mod16_year(fc, gee["mod16"], year), ["gauge_id", "start", "et_mm", "valid_frac"], f)
            print(f"MOD16 {year}: {len(df)} rows ({time.time() - t0:.0f} s)")
        for month in range(1, 13):
            f = out / "era5land" / f"{year}-{month:02d}.csv"
            if not f.exists():
                fetch(era5_month(fc, gee["era5land_daily"], year, month), ["gauge_id", "date", "e_mm", "snowfall_mm"], f)
        print(f"ERA5-Land {year} done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
