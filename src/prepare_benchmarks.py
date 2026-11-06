"""Download the static benchmark maps and write them as georeferenced GeoTIFFs for src/zonal.py.

Sources (files under <benchmarks>/):
- yang2016/: effective rooting depth (m), Yang et al. (2016), CSIRO collection 19813
  (doi:10.4225/08/5837b3aa9cb90). The published GeoTIFF has no CRS and its geotransform is half a cell
  off (bounds -180.25..179.75, -90.25..89.75). Registration tested 2026-10-09 against SoilGrids land
  fraction: on the standard 0.5 deg grid (edges at -180 and 90) 53 of 47,925 valid cells are under
  50% land, against >= 400 for every other half- or whole-cell offset. Written with that grid.
- zhao2025/: GRACE/GRACE-FO root-zone storage capacity Sr (mm), Zhao et al. (2025), Zenodo 14970062
  (Sr_GRACE_FO.nc, 0.5 deg cells of a 3 deg solution).
- stocker2023/: S_CWDX80 (mm), Stocker et al. (2023), Zenodo 10885724 (cwdx80.nc; the *_forcing.nc
  files there are modified forcing maps and are not used).
Zenodo files are checked against the record's MD5.

Usage: python src/prepare_benchmarks.py
"""
import hashlib
import json
import urllib.request

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from caravan_io import load_config

ZENODO = {"zhao2025": (14970062, "Sr_GRACE_FO.nc"), "stocker2023": (10885724, "cwdx80.nc")}
YANG_DATA = "https://data.csiro.au/dap/ws/v2/collections/19813/data"


def fetch(url, dest):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def fetch_zenodo(record, name, dest):
    meta = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{record}"))
    want = next(f["checksum"] for f in meta["files"] if f["key"] == name)
    fetch(f"https://zenodo.org/api/records/{record}/files/{name}/content", dest)
    got = "md5:" + hashlib.md5(dest.read_bytes()).hexdigest()
    if got != want:
        raise RuntimeError(f"{dest}: {got} != {want}")
    return dest


def write_tif(dest, data, transform, nodata=np.nan):
    with rasterio.open(dest, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform, nodata=nodata,
                       compress="deflate") as r:
        r.write(data.astype("float32"), 1)
    print(f"wrote {dest}")


def from_netcdf(nc, var, dest):
    """Lon/lat netCDF variable -> GeoTIFF, north-up, edges from the regular cell centres."""
    da = xr.open_dataset(nc)[var].squeeze(drop=True)
    lat, lon = [c for c in ("lat", "latitude") if c in da.dims][0], [c for c in ("lon", "longitude") if c in da.dims][0]
    da = da.transpose(lat, lon).sortby(lat, ascending=False).sortby(lon)
    dx = float(np.diff(da[lon]).mean())
    dy = float(-np.diff(da[lat]).mean())
    assert np.allclose(np.diff(da[lon]), dx) and np.allclose(-np.diff(da[lat]), dy), "irregular grid"
    west, north = float(da[lon][0]) - dx / 2, float(da[lat][0]) + dy / 2
    write_tif(dest, da.values, from_origin(west, north, dx, dy))


def main():
    base = load_config()["paths"]["benchmarks"]

    meta = json.load(urllib.request.urlopen(urllib.request.Request(YANG_DATA, headers={"Accept": "application/json"})))
    raw = fetch(meta["file"][0]["presignedLink"]["href"], base / "yang2016" / "Effective_Rooting_Depth.tif")
    with rasterio.open(raw) as r:
        assert r.shape == (360, 720) and np.isclose(r.bounds.left, -180.25) and np.isclose(r.bounds.top, 89.75), \
            "published grid changed; recheck the registration"
        y = r.read(1)
    y = np.where(y > -1e30, y, np.nan)
    write_tif(base / "yang2016" / "rooting_depth_m.tif", y, from_origin(-180.0, 90.0, 0.5, 0.5))

    zhao = fetch_zenodo(*ZENODO["zhao2025"], base / "zhao2025" / "Sr_GRACE_FO.nc")
    from_netcdf(zhao, "z", base / "zhao2025" / "sr_grace_mm.tif")

    stocker = fetch_zenodo(*ZENODO["stocker2023"], base / "stocker2023" / "cwdx80.nc")
    ds = xr.open_dataset(stocker)
    print("stocker2023 variables:", {v: dict(ds[v].attrs) for v in ds.data_vars})
    var = [v for v in ds.data_vars if ds[v].ndim >= 2][0]
    from_netcdf(stocker, var, base / "stocker2023" / "s_cwdx80_mm.tif")


if __name__ == "__main__":
    main()
