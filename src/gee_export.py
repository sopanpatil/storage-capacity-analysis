"""Full-sample MOD16 and ERA5-Land E from Earth Engine as batch exports to Google Drive.

Same computation as src/gee_extract.py (which uses interactive getInfo and is fine for the pilot), but
for thousands of catchments: catchments are split into chunks of --chunk, and each chunk gets two
Export.table.toDrive tasks (MOD16A2GF 8-day, ERA5-Land daily; 2003-2020). Tasks run on Google's
servers, so nothing here needs to stay running after submission.

Subcommands:
  submit   create and start the tasks (skips chunks whose task already exists and has not failed, so
           an interrupted submission can be rerun)
  status   print the state of every task (from Earth Engine's task list, matched by name)
  download fetch finished CSVs from Drive into <benchmarks>/gee/<list name>/{mod16,era5land}/,
           the layout src/benchmark_climate.py reads (Drive access uses the Earth Engine credentials,
           which include the Drive scope)
  cleanup  move the files in the Drive folder to the Drive bin once downloaded: an export is binned only
           if its local copy exists with the same size (duplicates of one export: any copy matching);
           other files in the folder (e.g. tests) are binned too. Lists what it would do unless --yes.
           Binned files still count against the Drive quota until the bin is emptied.

Usage: python src/gee_export.py submit|status|download|cleanup [--list data/catalog/screened_catchments.csv]
       [--chunk 200] [--first-chunks N] [--years 2003 2020] [--yes]
"""
import argparse
import io
from pathlib import Path

import ee
import pandas as pd

from caravan_io import ROOT, load_config, load_shapes
from gee_extract import catchments_fc, era5_range, mod16_year

DRIVE_FOLDER = "storage_capacity_gee"
PRODUCTS = {"mod16": ["gauge_id", "start", "et_mm", "valid_frac"],
            "era5land": ["gauge_id", "date", "e_mm", "snowfall_mm"]}


def submit(a, cfg, out):
    gee = cfg["earth_engine"]
    ids = pd.read_csv(ROOT / a.list, dtype={"gauge_id": str}).gauge_id.tolist()
    chunks = [ids[i:i + a.chunk] for i in range(0, len(ids), a.chunk)][:a.first_chunks]
    shapes = load_shapes(cfg, [g for c in chunks for g in c]).set_index("gauge_id")
    years = range(a.years[0], a.years[1] + 1)
    existing = {t["description"] for t in ee.data.getTaskList() if t["state"] not in ("FAILED", "CANCELLED")}
    for n, chunk in enumerate(chunks):
        names = {prod: f"{Path(a.list).stem}_{prod}_chunk{n:03d}" for prod in PRODUCTS}
        if all(v in existing for v in names.values()):
            continue
        fc = catchments_fc(shapes.loc[chunk].reset_index())
        colls = {"mod16": ee.FeatureCollection([mod16_year(fc, gee["mod16"], y) for y in years]).flatten(),
                 "era5land": era5_range(fc, gee["era5land_daily"], f"{years[0]}-01-01", f"{years[-1] + 1}-01-01")}
        for prod, coll in colls.items():
            name = names[prod]
            if name in existing:
                continue
            task = ee.batch.Export.table.toDrive(collection=coll, description=name, folder=DRIVE_FOLDER,
                                                 fileNamePrefix=name, fileFormat="CSV",
                                                 selectors=PRODUCTS[prod])
            task.start()
            print(f"started {name} ({task.id})", flush=True)
    print("all chunks submitted", flush=True)


def status(out, stem):
    """Latest task per name, from Earth Engine's task list."""
    rows = []
    for t in ee.data.getTaskList():   # newest first
        d = t["description"]
        if d.startswith(stem + "_") and d not in {r["name"] for r in rows}:
            prod, chunk = d[len(stem) + 1:].rsplit("_chunk", 1)
            rows.append(dict(name=d, product=prod, chunk=int(chunk), state=t["state"],
                             error=t.get("error_message", ""), file=d + ".csv"))
    tasks = pd.DataFrame(rows).sort_values(["product", "chunk"])
    print(tasks.groupby(["product", "state"]).size().to_string())
    bad = tasks[tasks.state.isin(["FAILED", "CANCELLED"])]
    if len(bad):
        print(bad[["file", "state", "error"]].to_string(index=False))
    return tasks


def download(out, stem):
    from googleapiclient.http import MediaIoBaseDownload
    tasks = status(out, stem)
    drive = drive_service()
    for t in tasks[tasks.state == "COMPLETED"].itertuples():
        dest = out / t.product / f"chunk{t.chunk:03d}.csv"
        if dest.exists():
            continue
        q = f"name = '{t.file}' and trashed = false"
        files = drive.files().list(q=q, fields="files(id, name, size)").execute()["files"]
        if len(files) != 1:
            print(f"{t.file}: {len(files)} matches on Drive, skipped")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        buf = io.FileIO(dest.with_suffix(".part"), "wb")
        dl = MediaIoBaseDownload(buf, drive.files().get_media(fileId=files[0]["id"]), chunksize=50 * 2**20)
        done = False
        while not done:
            _, done = dl.next_chunk()
        buf.close()
        dest.with_suffix(".part").rename(dest)
        print(f"downloaded {t.file} -> {dest}", flush=True)


def drive_service():
    """Drive client on the Earth Engine credentials, billed to our Cloud project (its Drive API must be
    enabled), not the EE client's."""
    from googleapiclient.discovery import build
    creds = ee.data.get_persistent_credentials().with_quota_project(load_config()["earth_engine"]["project"])
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def cleanup(out, stem, yes):
    drive = drive_service()
    q = f"name = '{DRIVE_FOLDER}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    folders = drive.files().list(q=q, fields="files(id)").execute()["files"]
    if len(folders) != 1:
        raise SystemExit(f"{len(folders)} Drive folders named {DRIVE_FOLDER}")
    files, token = [], None
    while True:
        r = drive.files().list(q=f"'{folders[0]['id']}' in parents and trashed = false", pageToken=token,
                               fields="nextPageToken, files(id, name, size)", pageSize=1000).execute()
        files += r["files"]
        token = r.get("nextPageToken")
        if not token:
            break
    keep = []
    for f in files:
        name = f["name"]
        if name.startswith(stem + "_"):
            prod, chunk = name[len(stem) + 1:].removesuffix(".csv").rsplit("_chunk", 1)
            dest = out / prod / f"chunk{int(chunk):03d}.csv"
            copies = [g for g in files if g["name"] == name]
            if not dest.exists() or all(int(g["size"]) != dest.stat().st_size for g in copies):
                keep.append(name)
                continue
        print(f"{'binning' if yes else 'would bin'} {name} ({int(f.get('size', 0)) / 2**20:.0f} MB)", flush=True)
        if yes:
            drive.files().update(fileId=f["id"], body={"trashed": True}).execute()
    for name in keep:
        print(f"kept {name}: no local copy of the same size")
    total = sum(int(f.get("size", 0)) for f in files if f["name"] not in keep) / 2**30
    print(f"{'binned' if yes else 'would bin'} {len(files) - len(keep)} files ({total:.2f} GB), kept {len(keep)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["submit", "status", "download", "cleanup"])
    ap.add_argument("--list", default="data/catalog/screened_catchments.csv")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--first-chunks", type=int, help="submit only the first N chunks (for testing)")
    ap.add_argument("--years", type=int, nargs=2, default=[2003, 2020])
    ap.add_argument("--yes", action="store_true", help="cleanup: actually bin the files")
    a = ap.parse_args()
    cfg = load_config()
    ee.Initialize(project=cfg["earth_engine"]["project"])
    out = cfg["paths"]["benchmarks"] / "gee" / Path(a.list).stem
    stem = Path(a.list).stem
    {"submit": lambda: submit(a, cfg, out), "status": lambda: status(out, stem),
     "download": lambda: download(out, stem), "cleanup": lambda: cleanup(out, stem, a.yes)}[a.action]()


if __name__ == "__main__":
    main()
