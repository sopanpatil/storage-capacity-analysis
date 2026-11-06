"""RQ3: storage each model actually uses, by the mass curve on its own fluxes.

For each catchment and model, runs the calibrated optimum and every behavioural Monte Carlo set
(KGE_cal >= the behavioural threshold) from <runs>/<tag>/<gauge_id>/ and, over 2003-2020:
- sr_10/20/40: mass-curve capacity (src/mass_curve.py) with inflow = model soil input (rain + melt) and
  outflow = model actual ET (src/model_fluxes.py), hydrological year from the model's own deficit;
- store_deficit_20: the 20-year Gumbel level of annual maxima of the simulated store deficit (capacity
  minus storage; same years), bounded by the parameter, so a diagnostic only.
GR6J results carry the prior_version of the calibration they came from.

Writes the full table (one row per catchment x model x parameter set; set = "opt" or the Monte Carlo
row index) to <runs>/rq3/state_capacity_<list name>.parquet (millions of rows for the full
sample, regenerable), and a per-catchment summary to data/rq3/state_capacity_<list name>_summary.csv
(one row per catchment x model: the optimum's values, the behavioural count, and the behavioural
5/50/95th percentiles of capacity, sr_10/20/40 and store_deficit_20).

With --task-index/--tasks (SLURM arrays) each task writes its own part to <runs>/rq3/; --combine joins
them, writes both outputs and deletes the parts.

Usage: python src/state_capacity.py [--list data/catalog/pilot_catchments.csv] [--tag calibration]
       [--task-index I --tasks N] | --combine
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from caravan_io import ROOT, load_config
from mass_curve import annual_maxima, gumbel_return_level, storage_capacity
from model_fluxes import FLUXES, forcing, window
from model_setups import SETUPS


def evaluate(f, name, p):
    dates, _, x = window(f, FLUXES[name](f, p))
    r = storage_capacity(x["soil_in"], x["aet"], dates)
    m = annual_maxima(x["store_deficit"], dates, r["start_month"])
    return dict(sr_10=r["sr_10"], sr_20=r["sr_20"], sr_40=r["sr_40"], sr_max=r["sr_max"],
                start_month=r["start_month"], lateral_inflow=r["lateral_inflow"],
                aet_mean=x["aet"].mean(), soil_in_mean=x["soil_in"].mean(),
                store_deficit_20=gumbel_return_level(m, 20))


SUMMARY_COLS = ["capacity", "sr_10", "sr_20", "sr_40", "store_deficit_20"]


def summarise(df):
    """One row per catchment x model: optimum values, behavioural count and percentiles."""
    key = ["gauge_id", "model"]
    opt = df[df.set == "opt"].set_index(key).drop(columns="set")
    beh = df[df.set != "opt"].groupby(key)[SUMMARY_COLS]
    q = beh.quantile([0.05, 0.5, 0.95]).unstack()
    q.columns = [f"beh_{c}_p{round(p * 100)}" for c, p in q.columns]
    out = opt.add_prefix("opt_").rename(columns={"opt_prior_version": "prior_version"})
    out = out.join(beh.size().rename("n_behavioural")).join(q)
    out["n_behavioural"] = out.n_behavioural.fillna(0).astype(int)
    return out.reset_index()


def write(df, cfg, stem):
    full = cfg["paths"]["runs"] / "rq3" / f"{stem}.parquet"
    full.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(full, index=False)
    f = ROOT / "data" / "rq3" / f"{stem}_summary.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    summarise(df).round(3).to_csv(f, index=False)
    print(f"{len(df)} rows -> {full}; summary -> {f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/pilot_catchments.csv")
    ap.add_argument("--tag", default="calibration")
    ap.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--combine", action="store_true", help="concatenate the per-task parts")
    a = ap.parse_args()
    cfg = load_config()
    parts_dir = cfg["paths"]["runs"] / "rq3"
    stem = f"state_capacity_{Path(a.list).stem}"
    if a.combine:
        parts = sorted(parts_dir.glob(f"{stem}_task*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
        print(f"{len(parts)} parts")
        write(df, cfg, stem)
        for f in parts:
            f.unlink()
        return
    thr = cfg["calibration"]["behavioural_kge"]
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    ids = pd.read_csv(ROOT / a.list, dtype={"gauge_id": str}).gauge_id.tolist()[a.task_index::a.tasks]
    rows = []
    for gid in ids:
        d = cfg["paths"]["runs"] / a.tag / gid
        summary = json.loads((d / "summary.json").read_text())
        f = forcing(cfg, gid, cat.loc[gid])
        for name, setup in SETUPS.items():
            m = summary["models"][name]
            sets = [("opt", np.array([m["params"][k] for k in setup.names]), m["kge_cal"])]
            mc = pd.read_parquet(d / f"{name}_mc.parquet")
            beh = mc[mc.kge_cal >= thr]
            sets += [(str(i), r[setup.names].to_numpy(float), r.kge_cal) for i, r in beh.iterrows()]
            for label, p, k in sets:
                rows.append(dict(gauge_id=gid, model=name, set=label, kge_cal=k,
                                 capacity=float(setup.capacity_of(p)[0]),
                                 prior_version=m.get("prior_version", ""), **evaluate(f, name, p)))
        print(f"{gid}: " + ", ".join(f"{n} {sum(1 for r in rows if r['gauge_id'] == gid and r['model'] == n) - 1}"
                                    for n in SETUPS) + " behavioural", flush=True)
    df = pd.DataFrame(rows).round(3)
    if a.tasks == 1:
        write(df, cfg, stem)
    else:
        parts_dir.mkdir(parents=True, exist_ok=True)
        f = parts_dir / f"{stem}_task{a.task_index:03d}.parquet"
        df.to_parquet(f, index=False)
        print(f"{len(df)} rows -> {f}")


if __name__ == "__main__":
    main()
