"""RQ1: identifiability of each model's capacity parameter.

Per catchment and model, from <runs>/<tag>/<gauge_id>/{summary.json,<model>_mc.parquet}:
- n_behavioural: Monte Carlo sets with KGE_cal >= the behavioural threshold (config);
- beh_p5/p50/p95 of the capacity parameter over the behavioural sets;
- spread_abs = log10(P95/P5) of the behavioural capacity, with a bootstrap 90% interval
  (spread_abs_lo/hi; behavioural counts differ a lot between models);
- spread_rel = behavioural (P95 - P5) / prior (P95 - P5); the prior percentiles come from all Monte
  Carlo samples, which are Latin-hypercube draws from the prior, so this also covers SAC-SMA's peaked
  UZTWM + LZTWM prior;
- opt_at_bound: the optimum's capacity parameter(s) within 1% of the prior range from a bound;
- opt_position: the optimum's capacity relative to the behavioural range, "below" P5, "inside" or
  "above" P95 (empty when spreads are NaN). Flags catchments where the behavioural ensemble misses the
  optimum's region, e.g. GR6J X1 < 20 mm under the linear-uniform prior;
- kge_cal/kge_val of the optimum and pass (>= 0.5 in both), for the catchment screen.
Spreads need >= --min-behavioural sets (default 10), else NaN. The unidentifiability rules are in
classify.py.

Writes data/rq1/identifiability_<list name>[_<task>].csv. With --task-index/--tasks (SLURM arrays) each
task writes its own part; combine with --combine. --behavioural-kge overrides the config threshold for the
sensitivity runs (0.3 and 0.7); the output name then ends in _beh<threshold>.

Usage: python src/identifiability.py [--list data/catalog/pilot_catchments.csv] [--tag calibration]
       [--behavioural-kge T] [--task-index I --tasks N] | --combine
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from caravan_io import ROOT, load_config
from model_setups import SETUPS

RNG = np.random.default_rng(20261009)


def spread(x):
    p5, p95 = np.percentile(x, [5, 95])
    return np.log10(p95 / p5) if p5 > 0 else np.nan


def opt_position(opt, p5, p95):
    if not (p5 == p5 and p95 == p95):   # NaN: too few behavioural sets
        return ""
    return "below" if opt < p5 else "above" if opt > p95 else "inside"


def one(d, gid, name, setup, thr, min_beh, n_boot=500):
    summary = json.loads((d / "summary.json").read_text())["models"][name]
    mc = pd.read_parquet(d / f"{name}_mc.parquet", columns=setup.capacity + ["capacity", "kge_cal"])
    beh = mc.capacity[mc.kge_cal >= thr].to_numpy(float)
    prior_p5, prior_p95 = np.percentile(mc.capacity, [5, 95])
    opt = summary["params"]
    lo = dict(zip(setup.names, setup.lo)); hi = dict(zip(setup.names, setup.hi))
    at_bound = any(min(opt[c] - lo[c], hi[c] - opt[c]) <= 0.01 * (hi[c] - lo[c]) for c in setup.capacity)
    row = dict(gauge_id=gid, model=name, prior_version=summary.get("prior_version", ""),
               capacity_opt=summary["capacity"], opt_at_bound=at_bound,
               kge_cal=summary["kge_cal"], kge_val=summary["kge_val"],
               passes=summary["kge_cal"] >= 0.5 and summary["kge_val"] >= 0.5,
               n_behavioural=len(beh), n_samples=len(mc), prior_p5=prior_p5, prior_p95=prior_p95)
    if len(beh) >= min_beh:
        p5, p50, p95 = np.percentile(beh, [5, 50, 95])
        boot = [spread(RNG.choice(beh, len(beh))) for _ in range(n_boot)]
        row.update(beh_p5=p5, beh_p50=p50, beh_p95=p95, spread_abs=spread(beh),
                   spread_abs_lo=np.nanpercentile(boot, 5), spread_abs_hi=np.nanpercentile(boot, 95),
                   spread_rel=(p95 - p5) / (prior_p95 - prior_p5),
                   opt_position=opt_position(summary["capacity"], p5, p95))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/pilot_catchments.csv")
    ap.add_argument("--tag", default="calibration")
    ap.add_argument("--min-behavioural", type=int, default=10)
    ap.add_argument("--behavioural-kge", type=float, help="default: config calibration.behavioural_kge")
    ap.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--combine", action="store_true", help="concatenate the per-task parts")
    a = ap.parse_args()
    cfg = load_config()
    out = ROOT / "data" / "rq1"
    thr = cfg["calibration"]["behavioural_kge"]
    stem = f"identifiability_{Path(a.list).stem}"
    if a.behavioural_kge is not None and a.behavioural_kge != thr:
        thr = a.behavioural_kge
        stem += f"_beh{thr:g}"
    if a.combine:
        parts = sorted(out.glob(f"{stem}_task*.csv"))
        df = pd.concat(pd.read_csv(f) for f in parts)
        df.to_csv(out / f"{stem}.csv", index=False)
        print(f"{len(parts)} parts, {len(df)} rows -> {out / (stem + '.csv')}")
        return
    ids = pd.read_csv(ROOT / a.list, dtype={"gauge_id": str}).gauge_id.tolist()[a.task_index::a.tasks]
    rows = []
    for gid in ids:
        d = cfg["paths"]["runs"] / a.tag / gid
        if not (d / "summary.json").exists():
            print(f"{gid}: no calibration results", flush=True)
            continue
        for name, setup in SETUPS.items():
            rows.append(one(d, gid, name, setup, thr, a.min_behavioural))
    out.mkdir(parents=True, exist_ok=True)
    f = out / (f"{stem}.csv" if a.tasks == 1 else f"{stem}_task{a.task_index:03d}.csv")
    pd.DataFrame(rows).round(4).to_csv(f, index=False)
    print(f"{len(rows)} rows -> {f}")


if __name__ == "__main__":
    main()
