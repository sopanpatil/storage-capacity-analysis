"""Robustness item J: FLEX on the fixed subset (robustness/subset.csv), against HBV's runs A and B on the same catchments.

Same inputs, split, warm-up, thresholds and SCE-UA settings as robustness_calibrate.py. Model and priors: flex_model.py.
For each catchment:
  so       SCE-UA on KGE(Q), full budget (compare hbv_so)
  mo       SCE-UA on the mean of KGE(Q) and KGE(log Q), full budget (compare hbv_mo)
  so_liang SCE-UA on KGE(Q) with Liang et al.'s Sumax 30-700 mm
  FLEX_mc.parquet: --samples Latin-hypercube sets over the wide prior, with both KGEs in both periods. Behavioural sets
  are selected afterwards: KGE(Q) >= 0.5, or both KGEs >= 0.5; Sumax 30-700 mm by filtering on the capacity column.
Each optimum's simulated deficit (state_capacity.evaluate, 2003-2020) goes in summary.json, so the summary step does
not need the forcing.

Writes <workspace>/robustness/flex/<gauge_id>/ (or --outdir). A catchment with an existing summary.json is skipped.

Usage: python src/robustness_flex.py --list robustness/subset.csv [--task-index I --tasks N] [--samples 40000]
       python src/robustness_flex.py [--outdir DIR] GAUGE_ID [...]
"""
import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from calibrate import catchment_inputs, kge
from caravan_io import ROOT, load_config
from flex_model import FLEX, FLEXLiang, NAMES, fluxes
from model_fluxes import FLUXES, forcing
from robustness_calibrate import kge_log, sce
from state_capacity import evaluate

FLUXES["FLEX"] = fluxes


def run_catchment(cfg, gid, row, n, outdir):
    d = outdir / gid
    if (d / "summary.json").exists():
        return "skipped"
    d.mkdir(parents=True, exist_ok=True)
    f, obs, cal, val, info = catchment_inputs(cfg, gid, row)
    window = forcing(cfg, gid, row)
    seed = zlib.crc32(gid.encode())
    eps = 0.01 * float(np.nanmean(obs[cal]))
    wide, narrow = FLEX(), FLEXLiang()

    def objective_so(setup):
        return lambda p: kge(setup.simulate(f, p)[cal], obs[cal])

    def objective_mo(p):
        s = wide.simulate(f, p)
        return 0.5 * (kge(s[cal], obs[cal]) + kge_log(s[cal], obs[cal], eps))

    optima = {}
    t0 = time.time()
    for label, setup, objective in [("so", wide, objective_so(wide)), ("mo", wide, objective_mo),
                                    ("so_liang", narrow, objective_so(narrow))]:
        best = sce(setup, objective, cfg, seed)
        s = setup.simulate(f, best)
        e = evaluate(window, "FLEX", best)
        optima[label] = dict(params=dict(zip(NAMES, map(float, best))), capacity=float(setup.capacity_of(best)[0]),
                             kge_cal=float(kge(s[cal], obs[cal])), kge_val=float(kge(s[val], obs[val])),
                             kgelog_cal=float(kge_log(s[cal], obs[cal], eps)),
                             kgelog_val=float(kge_log(s[val], obs[val], eps)),
                             sr_20=float(e["sr_20"]), sr_max=float(e["sr_max"]), aet_mean=float(e["aet_mean"]))
    t1 = time.time()
    params = wide.from_unit(qmc.LatinHypercube(d=len(NAMES), seed=seed).random(n))
    k = np.empty((n, 4))
    for i, p in enumerate(params):
        s = wide.simulate(f, p)
        k[i] = (kge(s[cal], obs[cal]), kge(s[val], obs[val]), kge_log(s[cal], obs[cal], eps), kge_log(s[val], obs[val], eps))
    mc = pd.DataFrame(params, columns=NAMES).astype("float32")
    mc["capacity"] = wide.capacity_of(params).astype("float32")
    for j, c in enumerate(["kge_cal", "kge_val", "kgelog_cal", "kgelog_val"]):
        mc[c] = k[:, j].astype("float32")
    mc.to_parquet(d / "FLEX_mc.parquet", index=False)
    t2 = time.time()
    summary = dict(gauge_id=gid, log_eps=eps, **info, prior_version=wide.prior_version, optima=optima,
                   n_samples=int(n), n_behavioural=int((mc.kge_cal >= 0.5).sum()),
                   n_behavioural_both=int(((mc.kge_cal >= 0.5) & (mc.kgelog_cal >= 0.5)).sum()),
                   seconds_optimise=round(t1 - t0, 1), seconds_monte_carlo=round(t2 - t1, 1))
    tmp = d / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1))
    tmp.replace(d / "summary.json")
    return "done"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gauges", nargs="*")
    ap.add_argument("--list", help="CSV with a gauge_id column")
    ap.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--samples", type=int, default=40000, help="Monte Carlo samples (Liang et al. 2024 used 40,000)")
    ap.add_argument("--outdir", help="default: <workspace>/robustness/flex")
    a = ap.parse_args()
    cfg = load_config()
    gauges = list(a.gauges)
    if a.list:
        gauges += pd.read_csv(ROOT / a.list if not os.path.isabs(a.list) else a.list).gauge_id.tolist()
    gauges = gauges[a.task_index::a.tasks]
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    outdir = Path(a.outdir) if a.outdir else cfg["paths"]["workspace"] / "robustness" / "flex"
    for gid in gauges:
        t0 = time.time()
        status = run_catchment(cfg, gid, cat.loc[gid], a.samples, outdir)
        print(f"{gid}: {status} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
