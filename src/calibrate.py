"""Calibrate, validate and Monte Carlo-sample the three models for a set of catchments.

Per catchment:
  - split the valid streamflow days in the analysis window (1981-2023): first 2/3 calibrate, last
    1/3 validate; run from 5 years before the first calibration day to the end of the window;
  - optimum per model with its own optimiser (SCE-UA for HBV and SAC-SMA, airGR's Michel algorithm
    for GR6J), objective KGE (Gupta et al. 2009) on calibration days;
  - Latin-hypercube Monte Carlo within each model's prior, KGE on calibration and validation days.
Writes <runs>/<tag>/<gauge_id>/: summary.json, <model>_mc.parquet, <model>_opt_q.npy.
A catchment with an existing summary.json is skipped (safe to resubmit).

Usage:
  python src/calibrate.py --tag pilot GAUGE_ID [GAUGE_ID ...]
  python src/calibrate.py --tag pilot --list data/catalog/pilot_catchments.csv \
      [--task-index I --tasks N]          # one share of the list, for SLURM job arrays
Options: --models HBV GR6J SACSMA, --samples N (all models; default per model from config),
         --sce-maxiter N, --redo MODEL [...] (rerun these models where results already exist;
         catchments without results get all --models), --forcing SOURCE (precipitation, PET and
         temperature from config `forcing_sources`, e.g. camelsgb_cehgear; streamflow unchanged; the run
         then ends at the last validation day, since these sources end before 2023)
"""
import argparse
import json
import os
import time
import zlib

import grsuite as gs
import numpy as np
import pandas as pd
from scipy.stats import qmc

from caravan_io import ROOT, load_config, load_forcing
from model_setups import SETUPS
from hbv_model.sceua import sceua  # noqa: E402  (path set by model_setups)


def kge(sim, obs):
    """Gupta et al. (2009) KGE; NaN if undefined."""
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    if len(o) < 2 or o.std() == 0 or o.mean() == 0 or s.std() == 0:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)


def catchment_inputs(cfg, gid, row, source=None):
    f = cfg["forcing"]
    ts = load_forcing(cfg, gid, ["streamflow", f["precip"], f["pet"], f["tmin"], f["tmean"], f["tmax"]], source)
    lo, hi = cfg["record"]["window"]
    q = ts["streamflow"]
    valid = q[(q.index.year >= lo) & (q.index.year <= hi)].dropna()
    k = int(np.ceil(cfg["record"]["calibration_fraction"] * len(valid)))
    cal_start, cal_end, val_end = valid.index[0], valid.index[k - 1], valid.index[-1]
    run_start = max(cal_start - pd.DateOffset(years=cfg["record"]["warmup_years"]), ts.index[0])
    run = ts.loc[run_start:val_end if source else f"{hi}-12-31"]
    forcing_cols = [f["precip"], f["pet"], f["tmin"], f["tmean"], f["tmax"]]
    if run[forcing_cols].isna().any().any():
        raise ValueError(f"{gid}: missing forcing in the run period")
    obs = run["streamflow"].to_numpy()
    idx = run.index
    cal = (idx >= cal_start) & (idx <= cal_end) & np.isfinite(obs)
    val = (idx > cal_end) & (idx <= val_end) & np.isfinite(obs)
    forcing = dict(P=run[f["precip"]].to_numpy(), E=run[f["pet"]].to_numpy(),
                   Tmin=run[f["tmin"]].to_numpy(), Tmean=run[f["tmean"]].to_numpy(),
                   Tmax=run[f["tmax"]].to_numpy(), dates=idx,
                   n_warm=int((idx < cal_start).sum()), lat=float(row.gauge_lat),
                   elev=float(row.ele_mt_sav), forest_frac=float(row.for_pc_sse) / 100.0)
    info = dict(run_start=str(run_start.date()), cal_start=str(cal_start.date()),
                cal_end=str(cal_end.date()), val_end=str(val_end.date()),
                n_cal_days=int(cal.sum()), n_val_days=int(val.sum()),
                cal_years=sorted(set(idx[cal].year.tolist())),
                cal_aridity=float(forcing["E"][cal].mean() / forcing["P"][cal].mean()),
                val_aridity=float(forcing["E"][val].mean() / forcing["P"][val].mean()))
    return forcing, obs, cal, val, info


def optimise(setup, f, obs, cal, cfg, seed):
    if setup.name == "GR6J":
        crit = gs.InputsCrit("KGE", obs[f["n_warm"]:], bool_crit=cal[f["n_warm"]:])
        opts = gs.CalibOptions(setup.model, search_ranges=setup.search_ranges())
        out = gs.calibration_michel(setup.im, setup.ro, crit, opts, verbose=False)
        return np.asarray(out.param_final_r, float), None

    def objective(p):
        k = kge(setup.simulate(f, p)[cal], obs[cal])
        return -k if np.isfinite(k) else 1e6
    best, _ = sceua(objective, setup.bounds(), n_complexes=cfg["calibration"]["sce_complexes"],
                    maxiter=cfg["calibration"]["sce_maxiter"], seed=seed, verbose=False)
    return np.asarray(best, float), None


def monte_carlo(setup, f, obs, cal, val, n, seed):
    u = qmc.LatinHypercube(d=len(setup.names), seed=seed).random(n)
    params = setup.from_unit(u)
    kc, kv = np.empty(n), np.empty(n)
    for i, p in enumerate(params):
        s = setup.simulate(f, p)
        kc[i], kv[i] = kge(s[cal], obs[cal]), kge(s[val], obs[val])
    df = pd.DataFrame(params, columns=setup.names).astype("float32")
    df["capacity"] = setup.capacity_of(params).astype("float32")
    df["kge_cal"], df["kge_val"] = kc.astype("float32"), kv.astype("float32")
    return df


def run_catchment(cfg, gid, row, models, n_samples, outdir, redo=(), source=None):
    """Run `models` for one catchment. If a summary already exists, run only the models in
    `redo` (replacing their outputs and summary entries); with no `redo`, skip the catchment."""
    d = outdir / gid
    if (d / "summary.json").exists():
        if not redo:
            return "skipped"
        summary = json.loads((d / "summary.json").read_text())
        todo = [m for m in models if m in redo]
    else:
        d.mkdir(parents=True, exist_ok=True)
        summary, todo = None, list(models)
    f, obs, cal, val, info = catchment_inputs(cfg, gid, row, source)
    seed = zlib.crc32(gid.encode())
    summary = summary or dict(gauge_id=gid, forcing=source or "caravan", **info, models={})
    for name in todo:
        setup = SETUPS[name]
        if hasattr(setup, "prepare"):
            setup.prepare(f)
        t0 = time.time()
        best, _ = optimise(setup, f, obs, cal, cfg, seed)
        sim = setup.simulate(f, best)
        t1 = time.time()
        mc = monte_carlo(setup, f, obs, cal, val, n_samples[name], seed)
        t2 = time.time()
        mc.to_parquet(d / f"{name}_mc.parquet", index=False)
        np.save(d / f"{name}_opt_q.npy", sim.astype("float32"))
        summary["models"][name] = dict(
            params=dict(zip(setup.names, map(float, best))),
            capacity=float(setup.capacity_of(best)[0]),
            kge_cal=float(kge(sim[cal], obs[cal])), kge_val=float(kge(sim[val], obs[val])),
            n_behavioural=int((mc.kge_cal >= cfg["calibration"]["behavioural_kge"]).sum()),
            n_samples=int(len(mc)), prior_version=setup.prior_version,
            seconds_optimise=round(t1 - t0, 1), seconds_monte_carlo=round(t2 - t1, 1))
    tmp = d / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1))
    tmp.replace(d / "summary.json")   # atomic, so a killed task never leaves a half-written summary
    return f"done ({', '.join(todo)})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gauges", nargs="*")
    ap.add_argument("--tag", required=True, help="output subfolder, e.g. pilot")
    ap.add_argument("--list", help="CSV with a gauge_id column")
    ap.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--models", nargs="+", default=list(SETUPS))
    ap.add_argument("--samples", type=int)
    ap.add_argument("--sce-maxiter", type=int)
    ap.add_argument("--redo", nargs="+", default=[],
                    help="models to rerun where a summary already exists (others are kept)")
    ap.add_argument("--forcing", help="alternative forcing source (config forcing_sources)")
    a = ap.parse_args()
    cfg = load_config()
    if a.sce_maxiter:
        cfg["calibration"]["sce_maxiter"] = a.sce_maxiter
    n = {m: a.samples or cfg["calibration"]["monte_carlo_samples"][m] for m in a.models}
    gauges = list(a.gauges)
    if a.list:
        gauges += pd.read_csv(ROOT / a.list if not os.path.isabs(a.list) else a.list).gauge_id.tolist()
    gauges = gauges[a.task_index::a.tasks]
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    outdir = cfg["paths"]["runs"] / a.tag
    for gid in gauges:
        t0 = time.time()
        status = run_catchment(cfg, gid, cat.loc[gid], a.models, n, outdir, a.redo, a.forcing)
        print(f"{gid}: {status} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
