"""Robustness runs A-D on the fixed subset (from robustness_subset.py).

Same inputs, split, warm-up and priors as calibrate.py; only what each run changes is new:
  hbv_mo    A. HBV, SCE-UA on the mean of KGE(Q) and KGE(log(Q + eps)), eps = 1% of the mean observed
               calibration flow (Pushpalatha et al. 2012), added to observed and simulated flow. Monte Carlo
               = the main run's 10,000-sample Latin hypercube (same seed), with both KGEs; behavioural
               needs both >= 0.5 on calibration days.
  hbv_so       Baseline for A: HBV optimum by SCE-UA on KGE(Q) alone, with the same full evaluation budget
               (the main run's SCE-UA stopped early; see sce()); no Monte Carlo.
  sacsma_so    Optional check: SAC-SMA optimum by SCE-UA (KGE) with the full evaluation budget, against the
               main run's optimum; no Monte Carlo.
  hbv_150k  B. HBV Monte Carlo only, 150,000-sample Latin hypercube (optimum as in the main run). Nested
               subsamples (the first 10k, 30k, 60k rows) are uniform draws from the prior. Also records
               KGE(log Q), so A's behavioural sets can come from 150k samples where 10k give too few.
               n_behavioural in summary.json counts KGE(Q) only.
  gr6j_sce  C. GR6J optimum by SCE-UA on the bounded prior (the main run uses airGR's Michel algorithm);
               no Monte Carlo. All SCE-UA runs here use the full evaluation budget (sce()).
  gr4j      D. GR4J-CemaNeige (grsuite, verified against airGR 1.7.9: verification/README.md), set up
               as GR6J (one snow layer, Michel, prior X1-X4 and CN1-CN2 as GR6J's); Monte Carlo of --samples.
Writes <workspace>/robustness/<run>/<gauge_id>/: summary.json, <model>_mc.parquet (not gr6j_sce, hbv_so, sacsma_so),
<model>_opt_q.npy (not hbv_150k). A catchment with an existing summary.json is skipped.

Usage:
  python src/robustness_calibrate.py --run hbv_mo --list robustness/subset.csv [--task-index I --tasks N]
  python src/robustness_calibrate.py --run gr4j --samples 50000 GAUGE_ID [...]
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

from calibrate import catchment_inputs, kge, monte_carlo
from caravan_io import ROOT, load_config
from model_setups import GR6J, SETUPS, _bounds
from hbv_model.sceua import sceua  # noqa: E402  (path set by model_setups)


class GR4J(GR6J):
    """GR4J-CemaNeige via grsuite, set up as GR6J (model_setups.GR6J): one snow layer at the catchment
    mean elevation, Michel optimisation, bounded uniform prior with GR6J's X1-X4, CN1 and CN2 ranges."""
    name = "GR4J"
    names = gs.param_names("CemaNeigeGR4J")
    model = "CemaNeigeGR4J"
    prior_version = "bounded-marrmot-as-gr6j-2026-10-29"
    lo, hi = _bounds(dict(zip(GR6J.names, zip(GR6J.lo, GR6J.hi))), ["X1", "X2", "X3", "X4", "CN1", "CN2"])

    def simulate(self, f, p):
        q = np.full(len(f["P"]), np.nan)
        q[self.n_warm:] = gs.run_model_cemaneige_gr4j(self.im, self.ro, np.asarray(p, float))["Qsim"]
        return q


def kge_log(sim, obs, eps):
    return kge(np.log(sim + eps), np.log(obs + eps))


def sce(setup, objective, cfg, seed):
    """SCE-UA with the full evaluation budget (tol=-1): hbv_model.sceua otherwise stops at the first shuffle
    (after 10) without improvement, which left GR6J and the multi-objective HBV short of their optima in
    tests."""
    def neg(p):
        k = objective(p)
        return -k if np.isfinite(k) else 1e6
    best, _ = sceua(neg, setup.bounds(),
                    n_complexes=cfg["calibration"]["sce_complexes"],
                    maxiter=cfg["calibration"]["sce_maxiter"], seed=seed, tol=-1, verbose=False)
    return np.asarray(best, float)


def michel(setup, obs, cal, n_warm):
    crit = gs.InputsCrit("KGE", obs[n_warm:], bool_crit=cal[n_warm:])
    opts = gs.CalibOptions(setup.model, search_ranges=setup.search_ranges())
    return np.asarray(gs.calibration_michel(setup.im, setup.ro, crit, opts, verbose=False).param_final_r, float)


def run_catchment(cfg, run, gid, row, n_samples, outdir):
    d = outdir / gid
    if (d / "summary.json").exists():
        return "skipped"
    d.mkdir(parents=True, exist_ok=True)
    f, obs, cal, val, info = catchment_inputs(cfg, gid, row)
    seed = zlib.crc32(gid.encode())
    setup = GR4J() if run == "gr4j" else SETUPS[{"gr6j_sce": "GR6J", "sacsma_so": "SACSMA"}.get(run, "HBV")]
    if hasattr(setup, "prepare"):
        setup.prepare(f)
    eps = 0.01 * float(np.nanmean(obs[cal]))
    out = dict(params=None)
    t0 = time.time()
    if run == "hbv_mo":
        best = sce(setup, lambda p: 0.5 * (kge((s := setup.simulate(f, p))[cal], obs[cal])
                                           + kge_log(s[cal], obs[cal], eps)), cfg, seed)
    elif run in ("gr6j_sce", "hbv_so", "sacsma_so"):
        best = sce(setup, lambda p: kge(setup.simulate(f, p)[cal], obs[cal]), cfg, seed)
    elif run == "gr4j":
        best = michel(setup, obs, cal, f["n_warm"])
    else:
        best = None
    t1 = time.time()
    if best is not None:
        sim = setup.simulate(f, best)
        np.save(d / f"{setup.name}_opt_q.npy", sim.astype("float32"))
        out = dict(params=dict(zip(setup.names, map(float, best))), capacity=float(setup.capacity_of(best)[0]),
                   kge_cal=float(kge(sim[cal], obs[cal])), kge_val=float(kge(sim[val], obs[val])))
        if run == "hbv_mo":
            out.update(kgelog_cal=float(kge_log(sim[cal], obs[cal], eps)),
                       kgelog_val=float(kge_log(sim[val], obs[val], eps)))
    if run not in ("gr6j_sce", "hbv_so", "sacsma_so"):
        n = n_samples
        if run in ("hbv_mo", "hbv_150k"):   # Latin hypercube as monte_carlo (same seed), with KGE(log Q) added
            params = setup.from_unit(qmc.LatinHypercube(d=len(setup.names), seed=seed).random(n))
            k = np.empty((n, 4))
            for i, p in enumerate(params):
                s = setup.simulate(f, p)
                k[i] = (kge(s[cal], obs[cal]), kge(s[val], obs[val]),
                        kge_log(s[cal], obs[cal], eps), kge_log(s[val], obs[val], eps))
            mc = pd.DataFrame(params, columns=setup.names).astype("float32")
            mc["capacity"] = setup.capacity_of(params).astype("float32")
            for j, c in enumerate(["kge_cal", "kge_val", "kgelog_cal", "kgelog_val"]):
                mc[c] = k[:, j].astype("float32")
            beh = mc.kge_cal >= cfg["calibration"]["behavioural_kge"]
            if run == "hbv_mo":
                beh &= mc.kgelog_cal >= cfg["calibration"]["behavioural_kge"]
        else:
            mc = monte_carlo(setup, f, obs, cal, val, n, seed)
            beh = mc.kge_cal >= cfg["calibration"]["behavioural_kge"]
        mc.to_parquet(d / f"{setup.name}_mc.parquet", index=False)
        out.update(n_behavioural=int(beh.sum()), n_samples=int(n))
    t2 = time.time()
    out.update(prior_version=setup.prior_version, seconds_optimise=round(t1 - t0, 1),
               seconds_monte_carlo=round(t2 - t1, 1))
    summary = dict(gauge_id=gid, run=run, log_eps=eps, **info, models={setup.name: out})
    tmp = d / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1))
    tmp.replace(d / "summary.json")
    return "done"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gauges", nargs="*")
    ap.add_argument("--run", required=True, choices=["hbv_mo", "hbv_so", "hbv_150k", "gr6j_sce", "gr4j", "sacsma_so"])
    ap.add_argument("--list", help="CSV with a gauge_id column")
    ap.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--samples", type=int, help="Monte Carlo samples (default: hbv_mo 10,000 as in the "
                    "main run, hbv_150k 150,000; required for gr4j)")
    ap.add_argument("--tag", help="output subfolder (default: the run name)")
    a = ap.parse_args()
    cfg = load_config()
    n = a.samples or {"hbv_mo": cfg["calibration"]["monte_carlo_samples"]["HBV"], "hbv_150k": 150000}.get(a.run)
    if a.run == "gr4j" and not n:
        ap.error("--samples is required for gr4j")
    gauges = list(a.gauges)
    if a.list:
        gauges += pd.read_csv(ROOT / a.list if not os.path.isabs(a.list) else a.list).gauge_id.tolist()
    gauges = gauges[a.task_index::a.tasks]
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    outdir = cfg["paths"]["workspace"] / "robustness" / (a.tag or a.run)
    for gid in gauges:
        t0 = time.time()
        status = run_catchment(cfg, a.run, gid, cat.loc[gid], n, outdir)
        print(f"{gid}: {status} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
