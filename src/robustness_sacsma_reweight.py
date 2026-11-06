"""Robustness item E: SAC-SMA behavioural capacity under a uniform prior on UZTWM + LZTWM.

UZTWM and LZTWM are each sampled uniformly on 1-800 mm, so their sum has a triangular prior on 2-1,600 mm (peak at
801). Each behavioural set (KGE_cal >= the config threshold) is weighted by 1 / (triangular density of its sum), which
turns the prior into a uniform one on 2-1,600 mm. Per screened catchment, as in identifiability.py but weighted:
P5/P50/P95, spread_abs = log10(P95/P5) with a bootstrap 90% interval (resampling sets, weights carried along),
spread_rel against the uniform prior's P5-P95 (81.9-1,520.1 mm), and the Kish effective sample size of the weights.
Spreads need >= 10 behavioural sets, as in the main analysis, so the assessed pairs are unchanged. The five classes
then follow classify.py with the same benchmarks.

Magnitudes: the median log10 ratio to each benchmark of the behavioural P50, unweighted and weighted, and of the
optimum (which does not depend on the prior shape), and the share of optima with UZTWM or LZTWM within 1% of 800 mm.

Writes robustness/E_sacsma_reweight.csv (per catchment), E_sacsma_reweight_classes.csv (shares per class and
criterion, unweighted vs weighted) and E_sacsma_reweight_levels.csv.

Usage: python src/robustness_sacsma_reweight.py [--task-index I --tasks N] | --combine
"""
import argparse
import os

import numpy as np
import pandas as pd

from caravan_io import ROOT, load_config
from classify import CLASSES, VARIANTS, classify
from rq2_parameter import PRIMARY, benchmarks

RNG = np.random.default_rng(20261030)
LO, HI = 1.0, 800.0                       # UZTWM and LZTWM prior bounds (model_setups.SAC_RANGES)
S_LO, S_HI = 2 * LO, 2 * HI
U_P5, U_P95 = S_LO + 0.05 * (S_HI - S_LO), S_LO + 0.95 * (S_HI - S_LO)
OUT = ROOT / "robustness"


def weight(s):
    """1 / triangular density of UZTWM + LZTWM (up to a constant)."""
    return 1.0 / ((HI - LO) - np.abs(s - (LO + HI)))


def wpercentile(x, w, q):
    """Weighted percentiles (q in %), interpolating between the midpoints of each set's cumulative weight;
    with equal weights this equals np.percentile's default for large samples."""
    i = np.argsort(x)
    x, w = x[i], w[i]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.asarray(q) / 100, c, x)


def one(d, n_boot=500, min_beh=10, thr=0.5):
    mc = pd.read_parquet(d / "SACSMA_mc.parquet", columns=["capacity", "kge_cal"])
    beh = mc.capacity[mc.kge_cal >= thr].to_numpy(float)
    row = dict(n_behavioural=len(beh))
    if len(beh) < min_beh:
        return row
    w = weight(beh)
    p5, p50, p95 = wpercentile(beh, w, [5, 50, 95])
    idx = RNG.integers(0, len(beh), size=(n_boot, len(beh)))
    boot = [np.log10(np.divide(*wpercentile(beh[i], w[i], [95, 5]))) for i in idx]
    row.update(ess=w.sum() ** 2 / (w ** 2).sum(), w_beh_p5=p5, w_beh_p50=p50, w_beh_p95=p95,
               w_spread_abs=np.log10(p95 / p5), w_spread_abs_lo=np.nanpercentile(boot, 5),
               w_spread_rel=(p95 - p5) / (U_P95 - U_P5), beh_p50=np.percentile(beh, 50))
    return row


def combine(cfg):
    parts = sorted(OUT.glob("E_sacsma_reweight_task*.csv"))
    e = pd.concat(pd.read_csv(f) for f in parts)
    name = "screened_catchments"
    c = pd.read_csv(ROOT / "data/classification" / f"classification_{name}.csv")
    c = c[c.model == "SACSMA"]
    ne, nc = e.set_index("gauge_id").n_behavioural, c.set_index("gauge_id").n_behavioural
    assert len(ne) == len(nc) and (ne.reindex(nc.index) == nc).all()
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", index_col="gauge_id")
    b = benchmarks(name, cat.loc[c.gauge_id], 0.5)
    r = c.drop(columns=["n_behavioural"]).merge(e, on="gauge_id", validate="1:1").merge(
        b[PRIMARY + ["env_lo_nosoil", "env_hi_nosoil", "n_benchmarks_nosoil"]], left_on="gauge_id",
        right_index=True, validate="1:1")
    w = r.rename(columns={"beh_p5": "u_beh_p5", "beh_p95": "u_beh_p95"}).assign(
        beh_p5=r.w_beh_p5, beh_p95=r.w_beh_p95, spread_abs=r.w_spread_abs)
    r["w_class"] = classify(w, w.spread_abs, w.env_lo, w.env_hi, w.n_benchmarks)
    r["w_class_ci"] = classify(w, w.w_spread_abs_lo, w.env_lo, w.env_hi, w.n_benchmarks)
    r["w_class_nosoil"] = classify(w, w.spread_abs, w.env_lo_nosoil, w.env_hi_nosoil, w.n_benchmarks_nosoil)
    r["w_class_fixed10"] = classify(w, w.spread_abs, w.env_lo, w.env_hi, w.n_benchmarks, width=1.0)
    r["w_class_halfprior"] = classify(w, w.w_spread_rel, w.env_lo, w.env_hi, w.n_benchmarks, width=0.5)
    r.round(4).to_csv(OUT / "E_sacsma_reweight.csv", index=False)

    rows = []
    for v in VARIANTS:
        for label, col in (("unweighted", v), ("weighted", "w_" + v)):
            counts = r[col].value_counts().reindex(CLASSES, fill_value=0)
            assessed = counts[["unidentifiable", "consistent", "inconsistent"]].sum()
            rows.append(dict(variant=v, prior=label, n=len(r), assessed=assessed, **counts.to_dict(),
                             unidentifiable_share=counts.unidentifiable / assessed))
    s = pd.DataFrame(rows)
    s.round(6).to_csv(OUT / "E_sacsma_reweight_classes.csv", index=False)

    ok = r[r.passes & r.w_beh_p50.notna()]
    p = pd.read_csv(ROOT / "data/calibration/calibration_summary.csv")
    p = p[p.model == "SACSMA"].set_index("gauge_id").reindex(r.gauge_id[r.passes])
    rows = []
    for bench in PRIMARY:
        m = ok[bench].notna()
        rows.append(dict(benchmark=bench, n=int(m.sum()),
                         opt=np.log10(ok.capacity_opt[m] / ok[bench][m]).median(),
                         beh_p50_unweighted=np.log10(ok.beh_p50[m] / ok[bench][m]).median(),
                         beh_p50_weighted=np.log10(ok.w_beh_p50[m] / ok[bench][m]).median()))
    lv = pd.DataFrame(rows)
    near = 0.01 * (HI - LO)
    lv["opt_uztwm_at_800"] = (p.p_UZTWM >= HI - near).mean()
    lv["opt_lztwm_at_800"] = (p.p_LZTWM >= HI - near).mean()
    lv["opt_either_at_800"] = ((p.p_UZTWM >= HI - near) | (p.p_LZTWM >= HI - near)).mean()
    lv.round(6).to_csv(OUT / "E_sacsma_reweight_levels.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{len(parts)} parts; ESS of the weights (assessed pairs): "
          f"{r.ess[r['class'].isin(CLASSES[3:])].describe(percentiles=[0.05, 0.5]).round(0).to_dict()}")
    print(s.round(3).to_string(index=False))
    print("median log10 ratio to benchmark (ratio = 10^x):")
    print(lv.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/screened_catchments.csv")
    ap.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    if a.combine:
        return combine(cfg)
    thr = cfg["calibration"]["behavioural_kge"]
    ids = pd.read_csv(ROOT / a.list, dtype={"gauge_id": str}).gauge_id.tolist()[a.task_index::a.tasks]
    rows = [dict(gauge_id=gid, **one(cfg["paths"]["runs"] / "calibration" / gid, thr=thr)) for gid in ids]
    OUT.mkdir(exist_ok=True)
    f = OUT / f"E_sacsma_reweight_task{a.task_index:03d}.csv"
    pd.DataFrame(rows).round(4).to_csv(f, index=False)
    print(f"{len(rows)} catchments -> {f}")


if __name__ == "__main__":
    main()
