"""Robustness item J: FLEX against HBV on the same catchments (robustness_flex.py, robustness_summary_ad.py).

Treated as in the main analysis (ident, add_classes, table of robustness_summary_ad.py). Variants:
  FLEX_so      KGE(Q), full budget; behavioural KGE(Q) >= 0.5 from all samples          (compare HBV_so)
  FLEX_so_10k  as FLEX_so with only the first 10,000 samples (HBV's sample size)
  FLEX_mo      mean of KGE(Q) and KGE(log Q); behavioural both >= 0.5                   (compare HBV_mo)
  FLEX_liang   Sumax 30-700 mm (Liang et al. 2024): its own optimum, samples filtered to that range
  HBV_so, HBV_mo  variants so and mo_both_150k of run A, from AD_per_catchment.csv
Also the Spearman correlation with the benchmark using E from P - Q (data/prior_tests, gao_sr_20), the closest to the design
of Gao et al. (2014), and paired bootstrap differences FLEX - HBV on the pairs where both optima pass.

Decision rule, fixed before the runs: see the "rule" columns of J_flex_paired.csv.

Writes robustness/J_per_catchment.csv, J_flex.csv, J_flex_paired.csv (or --out DIR).
Usage: python src/robustness_summary_flex.py [--rundir DIR] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from caravan_io import ROOT, load_config
from robustness_summary_ad import NAME, OUT, add_classes, ident, table
from rq2_parameter import PRIMARY, benchmarks, spearman_ci

RNG = np.random.default_rng(20261105)
ASSESSED = ["unidentifiable", "consistent", "inconsistent"]
PAIRS = [("FLEX_so", "HBV_so"), ("FLEX_mo", "HBV_mo")]


def flex_rows(rundir, ids):
    rows = []
    for gid in ids:
        if not (rundir / gid / "summary.json").exists():
            continue
        s = json.loads((rundir / gid / "summary.json").read_text())
        mc = pd.read_parquet(rundir / gid / "FLEX_mc.parquet", columns=["capacity", "kge_cal", "kgelog_cal"])
        small = mc.iloc[:10000]
        narrow = mc[mc.capacity.between(30, 700)]

        def add(variant, opt, d, sel):
            o = s["optima"][opt]
            rows.append(dict(gauge_id=gid, variant=variant, model="FLEX", sr_20=o["sr_20"], kgelog_cal=o["kgelog_cal"],
                             kgelog_val=o["kgelog_val"],
                             **ident(d.capacity[sel].to_numpy(float), d.capacity, o["capacity"], o["kge_cal"], o["kge_val"])))
        add("FLEX_so", "so", mc, mc.kge_cal >= 0.5)
        add("FLEX_so_10k", "so", small, small.kge_cal >= 0.5)
        add("FLEX_mo", "mo", mc, (mc.kge_cal >= 0.5) & (mc.kgelog_cal >= 0.5))
        add("FLEX_liang", "so_liang", narrow, narrow.kge_cal >= 0.5)
    return pd.DataFrame(rows)


def paired(r):
    """FLEX minus HBV on the same pairs: Spearman with each benchmark (paired bootstrap 90% interval, and each model's own
    interval, for the decision rule) and the unidentifiable share where both are assessed."""
    out = []
    for fx, hb in PAIRS:
        x, y = (r[r.variant == v].set_index("gauge_id") for v in (fx, hb))
        ids = x.index.intersection(y.index)
        x, y = x.loc[ids], y.loc[ids]
        both = x.passes.astype(bool) & y.passes.astype(bool)
        for bench in PRIMARY + ["gao"]:
            m = both & x[bench].notna()
            a, b, c = x.capacity_opt[m].to_numpy(), y.capacity_opt[m].to_numpy(), x[bench][m].to_numpy()
            if len(a) < 10:
                continue
            ra, lo_a, hi_a = spearman_ci(a, c)
            rb, lo_b, hi_b = spearman_ci(b, c)
            idx = RNG.integers(0, len(a), size=(2000, len(a)))
            diff = [spearmanr(a[i], c[i]).statistic - spearmanr(b[i], c[i]).statistic for i in idx]
            out.append(dict(comparison=f"{fx} - {hb}", benchmark=bench, n=len(a), rho_flex=ra, flex_lo=lo_a, flex_hi=hi_a,
                            rho_hbv=rb, hbv_lo=lo_b, hbv_hi=hi_b, diff=ra - rb,
                            diff_lo=np.nanpercentile(diff, 5), diff_hi=np.nanpercentile(diff, 95),
                            rule_intervals_separate=bool(lo_a > hi_b),
                            median_ratio_flex=np.median(a / c), median_ratio_hbv=np.median(b / c)))
        for klass in ["class", "class_ci", "class_nosoil", "class_fixed10", "class_halfprior"]:
            m = x[klass].isin(ASSESSED) & y[klass].isin(ASSESSED)
            ux, uy = (x[klass][m] == "unidentifiable").mean(), (y[klass][m] == "unidentifiable").mean()
            out.append(dict(comparison=f"{fx} - {hb}", benchmark=f"unidentifiable {klass}", n=int(m.sum()),
                            rho_flex=ux, rho_hbv=uy, diff=ux - uy,
                            rule_identifiable_in_most_pairs=bool(klass == "class" and ux < 0.5)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rundir")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = load_config()
    rundir = Path(a.rundir) if a.rundir else cfg["paths"]["workspace"] / "robustness" / "flex"
    out = Path(a.out) if a.out else OUT
    ids = pd.read_csv(OUT / "subset.csv").gauge_id.tolist()
    fx = flex_rows(rundir, ids)
    ids = fx.gauge_id.unique().tolist()
    ad = pd.read_csv(OUT / "AD_per_catchment.csv")
    hb = ad[(ad["item"] == "A") & ad.variant.isin(["so", "mo_both_150k"]) & ad.gauge_id.isin(ids)].copy()
    hb["variant"] = hb.variant.map({"so": "HBV_so", "mo_both_150k": "HBV_mo"})
    r = pd.concat([fx, hb.drop(columns="item")], ignore_index=True)
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    b = benchmarks(NAME, cat.loc[ids], 0.5)[PRIMARY + ["env_lo", "env_hi", "n_benchmarks", "env_lo_nosoil",
                                                     "env_hi_nosoil", "n_benchmarks_nosoil"]]
    g = pd.read_csv(ROOT / "data/prior_tests" / f"gao_benchmark_{NAME}.csv", index_col=0)
    b["gao"] = g.gao_sr_20.where(g.eligible & (g.gao_sr_20 > 0)).reindex(b.index)
    r = add_classes(r, b)
    t = table(r, "variant", ["sr_20"])
    for col, label in [("capacity_opt", "param"), ("sr_20", "sr_20")]:
        for variant, d in r[r.passes].groupby("variant"):
            m = d.gao.notna() & d[col].notna()
            rho, lo, hi = spearman_ci(d.loc[m, col].to_numpy(), d.loc[m, "gao"].to_numpy()) if m.sum() > 9 else (np.nan,) * 3
            t.loc[t.variant == variant, [f"rho_{label}_gao", f"rho_{label}_gao_lo", f"rho_{label}_gao_hi", f"n_{label}_gao"]] = \
                [rho, lo, hi, int(m.sum())]
    p = paired(r)
    out.mkdir(parents=True, exist_ok=True)
    r.round(6).to_csv(out / "J_per_catchment.csv", index=False)
    t.round(6).to_csv(out / "J_flex.csv", index=False)
    p.round(6).to_csv(out / "J_flex_paired.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    show = ["variant", "n", "passes", "median_n_behavioural", "median_spread_abs"] + \
           [c for c in t if c.startswith("unidentifiable_")] + \
           [f"rho_param_{x}" for x in ["climate", "stocker", "zhao", "soil", "gao"]] + ["rho_sr_20_climate", "rho_sr_20_gao"]
    print(t[show].round(3).T.to_string())
    print(p.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
