"""Robustness runs A-D: summaries on the subset, next to the main results there.

Subset: robustness/subset.csv (397 catchments, assessed for all three models in the main analysis). Runs from
src/robustness_calibrate.py in <workspace>/robustness/<run>/<gauge_id>/. Every variant is treated as in the main analysis:
optimum passes if KGE(Q) >= 0.5 in both periods; behavioural sets from the Monte Carlo; spreads as identifiability.py
(P5-P95, log10(P95/P5) with a 500-sample bootstrap 90% interval, spread_rel against the prior P5-P95 of all samples,
>= 10 sets); the five classes as classify.py; Spearman with each benchmark as rq2_parameter.py (bootstrap 90%
interval), on the pairs whose optimum passes.

- A. HBV: main (main run; SCE-UA stopped early), so (hbv_so: KGE(Q), full budget; behavioural sets as main), and
  multi-objective hbv_mo (mean of KGE(Q) and KGE(log Q)) with behavioural sets from the same 10,000 samples by
  both KGEs >= 0.5 (mo_both), their mean >= 0.5 (mo_mean), or both >= 0.5 in hbv_150k's 150,000 samples
  (mo_both_150k). Also the simulated deficit S_R,20 of each optimum (state_capacity.evaluate, 2003-2020), with its
  Spearman with each benchmark; the main one is data/rq3's.
- B. HBV, hbv_150k: behavioural sets (KGE(Q) >= 0.5) from the first 10k, 30k, 60k and all 150k samples (nested
  uniform draws), with the main optimum. Width = spread_abs.
- C. GR6J optimum, Michel (main) vs SCE-UA (gr6j_sce): KGE and X1.
- D. GR4J (gr4j, 100,000 samples) vs GR6J (main, 150,000): X1 identifiability, classes, Spearman.
- S. SAC-SMA optimum, main vs sacsma_so (full-budget SCE-UA): KGE and UZTWM + LZTWM.

Writes robustness/AD_per_catchment.csv (catchment x variant), A_hbv_multiobjective.csv, B_hbv_convergence.csv,
C_gr6j_optimiser.csv, D_gr4j.csv, S_sacsma_optimum.csv (each: per variant, classes under the five criteria and
Spearman with each benchmark, or the optimum comparison).

Usage: python src/robustness_summary_ad.py [--limit N]
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from caravan_io import ROOT, load_config
from classify import CLASSES, VARIANTS, classify
from identifiability import opt_position, spread
from model_fluxes import forcing
from model_setups import SETUPS
from rq2_parameter import PRIMARY, benchmarks, spearman_ci
from state_capacity import evaluate

RNG = np.random.default_rng(20261030)
NAME = "screened_catchments"
ASSESSED = ["unidentifiable", "consistent", "inconsistent"]
OUT = ROOT / "robustness"


def summary(path, model):
    return json.loads((path / "summary.json").read_text())["models"][model]


def ident(beh, prior, opt, kge_cal, kge_val, min_beh=10, n_boot=500):
    """One identifiability row (identifiability.one) from behavioural capacities and all-sample capacities."""
    prior_p5, prior_p95 = np.percentile(prior, [5, 95])
    row = dict(capacity_opt=opt, kge_cal=kge_cal, kge_val=kge_val, passes=kge_cal >= 0.5 and kge_val >= 0.5,
               n_behavioural=len(beh), prior_p5=prior_p5, prior_p95=prior_p95)
    if len(beh) >= min_beh:
        p5, p50, p95 = np.percentile(beh, [5, 50, 95])
        boot = [spread(RNG.choice(beh, len(beh))) for _ in range(n_boot)]
        row.update(beh_p5=p5, beh_p50=p50, beh_p95=p95, spread_abs=spread(beh),
                   spread_abs_lo=np.nanpercentile(boot, 5), spread_rel=(p95 - p5) / (prior_p95 - prior_p5),
                   opt_position=opt_position(opt, p5, p95))
    return row


def add_classes(r, b):
    r = r.merge(b, left_on="gauge_id", right_index=True, validate="m:1").astype({"passes": bool})
    r["class"] = classify(r, r.spread_abs, r.env_lo, r.env_hi, r.n_benchmarks)
    r["class_ci"] = classify(r, r.spread_abs_lo, r.env_lo, r.env_hi, r.n_benchmarks)
    r["class_nosoil"] = classify(r, r.spread_abs, r.env_lo_nosoil, r.env_hi_nosoil, r.n_benchmarks_nosoil)
    r["class_fixed10"] = classify(r, r.spread_abs, r.env_lo, r.env_hi, r.n_benchmarks, width=1.0)
    r["class_halfprior"] = classify(r, r.spread_rel, r.env_lo, r.env_hi, r.n_benchmarks, width=0.5)
    return r


def table(r, group, extra=()):
    """Per group: counts, median width, unidentifiable share (of assessed pairs) per criterion, Spearman per benchmark
    of the optimum capacity and of each column in `extra` (pairs that pass)."""
    rows = []
    for key, g in r.groupby(group, sort=False):
        row = dict(variant=key, n=len(g), passes=int(g.passes.sum()), n_beh_ge10=int(g.spread_abs.notna().sum()),
                   median_n_behavioural=g.n_behavioural.median(), median_spread_abs=g.spread_abs.median())
        for v in VARIANTS:
            c = g[v].value_counts().reindex(CLASSES, fill_value=0)
            row[f"assessed_{v}"] = int(c[ASSESSED].sum())
            row[f"unidentifiable_{v}"] = c.unidentifiable / c[ASSESSED].sum() if c[ASSESSED].sum() else np.nan
            row[f"consistent_{v}"] = c.consistent / c[ASSESSED].sum() if c[ASSESSED].sum() else np.nan
        p = g[g.passes]
        for col, label in [("capacity_opt", "param")] + [(e, e) for e in extra]:
            for bench in PRIMARY:
                m = p[bench].notna() & p[col].notna() & (p[col] > 0)
                rho, lo, hi = spearman_ci(p.loc[m, col].to_numpy(), p.loc[m, bench].to_numpy())
                row.update({f"rho_{label}_{bench}": rho, f"rho_{label}_{bench}_lo": lo,
                            f"rho_{label}_{bench}_hi": hi, f"n_{label}_{bench}": int(m.sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def optimum_pair(ref, new, cap_label):
    """Compare two optima per catchment (ref, new: DataFrames with kge_cal, kge_val, capacity_opt)."""
    d = ref.join(new, lsuffix="_ref", rsuffix="_new")
    lr = np.log10(d.capacity_opt_new / d.capacity_opt_ref)
    dk = d.kge_cal_new - d.kge_cal_ref
    return pd.DataFrame([dict(
        quantity=cap_label, n=len(d), median_kge_cal_ref=d.kge_cal_ref.median(), median_kge_cal_new=d.kge_cal_new.median(),
        median_kge_cal_change=dk.median(), share_new_better_by_0_01=(dk > 0.01).mean(),
        share_new_worse_by_0_01=(dk < -0.01).mean(), median_kge_val_change=(d.kge_val_new - d.kge_val_ref).median(),
        spearman_capacity=spearmanr(d.capacity_opt_ref, d.capacity_opt_new).statistic,
        median_abs_log10_ratio=lr.abs().median(), share_within_10pct=(lr.abs() <= np.log10(1.1)).mean(),
        share_beyond_factor_2=(lr.abs() > np.log10(2)).mean(), median_log10_ratio=lr.median())])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="first N catchments only (testing)")
    a = ap.parse_args()
    cfg = load_config()
    ws, runs = cfg["paths"]["workspace"] / "robustness", cfg["paths"]["runs"] / "calibration"
    ids = pd.read_csv(OUT / "subset.csv").gauge_id.tolist()[:a.limit]
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    b = benchmarks(NAME, cat.loc[ids], 0.5)[PRIMARY + ["env_lo", "env_hi", "n_benchmarks", "env_lo_nosoil",
                                                     "env_hi_nosoil", "n_benchmarks_nosoil"]]
    main_id = pd.read_csv(ROOT / "data/rq1" / f"identifiability_{NAME}.csv")
    main_id = main_id[main_id.gauge_id.isin(ids)].set_index(["gauge_id", "model"])
    rq3 = pd.read_csv(ROOT / "data/rq3" / f"state_capacity_{NAME}_summary.csv").set_index(["gauge_id", "model"])

    rows = []
    for k, gid in enumerate(ids):
        def add(item, variant, model, **kw):
            rows.append(dict(gauge_id=gid, item=item, variant=variant, model=model, **kw))

        f = forcing(cfg, gid, cat.loc[gid])
        # A and B: HBV
        hbv = SETUPS["HBV"]
        main_mc = pd.read_parquet(runs / gid / "HBV_mc.parquet", columns=["capacity", "kge_cal"])
        m = summary(runs / gid, "HBV")
        main_row = ident(main_mc.capacity[main_mc.kge_cal >= 0.5].to_numpy(float), main_mc.capacity, m["capacity"],
                         m["kge_cal"], m["kge_val"])
        assert main_row["n_behavioural"] == main_id.loc[(gid, "HBV"), "n_behavioural"]
        add("A", "main", "HBV", **main_row, sr_20=rq3.loc[(gid, "HBV"), "opt_sr_20"])
        so = summary(ws / "hbv_so" / gid, "HBV")
        p = np.array([so["params"][n] for n in hbv.names])
        add("A", "so", "HBV", **ident(main_mc.capacity[main_mc.kge_cal >= 0.5].to_numpy(float), main_mc.capacity,
                                      so["capacity"], so["kge_cal"], so["kge_val"]),
            sr_20=evaluate(f, "HBV", p)["sr_20"])
        mo = summary(ws / "hbv_mo" / gid, "HBV")
        p = np.array([mo["params"][n] for n in hbv.names])
        sr_mo = evaluate(f, "HBV", p)["sr_20"]
        mc = pd.read_parquet(ws / "hbv_mo" / gid / "HBV_mc.parquet", columns=["capacity", "kge_cal", "kgelog_cal"])
        big = pd.read_parquet(ws / "hbv_150k" / gid / "HBV_mc.parquet", columns=["capacity", "kge_cal", "kgelog_cal"])
        for variant, d, sel in [("mo_both", mc, (mc.kge_cal >= 0.5) & (mc.kgelog_cal >= 0.5)),
                                ("mo_mean", mc, (mc.kge_cal + mc.kgelog_cal) / 2 >= 0.5),
                                ("mo_both_150k", big, (big.kge_cal >= 0.5) & (big.kgelog_cal >= 0.5))]:
            add("A", variant, "HBV", **ident(d.capacity[sel].to_numpy(float), d.capacity, mo["capacity"],
                                             mo["kge_cal"], mo["kge_val"]),
                sr_20=sr_mo, kgelog_cal=mo["kgelog_cal"], kgelog_val=mo["kgelog_val"])
        for n in (10_000, 30_000, 60_000, 150_000):
            d = big.iloc[:n]
            add("B", f"{n // 1000}k", "HBV", **ident(d.capacity[d.kge_cal >= 0.5].to_numpy(float), d.capacity,
                                                      m["capacity"], m["kge_cal"], m["kge_val"]))
        # C and D: GR
        g6 = summary(runs / gid, "GR6J")
        mc6 = pd.read_parquet(runs / gid / "GR6J_mc.parquet", columns=["capacity", "kge_cal"])
        add("D", "GR6J", "GR6J", **ident(mc6.capacity[mc6.kge_cal >= 0.5].to_numpy(float), mc6.capacity,
                                         g6["capacity"], g6["kge_cal"], g6["kge_val"]))
        g4 = summary(ws / "gr4j" / gid, "GR4J")
        mc4 = pd.read_parquet(ws / "gr4j" / gid / "GR4J_mc.parquet", columns=["capacity", "kge_cal"])
        add("D", "GR4J", "GR4J", **ident(mc4.capacity[mc4.kge_cal >= 0.5].to_numpy(float), mc4.capacity,
                                         g4["capacity"], g4["kge_cal"], g4["kge_val"]))
        sce = summary(ws / "gr6j_sce" / gid, "GR6J")
        add("C", "michel", "GR6J", capacity_opt=g6["capacity"], kge_cal=g6["kge_cal"], kge_val=g6["kge_val"])
        add("C", "sce", "GR6J", capacity_opt=sce["capacity"], kge_cal=sce["kge_cal"], kge_val=sce["kge_val"])
        sac, sso = summary(runs / gid, "SACSMA"), summary(ws / "sacsma_so" / gid, "SACSMA")
        add("S", "main", "SACSMA", capacity_opt=sac["capacity"], kge_cal=sac["kge_cal"], kge_val=sac["kge_val"])
        add("S", "so", "SACSMA", capacity_opt=sso["capacity"], kge_cal=sso["kge_cal"], kge_val=sso["kge_val"])
        if k % 50 == 0:
            print(f"{k}/{len(ids)}", flush=True)

    r = pd.DataFrame(rows)
    r.round(6).to_csv(OUT / "AD_per_catchment.csv", index=False)
    pd.set_option("display.width", 250)
    for item, fname, extra in [("A", "A_hbv_multiobjective", ["sr_20"]), ("B", "B_hbv_convergence", []),
                               ("D", "D_gr4j", [])]:
        t = table(add_classes(r[r.item == item], b), "variant", extra)
        if item == "A":
            a = r[r.item == "A"].groupby("variant", sort=False)
            t["median_kge_cal"] = a.kge_cal.median().values
            t["median_kge_val"] = a.kge_val.median().values
            t["median_kgelog_cal"] = a.kgelog_cal.median().values
        t.round(6).to_csv(OUT / f"{fname}.csv", index=False)
        show = ["variant", "passes", "n_beh_ge10", "median_n_behavioural", "median_spread_abs"] + \
               [f"unidentifiable_{v}" for v in VARIANTS] + [f"rho_param_{x}" for x in PRIMARY] + \
               [f"rho_{e}_{x}" for e in extra for x in PRIMARY]
        print(f"\n{item}:\n" + t[show].round(3).to_string(index=False))
    for item, fname, label in [("C", "C_gr6j_optimiser", "X1"), ("S", "S_sacsma_optimum", "UZTWM+LZTWM")]:
        c = r[r.item == item]
        ref, new = (c[c.variant == v].set_index("gauge_id")[["kge_cal", "kge_val", "capacity_opt"]]
                    for v in c.variant.unique())
        t = optimum_pair(ref, new, label)
        cb = c.merge(b, left_on="gauge_id", right_index=True).assign(passes=lambda x: (x.kge_cal >= 0.5) & (x.kge_val >= 0.5))
        for v, g in cb.groupby("variant", sort=False):
            for bench in PRIMARY:
                p = g[g.passes & g[bench].notna()]
                t[f"rho_{v}_{bench}"] = spearmanr(p.capacity_opt, p[bench]).statistic
        t.round(6).to_csv(OUT / f"{fname}.csv", index=False)
        print(f"\n{item}:\n" + t.round(3).T.to_string())


if __name__ == "__main__":
    main()
