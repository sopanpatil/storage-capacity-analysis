"""Robustness item G: identifiability with the same number of behavioural sets per model.

Catchments assessed for all three models under the main criterion (data/classification: KGE >= 0.5 in both periods,
>= 10 behavioural sets, >= 2 benchmarks). In each, every model's behavioural capacities (KGE_cal >= the config
threshold) are subsampled without replacement to a common count N, and spread_abs (with a 200-sample bootstrap 90%
interval), spread_rel (against the prior P5-P95 of identifiability.py) and the P5-P95 range are recomputed; the five
classes then follow classify.py with the same benchmarks. N variants:
- min: per catchment, the smallest behavioural count of the three models (all catchments);
- 10 and 100: fixed, in the catchments where all three models have at least N sets.
Each subsample is repeated --reps times (seed 20261030); shares are averaged over repetitions.

Writes robustness/G_common_subset.parquet (per catchment x model x variant x repetition),
G_common_subset_classes.csv (mean shares per model x criterion x variant, next to the full-ensemble shares on the
same catchments) and G_common_subset_n.csv (the distribution of N).

Usage: python src/robustness_common_subset.py [--reps 20]
"""
import argparse

import numpy as np
import pandas as pd

from caravan_io import ROOT, load_config
from classify import CLASSES, VARIANTS, classify
from rq2_parameter import benchmarks
from model_setups import SETUPS

RNG = np.random.default_rng(20261030)
FIXED = [10, 100]
ASSESSED = ["unidentifiable", "consistent", "inconsistent"]
OUT = ROOT / "robustness"


def stats(x, n_boot=200):
    p5, p95 = np.percentile(x, [5, 95])   # capacities are >= 1 mm (prior lower bounds), so no zero P5
    b = np.percentile(x[RNG.integers(0, len(x), size=(n_boot, len(x)))], [5, 95], axis=1)
    return dict(beh_p5=p5, beh_p95=p95, spread_abs=np.log10(p95 / p5),
                spread_abs_lo=np.percentile(np.log10(b[1] / b[0]), 5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--limit", type=int, help="first N catchments only (testing)")
    a = ap.parse_args()
    cfg = load_config()
    thr = cfg["calibration"]["behavioural_kge"]
    name = "screened_catchments"
    c = pd.read_csv(ROOT / "data/classification" / f"classification_{name}.csv")
    ok = c[c["class"].isin(ASSESSED)].groupby("gauge_id").size()
    ids = ok[ok == len(SETUPS)].index.tolist()[:a.limit]
    c = c[c.gauge_id.isin(ids)].set_index(["gauge_id", "model"])
    prior = pd.read_csv(ROOT / "data/rq1" / f"identifiability_{name}.csv").set_index(["gauge_id", "model"])
    rows = []
    for k, gid in enumerate(ids):
        d = cfg["paths"]["runs"] / "calibration" / gid
        beh = {}
        for m in SETUPS:
            mc = pd.read_parquet(d / f"{m}_mc.parquet", columns=["capacity", "kge_cal"])
            beh[m] = mc.capacity[mc.kge_cal >= thr].to_numpy(float)
            assert len(beh[m]) == c.loc[(gid, m), "n_behavioural"], (gid, m)
        n_min = min(len(v) for v in beh.values())
        for variant, n in [("min", n_min)] + [(str(f), f) for f in FIXED if f <= n_min]:
            for rep in range(a.reps):
                for m, x in beh.items():
                    sub = x if n == len(x) else RNG.choice(x, n, replace=False)
                    p = prior.loc[(gid, m)]
                    s = stats(sub)
                    rows.append(dict(gauge_id=gid, model=m, variant=variant, rep=rep, n=n, **s,
                                     spread_rel=(s["beh_p95"] - s["beh_p5"]) / (p.prior_p95 - p.prior_p5)))
        if k % 250 == 0:
            print(f"{k}/{len(ids)} catchments", flush=True)
    r = pd.DataFrame(rows)

    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", index_col="gauge_id")
    b = benchmarks(name, cat.loc[ids], 0.5)
    r = r.merge(b[["env_lo", "env_hi", "n_benchmarks", "env_lo_nosoil", "env_hi_nosoil", "n_benchmarks_nosoil"]],
                left_on="gauge_id", right_index=True, validate="m:1")
    r["passes"] = True
    r["class"] = classify(r, r.spread_abs, r.env_lo, r.env_hi, r.n_benchmarks)
    r["class_ci"] = classify(r, r.spread_abs_lo, r.env_lo, r.env_hi, r.n_benchmarks)
    r["class_nosoil"] = classify(r, r.spread_abs, r.env_lo_nosoil, r.env_hi_nosoil, r.n_benchmarks_nosoil)
    r["class_fixed10"] = classify(r, r.spread_abs, r.env_lo, r.env_hi, r.n_benchmarks, width=1.0)
    r["class_halfprior"] = classify(r, r.spread_rel, r.env_lo, r.env_hi, r.n_benchmarks, width=0.5)
    OUT.mkdir(exist_ok=True)
    r.drop(columns="passes").to_parquet(OUT / "G_common_subset.parquet", index=False)

    out = []
    full = c.reset_index()
    for variant, g in r.groupby("variant"):
        sub_ids = g.gauge_id.unique()
        for m, gm in g.groupby("model"):
            fm = full[(full.model == m) & full.gauge_id.isin(sub_ids)]
            for v in VARIANTS:
                # shares among assessed pairs; one row per repetition
                sh = gm.groupby("rep")[v].value_counts().unstack().reindex(columns=CLASSES, fill_value=0).fillna(0)
                sh = sh[ASSESSED].div(sh[ASSESSED].sum(axis=1), axis=0)
                fs = fm[v].value_counts().reindex(ASSESSED, fill_value=0)
                fs = fs / fs.sum()
                out.append(dict(variant=variant, model=m, criterion=v, n_catchments=len(sub_ids),
                                median_n=gm.n.median(), full_unidentifiable=fs.unidentifiable,
                                subsampled_unidentifiable=sh.unidentifiable.mean(),
                                subsampled_unidentifiable_min=sh.unidentifiable.min(),
                                subsampled_unidentifiable_max=sh.unidentifiable.max(),
                                full_consistent=fs.consistent, subsampled_consistent=sh.consistent.mean()))
    s = pd.DataFrame(out)
    s.round(6).to_csv(OUT / "G_common_subset_classes.csv", index=False)
    nd = r[(r.variant == "min") & (r.rep == 0) & (r.model == "HBV")].n.describe(percentiles=[0.05, 0.25, 0.5, 0.75])
    nd.round(1).to_csv(OUT / "G_common_subset_n.csv", header=["N_min"])
    pd.set_option("display.width", 220)
    print(f"{len(ids)} catchments; N (min variant): {nd.round(0).to_dict()}")
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
