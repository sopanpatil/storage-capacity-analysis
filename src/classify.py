"""Catchment-model classification.

Per catchment x model, in order:
- fails_screen: optimum KGE < 0.5 in calibration or validation;
- few_behavioural: < 10 behavioural sets, so no spread;
- not_classifiable: < 2 benchmarks available, so the envelope has no width;
- unidentifiable: behavioural spread log10(P95/P5) wider than the benchmark envelope,
  log10(max/min) over the available benchmarks: calibration then cannot tell the benchmarks apart;
- consistent / inconsistent: identifiable, and the behavioural P5-P95 range does / does not overlap
  the envelope.
Sensitivity columns: class_ci (unidentifiable only if the lower bootstrap bound of the spread exceeds the
envelope width), class_nosoil (envelope without the soil benchmark), and two thresholds that do not depend on
the envelope width: class_fixed10 (unidentifiable if P95/P5 > 10, spread_abs > 1) and
class_halfprior (unidentifiable if spread_rel > 0.5). spread_rel (behavioural range
relative to the prior's) is carried along for reporting, not used in the class.

Inputs: data/rq1/identifiability_<list>.csv, data/rq2/parameter_benchmarks_<list>.csv (via the RQ2 benchmark
code). Writes data/classification/classification_<list>.csv and classification_summary_<list>.csv
(shares per model x stratum, also for --pilot catchments as a separate group).

Usage: python src/classify.py [--list data/catalog/screened_catchments.csv]
       [--pilot data/catalog/pilot_catchments.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from caravan_io import ROOT, load_config
from rq2_parameter import benchmarks

CLASSES = ["fails_screen", "few_behavioural", "not_classifiable", "unidentifiable", "consistent",
           "inconsistent"]
VARIANTS = ["class", "class_ci", "class_nosoil", "class_fixed10", "class_halfprior"]


def classify(r, spread, lo, hi, n_bench, width=None):
    """`width`: the identifiability threshold for `spread`; by default the envelope width log10(hi/lo)."""
    if width is None:
        width = np.log10(hi / lo)
    overlap = (r.beh_p5 <= hi) & (r.beh_p95 >= lo)
    return np.select(
        [~r.passes, r.spread_abs.isna(), n_bench < 2, spread > width, overlap],
        CLASSES[:5], default="inconsistent")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/screened_catchments.csv")
    ap.add_argument("--pilot", default="data/catalog/pilot_catchments.csv")
    ap.add_argument("--min-valid", type=float, default=0.5)
    a = ap.parse_args()
    cfg = load_config()
    name = Path(a.list).stem
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", index_col="gauge_id")
    ids = pd.read_csv(ROOT / a.list).gauge_id
    b = benchmarks(name, cat.loc[ids], a.min_valid)
    r = pd.read_csv(ROOT / "data" / "rq1" / f"identifiability_{name}.csv").merge(
        b[["env_lo", "env_hi", "n_benchmarks", "env_lo_nosoil", "env_hi_nosoil", "n_benchmarks_nosoil",
           "stratum"]], left_on="gauge_id", right_index=True, validate="m:1")
    r["envelope_width"] = np.log10(r.env_hi / r.env_lo)
    r["class"] = classify(r, r.spread_abs, r.env_lo, r.env_hi, r.n_benchmarks)
    r["class_ci"] = classify(r, r.spread_abs_lo, r.env_lo, r.env_hi, r.n_benchmarks)
    r["class_nosoil"] = classify(r, r.spread_abs, r.env_lo_nosoil, r.env_hi_nosoil, r.n_benchmarks_nosoil)
    r["class_fixed10"] = classify(r, r.spread_abs, r.env_lo, r.env_hi, r.n_benchmarks, width=1.0)
    r["class_halfprior"] = classify(r, r.spread_rel, r.env_lo, r.env_hi, r.n_benchmarks, width=0.5)
    pilot = set(pd.read_csv(ROOT / a.pilot).gauge_id)
    r["pilot"] = r.gauge_id.isin(pilot)
    out = ROOT / "data" / "classification"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["gauge_id", "model", "stratum", "pilot", "passes", "n_behavioural", "capacity_opt", "beh_p5",
            "beh_p95", "spread_abs", "spread_abs_lo", "spread_rel", "opt_position", "env_lo", "env_hi",
            "n_benchmarks", "envelope_width", "class", "class_ci", "class_nosoil", "class_fixed10", "class_halfprior"]
    r[cols].round(4).to_csv(out / f"classification_{name}.csv", index=False)

    rows = []
    groups = [("all", r), ("pilot", r[r.pilot])] + sorted(r.groupby("stratum"))
    for gname, g in groups:
        for model, gm in g.groupby("model"):
            for col in VARIANTS:
                counts = gm[col].value_counts().reindex(CLASSES, fill_value=0)
                rows.append(dict(group=gname, model=model, variant=col, n=len(gm), **counts.to_dict()))
    s = pd.DataFrame(rows)
    s.to_csv(out / f"classification_summary_{name}.csv", index=False)
    print(f"{len(r)} catchment-model pairs -> {out / f'classification_{name}.csv'}")
    show = s[s.group.isin(["all", "pilot"])].set_index(["group", "variant", "model"])
    print(show.to_string())


if __name__ == "__main__":
    main()
