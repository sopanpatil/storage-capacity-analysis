"""RQ3 vs benchmarks: does the storage each model uses agree better than its parameter?

Same sample, benchmarks, envelope, strata and statistics as src/rq2_parameter.py, computed twice on the
identical catchment-model pairs:
- level "parameter": optimum capacity parameter, behavioural P5-P95 of the parameter (as RQ2);
- level "state": mass-curve S_R,20 of the model's own fluxes for the optimum, behavioural P5-P95 of it
  (data/rq3/state_capacity_<list>_summary.csv; percentiles used only with >= 10 behavioural sets).
For the climate return-period sensitivities the state uses the same return period (sr_10 vs T10,
sr_40 vs T40), like for like. The bounded store_deficit_20 diagnostic is not compared.
- level "control" (if data/rq3/control_<list>.csv exists, src/rq3_control.py): the same mass curve with
  the model's soil input but PET scaled to the model's mean AET as outflow (optimum only, so no coverage).

Writes data/rq3/state_vs_benchmarks_summary_<list>.csv (level x model x benchmark x stratum).

Usage: python src/rq3_benchmarks.py [--list data/catalog/screened_catchments.csv] [--min-valid 0.5]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from caravan_io import ROOT, load_config
from rq2_parameter import PRIMARY, SENSITIVITY, benchmarks, summarise

STATE_T = {"climate_T10": 10, "climate_T40": 40}


def state_frame(pairs, rq3, t):
    """pairs with capacity_opt/beh_p5/beh_p95 replaced by the state S_R,t."""
    s = pairs.drop(columns=["capacity_opt", "beh_p5", "beh_p95"]).merge(
        rq3[["gauge_id", "model", f"opt_sr_{t}", f"beh_sr_{t}_p5", f"beh_sr_{t}_p95"]],
        on=["gauge_id", "model"], validate="1:1")
    s = s.rename(columns={f"opt_sr_{t}": "capacity_opt", f"beh_sr_{t}_p5": "beh_p5", f"beh_sr_{t}_p95": "beh_p95"})
    few = s.n_behavioural < 10
    s.loc[few, ["beh_p5", "beh_p95"]] = np.nan
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/screened_catchments.csv")
    ap.add_argument("--min-valid", type=float, default=0.5)
    a = ap.parse_args()
    cfg = load_config()
    name = Path(a.list).stem
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", index_col="gauge_id")
    ids = pd.read_csv(ROOT / a.list).gauge_id
    b = benchmarks(name, cat.loc[ids], a.min_valid)
    rq1 = pd.read_csv(ROOT / "data" / "rq1" / f"identifiability_{name}.csv")
    pairs = rq1[rq1.passes][["gauge_id", "model", "capacity_opt", "n_behavioural", "beh_p5", "beh_p95"]]
    rq3 = pd.read_csv(ROOT / "data" / "rq3" / f"state_capacity_{name}_summary.csv")
    frames = {"parameter": {20: pairs}, "state": {t: state_frame(pairs, rq3, t) for t in (10, 20, 40)}}
    cf = ROOT / "data" / "rq3" / f"control_{name}.csv"
    if cf.exists():
        ctrl = pd.read_csv(cf)
        frames["control"] = {t: pairs.drop(columns="capacity_opt").merge(
            ctrl[["gauge_id", "model", f"ctrl_sr_{t}"]].rename(columns={f"ctrl_sr_{t}": "capacity_opt"}),
            on=["gauge_id", "model"], validate="1:1").assign(beh_p5=np.nan, beh_p95=np.nan) for t in (10, 20, 40)}

    rows = []
    for level, by_t in frames.items():
        for bench in PRIMARY + ["envelope"] + SENSITIVITY:
            t = STATE_T.get(bench, 20) if level in ("state", "control") else 20
            df = by_t[t].merge(b, left_on="gauge_id", right_index=True, validate="m:1")
            for model, gm in df.groupby("model"):
                for stratum, g in [("all", gm)] + sorted(gm.groupby("stratum")):
                    rows.append(dict(level=level, model=model, stratum=stratum, benchmark=bench,
                                     **summarise(g, bench)))
    s = pd.DataFrame(rows)
    f = ROOT / "data" / "rq3" / f"state_vs_benchmarks_summary_{name}.csv"
    s.round(6).to_csv(f, index=False)
    print(f"{len(pairs)} catchment-model pairs -> {f}")
    keep = (s.stratum == "all") & s.benchmark.isin(PRIMARY + ["envelope"])
    print(s[keep].pivot_table(index=["model", "benchmark"], columns="level",
                              values=["spearman", "bias_log10", "coverage", "opt_inside"]).round(2).to_string())


if __name__ == "__main__":
    main()
