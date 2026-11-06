"""KGE threshold sensitivity (0.3 and 0.7, for the Supporting Information).

Two thresholds, moved one at a time and then together:
- screen: the optimum must reach KGE >= screen in calibration and validation (main 0.5);
- behavioural: Monte Carlo sets with KGE_cal >= behavioural (main 0.5); RQ1 tables for 0.3 / 0.7 come from
  `identifiability.py --behavioural-kge` (data/rq1/identifiability_<list>_beh<T>.csv). Combinations whose
  table does not exist yet are skipped.
Per combination and model:
- classification shares (src/classify.py rules: envelope, 10-fold, half prior);
- RQ2/RQ3 agreement, all catchments: parameter, state and control vs each primary benchmark and the envelope
  (src/rq2_parameter.py statistics). The behavioural state percentiles exist only at 0.5
  (src/state_capacity.py), so at other behavioural thresholds the state coverage is left empty; the optimum's
  state and the control do not depend on the behavioural threshold.

Writes data/sensitivity/kge_classification_<list>.csv and kge_agreement_<list>.csv.

Usage: python src/kge_sensitivity.py [--list data/catalog/screened_catchments.csv] [--min-valid 0.5]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from caravan_io import ROOT, load_config
from classify import CLASSES, classify
from rq2_parameter import PRIMARY, benchmarks, summarise
from rq3_benchmarks import state_frame

MAIN = 0.5
COMBOS = [(0.5, 0.5), (0.3, 0.5), (0.7, 0.5), (0.5, 0.3), (0.5, 0.7), (0.3, 0.3), (0.7, 0.7)]
RULES = {"envelope": ("spread_abs", None), "fixed10": ("spread_abs", 1.0), "halfprior": ("spread_rel", 0.5)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/screened_catchments.csv")
    ap.add_argument("--min-valid", type=float, default=0.5)
    a = ap.parse_args()
    cfg = load_config()
    assert cfg["calibration"]["behavioural_kge"] == MAIN
    name = Path(a.list).stem
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", index_col="gauge_id")
    ids = pd.read_csv(ROOT / a.list).gauge_id
    b = benchmarks(name, cat.loc[ids], a.min_valid)
    rq3 = pd.read_csv(ROOT / "data" / "rq3" / f"state_capacity_{name}_summary.csv")
    ctrl = pd.read_csv(ROOT / "data" / "rq3" / f"control_{name}.csv")

    cls_rows, agr_rows = [], []
    for screen, beh in COMBOS:
        f = ROOT / "data" / "rq1" / (f"identifiability_{name}.csv" if beh == MAIN
                                     else f"identifiability_{name}_beh{beh:g}.csv")
        if not f.exists():
            print(f"screen {screen}, behavioural {beh}: {f.name} missing, skipped")
            continue
        r = pd.read_csv(f)
        r["passes"] = (r.kge_cal >= screen) & (r.kge_val >= screen)
        r = r.merge(b, left_on="gauge_id", right_index=True, validate="m:1")
        for rule, (spread, width) in RULES.items():
            c = classify(r, r[spread], r.env_lo, r.env_hi, r.n_benchmarks, width=width)
            for model, g in pd.Series(c, index=r.index).groupby(r.model):
                counts = g.value_counts().reindex(CLASSES, fill_value=0)
                k = counts[["unidentifiable", "consistent", "inconsistent"]].sum()
                cls_rows.append(dict(screen=screen, behavioural=beh, rule=rule, model=model, n=len(g),
                                     **counts.to_dict(), unidentifiable_share=counts.unidentifiable / k if k else np.nan))
        pairs = r[r.passes][["gauge_id", "model", "capacity_opt", "n_behavioural", "beh_p5", "beh_p95"]]
        state = state_frame(pairs, rq3, 20)
        if beh != MAIN:
            state[["beh_p5", "beh_p95"]] = np.nan
        levels = {"parameter": pairs, "state": state,
                  "control": pairs.drop(columns="capacity_opt").merge(
                      ctrl[["gauge_id", "model", "ctrl_sr_20"]].rename(columns={"ctrl_sr_20": "capacity_opt"}),
                      on=["gauge_id", "model"], validate="1:1").assign(beh_p5=np.nan, beh_p95=np.nan)}
        for level, df in levels.items():
            df = df.merge(b, left_on="gauge_id", right_index=True, validate="m:1")
            for model, g in df.groupby("model"):
                for bench in PRIMARY + ["envelope"]:
                    agr_rows.append(dict(screen=screen, behavioural=beh, level=level, model=model, benchmark=bench,
                                         **summarise(g, bench)))
        print(f"screen {screen}, behavioural {beh}: done")
    out = ROOT / "data" / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    c, g = pd.DataFrame(cls_rows), pd.DataFrame(agr_rows)
    c.round(6).to_csv(out / f"kge_classification_{name}.csv", index=False)
    g.round(6).to_csv(out / f"kge_agreement_{name}.csv", index=False)
    pd.set_option("display.width", 200)
    print(c.pivot_table(index=["rule", "model"], columns=["screen", "behavioural"], values="unidentifiable_share")
          .round(2).to_string())
    print(g[g.benchmark.isin(["climate", "stocker"])].pivot_table(
        index=["benchmark", "model", "level"], columns=["screen", "behavioural"], values="spearman").round(2).to_string())
    print(g[g.benchmark == "climate"].pivot_table(index=["model", "level"], columns=["screen", "behavioural"],
                                                  values="n").astype(int).to_string())


if __name__ == "__main__":
    main()
