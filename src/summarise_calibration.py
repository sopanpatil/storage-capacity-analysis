"""Collect per-catchment calibration summaries into one table in the project.

Reads <runs>/<tag>/<gauge_id>/summary.json (from calibrate.py) and writes
data/calibration/<tag>_summary.csv: one row per catchment and model with the split, optimum
capacity, calibration/validation KGE, behavioural count and timings, plus the optimum parameters.
Small enough to keep in the repository.

Usage: python src/summarise_calibration.py [--tag calibration]
"""
import argparse
import json

import pandas as pd

from caravan_io import ROOT, load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="calibration")
    a = ap.parse_args()
    cfg = load_config()
    rows = []
    for f in sorted((cfg["paths"]["runs"] / a.tag).glob("*/summary.json")):
        s = json.loads(f.read_text())
        base = {k: s[k] for k in ["gauge_id", "cal_start", "cal_end", "val_end", "n_cal_days",
                                  "n_val_days", "cal_aridity", "val_aridity"]}
        for model, m in s["models"].items():
            rows.append({**base, "model": model,
                         **{k: m[k] for k in ["capacity", "kge_cal", "kge_val", "n_behavioural",
                                              "seconds_optimise", "seconds_monte_carlo"]},
                         **{f"p_{k}": v for k, v in m["params"].items()}})
    out = ROOT / "data" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{a.tag}_summary.csv", index=False)
    print(f"{df.gauge_id.nunique()} catchments, {len(df)} rows -> {out / (a.tag + '_summary.csv')}")


if __name__ == "__main__":
    main()
