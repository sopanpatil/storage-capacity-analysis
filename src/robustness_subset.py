"""Select the fixed subset for the robustness runs (runs A-D).

Draws from the catchments assessed for all three models under the main criterion (class consistent,
inconsistent or unidentifiable for GR6J, HBV and SAC-SMA), stratified by the climate classes in
data/classification. Classes where the climate-derived benchmark applies (snow fraction <= 0.2,
aridity <= 2) get `--per-eligible` catchments each, the others `--per-other`; a class with fewer
candidates gives all of them. Fixed seed.

Writes robustness/subset.csv (gauge_id, stratum, climate_eligible) and robustness/subset_by_class.csv.

Usage: python src/robustness_subset.py [--per-eligible 45 --per-other 25 --seed 20261029]
"""
import argparse

import pandas as pd

from caravan_io import ROOT

ASSESSED = ["consistent", "inconsistent", "unidentifiable"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-eligible", type=int, default=45)
    ap.add_argument("--per-other", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20261029)
    a = ap.parse_args()

    c = pd.read_csv(ROOT / "data/classification/classification_screened_catchments.csv")
    n = c[c["class"].isin(ASSESSED)].groupby("gauge_id").model.nunique()
    pool = c[c.gauge_id.isin(n[n == 3].index)].drop_duplicates("gauge_id")[["gauge_id", "stratum"]]
    elig = pd.read_csv(ROOT / "data/benchmarks/climate_screened_catchments.csv").set_index("gauge_id").eligible
    pool["climate_eligible"] = pool.gauge_id.map(elig).fillna(False).astype(bool)
    # eligibility is set by the class limits, so it is constant within a class
    assert (pool.groupby("stratum").climate_eligible.nunique() == 1).all()

    picks = []
    for stratum, g in pool.sort_values("gauge_id").groupby("stratum", sort=True):
        k = a.per_eligible if g.climate_eligible.iloc[0] else a.per_other
        picks.append(g.sample(n=min(k, len(g)), random_state=a.seed))
    sub = pd.concat(picks).sort_values(["stratum", "gauge_id"])

    out = ROOT / "robustness"
    out.mkdir(exist_ok=True)
    sub.to_csv(out / "subset.csv", index=False)
    by = pool.groupby("stratum").agg(candidates=("gauge_id", "size"), climate_eligible=("climate_eligible", "first"))
    by["selected"] = sub.groupby("stratum").size()
    by.to_csv(out / "subset_by_class.csv")
    print(by.to_string())
    print(f"{len(sub)} catchments, {sub.climate_eligible.sum()} with the climate-derived benchmark, "
          f"{sub.gauge_id.str.split('_').str[0].nunique()} sources")


if __name__ == "__main__":
    main()
