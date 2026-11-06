"""Water-balance check of the climate-derived benchmark's satellite E, 2003-2020.

For a gauged catchment, long-term E cannot exceed P - Q. Per catchment: runoff ratio Q/P over the days
in 2003-2020 with observed streamflow (P from the same Caravan ERA5-Land forcing as the benchmark), then
the water-balance E = P_mean * (1 - Q/P) with P_mean over all of 2003-2020 (the benchmark's period), and
the ratio of each satellite E (from data/benchmarks/climate_<list>.csv) to it. Using the ratio rather
than mean Q keeps catchments with partial flow records comparable.
The check uses streamflow only to judge the benchmark's inputs, not to compute the benchmark.

Writes data/benchmarks/water_balance_<list name>.csv.

Usage: python src/benchmark_water_balance.py [--list data/catalog/screened_catchments.csv] [--workers 4]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd

from caravan_io import ROOT, load_config, load_timeseries

PERIOD = ("2003-01-01", "2020-12-31")


def one(cfg, gid):
    ts = load_timeseries(cfg, gid, [cfg["forcing"]["precip"], "streamflow"]).loc[PERIOD[0]:PERIOD[1]]
    p, q = ts[cfg["forcing"]["precip"]], ts["streamflow"]
    ok = q.notna() & p.notna()
    return dict(gauge_id=gid, q_days=int(ok.sum()), q_frac=ok.mean(), p_mean=p.mean(),
                runoff_ratio=q[ok].sum() / p[ok].sum() if ok.any() else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/screened_catchments.csv")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config()
    name = Path(a.list).stem
    ids = pd.read_csv(ROOT / a.list, dtype={"gauge_id": str}).gauge_id.tolist()
    with ProcessPoolExecutor(a.workers) as ex:
        wb = pd.DataFrame(ex.map(partial(one, cfg), ids, chunksize=50)).set_index("gauge_id")
    clim = pd.read_csv(ROOT / "data" / "benchmarks" / f"climate_{name}.csv", index_col=0)
    assert (clim.p_mean - wb.p_mean).abs().max() < 1e-3, "P differs from the climate benchmark's"
    wb["e_wb"] = wb.p_mean * (1 - wb.runoff_ratio)
    for k in ("ens", "mod16", "ssebop"):
        wb[f"{k}_e_ratio"] = clim[f"{k}_e_mean"] / wb.e_wb
    wb.insert(0, "eligible", clim.eligible)
    f = ROOT / "data" / "benchmarks" / f"water_balance_{name}.csv"
    wb.round(4).to_csv(f)
    print(f"{len(wb)} catchments -> {f}")


if __name__ == "__main__":
    main()
