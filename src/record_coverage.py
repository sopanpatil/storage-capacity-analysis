"""Fraction of valid streamflow days per calendar year, for every catchment (1950-2023).

Used to choose the analysis window and minimum record length.
Writes catalog/annual_valid_fraction.csv: one row per gauge_id, one column per year.

Usage: python src/record_coverage.py [--workers N]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from caravan_io import load_attributes, load_config, load_timeseries


def annual_fraction(args):
    cfg, gid = args
    q = load_timeseries(cfg, gid, ["streamflow"])["streamflow"]
    frac = q.notna().groupby(q.index.year).mean()
    return gid, frac


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config()
    gids = load_attributes(cfg).gauge_id
    with ProcessPoolExecutor(args.workers) as ex:
        rows = dict(ex.map(annual_fraction, [(cfg, g) for g in gids], chunksize=50))
    out = pd.DataFrame(rows).T.sort_index(axis=1)
    out.index.name = "gauge_id"
    out.round(4).to_csv(cfg["paths"]["catalog"] / "annual_valid_fraction.csv")
    print(out.shape)


if __name__ == "__main__":
    main()
