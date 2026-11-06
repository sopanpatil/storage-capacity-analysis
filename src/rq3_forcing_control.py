"""Forcing-only storage-free deficit: a sensitivity check on the RQ3 control (src/rq3_control.py).

The main control scales FAO-PM PET to each model's mean actual ET, which after calibration is close to P - Q, so it
carries streamflow information through the water balance. This version uses the forcing alone: inflow is ERA5-Land
precipitation P, and PET is scaled to the long-term evaporation that Budyko's (1974) parameter-free curve gives for
the catchment's mean P and PET over 2003-2020:

    E_ctrl(t) = PET(t) * E_B / mean(PET),   E_B = mean(P) * sqrt(phi * tanh(1/phi) * (1 - exp(-phi))),
    phi = mean(PET) / mean(P).

No model, no streamflow and no satellite data. One value per catchment (S_R,20 and S_R,10/40).

Writes data/rq3/forcing_control_<list name>.csv.

Usage: python src/rq3_forcing_control.py [--list data/catalog/screened_catchments.csv] [--workers 4] [--limit N]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from caravan_io import ROOT, load_config
from mass_curve import storage_capacity
from model_fluxes import WINDOW_START, forcing


def budyko(phi):
    """Budyko (1974): long-term evaporation ratio E/P for aridity index phi = PET/P."""
    return np.sqrt(phi * np.tanh(1.0 / phi) * (1.0 - np.exp(-phi)))


def one(cfg, item):
    gid, row = item
    f = forcing(cfg, gid, row)
    keep = f["dates"] >= WINDOW_START
    p, pet, dates = f["P"][keep], f["E"][keep], f["dates"][keep]
    phi = pet.mean() / p.mean()
    e_b = p.mean() * budyko(phi)
    c = storage_capacity(p, pet * e_b / pet.mean(), dates)
    return dict(gauge_id=gid, fctrl_sr_10=c["sr_10"], fctrl_sr_20=c["sr_20"], fctrl_sr_40=c["sr_40"],
                p_mean=p.mean(), pet_mean=pet.mean(), e_budyko=e_b, lateral_inflow=c["lateral_inflow"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/screened_catchments.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="first N catchments only, printed, no output file")
    a = ap.parse_args()
    cfg = load_config()
    name = Path(a.list).stem
    ids = pd.read_csv(ROOT / a.list, dtype={"gauge_id": str}).gauge_id[:a.limit]
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    with ProcessPoolExecutor(a.workers) as ex:
        df = pd.DataFrame(ex.map(partial(one, cfg), cat.loc[ids].iterrows(), chunksize=20))
    if a.limit is not None:
        print(df.round(2).to_string(index=False))
        return
    f = ROOT / "data" / "rq3" / f"forcing_control_{name}.csv"
    df.round(6).to_csv(f, index=False)
    print(f"{len(df)} rows -> {f}")


if __name__ == "__main__":
    main()
