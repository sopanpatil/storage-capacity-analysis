"""Forcing-only control for the RQ3 state-level mass curve.

The state mass curve (src/state_capacity.py) and the climate benchmark share the precipitation forcing
and the method, so part of their agreement could come from the shared input. The control keeps each
model's own soil input (rain + melt, optimum parameters) and replaces its actual ET by FAO-PM PET scaled
to the model's mean actual ET over 2003-2020:  E_ctrl(t) = PET(t) * mean(AET) / mean(PET).
Same long-term ET as the model, PET timing, no soil-moisture limitation. If the control agrees with
the benchmarks as well as the state does, the agreement comes from the forcing and the mean ET, not
from the model's storage dynamics.

Also recomputes the optimum's state S_R,20 as a check against data/rq3/state_capacity_<list>_summary.csv.

Writes data/rq3/control_<list name>.csv (one row per catchment x model, optimum only).

Usage: python src/rq3_control.py [--list data/catalog/screened_catchments.csv] [--tag calibration]
       [--workers 4]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from caravan_io import ROOT, load_config
from mass_curve import storage_capacity
from model_fluxes import FLUXES, WINDOW_START, forcing, window
from model_setups import SETUPS


def one(cfg, tag, item):
    gid, row = item
    summary = json.loads((cfg["paths"]["runs"] / tag / gid / "summary.json").read_text())["models"]
    f = forcing(cfg, gid, row)
    pet = f["E"][f["dates"] >= WINDOW_START]
    rows = []
    for name, setup in SETUPS.items():
        p = np.array([summary[name]["params"][k] for k in setup.names])
        dates, _, x = window(f, FLUXES[name](f, p))
        e_ctrl = pet * x["aet"].mean() / pet.mean()
        r = storage_capacity(x["soil_in"], x["aet"], dates)
        c = storage_capacity(x["soil_in"], e_ctrl, dates)
        rows.append(dict(gauge_id=gid, model=name, state_sr_20=r["sr_20"], ctrl_sr_10=c["sr_10"],
                         ctrl_sr_20=c["sr_20"], ctrl_sr_40=c["sr_40"], ctrl_lateral_inflow=c["lateral_inflow"],
                         aet_mean=x["aet"].mean(), pet_mean=pet.mean()))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/screened_catchments.csv")
    ap.add_argument("--tag", default="calibration")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config()
    name = Path(a.list).stem
    ids = pd.read_csv(ROOT / a.list, dtype={"gauge_id": str}).gauge_id
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(partial(one, cfg, a.tag), cat.loc[ids].iterrows(), chunksize=20))
    df = pd.DataFrame([r for rows in res for r in rows])
    f = ROOT / "data" / "rq3" / f"control_{name}.csv"
    df.round(3).to_csv(f, index=False)
    print(f"{len(df)} rows -> {f}")
    sf = ROOT / "data" / "rq3" / f"state_capacity_{name}_summary.csv"
    if sf.exists():
        m = df.merge(pd.read_csv(sf)[["gauge_id", "model", "opt_sr_20"]], on=["gauge_id", "model"])
        print(f"check: max |state S_R,20 - RQ3 summary| = {(m.state_sr_20 - m.opt_sr_20).abs().max():.3f} mm")


if __name__ == "__main__":
    main()
