"""Robustness item H: why the simulated deficit can exceed the calibrated capacity.

Pairs as in rq3_coupling.py: optimum, KGE >= 0.5 in both periods, positive capacity and S_R,20. The simulated deficit
S_R,20 is the Gumbel 20-year level of annual maxima of D(t) = max(0, D + AET - soil input) (mass_curve.py), so it can
exceed the capacity in two ways:
- gumbel: the observed maximum deficit (sr_max, 2003-2020) stays within the capacity and only the extrapolated
  20-year level exceeds it;
- other_stores: the observed maximum itself exceeds the capacity, so evaporation drew on water outside the capacity
  store. For HBV and GR6J this cannot happen: AET comes only from the capacity store (GR's net interception min(P, E)
  enters both soil input and AET and cancels), so D never exceeds capacity minus storage. In SAC-SMA, D is bounded
  only by the deficit of all five stores (dD <= d(total deficit) because routed inflow leaves the stores but not D),
  so the excess over UZTWM + LZTWM is water drawn from the free-water stores (UZFWM, LZFPM, LZFSM: upper-zone
  free-water evaporation, percolation into lower-zone tension water, and lower-zone free-to-tension transfer).
  For these pairs the optimum is rerun (model_fluxes.py) and, on the day of the maximum deficit, the tension-water
  and free-water deficits are recorded; the check is sr_max <= tension + free-water deficit there.

Writes robustness/H_deficit_excess.csv (per model) and robustness/H_sacsma_free_water.csv (per SAC-SMA pair
whose observed maximum exceeds the capacity).

Usage: python src/robustness_deficit_excess.py
"""
import json

import numpy as np
import pandas as pd

from caravan_io import ROOT, load_config
from mass_curve import deficit, lateral_inflow
from model_fluxes import forcing, sacsma, window
from model_setups import SETUPS

FREE = ["UZFWM", "LZFPM", "LZFSM"]


def free_water(cfg, cat, gid):
    """Rerun the SAC-SMA optimum over 2003-2020 and split the storage deficit on the day of the maximum D."""
    s = json.loads((cfg["paths"]["runs"] / "calibration" / gid / "summary.json").read_text())["models"]["SACSMA"]
    setup = SETUPS["SACSMA"]
    p = np.array([s["params"][k] for k in setup.names])
    f = forcing(cfg, gid, cat.loc[gid])
    _, _, x = window(f, sacsma(f, p, states=["uzfwc", "lzfpc", "lzfsc"]))
    dd = deficit(x["soil_in"] + lateral_inflow(x["soil_in"], x["aet"]), x["aet"])
    t = int(np.argmax(dd))
    free_cap = sum(s["params"][k] for k in FREE)
    free_def = free_cap - (x["uzfwc"][t] + x["lzfpc"][t] + x["lzfsc"][t])
    return dict(gauge_id=gid, capacity=setup.capacity_of(p)[0], free_capacity=free_cap, sr_max_rerun=dd.max(),
                tension_deficit_at_max=x["store_deficit"][t], free_deficit_at_max=free_def,
                excess=dd.max() - setup.capacity_of(p)[0],
                bound_holds=dd.max() <= x["store_deficit"][t] + free_def + 1e-3)


def main():
    cfg = load_config()
    rq1 = pd.read_csv(ROOT / "data/rq1/identifiability_screened_catchments.csv")
    t = pd.read_parquet(cfg["paths"]["runs"] / "rq3" / "state_capacity_screened_catchments.parquet",
                        filters=[("set", "==", "opt")])
    d = rq1[rq1.passes][["gauge_id", "model"]].merge(t, on=["gauge_id", "model"], validate="1:1")
    d = d[(d.capacity > 0) & (d.sr_20 > 0)]
    rows = []
    for model, g in d.groupby("model"):
        ex = g.sr_20 > g.capacity
        other = ex & (g.sr_max > g.capacity * (1 + 1e-3))   # the table rounds to 3 decimals
        gum = ex & ~other
        rows.append(dict(model=model, n=len(g), exceed=int(ex.sum()), exceed_share=ex.mean(),
                         gumbel=int(gum.sum()), other_stores=int(other.sum()),
                         max_srmax_over_capacity=(g.sr_max / g.capacity).max(),
                         gumbel_median_sr20_over_capacity=(g.sr_20 / g.capacity)[gum].median(),
                         gumbel_median_sr20_over_srmax=(g.sr_20 / g.sr_max)[gum].median(),
                         all_median_sr20_over_srmax=(g.sr_20 / g.sr_max).median()))
    s = pd.DataFrame(rows)
    out = ROOT / "robustness"
    s.round(6).to_csv(out / "H_deficit_excess.csv", index=False)
    print(s.round(3).to_string(index=False))

    sac = d[(d.model == "SACSMA") & (d.sr_max > d.capacity * (1 + 1e-3))]
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    fw = pd.DataFrame([free_water(cfg, cat, gid) for gid in sac.gauge_id])
    fw = fw.merge(sac[["gauge_id", "sr_max", "sr_20"]], on="gauge_id")
    fw["excess_share_of_free_deficit"] = fw.excess / fw.free_deficit_at_max
    fw.round(4).to_csv(out / "H_sacsma_free_water.csv", index=False)
    print(f"SAC-SMA, observed maximum > capacity: {len(fw)} pairs; rerun reproduces sr_max within "
          f"{(fw.sr_max_rerun - fw.sr_max).abs().max():.3f} mm; bound sr_max <= tension + free deficit holds in "
          f"{fw.bound_holds.sum()}/{len(fw)}")
    print(fw[["capacity", "free_capacity", "sr_max", "excess", "free_deficit_at_max",
              "excess_share_of_free_deficit"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
