"""Daily soil-routine input and actual ET from each model, for the RQ3 state-level mass curve.

RQ3: the mass curve of the climate-derived benchmark is applied to each model's own fluxes, with
inflow = liquid water reaching the model's soil routine (rain + snowmelt) and outflow = the model's
actual ET, i.e. the model analogue of precipitation and satellite E in the climate-derived benchmark.
(Using only the part that infiltrates the capacity store would make the deficit bounded by the
parameter; that bounded version is reported separately as the store deficit, a diagnostic.)

- HBV (hbv-model): soil input from the water balance below the snow routine,
  in_soil = d(SM + UZ + LZ) + ETact + Qgen (exact: nothing else enters or leaves those stores).
  Store deficit = FC - SM.
- GR6J-CemaNeige (grsuite): soil input = CemaNeige PliqAndMelt (one layer); AET = AE (interception +
  production-store evaporation). grsuite's `Precip` output is raw precipitation, so it is not used.
  Store deficit = X1 - production store.
- SAC-SMA + SNOW-17: soil input = raim (rain + melt), AET = aet. Store deficit = tension-water deficit,
  (UZTWM - uztwc) + (LZTWM - lztwc).

Runs use the calibrated parameters unchanged (model_setups.py), from 1998-01-01 (5 years of warm-up,
as in calibration) to 2020-12-31; fluxes are returned for 2003-2020.

Usage: python src/model_fluxes.py --selftest
"""
import argparse

import grsuite as gs
import numpy as np
import pandas as pd

from caravan_io import load_config, load_forcing
from model_setups import SETUPS, HBVModel, sacsnow

RUN_START, WINDOW_START, WINDOW_END = "1998-01-01", "2003-01-01", "2020-12-31"


def forcing(cfg, gid, row, source=None, end=WINDOW_END):
    """Forcing dict as in calibrate.catchment_inputs, over RUN_START..end (source: alternative forcing,
    config `forcing_sources`; end: earlier than WINDOW_END for sources that stop sooner)."""
    f = cfg["forcing"]
    cols = [f["precip"], f["pet"], f["tmin"], f["tmean"], f["tmax"]]
    run = load_forcing(cfg, gid, cols, source).loc[RUN_START:end]
    if run.isna().any().any():
        raise ValueError(f"{gid}: missing forcing")
    idx = run.index
    return dict(P=run[f["precip"]].to_numpy(), E=run[f["pet"]].to_numpy(), Tmin=run[f["tmin"]].to_numpy(),
                Tmean=run[f["tmean"]].to_numpy(), Tmax=run[f["tmax"]].to_numpy(), dates=idx,
                n_warm=int((idx < WINDOW_START).sum()), lat=float(row.gauge_lat),
                elev=float(row.ele_mt_sav), forest_frac=float(row.for_pc_sse) / 100.0)


def hbv(f, p):
    v = dict(zip(SETUPS["HBV"].names, p))
    _, s = HBVModel(v).run(f["P"], f["Tmean"], f["E"])
    store = s["SM"] + s["UZ"] + s["LZ"]
    in_soil = np.diff(store, prepend=np.nan) + s["ETact"] + s["Qgen"]
    # SP is the frozen part only; liquid water held in the pack (WC <= CWH*SP) needs SP > 0
    return dict(soil_in=in_soil, aet=s["ETact"], store_deficit=v["FC"] - s["SM"], snow=s["SP"])


def gr6j(f, p):
    setup = SETUPS["GR6J"]
    setup.prepare(f)
    out = gs.run_model_cemaneige_gr6j(setup.im, setup.ro, np.asarray(p, float))
    pad = np.full(f["n_warm"], np.nan)
    soil_in = np.concatenate([pad, np.asarray(out["CemaNeigeLayers"][0]["PliqAndMelt"])])
    aet = np.concatenate([pad, out["AE"]])
    deficit = np.concatenate([pad, p[setup.names.index("X1")] - out["Prod"]])
    snow = np.concatenate([pad, np.asarray(out["CemaNeigeLayers"][0]["SnowPack"])])
    return dict(soil_in=soil_in, aet=aet, store_deficit=deficit, snow=snow)


def sacsma(f, p, states=()):
    """`states`: further sacsnow outputs to return as well (e.g. the free-water contents)."""
    setup = SETUPS["SACSMA"]
    v = dict(zip(setup.names, p))
    sac = {k.lower(): v[k] for k in ["UZTWM", "UZFWM", "LZTWM", "LZFPM", "LZFSM", "UZK", "LZPK",
                                     "LZSK", "ZPERC", "REXP", "PFREE"]}
    sac.update(setup.fixed_sac, efc=f["forest_frac"])
    snow = {k.lower(): v[k] for k in ["SCF", "MFMAX", "MFMIN", "UADJ", "SI"]}
    snow.update(setup.fixed_snow)
    ptps = (f["Tmean"] <= v["PXTEMP"]).astype(float)
    d = f["dates"]
    o = sacsnow.run(24, d.year, d.month, d.day, np.full(len(d), 12), [f["lat"]], [f["elev"]],
                    np.array([[sac[k]] for k in sacsnow.SAC_NAMES]),
                    np.array([[snow[k]] for k in sacsnow.SNOW_NAMES]),
                    [1.0], [1.0], [0.0], f["P"], ptps, f["Tmean"], f["E"], return_states=True)
    deficit = (v["UZTWM"] - o["uztwc"][:, 0]) + (v["LZTWM"] - o["lztwc"][:, 0])
    return dict(soil_in=o["raim"][:, 0], aet=o["aet"][:, 0], store_deficit=deficit, snow=o["swe"][:, 0],
                **{k: o[k][:, 0] for k in states})


FLUXES = {"HBV": hbv, "GR6J": gr6j, "SACSMA": sacsma}


def window(f, fluxes):
    """Trim forcing P and the flux dict to WINDOW_START..WINDOW_END; returns (dates, P, fluxes)."""
    keep = f["dates"] >= WINDOW_START
    return f["dates"][keep], f["P"][keep], {k: np.asarray(v)[keep] for k, v in fluxes.items()}


def selftest():
    cfg = load_config()
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    import json
    for gid in ["camelsgb_39088", "lamah_212761"]:
        f = forcing(cfg, gid, cat.loc[gid])
        s = json.loads((cfg["paths"]["runs"] / "calibration" / gid / "summary.json").read_text())
        for name, fn in FLUXES.items():
            p = np.array(list(s["models"][name]["params"].values()))
            dates, P, x = window(f, fn(f, p))
            soil_in, aet, deficit = x["soil_in"], x["aet"], x["store_deficit"]
            assert np.isfinite(soil_in).all() and np.isfinite(aet).all(), (gid, name)
            assert (aet >= -1e-9).all() and (soil_in >= -1e-6).all(), (gid, name, soil_in.min(), aet.min())
            # with (almost) no snow in the pack before and after the step, on a day too warm for any
            # snowfall in any of the models' rain/snow splits, soil input equals precipitation
            # (CemaNeige keeps residuals of ~1e-14..1e-3 mm, hence the 1e-3 threshold and tolerance)
            t = window(f, {"tmin": f["Tmin"], "tmean": f["Tmean"]})[2]
            nosnow = (x["snow"] < 1e-3) & (np.roll(x["snow"], 1) < 1e-3) & (t["tmin"] > 0) & (t["tmean"] > 3)
            nosnow[0] = False
            err = np.abs(soil_in[nosnow] - P[nosnow]).max()
            cap = SETUPS[name].capacity_of(p)[0]
            print(f"{gid} {name}: soil input / P = {soil_in.sum() / P.sum():.4f}; max |in - P| on "
                  f"{nosnow.sum()} snow-free days = {err:.2e}; mean AET {aet.mean():.2f} mm/d; "
                  f"store deficit {deficit.min():.1f}..{deficit.max():.1f} (capacity {cap:.0f})")
            assert err < 2e-3, err
            assert (deficit >= -1e-6).all() and (deficit <= cap + 1e-6).all()
    print("model_fluxes selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
