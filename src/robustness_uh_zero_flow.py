"""Robustness item I: why much of the SAC-SMA routing range gives no routed flow.

DUAMEL (third_party/nwsrfsr-1.0.3/src/duamel.f; the same loop as NCAR's NWS_hydro_models Snow17SacUH/src_bin/sac/
duamel.f, which also passes the step in days) sets ordinate i to the gamma density at t = i days and stops at the
first i > 1 where the log density is <= -8, a rule meant to cut the receding tail. When the density is below e^-8
at both day 1 and day 2, i.e. the rising limb starts below the threshold (large scale for the shape, or a scale
so small that day 1 is already deep in the tail), the loop stops at day 2 with every ordinate zero; the sum is then
replaced by 1e-5 and the hydrograph stays zero, so no flow is routed. Part 1 checks this rule against the compiled
routine on random draws from the prior (shape 1-5, scale 0.001-150 d) and gives the zero-flow share and, per shape,
the working scale range.

Part 2 tests whether discarding those sets loses anything: in three catchments of the robustness subset (wet, snowy,
arid), Monte Carlo sets from the main run are rerun with a unit hydrograph integrated over each day (ordinate i =
F(i) - F(i - 1) of the gamma CDF, renormalised; one possible fix), for a random sample of zero-flow sets and of
flowing sets, and the behavioural counts (KGE_cal >= 0.5) are compared with the stored DUAMEL ones. The main pipeline
is not changed (the routing function is swapped only inside this script).

Writes robustness/I_uh_zero_flow.csv (part 1) and I_uh_fix_test.csv (part 2).

Usage: python src/robustness_uh_zero_flow.py [--n-sets 10000]
"""
import argparse
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy.stats import gamma

from calibrate import catchment_inputs, kge
from caravan_io import ROOT, load_config
from model_setups import SAC_RANGES, SETUPS, sacsnow

RNG = np.random.default_rng(20261030)
CATCHMENTS = ["hysets_14241490", "GRDC_6854570", "camels_11124500"]   # wet, snowy, arid; median SAC-SMA counts
OUT = ROOT / "robustness"


def zero_flow(shape, scale):
    """DUAMEL routes nothing when the gamma density is <= e^-8 at both day 1 and day 2."""
    return (gamma.logpdf(1, shape, scale=scale) <= -8) & (gamma.logpdf(2, shape, scale=scale) <= -8)


def integrated_uh(tci, shape, scale, dt_hours=24, start_of_timestep=False, max_len=1000):
    """Drop-in for sacsnow.uh (one zone, no start-of-step shift): ordinates integrated over each step."""
    assert not start_of_timestep and dt_hours == 24
    edges = np.arange(max_len + 1, dtype=float)
    u = np.diff(gamma.cdf(edges, shape, scale=scale))
    u /= u.sum()
    tci = np.asarray(tci, float).reshape(len(tci), -1)[:, 0]
    return np.convolve(tci, u)[:len(tci)][:, None]


@contextmanager
def routing(fn):
    old = sacsnow.uh
    sacsnow.uh = fn
    try:
        yield
    finally:
        sacsnow.uh = old


def part1(n=20000):
    shape = RNG.uniform(*SAC_RANGES["USHAPE"], n)
    scale = RNG.uniform(*SAC_RANGES["USCALE"], n)
    imp = np.zeros(1100)
    imp[0] = 1.0
    routed = np.array([sacsnow.uh(imp, a, b, 24, start_of_timestep=False)[:, 0].sum() for a, b in zip(shape, scale)])
    pred = zero_flow(shape, scale)
    rows = [dict(item="prior_zero_share_duamel", value=(routed < 1e-6).mean()),
            dict(item="prior_zero_share_rule", value=pred.mean()),
            dict(item="rule_agrees_with_duamel", value=((routed < 1e-6) == pred).mean()),
            dict(item="min_routed_volume_flowing", value=routed[~pred].min())]
    s = np.linspace(SAC_RANGES["USCALE"][0], SAC_RANGES["USCALE"][1], 150001)
    for k in (1, 2, 3, 4, 5):
        ok = ~zero_flow(k, s)
        rows += [dict(item=f"shape_{k}_working_scale_min_d", value=s[ok].min()),
                 dict(item=f"shape_{k}_working_scale_max_d", value=s[ok].max())]
    return pd.DataFrame(rows)


def part2(cfg, n_sets):
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    thr = cfg["calibration"]["behavioural_kge"]
    setup = SETUPS["SACSMA"]
    rows = []
    for gid in CATCHMENTS:
        f, obs, cal, val, _ = catchment_inputs(cfg, gid, cat.loc[gid])
        mc = pd.read_parquet(cfg["paths"]["runs"] / "calibration" / gid / "SACSMA_mc.parquet")
        z = zero_flow(mc.USHAPE.astype(float), mc.USCALE.astype(float))
        assert (mc.kge_cal[z].isna()).all() and mc.kge_cal[~z].notna().all(), gid
        for group, sets in (("zero_flow", mc[z]), ("flowing", mc[~z])):
            sets = sets.sample(min(n_sets, len(sets)), random_state=1)
            with routing(integrated_uh):
                k = np.array([kge(setup.simulate(f, p)[cal], obs[cal]) for p in sets[setup.names].to_numpy(float)])
            rows.append(dict(gauge_id=gid, group=group, n_in_mc=len(mc[z] if group == "zero_flow" else mc[~z]),
                             n_rerun=len(sets), behavioural_duamel=int((sets.kge_cal >= thr).sum()),
                             behavioural_integrated=int((k >= thr).sum()), best_kge_integrated=np.nanmax(k),
                             median_kge_integrated=np.nanmedian(k),
                             median_kge_change=np.nanmedian(k - sets.kge_cal) if group == "flowing" else np.nan))
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sets", type=int, default=10000, help="sets rerun per catchment and group")
    a = ap.parse_args()
    cfg = load_config()
    OUT.mkdir(exist_ok=True)
    p1 = part1()
    p1.round(6).to_csv(OUT / "I_uh_zero_flow.csv", index=False)
    print(p1.round(4).to_string(index=False))
    p2 = part2(cfg, a.n_sets)
    p2.round(6).to_csv(OUT / "I_uh_fix_test.csv", index=False)
    print(p2.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
