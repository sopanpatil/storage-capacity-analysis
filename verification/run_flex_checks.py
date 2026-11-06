"""Checks of src/flex_model.py (robustness item J). No reference code implements this exact structure, so instead of a
machine-precision comparison with a package (as for GR6J and SAC-SMA) this tests the model against independent
calculations:
  1. Xinanjiang runoff against numerical integration of the capacity distribution it is derived from.
  2. Water balance over a real 40-year forcing, for random parameter sets across the prior.
  3. A separate plain-Python implementation written from Liang et al. (2024) Table 2, day by day.
  4. Limits with known answers: beta = 0 bucket, a full store, an empty store, a linear-reservoir recession, lag weights.

Usage: python verification/run_flex_checks.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from caravan_io import load_config, load_forcing  # noqa: E402
from flex_model import COLUMNS, FLEX, NAMES, run, xinanjiang_runoff  # noqa: E402

RNG = np.random.default_rng(20261105)


def check_xinanjiang(n=2000):
    """The point capacity w has distribution F(w) = 1 - (1 - w / wm)^beta on 0..wm, with wm = (1 + beta) Sumax. The tension
    level `a` holds storage S(a) = integral of (1 - F) from 0 to a. Input Pe raises a to a + Pe. Runoff is the input
    on the saturated fraction: the integral of F from a to a + Pe (F = 1 above wm)."""
    worst = 0.0
    for _ in range(n):
        Sumax, beta = RNG.uniform(1, 2000), RNG.uniform(0.02, 1)
        Su, Pe = RNG.uniform(0, 1) * Sumax, RNG.uniform(0, 1.5) * Sumax
        wm = (1 + beta) * Sumax
        grid = np.linspace(0, wm, 400001)
        survive = (1 - grid / wm) ** beta
        stored = np.concatenate([[0.0], np.cumsum((survive[1:] + survive[:-1]) / 2 * np.diff(grid))])
        a = np.interp(Su, stored, grid)     # tension level from storage, by numerical inversion only
        top = a + Pe
        inside = np.linspace(a, min(top, wm), 200001)
        F = 1 - (1 - inside / wm) ** beta
        numeric = np.sum((F[1:] + F[:-1]) / 2 * np.diff(inside)) + max(top - wm, 0.0)
        worst = max(worst, abs(numeric - xinanjiang_runoff(Pe, Su, Sumax, beta)) / max(Sumax, 1.0))
    print(f"1. Xinanjiang closed form vs numerical integral, {n} cases: worst error {worst:.1e} of Sumax")
    assert worst < 1e-5


def forcing():
    cfg = load_config()
    f = cfg["forcing"]
    ts = load_forcing(cfg, "camelsgb_39088", [f["precip"], f["tmean"], f["pet"]]).loc["1981":"2020"]
    return ts[f["precip"]].to_numpy(), ts[f["tmean"]].to_numpy(), ts[f["pet"]].to_numpy()


def check_balance(P, T, E, n=500):
    setup, worst, worst_su = FLEX(), 0.0, 0.0
    for u in RNG.random((n, len(NAMES))):
        p = setup.from_unit(u[None])[0]
        o = run(P, T, E, p)
        store = o["Sw"] + o["Si"] + o["Su"] + o["Sf"] + o["Ss"] + o["lag"]
        initial = 0.5 * p[NAMES.index("Sumax")]
        residual = P.sum() - (o["Q"] + o["Ei"] + o["Ea"]).sum() - (store[-1] - initial)
        worst = max(worst, abs(residual) / P.sum())
        worst_su = max(worst_su, (o["Su"] - p[NAMES.index("Sumax")]).max(), -o["Su"].min())
        assert all((o[k] >= -1e-9).all() for k in COLUMNS), p
    print(f"2. Water balance, {n} sets, {len(P)} days: worst residual {worst:.1e} of total P; "
          f"Su outside 0..Sumax by at most {worst_su:.1e} mm")
    assert worst < 1e-10 and worst_su < 1e-9


def plain(P, T, E, FDD, Tt, Imax, Sumax, Ce, beta, D, Kf, Ks, Tlag):
    """Written from the equations, not from flex_model.py: lists and Python floats, runoff expression inline."""
    n = max(round(Tlag), 1)
    c = [i / sum(range(1, n + 1)) for i in range(1, n + 1)]
    fast_in, sw, si, su, sf, ss, qs = [0.0] * (len(P) + n), 0.0, 0.0, Sumax / 2, 0.0, 0.0, []
    for t, (pr, tm, ep) in enumerate(zip(P, T, E)):
        ep = max(ep, 0.0)
        snowing = tm <= Tt
        melt = 0.0 if snowing else min(sw, FDD * (tm - Tt))
        sw += (pr if snowing else 0.0) - melt
        si += 0.0 if snowing else pr
        spill = max(si - Imax, 0.0)
        si -= spill
        ei = min(ep, si)
        si -= ei
        pe = spill + melt
        wm = (1 + beta) * Sumax
        au = wm * (1 - max(1 - su / Sumax, 0.0) ** (1 / (1 + beta)))
        ru = pe - Sumax + su + (Sumax * (1 - (pe + au) / wm) ** (1 + beta) if pe + au < wm else 0.0)
        ru = min(max(ru, 0.0), pe)
        su += pe - ru
        if su > Sumax:
            ru, su = ru + su - Sumax, Sumax
        ea = min((ep - ei) * min(su / (Sumax * Ce), 1.0), su)
        su -= ea
        for i in range(n):
            fast_in[t + i] += D * ru * c[i]
        sf += fast_in[t]
        ss += (1 - D) * ru
        qf, q_slow = sf / Kf, ss / Ks
        sf, ss = sf - qf, ss - q_slow
        qs.append(qf + q_slow)
    return np.array(qs)


def check_plain(P, T, E, n=20):
    setup, worst = FLEX(), 0.0
    for u in RNG.random((n, len(NAMES))):
        p = setup.from_unit(u[None])[0]
        worst = max(worst, np.abs(plain(P, T, E, *p) - run(P, T, E, p)["Q"]).max())
    print(f"3. Plain-Python implementation vs flex_model, {n} sets: worst daily difference in Q {worst:.1e} mm")
    assert worst < 1e-9


def check_limits():
    assert abs(xinanjiang_runoff(30.0, 80.0, 100.0, 0.0) - 10.0) < 1e-12   # beta = 0: a bucket spills only the excess
    assert xinanjiang_runoff(10.0, 50.0, 100.0, 0.0) == 0.0
    assert abs(xinanjiang_runoff(25.0, 100.0, 100.0, 0.5) - 25.0) < 1e-9     # full store: all input runs off
    assert xinanjiang_runoff(25.0, 0.0, 100.0, 0.5) < 25.0 * 0.5             # empty store: most input is stored
    n = 400
    zero, warm = np.zeros(n), np.full(n, 20.0)
    P = zero.copy()
    P[0] = 5000.0
    #          FDD  Tt  Imax Sumax Ce  beta D    Kf   Ks     Tlag
    p_slow = [3.0, 0.0, 1.0, 100.0, 0.5, 0.0, 0.0, 1.0, 50.0, 1.0]
    q = run(P, warm, zero, p_slow)["Q"]
    assert np.allclose(q[1:] / q[:-1], 1 - 1 / 50.0, atol=1e-12)           # linear recession with K = 50 d, no evaporation
    p_fast = [3.0, 0.0, 1.0, 100.0, 0.5, 0.0, 1.0, 1.0, 50.0, 4.0]
    q = run(P, warm, zero, p_fast)["Q"]
    assert np.allclose(q[:5] / q[:4].sum(), [0.1, 0.2, 0.3, 0.4, 0.0], atol=1e-12)   # Kf = 1 d: Q shows the lag weights
    snow = run(np.r_[100.0, np.zeros(19)], np.r_[-5.0, np.full(19, 5.0)], np.zeros(20), p_fast)
    assert abs(snow["Sw"][0] - 100.0) < 1e-12 and abs(snow["soil_in"][1:6].sum() - 75.0) < 1e-9   # 15 mm/d melt
    print("4. Limits: bucket at beta = 0, full and empty store, recession, lag weights, degree-day melt: all as expected")


if __name__ == "__main__":
    check_xinanjiang()
    P, T, E = forcing()
    check_balance(P, T, E)
    check_plain(P, T, E)
    check_limits()
    print("All FLEX checks passed")
