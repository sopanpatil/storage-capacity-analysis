"""FLEX-type lumped model for the robustness check (item J).

Structure and equations as Liang et al. (2024, egusphere-2024-550, Table 2), who take them from Fenicia et al.
(2008, 2011) and Gao et al. (2014): snow, interception, root zone (Su), fast and slow reservoirs. Gao et al.
(2014) state that the root zone uses the tension-water capacity curve of the Xinanjiang model (Zhao and Liu 1995).

Daily sequence (the paper gives the equations, not the numerical scheme, so these are our choices, the same kind
of explicit daily step as hbv-model):
- Snow: T <= Tt all precipitation is snow; T > Tt all rain, melt = min(Sw, FDD (T - Tt)).
- Interception (rain only): Si += Pr, spill Ptf = max(Si - Imax, 0), Ei = min(Ep, Si). Melt bypasses it.
- Root zone: Pe = Ptf + melt. Ru from the Xinanjiang curve (exact integral), Su += Pe - Ru, then
  Ea = (Ep - Ei) min(Su / (Sumax Ce), 1), limited to Su.
- Split: Rf = D Ru through the lag, Rs = (1 - D) Ru straight to the slow reservoir (Liang's Eqs. 10-13). The lag has
  weights c(i) = i / sum(u = 1..n), i = 1..n, with n = round(Tlag), at least 1, as hbv-model rounds MAXBAS.
- Reservoirs: S += inflow, Q = S / K, S -= Q; K >= 1 d under the priors, so this cannot go negative.
Initial states: Su = Sumax / 2, all others 0 (hbv-model's convention), followed by the common 5-year warm-up.

Priors: Liang et al. (2024) Table 3, except Sumax 1-2000 mm in `FLEX`, the range of HBV's FC and GR6J's X1 in the main
analysis, so that all three are tested from the same wide range. `FLEXLiang` keeps their Sumax 30-700 mm.

Usage: FLEX().simulate(f, p), or fluxes(f, p) for the deficit analysis. Checks: verification/run_flex_checks.py.
"""
import numpy as np
from numba import njit

from model_setups import Setup, _bounds

RANGES = {"FDD": (1.0, 7.0), "Tt": (-2.0, 4.0), "Imax": (1.0, 5.0), "Sumax": (1.0, 2000.0), "Ce": (0.1, 1.0),
          "beta": (0.0, 1.0), "D": (0.0, 1.0), "Kf": (1.0, 10.0), "Ks": (10.0, 200.0), "Tlag": (0.8, 10.0)}
NAMES = list(RANGES)
COLUMNS = ["Q", "soil_in", "Ei", "Ea", "Ru", "Sw", "Si", "Su", "Sf", "Ss", "lag"]


@njit(cache=True)
def xinanjiang_runoff(Pe, Su, Sumax, beta):
    """Runoff from input Pe on storage Su (Liang et al. 2024, Table 2; Zhao and Liu 1995)."""
    wm = (1.0 + beta) * Sumax
    ratio = max(1.0 - Su / Sumax, 0.0)
    au = wm * (1.0 - ratio ** (1.0 / (1.0 + beta)))
    if Pe + au < wm:
        ru = Pe - Sumax + Su + Sumax * (1.0 - (Pe + au) / wm) ** (1.0 + beta)
    else:
        ru = Pe - Sumax + Su
    return min(max(ru, 0.0), Pe)


@njit(cache=True)
def _run(P, T, E, FDD, Tt, Imax, Sumax, Ce, beta, D, Kf, Ks, n_lag):
    """Daily loop. Returns one row per day with COLUMNS."""
    n = len(P)
    out = np.empty((n, 11))
    weights = np.arange(1.0, n_lag + 1.0)
    weights /= weights.sum()
    buf = np.zeros(n_lag)
    Sw = Si = Sf = Ss = 0.0
    Su = 0.5 * Sumax
    for t in range(n):
        ep = max(E[t], 0.0)
        if T[t] > Tt:
            rain = P[t]
            melt = min(Sw, FDD * (T[t] - Tt))
            Sw -= melt
        else:
            rain = 0.0
            melt = 0.0
            Sw += P[t]
        Si += rain
        ptf = max(Si - Imax, 0.0)
        Si -= ptf
        ei = min(ep, Si)
        Si -= ei

        pe = ptf + melt
        ru = xinanjiang_runoff(pe, Su, Sumax, beta)
        Su += pe - ru
        if Su > Sumax:   # rounding only: the curve fills the store no further than Sumax
            ru += Su - Sumax
            Su = Sumax
        ea = min((ep - ei) * min(Su / (Sumax * Ce), 1.0), Su)
        Su -= ea

        rf = D * ru
        for i in range(n_lag):
            buf[i] += rf * weights[i]
        rfl = buf[0]
        buf[:-1] = buf[1:]
        buf[-1] = 0.0

        Sf += rfl
        qf = Sf / Kf
        Sf -= qf
        Ss += ru - rf
        qs = Ss / Ks
        Ss -= qs
        out[t, 0] = qf + qs
        out[t, 1] = rain + melt
        out[t, 2] = ei
        out[t, 3] = ea
        out[t, 4] = ru
        out[t, 5] = Sw
        out[t, 6] = Si
        out[t, 7] = Su
        out[t, 8] = Sf
        out[t, 9] = Ss
        out[t, 10] = buf.sum()
    return out


def run(P, T, E, p):
    """Run one parameter vector (order NAMES). Returns a dict of the COLUMNS."""
    v = dict(zip(NAMES, map(float, p)))
    out = _run(np.asarray(P, float), np.asarray(T, float), np.asarray(E, float), v["FDD"], v["Tt"], v["Imax"],
               v["Sumax"], v["Ce"], v["beta"], v["D"], v["Kf"], v["Ks"], max(int(round(v["Tlag"])), 1))
    return dict(zip(COLUMNS, out.T))


def fluxes(f, p):
    """Soil input, actual evaporation and store deficit for state_capacity.evaluate, as model_fluxes.py: inflow =
    liquid water below the snow routine, outflow = all actual evaporation (Ei + Ea, as GR6J's AE)."""
    o = run(f["P"], f["Tmean"], f["E"], p)
    return dict(soil_in=o["soil_in"], aet=o["Ei"] + o["Ea"], store_deficit=p[NAMES.index("Sumax")] - o["Su"],
                snow=o["Sw"])


class FLEX(Setup):
    name = "FLEX"
    names = NAMES
    prior_version = "liang2024-sumax-1-2000-2026-11-05"
    lo, hi = _bounds(RANGES, NAMES)
    capacity = ["Sumax"]

    def simulate(self, f, p):
        return run(f["P"], f["Tmean"], f["E"], p)["Q"]


class FLEXLiang(FLEX):
    """Liang et al.'s own prior for Sumax (30-700 mm)."""
    prior_version = "liang2024-2026-11-05"
    lo, hi = _bounds({**RANGES, "Sumax": (30.0, 700.0)}, NAMES)
