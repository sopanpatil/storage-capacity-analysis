"""The three models in their standard setups.

Each setup gives parameter names, a prior (mapping from the unit hypercube to parameter values,
used for Monte Carlo and as SCE-UA bounds), a `simulate` function returning daily streamflow over
the run period, and its capacity parameter(s). The forcing dict holds numpy arrays over the run
period (warm-up included): P, E (FAO-PM PET), Tmin, Tmean, Tmax, plus `dates`, `n_warm`
(warm-up length in days), and catchment constants (lat, elev, forest_frac).
"""
import sys

import grsuite as gs
import numpy as np

from caravan_io import ROOT, load_config

_cfg = load_config()
sys.path.insert(0, str(_cfg["paths"]["hbv_model"]))
sys.path.insert(0, str(ROOT / "models" / "sacsma"))
from hbv_model.hbv import HBVModel  # noqa: E402
import sacsnow  # noqa: E402


HBV_RANGES = {"TT": (-2.5, 2.5), "CFMAX": (0.5, 10), "CFR": (0, 0.1), "CWH": (0, 0.2),
              "FC": (1, 2000), "LP": (0.2, 1), "BETA": (1, 6), "K0": (0.05, 0.9), "K1": (0.01, 0.5),
              "K2": (0.001, 0.2), "UZL": (0, 100), "PERC": (0, 10), "MAXBAS": (1, 7)}
SAC_RANGES = {"MFMAX": (0.8, 3.0), "MFMIN": (0.01, 0.79), "UADJ": (0.01, 0.40), "SI": (1.0, 3500.0),
              "SCF": (0.1, 5.0), "PXTEMP": (-1.0, 3.0),
              "UZTWM": (1.0, 800.0), "UZFWM": (1.0, 800.0), "LZTWM": (1.0, 800.0),
              "LZFPM": (1.0, 1000.0), "LZFSM": (1.0, 1000.0), "UZK": (0.1, 0.7),
              "LZPK": (0.00001, 0.025), "LZSK": (0.001, 0.25), "ZPERC": (1.0, 250.0),
              "REXP": (0.0, 6.0), "PFREE": (0.0, 1.0), "USHAPE": (1.0, 5.0), "USCALE": (0.001, 150.0)}


def _bounds(ranges, names):
    return (np.array([ranges[n][0] for n in names], float),
            np.array([ranges[n][1] for n in names], float))


class Setup:
    name = ""
    names = []
    lo = np.array([])
    hi = np.array([])
    capacity = []      # parameter(s) compared with the benchmarks (summed if several)

    def from_unit(self, u):
        """Unit hypercube (n, k) -> parameter values (n, k): uniform within the prior."""
        return self.lo + np.asarray(u) * (self.hi - self.lo)

    def bounds(self):
        return list(zip(self.lo, self.hi))

    def capacity_of(self, params):
        params = np.atleast_2d(params)
        return sum(params[:, self.names.index(n)] for n in self.capacity)


class HBV(Setup):
    """hbv-model v1.2.0. Ranges: Beck et al. (2020) Table 2; FC 1-2000 mm (MARRMoT standard
    range, Knoben et al. 2019); MAXBAS 1-7 d (Seibert 1999)."""
    name = "HBV"
    prior_version = "2026-10-08"
    names = HBVModel.PARAM_NAMES
    lo, hi = _bounds(HBV_RANGES, names)
    capacity = ["FC"]

    def simulate(self, f, p):
        q, _ = HBVModel(dict(zip(self.names, p))).run(f["P"], f["Tmean"], f["E"])
        return q


class GR6J(Setup):
    """GR6J-CemaNeige via grsuite (= airGR 1.7.9), one snow layer at the catchment mean elevation.
    Prior: bounded and uniform in real space, used for both
    the Monte Carlo and the Michel optimisation. X1 1-2000 mm and X2-X4 from MARRMoT's GR4J ranges
    (Knoben et al. 2019); X6 as X3 (MARRMoT's rule that similar stores share a range); X5 and CN1
    airGR's own default bounds (already finite); CN2 MARRMoT's degree-day factor range (0-20; lower
    bound 0.01 because airGR log-transforms CN2 internally).
    airGR's default transformed space is not used because ~97% of its samples were unusable."""
    name = "GR6J"
    names = gs.param_names("CemaNeigeGR6J")
    model = "CemaNeigeGR6J"
    prior_version = "bounded-marrmot-2026-10-09"
    capacity = ["X1"]
    _airgr = gs.transfo_param(np.array([[-9.99] * 8, [9.99] * 8]), "TR", "CemaNeigeGR6J")
    lo, hi = _bounds({"X1": (1.0, 2000.0), "X2": (-20.0, 20.0), "X3": (1.0, 300.0),
                      "X4": (0.5, 15.0), "X5": (_airgr[0, 4], _airgr[1, 4]), "X6": (1.0, 300.0),
                      "CN1": (_airgr[0, 6], _airgr[1, 6]), "CN2": (0.01, 20.0)},
                     ["X1", "X2", "X3", "X4", "X5", "X6", "CN1", "CN2"])

    def search_ranges(self):
        return np.vstack([self.lo, self.hi])

    def prepare(self, f):
        """Build grsuite inputs once per catchment (they do not depend on parameters)."""
        dates = np.asarray(f["dates"].values, dtype="datetime64[D]")
        ext = gs.data_alti_extrapolation_valery(dates, f["P"], f["Tmean"], f["elev"], None, 1,
                                                temp_min=f["Tmin"], temp_max=f["Tmax"])
        self.im = gs.InputsModel(dates, f["P"], f["E"], f["Tmean"], layer_precip=ext["LayerPrecip"],
                                 layer_temp_mean=ext["LayerTempMean"],
                                 layer_frac_solid_precip=ext["LayerFracSolidPrecip"],
                                 z_layers=ext["ZLayers"])
        n, w = len(dates), f["n_warm"]
        self.ro = gs.RunOptions(self.im, self.model, np.arange(w, n), ind_period_warmup=np.arange(w))
        self.n_warm = w

    def simulate(self, f, p):
        q = np.full(len(f["P"]), np.nan)
        q[self.n_warm:] = gs.run_model_cemaneige_gr6j(self.im, self.ro, np.asarray(p, float))["Qsim"]
        return q


class SACSMA(Setup):
    """SAC-SMA + SNOW-17 + 2-parameter gamma UH (NWS Fortran via models/sacsma).
    Calibrated parameters and ranges: Newman et al. (2015) Table 1 (without the Priestley-Taylor
    coefficient; PET is FAO-PM). Fixed: SAC-SMA PCTIM = ADIMP = RIVA = SIDE = 0, RSERV = 0.3
    (Anderson 2002); EFC = forest cover fraction (HydroATLAS); SNOW-17 NMF 0.15, MBASE 0,
    PLWHC 0.04, DAYGM 0 (NWS SNOW-17 manual), TIPM 0.12 and the NRKW1 zone 1 depletion curve
    (nwsrfsr's bundled NWS calibration). Snow fraction: Tmean <= PXTEMP. PE/precip adjustment 1.
    UH without nwsrfsr's start-of-step shift; scale in days."""
    name = "SACSMA"
    prior_version = "2026-10-08"
    names = list(SAC_RANGES)
    lo, hi = _bounds(SAC_RANGES, names)
    capacity = ["UZTWM", "LZTWM"]
    fixed_sac = {"adimp": 0.0, "pctim": 0.0, "riva": 0.0, "side": 0.0, "rserv": 0.3}
    fixed_snow = {"nmf": 0.15, "tipm": 0.12, "mbase": 0.0, "plwhc": 0.04, "daygm": 0.0,
                  "adc_a": 0.106302, "adc_b": 44.517337, "adc_c": 3.824784}

    def simulate(self, f, p):
        v = dict(zip(self.names, p))
        sac = {k.lower(): v[k] for k in ["UZTWM", "UZFWM", "LZTWM", "LZFPM", "LZFSM", "UZK", "LZPK",
                                         "LZSK", "ZPERC", "REXP", "PFREE"]}
        sac.update(self.fixed_sac, efc=f["forest_frac"])
        snow = {k.lower(): v[k] for k in ["SCF", "MFMAX", "MFMIN", "UADJ", "SI"]}
        snow.update(self.fixed_snow)
        ptps = (f["Tmean"] <= v["PXTEMP"]).astype(float)
        d = f["dates"]
        o = sacsnow.run(24, d.year, d.month, d.day, np.full(len(d), 12), [f["lat"]], [f["elev"]],
                        np.array([[sac[k]] for k in sacsnow.SAC_NAMES]),
                        np.array([[snow[k]] for k in sacsnow.SNOW_NAMES]),
                        [1.0], [1.0], [0.0], f["P"], ptps, f["Tmean"], f["E"], return_states=False)
        return sacsnow.uh(o["tci"], v["USHAPE"], v["USCALE"], 24, start_of_timestep=False)[:, 0]


SETUPS = {s.name: s for s in (HBV(), GR6J(), SACSMA())}
