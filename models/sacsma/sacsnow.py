"""Thin ctypes wrapper around the NWS SAC-SMA + SNOW-17 Fortran (via nwsrfsr's sacsnow).

The Fortran is compiled unchanged from the nwsrfsr package sources into libsacsnow.so,
so Python runs the same code as the R reference rather than a reimplementation.
"""
import ctypes as ct
import os
import numpy as np

_lib = ct.CDLL(os.path.join(os.path.dirname(__file__), "build", "libsacsnow.so"))
_f = _lib.sacsnow_
_f.restype = None

SAC_NAMES = ["uztwm", "uzfwm", "lztwm", "lzfpm", "lzfsm", "adimp", "uzk", "lzpk", "lzsk",
             "zperc", "rexp", "pctim", "pfree", "riva", "side", "rserv", "efc"]
SNOW_NAMES = ["scf", "mfmax", "mfmin", "uadj", "si", "nmf", "tipm", "mbase", "plwhc",
              "daygm", "adc_a", "adc_b", "adc_c"]
OUT_NAMES = ["tci", "aet", "uztwc", "uzfwc", "lztwc", "lzfsc", "lzfpc", "adimc",
             "roimp", "sdro", "ssur", "sif", "bfs", "bfp",
             "swe", "aesc", "neghs", "liqw", "raim", "psfall", "prain"]


def _d(a, shape=None):
    a = np.asarray(a, dtype=np.float64)
    if shape is not None:
        a = a.reshape(shape, order="F")
    return np.asfortranarray(a)


def _ptr(a):
    return a.ctypes.data_as(ct.c_void_p)


_uh = _lib.duamel_
_uh.restype = None


def uh(tci, shape, scale, dt_hours, start_of_timestep=True, backfill=True, max_len=1000):
    """Route channel inflow through a 2-parameter gamma unit hydrograph (nwsrfsr `uh`).

    Calls the NWS DUAMEL Fortran (single precision), as nwsrfsr's `uh()` does with `unit_shape`
    and `unit_scale` given. tci: (n_steps,) or (n_steps, n_zones), depth per step; shape, scale
    (days): one per zone. A negative shape means no routing. Returns depth per step for each zone,
    (n_steps, n_zones); unlike nwsrfsr there is no cfs conversion and no summing over zones.
    start_of_timestep / backfill: nwsrfsr's one-step shift for start-of-step forcing (its default).
    """
    tci = np.atleast_2d(np.asarray(tci, float).T).T
    n_steps, nz = tci.shape
    shape, scale = np.broadcast_to(shape, (nz,)), np.broadcast_to(scale, (nz,))
    n = n_steps + max_len
    out = np.empty((n_steps, nz))
    for i in range(nz):
        q = np.ascontiguousarray(tci[:, i], dtype=np.float32)
        qr = np.zeros(n, dtype=np.float32)
        _uh(_ptr(q), ct.byref(ct.c_float(shape[i])), ct.byref(ct.c_float(scale[i])),
            ct.byref(ct.c_float(dt_hours / 24.0)), ct.byref(ct.c_int(n)), ct.byref(ct.c_int(max_len)),
            ct.byref(ct.c_int(1)), ct.byref(ct.c_int(0)), _ptr(qr))
        out[:, i] = qr[:n_steps]
    if start_of_timestep:
        first = out[:1] if backfill else np.full((1, nz), np.nan)
        out = np.vstack([first, out[:-1]])
    return out


def run(dt_hours, year, month, day, hour, latitude, elev, sac_pars, snow_pars,
        peadj, pxadj, init_swe, map_mm, ptps, mat_degc, etd_mm, return_states=True):
    """Run SNOW-17 + SAC-SMA for one or more zones.

    sac_pars: (17, n_zones) in SAC_NAMES order; snow_pars: (13, n_zones) in SNOW_NAMES order.
    Forcings: (n_steps, n_zones). Returns dict of (n_steps, n_zones) arrays.
    """
    map_mm = np.atleast_2d(np.asarray(map_mm, float).T).T
    n, nz = map_mm.shape
    ints = [np.ascontiguousarray(np.asarray(x, dtype=np.int32)) for x in (year, month, day, hour)]
    zone = [_d(x, (nz,)) for x in (latitude, elev)]
    sac = _d(sac_pars, (17, nz)); snow = _d(snow_pars, (13, nz))
    adj = [_d(x, (nz,)) for x in (peadj, pxadj, init_swe)]
    forc = [_d(x, (n, nz)) for x in (map_mm, ptps, mat_degc, etd_mm)]
    outs = {k: np.zeros((n, nz), dtype=np.float64, order="F") for k in OUT_NAMES}
    n_c, dt_c, len_c = ct.c_int(nz), ct.c_int(int(dt_hours * 3600)), ct.c_int(n)
    rs = ct.c_int(1 if return_states else 0)
    _f(ct.byref(n_c), ct.byref(dt_c), ct.byref(len_c), *[_ptr(a) for a in ints],
       *[_ptr(a) for a in zone], _ptr(sac), _ptr(adj[0]), _ptr(adj[1]), _ptr(snow), _ptr(adj[2]),
       *[_ptr(a) for a in forc], ct.byref(rs), *[_ptr(outs[k]) for k in OUT_NAMES])
    return outs
