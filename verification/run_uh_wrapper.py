"""Compare sacsnow.uh() with nwsrfsr::uh() output from run_nwsrfsr_uh.R.

Usage: python verification/run_uh_wrapper.py <verification dir containing uh/> <models/sacsma dir>
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[2]); import sacsnow  # noqa: E402

d = sys.argv[1] + "/uh"
tci = pd.read_csv(f"{d}/tci.csv").drop(columns="step").to_numpy(float)
pars = pd.read_csv(f"{d}/pars.csv")
ref = pd.read_csv(f"{d}/nwsrfsr_uh.csv")
rows = []
for (dt, sot), r in ref.groupby(["dt", "start_of_timestep"]):
    got = sacsnow.uh(tci, pars["shape"].values, pars["scale"].values, dt, start_of_timestep=bool(sot))
    want = r.drop(columns=["dt", "start_of_timestep", "step"]).to_numpy(float)
    for z, p in pars.iterrows():
        rows.append(dict(dt_h=dt, start_of_timestep=bool(sot), shape=p["shape"], scale=p["scale"],
                         max_abs_mm=np.max(np.abs(got[:, z] - want[:, z])),
                         max_rel=np.max(np.abs(got[:, z] - want[:, z])) / np.max(np.abs(want[:, z]))))
res = pd.DataFrame(rows)
res.to_csv(f"{sys.argv[1]}/uh_wrapper_vs_nwsrfsr.csv", index=False)
pd.set_option("display.width", 200)
print(res.to_string(index=False))
print(f"\nmax abs difference over all cases: {res.max_abs_mm.max():.2e} mm "
      f"(max relative {res.max_rel.max():.2e}; DUAMEL runs in single precision)")
