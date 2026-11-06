import sys, numpy as np, pandas as pd, grsuite as gs
d = sys.argv[1]
f = pd.read_csv(f"{d}/forcing_L0123002.csv")
dates = pd.to_datetime(f.date).values.astype("datetime64[D]")
hyp = pd.read_csv(f"{d}/hypso_L0123002.csv").iloc[:,0].values.astype(float)
z = float(open(f"{d}/zinputs.txt").read().split()[0])
iw0, iw1, ir0, ir1 = [int(x) for x in open(f"{d}/periods.txt").read().split()]
ext = gs.data_alti_extrapolation_valery(dates, f.P.values, f["T"].values, z, hyp, 5)
print(type(ext), (list(ext.keys()) if isinstance(ext, dict) else [a for a in dir(ext) if not a.startswith('_')]))
im = gs.InputsModel(dates, f.P.values, f.E.values, f["T"].values,
                    layer_precip=ext["LayerPrecip"], layer_temp_mean=ext["LayerTempMean"],
                    layer_frac_solid_precip=ext["LayerFracSolidPrecip"], z_layers=ext["ZLayers"])
ro = gs.RunOptions(im, "CemaNeigeGR6J", np.arange(ir0, ir1+1), ind_period_warmup=np.arange(iw0, iw1+1))
pars = pd.read_csv(f"{d}/params.csv").values
import time
ref = pd.read_csv(f"{d}/airgr_out.csv")
rows = []; t0 = time.time()
for i, p in enumerate(pars, start=1):
    o = gs.run_model_cemaneige_gr6j(im, ro, p)
    swe = np.mean([np.asarray(l["SnowPack"]) for l in o["CemaNeigeLayers"]], axis=0)
    r = ref[ref.set == i]
    row = {"set": i}
    for k, v in [("Qsim", o["Qsim"]), ("Prod", o["Prod"]), ("Rout", o["Rout"]), ("Exp", o["Exp"]), ("SWE", swe)]:
        a = np.asarray(v, float); b = r[k].values
        row[k + "_maxabs"] = np.nanmax(np.abs(a - b))
        row[k + "_maxrel"] = np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-6))
    rows.append(row)
dt = time.time() - t0
res = pd.DataFrame(rows)
res.to_csv(f"{d}/grsuite_vs_airgr.csv", index=False)
pd.set_option("display.width", 200)
print(res.describe().loc[["max"]].T)
print(f"{len(pars)} runs x {ir1-iw0+1} days in {dt:.2f}s (incl. JIT compile)")
t0 = time.time()
for k in range(8):
    for p in pars: gs.run_model_cemaneige_gr6j(im, ro, p)
print(f"steady-state: {(time.time()-t0)/(8*len(pars))*1000:.1f} ms per 12-year CemaNeige-GR6J run (5 snow layers)")
