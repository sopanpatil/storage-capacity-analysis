import sys, time, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[2]); import sacsnow
d = sys.argv[1] + "/sac"
P = pd.read_csv(f"{d}/pars.csv")
rows = []; timing = {}
for dt, tag in [(6, ""), (24, "daily_")]:
    F = [pd.read_csv(f"{d}/forcing_{tag}z{z}.csv") for z in (1, 2)]
    ref = pd.read_csv(f"{d}/nwsrfsr_out_dt{dt}.csv")
    st = lambda c: np.column_stack([f[c].values for f in F])
    t = 0
    for s in sorted(P.set.unique()):
        p = P[P.set == s].sort_values("zone")
        t0 = time.time()
        o = sacsnow.run(dt, F[0].year, F[0].month, F[0].day, F[0].hour, p.alat.values, p.elev.values,
                        p[sacsnow.SAC_NAMES].values.T, p[sacsnow.SNOW_NAMES].values.T,
                        p.peadj.values, p.pxadj.values, p.init_swe.values,
                        st("map_mm"), st("ptps"), st("mat_degc"), st("etd_mm"))
        t += time.time() - t0
        r = ref[ref.set == s]
        row = {"dt_h": dt, "set": s}
        for v in ["tci", "aet", "uztwc", "lztwc", "lzfpc", "swe"]:
            a = o[v]; b = np.column_stack([r[f"{v}_1"].values, r[f"{v}_2"].values])
            row[v] = np.max(np.abs(a - b))
        rows.append(row)
    timing[dt] = t / P.set.nunique()
res = pd.DataFrame(rows)
res.to_csv(f"{d}/wrapper_vs_nwsrfsr.csv", index=False)
print("max abs difference (mm) over all parameter sets, both zones, all steps:")
print(res.groupby("dt_h")[["tci","aet","uztwc","lztwc","lzfpc","swe"]].max().to_string())
print("seconds per run (2 zones):", {k: round(v, 3) for k, v in timing.items()},
      "| steps:", {6: len(pd.read_csv(f'{d}/forcing_z1.csv')), 24: len(pd.read_csv(f'{d}/forcing_daily_z1.csv'))})
