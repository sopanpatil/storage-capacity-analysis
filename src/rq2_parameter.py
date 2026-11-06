"""RQ2: calibrated capacity parameters vs the independent benchmarks.

Sample: catchment-model pairs whose optimum passes the KGE screen (>= 0.5 in calibration and validation;
from data/rq1/identifiability_<list>.csv, which also gives the behavioural P5/P95).
Benchmarks, each used only where it covers >= --min-valid of the catchment:
- climate: ensemble S_R,20 (data/benchmarks/climate_<list>.csv), only where eligible (frac_snow <= 0.2,
  aridity <= 2) and satellite E <= P (no lateral-inflow correction);
- soil: rooting depth x SoilGrids WHC, not for soil_sr_excluded (North America > 60 deg N);
- stocker: Stocker et al. (2023) S_CWDX80;  zhao: Zhao et al. (2025) Sr (regional, ~300 km).
Envelope per catchment: min-max over the benchmarks available there (n_benchmarks of them).
Sensitivity envelope "envelope_nosoil": the same without the soil benchmark, which sets the envelope's
lower edge in two thirds of catchments and is unrelated to the other benchmarks.
Sensitivity benchmarks (not in the envelope): climate at T = 10 and 40 years, climate including the
E > P catchments, MOD16-only and SSEBop-only climate S_R,20.

Per model x benchmark x stratum (all catchments, then each aridity x snow cell from config `pilot`,
with the cells aridity >= 2 and snow >= 0.05 merged):
- spearman between optimum capacity and benchmark, with a bootstrap 90% interval;
- bias: median log10(optimum / benchmark), with its interquartile range;
- opt_inside (envelope rows only): share of catchments whose optimum lies inside the envelope;
- coverage: share (of those with >= 10 behavioural sets) whose behavioural P5-P95 range contains the
  benchmark value (single benchmarks) or overlaps the envelope (envelope rows).

Writes data/rq2/parameter_benchmarks_<list>.csv (per catchment x model) and
data/rq2/parameter_summary_<list>.csv.

Usage: python src/rq2_parameter.py [--list data/catalog/screened_catchments.csv] [--min-valid 0.5]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from caravan_io import ROOT, load_config

RNG = np.random.default_rng(20261012)
PRIMARY = ["climate", "soil", "stocker", "zhao"]
ENVELOPES = {"envelope": "", "envelope_nosoil": "_nosoil"}
SENSITIVITY = ["envelope_nosoil", "climate_T10", "climate_T40", "climate_incl_EgtP", "climate_mod16", "climate_ssebop"]


def benchmarks(name, cat, min_valid, clim=None, stat=None):
    """Benchmark table per catchment; `clim`/`stat` override the files for list `name` (e.g. a climate
    benchmark recomputed with other forcing)."""
    if clim is None:
        clim = pd.read_csv(ROOT / "data" / "benchmarks" / f"climate_{name}.csv", index_col=0)
    if stat is None:
        stat = pd.read_csv(ROOT / "data" / "benchmarks" / f"static_{name}.csv", index_col=0)
    clim, stat = clim.reindex(cat.index), stat.reindex(cat.index)
    ok = clim.eligible & (clim.ens_lateral_inflow == 0)
    b = pd.DataFrame(index=clim.index)
    b["climate"] = clim.ens_sr_20.where(ok)
    b["soil"] = stat.soil_sr_mm.where((stat.rooting_depth_valid_frac >= min_valid) & ~stat.soil_sr_excluded)
    b["stocker"] = stat.stocker_scwdx80_mm.where(stat.stocker_scwdx80_valid_frac >= min_valid)
    b["zhao"] = stat.zhao_sr_mm.where(stat.zhao_sr_valid_frac >= min_valid)
    b["climate_T10"] = clim.ens_sr_10.where(ok)
    b["climate_T40"] = clim.ens_sr_40.where(ok)
    b["climate_incl_EgtP"] = clim.ens_sr_20.where(clim.eligible)
    b["climate_mod16"] = clim.mod16_sr_20.where(ok)
    b["climate_ssebop"] = clim.ssebop_sr_20.where(ok)
    b = b.where(b > 0)   # a zero benchmark has no log ratio; none expected
    b["env_lo"] = b[PRIMARY].min(axis=1)
    b["env_hi"] = b[PRIMARY].max(axis=1)
    b["n_benchmarks"] = b[PRIMARY].notna().sum(axis=1)
    nosoil = [k for k in PRIMARY if k != "soil"]
    b["env_lo_nosoil"] = b[nosoil].min(axis=1)
    b["env_hi_nosoil"] = b[nosoil].max(axis=1)
    b["n_benchmarks_nosoil"] = b[nosoil].notna().sum(axis=1)
    p = load_config()["pilot"]
    arid = pd.cut(cat.aridity_FAO_PM, p["aridity_bins"], right=False).astype(str)
    snow = pd.cut(cat.frac_snow, p["snow_bins"], right=False, include_lowest=True).astype(str)
    merged = (cat.aridity_FAO_PM >= 2) & (cat.frac_snow >= 0.05)
    b["stratum"] = ("aridity " + arid + ", snow " + snow).where(~merged, "aridity [2.0, inf), snow [0.05, 1.0)")
    return b


def spearman_ci(x, y, n_boot=500):
    rho = spearmanr(x, y).statistic
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    boot = [spearmanr(x[i], y[i]).statistic for i in idx]
    return rho, *np.nanpercentile(boot, [5, 95])


def summarise(g, bench):
    out = dict(n=0)
    if bench in ENVELOPES:
        x = ENVELOPES[bench]
        d = g[g[f"n_benchmarks{x}"] > 0]
        lo, hi = d[f"env_lo{x}"], d[f"env_hi{x}"]
        out.update(n=len(d), opt_inside=((d.capacity_opt >= lo) & (d.capacity_opt <= hi)).mean())
        c = d.beh_p5.notna()
        out.update(n_coverage=int(c.sum()),
                   coverage=((d.beh_p5[c] <= hi[c]) & (d.beh_p95[c] >= lo[c])).mean())
        return out
    d = g[g[bench].notna()]
    if len(d) < 10:
        return dict(n=len(d))
    lr = np.log10(d.capacity_opt / d[bench])
    rho, lo, hi = spearman_ci(d.capacity_opt.to_numpy(), d[bench].to_numpy())
    c = d[d.beh_p5.notna()]
    out.update(n=len(d), spearman=rho, spearman_lo=lo, spearman_hi=hi, bias_log10=lr.median(),
               bias_log10_q25=lr.quantile(0.25), bias_log10_q75=lr.quantile(0.75),
               n_coverage=len(c),
               coverage=((c.beh_p5 <= c[bench]) & (c.beh_p95 >= c[bench])).mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/screened_catchments.csv")
    ap.add_argument("--min-valid", type=float, default=0.5)
    a = ap.parse_args()
    cfg = load_config()
    name = Path(a.list).stem
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", index_col="gauge_id")
    ids = pd.read_csv(ROOT / a.list).gauge_id
    b = benchmarks(name, cat.loc[ids], a.min_valid)
    rq1 = pd.read_csv(ROOT / "data" / "rq1" / f"identifiability_{name}.csv")
    rq1 = rq1[rq1.passes]
    df = rq1[["gauge_id", "model", "capacity_opt", "n_behavioural", "beh_p5", "beh_p50", "beh_p95",
              "opt_position"]].merge(b, left_on="gauge_id", right_index=True, validate="m:1")
    out = ROOT / "data" / "rq2"
    out.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(out / f"parameter_benchmarks_{name}.csv", index=False)

    rows = []
    for model, gm in df.groupby("model"):
        for stratum, g in [("all", gm)] + sorted(gm.groupby("stratum")):
            for bench in PRIMARY + ["envelope"] + SENSITIVITY:
                rows.append(dict(model=model, stratum=stratum, benchmark=bench, **summarise(g, bench)))
    s = pd.DataFrame(rows)
    s.round(6).to_csv(out / f"parameter_summary_{name}.csv", index=False)
    print(f"{len(df)} catchment-model pairs -> {out / f'parameter_benchmarks_{name}.csv'}")
    print(s[(s.stratum == "all") & s.benchmark.isin(PRIMARY + ["envelope"])]
          .drop(columns="stratum").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
