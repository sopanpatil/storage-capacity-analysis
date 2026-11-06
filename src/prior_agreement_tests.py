"""Why earlier studies found agreement between calibrated and climate-derived capacity: two tests.

1. gao: a water-balance benchmark as in the agreement studies (Gao et al. 2014; Nijzink et al. 2016; de
   Boer-Euser et al. 2016), which derive transpiration from observed streamflow. Per catchment, 2003-2020
   (the main benchmark's window):
     interception E_i = min(P, E_p, 2 mm/d): Gao et al.'s 2 mm/d threshold, capped by the evaporative
     demand as in Nijzink et al. (Eq. 2) and de Boer-Euser et al.'s interception reservoir (without the cap,
     winter rain is mostly removed); effective precipitation P_e = P - E_i;
     long-term transpiration T = mean(P_e) - mean(Q) over the days with observed flow (>= 5 years of them);
     daily T(t) = E_p(t) / mean(E_p) * T, with E_p the FAO-PM PET of the model forcing (Nijzink et al. Eq. 6);
     S_R from the same mass curve, hydrological year and Gumbel method as the main benchmark
     (src/mass_curve.py). Same eligibility (frac_snow <= 0.2, aridity <= 2). NaN where T <= 0.
   This benchmark shares observed Q with the calibration, as theirs did; the main one uses no streamflow.
2. clusters: as Liang et al. (2024), compare cluster averages instead of catchments. Clusters: the aridity x
   snow strata, and k-means (k = 10 as Liang et al., and 50) on standardised Caravan climate attributes of
   the screened catchments (log P, log PET, log aridity, snow fraction, seasonality, high/low precipitation
   frequency). Spearman between cluster medians of log10 capacity and log10 benchmark (clusters with >= 5
   pairs), with a bootstrap 90% interval over catchments. Also, as Liang et al. chose the best-matching return
   period per cluster: median |log10 bias| across clusters at T = 20 vs the best of T = 10/20/40 per cluster.
3. constrained: as Ibrahim et al. (2026), calibrate with the capacity prior confined to the benchmark's own
   range. Emulated from the existing Monte Carlo samples (Latin-hypercube draws from the full prior): the
   samples whose capacity lies in the band are draws from the truncated prior. Bands from the main climate
   benchmark (ensemble, eligible, no lateral inflow): narrow = S_R,10 to S_R,40; wide = S_R,5 to S_R,80, their
   return-period range, extrapolated exactly along our Gumbel fit (return level is linear in the reduced
   variate; no bootstrap bounds, so still narrower than theirs). Sample: pairs passing the KGE screen. Per
   pair and band, estimates compared with S_R,20:
     band_best: capacity of the in-band set with the highest KGE_cal (calibration under the constrained prior);
     band_beh_median: median over in-band behavioural sets (KGE_cal >= config threshold; their ensemble median);
     band_prior_median: median over all in-band sets, with no streamflow information (the null: agreement
       from the constraint alone);
     free_opt / free_beh_median: the unconstrained optimum and behavioural median (data/rq1), for reference.
   Summary on the pairs with >= 10 behavioural sets in the band (the same pairs for every estimate):
   Spearman (bootstrap 90% interval), Pearson (as Ibrahim et al.), median log10 bias; and the performance
   cost: KGE_cal of band_best minus the best KGE_cal of all samples, and the share of band_best sets that
   pass the KGE screen (>= 0.5 in calibration and validation).

Subcommands:
  gao [--workers N]          writes data/prior_tests/gao_benchmark_<list>.csv
  compare                    writes data/prior_tests/gao_agreement_<list>.csv and cluster_agreement_<list>.csv
  constrained [--workers N]  writes data/prior_tests/constrained_prior_<list>.csv (per pair x band) and
                             constrained_prior_summary_<list>.csv; with --limit N, the first N catchments,
                             printed only
  constrained-summary        rebuilds constrained_prior_summary_<list>.csv from the saved per-pair table

Usage: python src/prior_agreement_tests.py gao --workers 4
       python src/prior_agreement_tests.py compare
       python src/prior_agreement_tests.py constrained --workers 4
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

from caravan_io import ROOT, load_config, load_timeseries
from mass_curve import storage_capacity
from rq2_parameter import benchmarks, spearman_ci, summarise
from rq3_benchmarks import state_frame

PERIOD = ("2003-01-01", "2020-12-31")
EI_MAX = 2.0           # mm/d, Gao et al. (2014)
MIN_Q_DAYS = 5 * 365
OUT = ROOT / "data" / "prior_tests"
CLUSTER_ATTRS = ["p_mean", "pet_mean_FAO_PM", "aridity_FAO_PM", "frac_snow", "seasonality_FAO_PM",
                 "high_prec_freq", "low_prec_freq"]
LOGGED = {"p_mean", "pet_mean_FAO_PM", "aridity_FAO_PM"}
RNG = np.random.default_rng(20261012)


def gao_one(cfg, gid):
    f = cfg["forcing"]
    ts = load_timeseries(cfg, gid, [f["precip"], f["pet"], "streamflow"]).loc[PERIOD[0]:PERIOD[1]]
    p, ep, q = ts[f["precip"]], ts[f["pet"]], ts["streamflow"]
    pe = p - np.minimum(np.minimum(p, ep), EI_MAX)
    ok = q.notna()
    row = dict(gauge_id=gid, q_days=int(ok.sum()))
    if ok.sum() < MIN_Q_DAYS:
        return row
    t_mean = pe[ok].mean() - q[ok].mean()
    row.update(pe_mean=pe.mean(), t_mean=t_mean)
    if t_mean <= 0:
        return row
    r = storage_capacity(pe.to_numpy(), (ep / ep.mean() * t_mean).to_numpy(), p.index)
    row.update({f"gao_sr_{t}": r[f"sr_{t}"] for t in (10, 20, 40)}, gao_lateral_inflow=r["lateral_inflow"])
    return row


def gao(cfg, name, ids, workers):
    with ProcessPoolExecutor(workers) as ex:
        g = pd.DataFrame(ex.map(partial(gao_one, cfg), ids, chunksize=25)).set_index("gauge_id")
    clim = pd.read_csv(ROOT / "data" / "benchmarks" / f"climate_{name}.csv", index_col=0)
    g.insert(0, "eligible", clim.eligible.reindex(g.index))
    OUT.mkdir(parents=True, exist_ok=True)
    g.round(4).to_csv(OUT / f"gao_benchmark_{name}.csv")
    e = g[g.eligible]
    print(f"{len(g)} catchments, {len(e)} eligible, S_R,20 for {e.gao_sr_20.notna().sum()}; median "
          f"{e.gao_sr_20.median():.0f} mm (main benchmark {clim[clim.eligible].ens_sr_20.median():.0f} mm)")


def clusters(cat):
    x = cat[CLUSTER_ATTRS].copy()
    for c in LOGGED:
        x[c] = np.log(x[c])
    x = (x - x.mean()) / x.std()
    out = pd.DataFrame(index=cat.index)
    for k in (10, 50):
        out[f"kmeans{k}"] = KMeans(k, n_init=20, random_state=0).fit_predict(x.to_numpy())
    return out


def cluster_rho(d, value, bench, key, min_size=5):
    lv, lb = np.log10(d[value]), np.log10(d[bench])
    def rho(idx):
        m = pd.DataFrame({"v": lv.to_numpy()[idx], "b": lb.to_numpy()[idx], "k": d[key].to_numpy()[idx]})
        g = m.groupby("k").agg(v=("v", "median"), b=("b", "median"), n=("v", "size"))
        g = g[g.n >= min_size]
        return spearmanr(g.v, g.b).statistic, len(g)
    r, n = rho(np.arange(len(d)))
    boot = [rho(RNG.integers(0, len(d), len(d)))[0] for _ in range(200)]
    return r, *np.nanpercentile(boot, [5, 95]), n


def compare(cfg, name, ids):
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", index_col="gauge_id").loc[ids]
    b = benchmarks(name, cat, 0.5)
    g = pd.read_csv(OUT / f"gao_benchmark_{name}.csv", index_col=0)
    b["gao"] = g.gao_sr_20.where(g.eligible & (g.gao_sr_20 > 0))
    for t in (10, 40):
        b[f"gao_T{t}"] = g[f"gao_sr_{t}"].where(g.eligible & (g[f"gao_sr_{t}"] > 0))
    b = b.join(clusters(cat))
    b["catchment"] = np.arange(len(b))
    both = b.climate.notna() & b.gao.notna()
    print(f"gao vs main climate benchmark (eligible, both defined, n={both.sum()}): Spearman "
          f"{spearmanr(b.climate[both], b.gao[both]).statistic:.2f}; median log10(gao/climate) "
          f"{np.log10(b.gao[both] / b.climate[both]).median():.2f}")

    rq1 = pd.read_csv(ROOT / "data" / "rq1" / f"identifiability_{name}.csv")
    pairs = rq1[rq1.passes][["gauge_id", "model", "capacity_opt", "n_behavioural", "beh_p5", "beh_p95"]]
    rq3 = pd.read_csv(ROOT / "data" / "rq3" / f"state_capacity_{name}_summary.csv")
    ctrl = pd.read_csv(ROOT / "data" / "rq3" / f"control_{name}.csv")
    levels = {"parameter": pairs, "state": state_frame(pairs, rq3, 20),
              "control": pairs.drop(columns="capacity_opt").merge(
                  ctrl[["gauge_id", "model", "ctrl_sr_20"]].rename(columns={"ctrl_sr_20": "capacity_opt"}),
                  on=["gauge_id", "model"], validate="1:1").assign(beh_p5=np.nan, beh_p95=np.nan)}

    gao_rows, cl_rows = [], []
    for level, df in levels.items():
        df = df.merge(b, left_on="gauge_id", right_index=True, validate="m:1")
        for model, gm in df.groupby("model"):
            common = gm[gm.climate.notna() & gm.gao.notna()]
            for sample, gg in (("all_gao", gm), ("common", common)):
                for bench in (["gao"] if sample == "all_gao" else ["climate", "gao"]):
                    gao_rows.append(dict(level=level, model=model, sample=sample, benchmark=bench,
                                         **summarise(gg, bench)))
            for bench in ("climate", "gao"):
                d = gm[gm[bench].notna()]
                for key in ("catchment", "stratum", "kmeans10", "kmeans50"):
                    r, lo, hi, n = cluster_rho(d, "capacity_opt", bench, key, 1 if key == "catchment" else 5)
                    row = dict(level=level, model=model, benchmark=bench, grouping=key, n_groups=n,
                               spearman=r, spearman_lo=lo, spearman_hi=hi)
                    if level == "parameter" and key != "catchment":
                        tb = {20: bench, 10: f"{bench}_T10", 40: f"{bench}_T40"}
                        bias = pd.DataFrame({t: np.log10(d.capacity_opt / d[c]) for t, c in tb.items()}) \
                            .groupby(d[key]).median()
                        bias = bias[d.groupby(key).size().reindex(bias.index) >= 5].abs()
                        row.update(abs_bias_T20=bias[20].median(), abs_bias_bestT=bias.min(axis=1).median(),
                                   share_T10_best=(bias.idxmin(axis=1) == 10).mean(),
                                   share_T40_best=(bias.idxmin(axis=1) == 40).mean())
                    cl_rows.append(row)
    ga, cl = pd.DataFrame(gao_rows), pd.DataFrame(cl_rows)
    ga.round(6).to_csv(OUT / f"gao_agreement_{name}.csv", index=False)
    cl.round(6).to_csv(OUT / f"cluster_agreement_{name}.csv", index=False)
    pd.set_option("display.width", 220)
    print(ga.pivot_table(index=["model", "level"], columns=["sample", "benchmark"],
                         values=["spearman", "bias_log10", "n"]).round(2).to_string())
    print(cl.pivot_table(index=["benchmark", "model", "level"], columns="grouping", values="spearman")
          [["catchment", "kmeans50", "stratum", "kmeans10"]].round(2).to_string())
    p = cl[cl.level == "parameter"]
    print(p[p.grouping != "catchment"][["benchmark", "model", "grouping", "n_groups", "spearman", "spearman_lo",
                                        "spearman_hi", "abs_bias_T20", "abs_bias_bestT"]].round(2).to_string(index=False))


def gumbel_y(t):
    return -np.log(-np.log(1.0 - 1.0 / t))


def return_level(s10, s40, t):
    """S_R at return period t from S_R,10 and S_R,40 (linear in the Gumbel reduced variate)."""
    y10, y40 = gumbel_y(10), gumbel_y(40)
    return s10 + (s40 - s10) * (gumbel_y(t) - y10) / (y40 - y10)


def constrained_one(runs, thr, pairs):
    """pairs: rows of one catchment (gauge_id, model, lo/hi per band). Reads its Monte Carlo files."""
    rows = []
    for _, p in pairs.iterrows():
        mc = pd.read_parquet(runs / p.gauge_id / f"{p.model}_mc.parquet", columns=["capacity", "kge_cal", "kge_val"])
        kge = mc.kge_cal.fillna(-np.inf)   # failed runs (NaN) are never behavioural
        for band in ("narrow", "wide"):
            lo, hi = p[f"{band}_lo"], p[f"{band}_hi"]
            m = mc[(mc.capacity >= lo) & (mc.capacity <= hi)]
            k = kge[m.index]
            row = dict(gauge_id=p.gauge_id, model=p.model, band=band, band_lo=lo, band_hi=hi, n_in=len(m),
                       n_beh_in=int((k >= thr).sum()), kge_cal_best_all=kge.max())
            if len(m):
                best = k.idxmax()
                row.update(band_best=m.capacity[best], kge_cal_band_best=k[best], kge_val_band_best=m.kge_val[best],
                           band_prior_median=m.capacity.median())
                if (k >= thr).any():
                    row.update(band_beh_median=m.capacity[k >= thr].median())
            rows.append(row)
    return rows


ESTIMATES = ["free_opt", "free_beh_median", "band_best", "band_beh_median", "band_prior_median"]


def constrained(cfg, name, ids, workers, limit=None):
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", index_col="gauge_id").loc[ids]
    clim = pd.read_csv(ROOT / "data" / "benchmarks" / f"climate_{name}.csv", index_col=0).reindex(cat.index)
    b = benchmarks(name, cat, 0.5)[["climate"]]
    s10, s40 = clim.ens_sr_10.where(b.climate.notna()), clim.ens_sr_40.where(b.climate.notna())
    b["narrow_lo"], b["narrow_hi"] = s10, s40
    b["wide_lo"], b["wide_hi"] = return_level(s10, s40, 5).clip(lower=0), return_level(s10, s40, 80)
    rq1 = pd.read_csv(ROOT / "data" / "rq1" / f"identifiability_{name}.csv")
    pairs = rq1[rq1.passes][["gauge_id", "model", "capacity_opt", "beh_p50"]] \
        .merge(b.dropna(), left_on="gauge_id", right_index=True, validate="m:1")
    gids = pairs.gauge_id.unique()[:limit]
    pairs = pairs[pairs.gauge_id.isin(gids)]
    thr = cfg["calibration"]["behavioural_kge"]
    groups = [g for _, g in pairs.groupby("gauge_id")]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rs in ex.map(partial(constrained_one, cfg["paths"]["runs"] / "calibration", thr), groups,
                                   chunksize=10) for r in rs]
    d = pd.DataFrame(rows).merge(pairs[["gauge_id", "model", "climate", "capacity_opt", "beh_p50"]]
                                 .rename(columns={"capacity_opt": "free_opt", "beh_p50": "free_beh_median"}),
                                 on=["gauge_id", "model"], validate="m:1")
    d["dkge_cal"] = d.kge_cal_band_best - d.kge_cal_best_all
    d["band_best_passes"] = (d.kge_cal_band_best >= 0.5) & (d.kge_val_band_best >= 0.5)
    s = constrained_summary(d)
    pd.set_option("display.width", 220)
    if limit is not None and limit <= 5:
        print(d[["gauge_id", "model", "band", "band_lo", "band_hi", "n_in", "n_beh_in", "climate"] + ESTIMATES
                + ["dkge_cal", "band_best_passes"]].round(2).to_string(index=False))
    print_constrained(s)
    if limit is None:
        OUT.mkdir(parents=True, exist_ok=True)
        d.round(4).to_csv(OUT / f"constrained_prior_{name}.csv", index=False)
        s.round(6).to_csv(OUT / f"constrained_prior_summary_{name}.csv", index=False)
        print(f"{len(d)} rows -> {OUT / f'constrained_prior_{name}.csv'}")


def constrained_summary(d):
    """Per model x band: performance of the restricted best set and agreement of each estimate with the
    climate-derived benchmark. d is the per-pair table written by `constrained`."""
    rows = []
    for (model, band), g in d.groupby(["model", "band"]):
        a = g[g.n_in > 0]
        perf = dict(model=model, band=band, n_pairs=len(g), share_in=len(a) / len(g),
                    share_any_beh=(g.n_beh_in > 0).mean(), share_beh_ge10=(g.n_beh_in >= 10).mean(),
                    median_n_in=g.n_in.median(), median_dkge_cal=a.dkge_cal.median(),
                    share_band_best_passes=a.band_best_passes.mean())
        for sample, s, ests in (("in_band", a, ["free_opt", "band_best", "band_prior_median"]),
                                ("beh_ge10", g[g.n_beh_in >= 10], ESTIMATES)):
            for est in ests:
                row = dict(perf, sample=sample, estimate=est, n=len(s))
                if len(s) >= 10:
                    x, y = s[est].to_numpy(), s.climate.to_numpy()
                    rho, lo, hi = spearman_ci(x, y)
                    row.update(spearman=rho, spearman_lo=lo, spearman_hi=hi, pearson=np.corrcoef(x, y)[0, 1],
                               pearson_log=np.corrcoef(np.log10(x), np.log10(y))[0, 1],
                               bias_log10=np.median(np.log10(x / y)))
                rows.append(row)
    return pd.DataFrame(rows)


def print_constrained(s):
    if "spearman" in s:
        print(s.pivot_table(index=["sample", "band", "model"], columns="estimate", values=["spearman", "pearson"])
              .reindex(columns=ESTIMATES, level=1).round(2).to_string())
        print(s.pivot_table(index=["sample", "band", "model"], columns="estimate", values="n")
              .reindex(columns=ESTIMATES).to_string())
    print(s.drop_duplicates(["model", "band"])[["band", "model", "n_pairs", "share_in", "share_any_beh",
          "share_beh_ge10", "median_n_in", "median_dkge_cal", "share_band_best_passes"]].round(3).to_string(index=False))


def constrained_summary_only(name):
    """Rebuild the summary from the saved per-pair table, without re-reading the run outputs."""
    d = pd.read_csv(OUT / f"constrained_prior_{name}.csv")
    s = constrained_summary(d)
    print_constrained(s)
    s.round(6).to_csv(OUT / f"constrained_prior_summary_{name}.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["gao", "compare", "constrained", "constrained-summary"])
    ap.add_argument("--list", default="data/catalog/screened_catchments.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="constrained: first N catchments only, no output files")
    a = ap.parse_args()
    cfg = load_config()
    name = Path(a.list).stem
    ids = pd.read_csv(ROOT / a.list, dtype={"gauge_id": str}).gauge_id.tolist()
    if a.action == "constrained-summary":
        return constrained_summary_only(name)
    if a.action == "gao":
        gao(cfg, name, ids, a.workers)
    elif a.action == "constrained":
        constrained(cfg, name, ids, a.workers, a.limit)
    else:
        compare(cfg, name, ids)


if __name__ == "__main__":
    main()
