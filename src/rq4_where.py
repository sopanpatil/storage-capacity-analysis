"""RQ4: where does it break? Identifiability and agreement against aridity, snow and the water-balance ratio

The two-way aridity x snow strata are already in the classification, RQ2 and RQ3 summaries. This adds:
- profiles: one-way bins of each covariate (aridity and snow on the pilot bins of config `pilot`; the
  water-balance ratio on [0, 0.8, 1, 1.2, 1.5, inf)). Per model x bin: screen survival (passes, >= 10
  behavioural sets), class shares among classifiable pairs, median behavioural spread, and agreement of the
  parameter, state and control (as src/rq3_benchmarks.py, T = 20) with the climate and Stocker benchmarks:
  (unidentifiable shares also under the fixed thresholds of src/classify.py, class_fixed10/class_halfprior),
  Spearman with bootstrap 90% interval, bias and median absolute log10 error.
- covariates: per model x outcome, Spearman with each covariate alone, and a joint model with all three as
  percentile ranks over the screened catchments (0 = lowest, 1 = highest), so each effect is the change
  across the covariate's range with the other two held fixed, whatever their skew or outliers.
  Unidentifiable (among classifiable pairs): logistic, average marginal effect in probability. Continuous
  outcomes (behavioural spread, envelope width, |log10(level / benchmark)| for parameter, state and control
  vs climate and Stocker): OLS with HC3 errors. 90% intervals. Sensitivity spec "+runoff_ratio" adds the
  2003-2020 runoff ratio, since the water-balance ratio rises with it (a wetness effect, not only P error).

Water-balance ratio = satellite E / (P (1 - Q/P)), 2003-2020 (src/benchmark_water_balance.py). Where Q >= P
(10 catchments) it is set to +inf (P too low for the observed flow alone); where there is no flow record in
2003-2020 (130) it is missing, and those catchments drop out of the ratio profile and the joint models.

Writes data/rq4/profiles_<list>.csv and data/rq4/covariates_<list>.csv.

Usage: python src/rq4_where.py [--list data/catalog/screened_catchments.csv] [--min-valid 0.5]
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from caravan_io import ROOT, load_config
from rq2_parameter import benchmarks, spearman_ci
from rq3_benchmarks import state_frame

COVARIATES = ["aridity", "snow", "wb_ratio"]
RATIO_BINS = [0, 0.8, 1.0, 1.2, 1.5, np.inf]
BENCH = ["climate", "stocker"]
LEVELS = ["parameter", "state", "control"]
DECIDED = ["unidentifiable", "consistent", "inconsistent"]


def load(name, cfg, min_valid):
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", index_col="gauge_id")
    ids = pd.read_csv(ROOT / "data" / "catalog" / f"{name}.csv").gauge_id
    wb = pd.read_csv(ROOT / "data" / "benchmarks" / f"water_balance_{name}.csv", index_col="gauge_id").loc[ids]
    x = pd.DataFrame({"aridity": cat.loc[ids, "aridity_FAO_PM"], "snow": cat.loc[ids, "frac_snow"],
                      "wb_ratio": wb.ens_e_ratio.where(wb.e_wb > 0, np.inf).where(wb.q_days > 0),
                      "runoff_ratio": wb.runoff_ratio})
    b = benchmarks(name, cat.loc[ids], min_valid)
    cls = pd.read_csv(ROOT / "data" / "classification" / f"classification_{name}.csv")
    pairs = cls[cls.passes][["gauge_id", "model", "capacity_opt", "n_behavioural", "beh_p5", "beh_p95"]]
    rq3 = pd.read_csv(ROOT / "data" / "rq3" / f"state_capacity_{name}_summary.csv")
    ctrl = pd.read_csv(ROOT / "data" / "rq3" / f"control_{name}.csv")
    lv = {"parameter": pairs, "state": state_frame(pairs, rq3, 20),
          "control": pairs.drop(columns="capacity_opt").merge(
              ctrl[["gauge_id", "model", "ctrl_sr_20"]].rename(columns={"ctrl_sr_20": "capacity_opt"}),
              on=["gauge_id", "model"], validate="1:1")}
    # one row per catchment x model: class table + each level's value and absolute log error vs each benchmark
    r = cls.merge(b[BENCH], left_on="gauge_id", right_index=True, validate="m:1")
    for level, f in lv.items():
        r = r.merge(f[["gauge_id", "model", "capacity_opt"]].rename(columns={"capacity_opt": f"{level}_value"}),
                    on=["gauge_id", "model"], how="left", validate="1:1")
        for bench in BENCH:
            r[f"abserr_{level}_{bench}"] = np.abs(np.log10(r[f"{level}_value"] / r[bench]))
    r = r.merge(x, left_on="gauge_id", right_index=True, validate="m:1")
    ranks = x.rank(pct=True)   # over the screened catchments; inf ranks highest, missing stays missing
    r = r.merge(ranks.add_suffix("_rank"), left_on="gauge_id", right_index=True, validate="m:1")
    return r


def profile_rows(r, cfg):
    p = cfg["pilot"]
    bins = {"aridity": p["aridity_bins"], "snow": p["snow_bins"], "wb_ratio": RATIO_BINS}
    rows = []
    for cov, edges in bins.items():
        b = pd.cut(r[cov], edges, right=False, include_lowest=True)
        for (model, interval), g in r.groupby(["model", b], observed=True):
            dec = g[g["class"].isin(DECIDED)]
            row = dict(covariate=cov, bin=str(interval), model=model, n=len(g),
                       passes=g.passes.mean(), behavioural_10=(g.passes & (g.n_behavioural >= 10)).mean(),
                       n_classified=len(dec), spread_median=g.spread_abs.median(),
                       **{f"share_{c}": (dec["class"] == c).mean() if len(dec) else np.nan for c in DECIDED})
            for v in ("class_fixed10", "class_halfprior"):   # thresholds independent of the envelope width
                d = g[g[v].isin(DECIDED)]
                row[f"share_unidentifiable_{v[6:]}"] = (d[v] == "unidentifiable").mean() if len(d) else np.nan
            for level in LEVELS:
                for bench in BENCH:
                    d = g[g[bench].notna() & g[f"{level}_value"].notna()]
                    k = f"{level}_{bench}"
                    row[f"n_{k}"] = len(d)
                    if len(d) >= 30:
                        rho, lo, hi = spearman_ci(d[f"{level}_value"].to_numpy(), d[bench].to_numpy())
                        lr = np.log10(d[f"{level}_value"] / d[bench])
                        row.update({f"rho_{k}": rho, f"rho_{k}_lo": lo, f"rho_{k}_hi": hi,
                                    f"bias_{k}": lr.median(), f"abserr_{k}": lr.abs().median()})
            rows.append(row)
    return pd.DataFrame(rows)


def joint(y, X, binary):
    X = sm.add_constant(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if binary:
            m = sm.Logit(y, X).fit(disp=0)
            f = m.get_margeff(at="overall").summary_frame(alpha=0.1)
            return f["dy/dx"], f.iloc[:, 4], f.iloc[:, 5]
        m = sm.OLS(y, X).fit(cov_type="HC3")
        ci = m.conf_int(0.1)
        return m.params.drop("const"), ci[0].drop("const"), ci[1].drop("const")


def covariate_rows(r):
    outcomes = {"unidentifiable": (r["class"].isin(DECIDED), lambda g: (g["class"] == "unidentifiable").astype(float), True),
                "spread_abs": (r.spread_abs.notna(), lambda g: g.spread_abs, False),
                "envelope_width": (r["class"].isin(DECIDED), lambda g: g.envelope_width, False)}
    for level in LEVELS:
        for bench in BENCH:
            c = f"abserr_{level}_{bench}"
            outcomes[c] = (r.passes & r[c].notna(), lambda g, c=c: g[c], False)
    specs = {"base": COVARIATES, "+runoff_ratio": COVARIATES + ["runoff_ratio"]}
    rows = []
    for model, gm in r.groupby("model"):
        for out, (sel, get, binary) in outcomes.items():
            g = gm[sel.loc[gm.index]]
            y = get(g)
            uni = {c: spearmanr(g[c], y, nan_policy="omit").statistic for c in specs["+runoff_ratio"]}
            for spec, covs in specs.items():
                X = g[[f"{c}_rank" for c in covs]].dropna()
                eff, lo, hi = joint(y.loc[X.index], X, binary)
                for c in covs:
                    rows.append(dict(model=model, outcome=out, spec=spec, n=len(X), covariate=c,
                                     spearman=uni[c], effect=eff[f"{c}_rank"], effect_lo=lo[f"{c}_rank"],
                                     effect_hi=hi[f"{c}_rank"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/screened_catchments.csv")
    ap.add_argument("--min-valid", type=float, default=0.5)
    a = ap.parse_args()
    cfg = load_config()
    name = Path(a.list).stem
    r = load(name, cfg, a.min_valid)
    out = ROOT / "data" / "rq4"
    out.mkdir(parents=True, exist_ok=True)
    prof = profile_rows(r, cfg)
    prof.round(6).to_csv(out / f"profiles_{name}.csv", index=False)
    cov = covariate_rows(r)
    cov.round(6).to_csv(out / f"covariates_{name}.csv", index=False)
    print(f"{len(r)} catchment-model pairs -> {out}")
    pd.set_option("display.width", 200)
    show = ["covariate", "bin", "model", "n", "passes", "behavioural_10", "n_classified", "share_unidentifiable",
            "rho_parameter_climate", "rho_state_climate", "rho_control_climate"]
    print(prof[show].round(2).to_string(index=False))
    c = cov[cov.spec == "base"]
    c = c.assign(eff=c.effect.round(2).astype(str) + " [" + c.effect_lo.round(2).astype(str) + ", "
                 + c.effect_hi.round(2).astype(str) + "]")
    print(c.pivot_table(index=["outcome", "model"], columns="covariate", values="eff", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
