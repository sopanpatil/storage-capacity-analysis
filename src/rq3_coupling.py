"""RQ3 diagnostic: how much of the state-level S_R comes from the capacity parameter and how much from the forcing.
Optimum only; pairs passing the KGE screen.

Model AET is drawn from a store whose size is the parameter, so the flux-based state S_R,20 is close to bounded by it
(Heckman and Lowman 2026 make the same point for land-surface models). The forcing-only control (src/rq3_control.py:
the model's soil input with PET scaled to its mean AET) carries the forcing part without the store.

Per model:
- coupling: Spearman between state, parameter and control; share of pairs with state > parameter; rank regression
  of the state on the parameter and the control (ranks standardised): coefficients, R^2, and each predictor's unique
  R^2 (drop in R^2 when it is left out);
- partial: per primary benchmark (rq2_parameter.benchmarks), Spearman with the benchmark and partial Spearman
  (Pearson of rank residuals) of the state given the control, the state given the parameter and the control given
  the state, each with a bootstrap 90% interval.

Writes data/rq3/coupling_<list>.csv and data/rq3/coupling_partial_<list>.csv.

Usage: python src/rq3_coupling.py [--list data/catalog/screened_catchments.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from caravan_io import ROOT, load_config
from rq2_parameter import PRIMARY, benchmarks

RNG = np.random.default_rng(20261013)


def z(x):
    r = rankdata(x)
    return (r - r.mean()) / r.std()


def r2(y, xs):
    x = np.column_stack(xs)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return 1 - ((y - x @ beta) ** 2).sum() / (y ** 2).sum(), beta


def partial(x, y, given):
    """Pearson correlation of the rank residuals of x and y after regressing each on the ranks of `given`."""
    zx, zy, zg = z(x), z(y), z(given)
    rx = zx - zg * (zx @ zg) / (zg @ zg)
    ry = zy - zg * (zy @ zg) / (zg @ zg)
    return np.corrcoef(rx, ry)[0, 1]


def with_ci(f, cols, n_boot=500):
    est = f(*cols)
    idx = RNG.integers(0, len(cols[0]), size=(n_boot, len(cols[0])))
    boot = [f(*(c[i] for c in cols)) for i in idx]
    return est, *np.nanpercentile(boot, [5, 95])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/screened_catchments.csv")
    a = ap.parse_args()
    cfg = load_config()
    name = Path(a.list).stem
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", index_col="gauge_id")
    ids = pd.read_csv(ROOT / a.list).gauge_id
    b = benchmarks(name, cat.loc[ids], 0.5)
    rq1 = pd.read_csv(ROOT / "data" / "rq1" / f"identifiability_{name}.csv")
    rq3 = pd.read_csv(ROOT / "data" / "rq3" / f"state_capacity_{name}_summary.csv")
    ctrl = pd.read_csv(ROOT / "data" / "rq3" / f"control_{name}.csv")
    d = rq1[rq1.passes][["gauge_id", "model", "capacity_opt"]] \
        .merge(rq3[["gauge_id", "model", "opt_sr_20"]], on=["gauge_id", "model"], validate="1:1") \
        .merge(ctrl[["gauge_id", "model", "ctrl_sr_20"]], on=["gauge_id", "model"], validate="1:1") \
        .rename(columns={"capacity_opt": "parameter", "opt_sr_20": "state", "ctrl_sr_20": "control"})
    d = d[(d[["parameter", "state", "control"]] > 0).all(axis=1)].merge(b[PRIMARY], left_on="gauge_id",
                                                                         right_index=True, validate="m:1")
    rows, prows = [], []
    for model, g in d.groupby("model"):
        s, p, c = (z(g[k].to_numpy()) for k in ("state", "parameter", "control"))
        full, beta = r2(s, [p, c])
        rows.append(dict(model=model, n=len(g), rho_state_param=np.corrcoef(s, p)[0, 1],
                         rho_state_control=np.corrcoef(s, c)[0, 1], rho_param_control=np.corrcoef(p, c)[0, 1],
                         share_state_gt_param=(g.state > g.parameter).mean(),
                         median_state_over_param=(g.state / g.parameter).median(),
                         beta_param=beta[0], beta_control=beta[1], r2=full,
                         unique_r2_param=full - r2(s, [c])[0], unique_r2_control=full - r2(s, [p])[0]))
        for bench in PRIMARY:
            h = g[g[bench].notna()]
            st, pa, co, be = (h[k].to_numpy() for k in ("state", "parameter", "control", bench))
            row = dict(model=model, benchmark=bench, n=len(h))
            for key, f, cols in (("state", lambda x, y: np.corrcoef(z(x), z(y))[0, 1], (st, be)),
                                 ("control", lambda x, y: np.corrcoef(z(x), z(y))[0, 1], (co, be)),
                                 ("state_given_control", partial, (st, be, co)),
                                 ("state_given_param", partial, (st, be, pa)),
                                 ("control_given_state", partial, (co, be, st))):
                row[key], row[f"{key}_lo"], row[f"{key}_hi"] = with_ci(f, cols)
            prows.append(row)
    out = ROOT / "data" / "rq3"
    cp, pp = pd.DataFrame(rows), pd.DataFrame(prows)
    cp.round(6).to_csv(out / f"coupling_{name}.csv", index=False)
    pp.round(6).to_csv(out / f"coupling_partial_{name}.csv", index=False)
    pd.set_option("display.width", 220)
    print(cp.round(2).to_string(index=False))
    print(pp[["model", "benchmark", "n", "state", "control", "state_given_control", "state_given_control_lo",
              "state_given_control_hi", "state_given_param", "control_given_state"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
