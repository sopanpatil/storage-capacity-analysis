"""Robustness item F: HBV FC*LP and FC*(1 - LP) against the benchmarks.

FC*LP is the soil moisture below which HBV's actual evaporation falls below potential; FC*(1 - LP) is the
storage over which it runs at the full potential rate. For the optimum of every assessed-sample HBV pair
(data/rq2/parameter_benchmarks_screened_catchments.csv: KGE >= 0.5 in both periods, benchmarks as in
rq2_parameter.py), reports Spearman with each benchmark (bootstrap 90% interval, rq2_parameter.spearman_ci),
the median log10 ratio, and the share inside the benchmark range, for FC, FC*LP and FC*(1 - LP).

Writes robustness/F_hbv_lp.csv.  Usage: python src/robustness_hbv_lp.py
"""
import numpy as np
import pandas as pd

from caravan_io import ROOT
from rq2_parameter import PRIMARY, spearman_ci


def main():
    d = pd.read_csv(ROOT / "data/rq2/parameter_benchmarks_screened_catchments.csv")
    d = d[d.model == "HBV"].set_index("gauge_id")
    p = pd.read_csv(ROOT / "data/calibration/calibration_summary.csv")
    p = p[p.model == "HBV"].set_index("gauge_id")
    assert np.allclose(d.capacity_opt, p.p_FC.reindex(d.index), rtol=1e-3)   # rq2 rounds to 3 decimals
    q = {"FC": p.p_FC, "FC*LP": p.p_FC * p.p_LP, "FC*(1-LP)": p.p_FC * (1 - p.p_LP)}
    rows = []
    for name, v in q.items():
        v = v.reindex(d.index)
        for bench in PRIMARY:
            m = d[bench].notna()
            rho, lo, hi = spearman_ci(v[m].to_numpy(), d.loc[m, bench].to_numpy())
            pos = m & (v > 0)   # FC*(1 - LP) is 0 where LP is at its upper bound (1): no log ratio
            rows.append(dict(quantity=name, benchmark=bench, n=int(m.sum()), n_zero=int((m & (v <= 0)).sum()),
                             spearman=rho, spearman_lo=lo, spearman_hi=hi,
                             bias_log10=np.log10(v[pos] / d.loc[pos, bench]).median()))
        m = d.n_benchmarks > 0
        rows.append(dict(quantity=name, benchmark="envelope", n=int(m.sum()),
                         inside=((v[m] >= d.env_lo[m]) & (v[m] <= d.env_hi[m])).mean()))
    s = pd.DataFrame(rows)
    out = ROOT / "robustness"
    out.mkdir(exist_ok=True)
    s.round(6).to_csv(out / "F_hbv_lp.csv", index=False)
    print(f"LP of the optimum: median {p.p_LP.reindex(d.index).median():.2f}, "
          f"{(p.p_LP.reindex(d.index) >= 0.95).mean():.0%} >= 0.95, {(p.p_LP.reindex(d.index) <= 0.25).mean():.0%} <= 0.25")
    print(s.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
