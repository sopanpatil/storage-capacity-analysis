"""Climate-derived (mass-curve) capacity benchmark per catchment, 2003-2020.

1. Satellite E: MOD16A2GF 8-day totals (src/gee_extract.py) and SSEBop v5 monthly totals
   (src/benchmark_ssebop.py), each spread over its days in proportion to ERA5-Land daily evaporation
   (negative ERA5-Land values, i.e. dew, get zero weight; a period with no positive ERA5-Land E is
   spread uniformly). Monthly/period totals stay satellite-based.
2. A product counts for a catchment only if its mean valid-pixel fraction is >= --min-valid (0.5). E = mean of the products that count (Wang-Erlandsson et al.
   2016 used the SSEBop + MOD16 ensemble mean).
3. Inflow = Caravan ERA5-Land precipitation (the model forcing). Mass curve with src/mass_curve.py,
   return periods 10/20/40 years, hydrological year starting in the lowest-mean-deficit month; S_R,20 with calendar years is kept as a sensitivity column.
4. Snow-aware sensitivity variant (`ens_snow_*`, where ERA5-Land snowfall was extracted): inflow =
   rain + snowmelt, as in Stocker et al. (2023): snowfall from the reanalysis (ERA5-Land daily
   snowfall, capped at the day's precipitation because the two use UTC and local days), a snowpack
   melting at 1 mm/d per degree C above 1 degree C (Caravan daily mean temperature), rain = P - snowfall.
Also reported per product (MOD16 only, SSEBop only). `eligible` marks frac_snow <= 0.2 and aridity <= 2
(Gao et al. 2014), the only catchments where the benchmark is used; it is computed for all.

Writes data/benchmarks/climate_<list name>.csv.

Usage: python src/benchmark_climate.py [--list data/catalog/pilot_catchments.csv] [--min-valid 0.5]
       [--gee-from screened_catchments]   (read Earth Engine data extracted for another list, e.g.
                                           the full-sample export, which includes the pilot)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from caravan_io import ROOT, load_config, load_timeseries
from mass_curve import storage_capacity

DATES = pd.date_range("2003-01-01", "2020-12-31", freq="D")


def disaggregate(starts, ends, totals, weights):
    """Spread period totals over days in proportion to `weights` (pd.Series on DATES)."""
    out = pd.Series(np.nan, index=DATES)
    w = weights.clip(lower=0.0)
    for s, e, tot in zip(starts, ends, totals):
        days = out.loc[s:e].index
        if len(days) == 0:
            continue
        ws = w.loc[days]
        out.loc[days] = tot * (ws / ws.sum() if ws.sum() > 0 else 1.0 / len(days))
    return out


def mod16_daily(m, weights):
    """MOD16 composite starts -> end = day before the next start, last composite of a year to 31 Dec."""
    starts = pd.to_datetime(m.start).sort_values().reset_index(drop=True)
    nxt = starts.shift(-1) - pd.Timedelta(days=1)
    year_end = pd.to_datetime(starts.dt.year.astype(str) + "-12-31")
    ends = nxt.where(nxt.notna() & (nxt <= year_end), year_end)
    m = m.assign(start=pd.to_datetime(m.start)).sort_values("start")
    return disaggregate(m.start, ends.values, m.et_mm.values, weights)


def ssebop_daily(s, weights):
    months = pd.PeriodIndex(s.month, freq="M")
    return disaggregate(months.start_time, months.end_time.normalize(), s.et_mm.values, weights)


def snow_inflow(p, snowfall, tmean, melt_t=1.0, ddf=1.0):
    """Rain + snowmelt (mm/d): snowfall (capped at p) accumulates; melt = ddf * (T - melt_t) when T > melt_t."""
    sf = np.minimum(np.asarray(snowfall, float), p)
    pack, out = 0.0, np.empty(len(p))
    for i in range(len(p)):
        pack += sf[i]
        melt = min(pack, ddf * max(tmean[i] - melt_t, 0.0))
        pack -= melt
        out[i] = p[i] - sf[i] + melt
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="data/catalog/pilot_catchments.csv")
    ap.add_argument("--min-valid", type=float, default=0.5)
    ap.add_argument("--gee-from", help="list name whose Earth Engine extract to read (default: --list)")
    a = ap.parse_args()
    cfg = load_config()
    name = Path(a.list).stem
    ids = pd.read_csv(ROOT / a.list, dtype={"gauge_id": str}).gauge_id
    cat = pd.read_csv(cfg["paths"]["catalog"] / "catchments.csv", dtype={"gauge_id": str}).set_index("gauge_id").loc[ids]
    gee = cfg["paths"]["benchmarks"] / "gee" / (a.gee_from or name)
    mod16 = pd.concat(pd.read_csv(f) for f in sorted((gee / "mod16").glob("*.csv")))
    era5 = pd.concat(pd.read_csv(f) for f in sorted((gee / "era5land").glob("*.csv")))
    era5["date"] = pd.to_datetime(era5.date)
    sseb = pd.concat(pd.read_csv(f) for f in sorted((cfg["paths"]["benchmarks"] / "ssebop_v5_catchments" / name).glob("*.csv")))
    assert sseb.month.nunique() == 216, "SSEBop months missing; run src/benchmark_ssebop.py for all years"
    has_snow = "snowfall_mm" in era5.columns
    era5, mod16, sseb = (dict(tuple(x.groupby("gauge_id"))) for x in (era5, mod16, sseb))

    rows = []
    for gid in cat.index:
        e5 = era5[gid].set_index("date").reindex(DATES)
        w = e5.e_mm
        assert w.notna().all(), f"{gid}: ERA5-Land E incomplete"
        m, s = mod16[gid], sseb[gid]
        products = {}
        row = {"gauge_id": gid, "mod16_valid_frac": m.valid_frac.mean(), "ssebop_valid_frac": s.valid_frac.mean()}
        if row["mod16_valid_frac"] >= a.min_valid and m.et_mm.notna().all():
            products["mod16"] = mod16_daily(m, w)
        if row["ssebop_valid_frac"] >= a.min_valid and s.et_mm.notna().all():
            products["ssebop"] = ssebop_daily(s, w)
        ts = load_timeseries(cfg, gid, [cfg["forcing"]["precip"], cfg["forcing"]["tmean"]]).reindex(DATES)
        p = ts[cfg["forcing"]["precip"]]
        assert p.notna().all(), f"{gid}: precipitation incomplete"
        row.update(products="+".join(products) or "none", p_mean=p.mean(), era5land_e_mean=w.mean())
        if products:
            variants = dict(products)
            variants["ens"] = pd.concat(products.values(), axis=1).mean(axis=1)
            for k, e in variants.items():
                row[f"{k}_e_mean"] = e.mean()
                r = storage_capacity(p.values, e.values, DATES)
                row.update({f"{k}_sr_{t}": r[f"sr_{t}"] for t in (10, 20, 40)})
                if k == "ens":
                    row.update(ens_sr_max=r["sr_max"], ens_lateral_inflow=r["lateral_inflow"],
                               ens_start_month=r["start_month"], ens_n_years=r["n_years"])
                    rc = storage_capacity(p.values, e.values, DATES, start_month=1)
                    row.update(ens_sr_20_calendar=rc["sr_20"], ens_n_years_calendar=rc["n_years"])
                    if has_snow:
                        inflow = snow_inflow(p.values, e5.snowfall_mm.values, ts[cfg["forcing"]["tmean"]].values)
                        rs = storage_capacity(inflow, e.values, DATES)
                        row.update({f"ens_snow_sr_{t}": rs[f"sr_{t}"] for t in (10, 20, 40)})
                        row.update(snowfall_frac=float(np.minimum(e5.snowfall_mm, p).sum() / p.sum()))
        rows.append(row)
    out = pd.DataFrame(rows).set_index("gauge_id")
    out.insert(0, "eligible", (cat.frac_snow <= 0.2) & (cat.aridity_FAO_PM <= 2))
    f = ROOT / "data" / "benchmarks" / f"climate_{name}.csv"
    out.round(3).to_csv(f)
    print(f"{len(out)} catchments ({out.eligible.sum()} eligible) -> {f}")


if __name__ == "__main__":
    main()
