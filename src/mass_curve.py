"""Mass-curve (root-zone storage deficit) estimate of storage capacity.

Wang-Erlandsson et al. (2016, HESS 20, 1459-1481), sections 2.1 and 2.3.
Used for the climate-derived benchmark (inflow = precipitation, outflow =
satellite E) and for RQ3, where the same algorithm runs on each model's own fluxes.

- Deficit (their Eq. 4): D(t) = max(0, D(t-1) + out(t) - in(t)), starting from 0 and never reset.
- Lateral inflow: where mean outflow exceeds mean inflow over the period, the mean difference is added
  to the inflow on every day (their section 2.1), and the result is flagged.
- One maximum per hydrological year; complete years only. The year start is an argument;
  `driest_start_month` gives the start used for the benchmarks.
- Gumbel's method with finite-sample reduced mean y_n and standard deviation sigma_n (their Eq. 8-9).

Usage: python src/mass_curve.py --selftest
"""
import argparse

import numpy as np
import pandas as pd


def deficit(inflow, outflow):
    """Daily deficit D (mm) for daily inflow and outflow (mm/d), time along axis 0.

    Uses the closed form of the recursion: with S the cumulative sum of (out - in),
    D(t) = S(t) - min(0, min over s <= t of S(s)). Works on 1-D series or 2-D arrays (time x runs).
    """
    inflow, outflow = np.asarray(inflow, float), np.asarray(outflow, float)
    if np.isnan(inflow).any() or np.isnan(outflow).any():
        raise ValueError("inflow and outflow must be complete (no NaN)")
    s = np.cumsum(outflow - inflow, axis=0)
    return s - np.minimum(np.minimum.accumulate(s, axis=0), 0.0)


def lateral_inflow(inflow, outflow):
    """Mean (out - in) where positive, else 0: the per-day amount added to the inflow."""
    return max(float(np.mean(outflow) - np.mean(inflow)), 0.0)


def hydro_year(dates, start_month):
    """Hydrological year of each date, labelled by the calendar year in which it starts."""
    dates = pd.DatetimeIndex(dates)
    return dates.year - (dates.month < start_month).astype(int)


def annual_maxima(d, dates, start_month):
    """Maximum deficit in each complete hydrological year (pd.Series indexed by year)."""
    dates = pd.DatetimeIndex(dates)
    s = pd.Series(np.asarray(d, float), index=dates)
    years = hydro_year(dates, start_month)
    out = {}
    for y, part in s.groupby(years):
        start = pd.Timestamp(year=y, month=start_month, day=1)
        n_days = (start + pd.DateOffset(years=1) - start).days
        if part.index[0] == start and len(part) == n_days:
            out[y] = part.max()
    return pd.Series(out, name="annual_max_deficit", dtype=float)


def driest_start_month(d, dates):
    """Hydrological-year start: the calendar month with
    the lowest mean deficit, when the store is usually full."""
    s = pd.Series(np.asarray(d, float), index=pd.DatetimeIndex(dates))
    return int(s.groupby(s.index.month).mean().idxmin())


def gumbel_reduced_stats(n):
    """Reduced mean y_n and standard deviation sigma_n for n annual maxima (Weibull plotting
    positions, population std): 0.4952 and 0.9496 for n = 10, as in Wang-Erlandsson et al."""
    y = -np.log(-np.log(np.arange(1, n + 1) / (n + 1)))
    return y.mean(), y.std(ddof=0)


def gumbel_return_level(maxima, return_period):
    """Deficit at a return period (years) by Gumbel's method (Wang-Erlandsson et al. Eq. 8-9)."""
    x = np.asarray(maxima, float)
    y_n, sigma_n = gumbel_reduced_stats(len(x))
    y_l = -np.log(-np.log(1.0 - 1.0 / return_period))
    return x.mean() + x.std(ddof=1) / sigma_n * (y_l - y_n)


def storage_capacity(inflow, outflow, dates, return_periods=(10, 20, 40), start_month=None):
    """Mass-curve storage capacity at each return period for one catchment.

    `start_month` None uses `driest_start_month` (computed after the lateral-inflow correction).
    Returns a dict with the capacities ('sr_<T>'), the plain maximum deficit ('sr_max', their Eq. 5),
    the lateral inflow added (mm/d), the year start and the number of annual maxima.
    """
    inflow, outflow = np.asarray(inflow, float), np.asarray(outflow, float)
    lat = lateral_inflow(inflow, outflow)
    d = deficit(inflow + lat, outflow)
    if start_month is None:
        start_month = driest_start_month(d, dates)
    maxima = annual_maxima(d, dates, start_month)
    out = {f"sr_{t}": gumbel_return_level(maxima, t) for t in return_periods}
    out.update(sr_max=float(d.max()), lateral_inflow=lat, start_month=start_month,
               n_years=len(maxima))
    return out


def selftest():
    # Recursion vs closed form, 1-D and 2-D
    rng = np.random.default_rng(1)
    p = rng.gamma(0.3, 10.0, size=(3000, 4))
    e = rng.uniform(0, 5, size=(3000, 4))
    loop = np.zeros_like(p)
    prev = np.zeros(4)
    for t in range(len(p)):
        prev = np.maximum(0.0, prev + e[t] - p[t])
        loop[t] = prev
    assert np.allclose(deficit(p, e), loop, atol=1e-9)
    assert np.allclose(deficit(p[:, 0], e[:, 0]), loop[:, 0], atol=1e-9)

    # Hand example: deficit builds 1/day for 3 days, 2 mm of rain recovers part of it
    assert np.allclose(deficit([0, 0, 0, 2, 0, 10], [1, 1, 1, 0, 1, 1]), [1, 2, 3, 1, 2, 0])

    # Published reduced statistics (Wang-Erlandsson et al. 2016, section 2.3)
    for n, ym, sd in [(10, 0.4952, 0.9496), (11, 0.4996, 0.9676)]:
        y_n, s_n = gumbel_reduced_stats(n)
        assert round(y_n, 4) == ym and round(s_n, 4) == sd, (n, y_n, s_n)

    # Gumbel's method recovers a Gumbel return level from a long sample
    mu, beta, t = 100.0, 30.0, 20
    x = mu - beta * np.log(-np.log(rng.uniform(size=20000)))
    true = mu - beta * np.log(-np.log(1 - 1 / t))
    assert abs(gumbel_return_level(x, t) / true - 1) < 0.01

    # Synthetic seasonal climate: wet winter, dry summer, 18 calendar years (as 2003-2020)
    dates = pd.date_range("2003-01-01", "2020-12-31", freq="D")
    doy = dates.dayofyear.values
    wet = np.cos(2 * np.pi * (doy - 15) / 365.25) > 0
    p = np.where(wet, 6.0, 0.5)
    e = 2.0 + 1.5 * np.sin(2 * np.pi * (doy - 100) / 365.25)
    r = storage_capacity(p, e, dates)
    assert r["lateral_inflow"] == 0.0 and r["n_years"] in (17, 18)
    assert r["sr_10"] < r["sr_20"] < r["sr_40"]
    # Identical years: all maxima equal, so every return level equals the maximum deficit
    assert np.isclose(r["sr_20"], r["sr_max"], rtol=1e-6)
    # The year starts in the wet season (mid-Oct to mid-Apr here), once the store has refilled
    assert r["start_month"] in (1, 2, 3, 4), r["start_month"]

    # A dry season split by the calendar year: a calendar start counts the same drought twice
    # (Southern-Hemisphere-like timing), the driest-month start does not
    p_sh = np.roll(p, 182)
    e_sh = np.roll(e, 182)
    d = deficit(p_sh, e_sh)
    m_cal = annual_maxima(d, dates, 1)
    m_hyd = annual_maxima(d, dates, driest_start_month(d, dates))
    assert len(m_cal) == 18 and len(m_hyd) == 17

    # Lateral inflow: outflow exceeding inflow on average is offset, and flagged
    r = storage_capacity(np.full(len(dates), 1.0), np.full(len(dates), 2.0), dates)
    assert np.isclose(r["lateral_inflow"], 1.0) and np.isclose(r["sr_max"], 0.0)

    # Incomplete first/last years are dropped: 15 Mar 2003 - 17 Apr 2004 holds only the April-start
    # year 2003 (the 2002 and 2004 years are partial)
    m = annual_maxima(np.arange(400.0), pd.date_range("2003-03-15", periods=400), 4)
    assert list(m.index) == [2003] and m[2003] == 382.0
    print("mass_curve selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
