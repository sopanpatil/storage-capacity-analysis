"""Build the catchment catalogue for Caravan v1.6 + GRDC-Caravan v0.6 and flag duplicates.

Writes to paths.catalog:
  record_stats.csv     streamflow record statistics per catchment (cached; --refresh to redo)
  duplicate_pairs.csv  every candidate duplicate pair, how it was found, and the flow check
  catchments.csv       attributes + record stats + `duplicate_of` (NaN = kept)

Duplicate candidates:
  id       GRDC nat_id equals the local part of a Caravan gauge_id, or a camels id equals a
           hysets id, with gauges within dedup.id_match_max_km
  spatial  gauges from different sources within dedup.spatial_max_km and areas within
           dedup.spatial_max_area_diff
Every candidate is treated as a duplicate: same gauge ID, or same location and area, means the
same forcing and polygon, so keeping both would double-count the catchment. The flow check (r over
common days) is recorded as a diagnostic only; copies of one gauge can disagree (e.g. a one-day
offset between CAMELS-AUS and GRDC, different record revisions).
Each duplicate is dropped in favour of the member with higher source priority, then longer record.

Usage: python src/build_catalog.py [--workers N] [--refresh]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from caravan_io import load_attributes, load_config, load_timeseries

EARTH_KM = 6371.0


def record_stats(args):
    cfg, gid = args
    f = cfg["forcing"]
    df = load_timeseries(cfg, gid, ["streamflow", f["precip"], f["pet"]])
    q = df["streamflow"]
    ok = q.notna()
    yr = df.index.year
    days_per_year = pd.Series(1, index=df.index).groupby(yr).size()
    valid_per_year = ok.groupby(yr).sum()
    is_complete = valid_per_year >= cfg["record"]["complete_year_frac"] * days_per_year
    complete = lambda lo, hi: int(is_complete[(is_complete.index >= lo) & (is_complete.index <= hi)].sum())
    return dict(
        gauge_id=gid,
        q_days=int(ok.sum()),
        q_first=q[ok].index.min(),
        q_last=q[ok].index.max(),
        complete_years=complete(*cfg["record"]["window"]),
        q_negative_days=int((q < 0).sum()),
        zero_flow_frac=float((q[ok] < 0.001).mean()) if ok.any() else np.nan,
        runoff_ratio=float(q[ok].mean() / df.loc[ok, f["precip"]].mean()) if ok.any() else np.nan,
        p_mean_qdays=float(df.loc[ok, f["precip"]].mean()) if ok.any() else np.nan,
        pet_mean_qdays=float(df.loc[ok, f["pet"]].mean()) if ok.any() else np.nan,
    )


def flow_check(args):
    cfg, a, b = args
    qa = load_timeseries(cfg, a, ["streamflow"])["streamflow"]
    qb = load_timeseries(cfg, b, ["streamflow"])["streamflow"]
    j = pd.concat([qa, qb], axis=1, keys=["a", "b"]).dropna()
    r = j.a.corr(j.b) if len(j) >= 2 else np.nan
    ratio = j.b.mean() / j.a.mean() if len(j) and j.a.mean() > 0 else np.nan
    return dict(overlap_days=len(j), flow_r=r, flow_mean_ratio=ratio)


def unit_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def gauge_km(att, a, b):
    xyz = unit_xyz(att.gauge_lat.values, att.gauge_lon.values)
    chord = np.linalg.norm(xyz[a] - xyz[b], axis=1)
    return 2 * EARTH_KM * np.arcsin(np.clip(chord / 2, 0, 1))


def candidate_pairs(att, cfg):
    d = cfg["dedup"]
    att = att.reset_index(drop=True)
    key = lambda s: s.astype(str).str.strip().str.upper().str.lstrip("0")
    local = key(att.gauge_id.str.split("_", n=1).str[1])
    nat = key(att.nat_id.where(att.source == "GRDC"))

    pairs = []
    # id: Caravan local id vs GRDC nat_id; camels id vs hysets id
    caravan = att.index[att.source != "GRDC"]
    grdc = att.index[(att.source == "GRDC") & att.nat_id.notna()]
    lk = pd.DataFrame({"i": caravan, "k": local[caravan].values})
    m = lk.merge(pd.DataFrame({"j": grdc, "k": nat[grdc].values}), on="k")
    pairs.append(m[["i", "j"]])
    cam = pd.DataFrame({"i": att.index[att.source == "camels"], "k": local[att.source == "camels"].values})
    hys = pd.DataFrame({"j": att.index[att.source == "hysets"], "k": local[att.source == "hysets"].values})
    pairs.append(cam.merge(hys, on="k")[["i", "j"]])
    idp = pd.concat(pairs, ignore_index=True)
    idp["km"] = gauge_km(att, idp.i.values, idp.j.values)
    idp = idp[idp.km < d["id_match_max_km"]].assign(found_by="id")

    # spatial: cross-source gauges close together with similar area
    tree = cKDTree(unit_xyz(att.gauge_lat.values, att.gauge_lon.values))
    sp = pd.DataFrame(sorted(tree.query_pairs(d["spatial_max_km"] / EARTH_KM)), columns=["i", "j"])
    sp = sp[att.source.values[sp.i] != att.source.values[sp.j]]
    area_diff = np.abs(att.area.values[sp.j] / att.area.values[sp.i] - 1)
    sp = sp[area_diff < d["spatial_max_area_diff"]]
    sp["km"] = gauge_km(att, sp.i.values, sp.j.values)
    sp = sp.assign(found_by="spatial")

    allp = pd.concat([idp, sp], ignore_index=True)
    # order each pair so `a` is the higher-priority source, then merge id/spatial duplicates
    prio = att.source.map(lambda s: d["source_priority"].get(s, d["source_priority"]["default"])).values
    swap = prio[allp.j.values] < prio[allp.i.values]
    allp.loc[swap, ["i", "j"]] = allp.loc[swap, ["j", "i"]].values
    allp = (allp.groupby(["i", "j"], as_index=False)
            .agg(km=("km", "first"), found_by=("found_by", lambda s: "+".join(sorted(set(s))))))
    allp.insert(0, "a", att.gauge_id.values[allp.i])
    allp.insert(1, "b", att.gauge_id.values[allp.j])
    allp["area_ratio"] = att.area.values[allp.j] / att.area.values[allp.i]
    return allp.drop(columns=["i", "j"])


def resolve(att, pairs, cfg):
    """Mark each duplicate with the gauge it is dropped in favour of."""
    d = cfg["dedup"]
    prio = att.set_index("gauge_id").source.map(
        lambda s: d["source_priority"].get(s, d["source_priority"]["default"]))
    days = att.set_index("gauge_id").q_days
    rank = lambda g: (prio[g], -days[g], g)  # lower is better
    better = {}
    for a, b in zip(pairs.a, pairs.b):
        keep, drop = sorted([a, b], key=rank)
        better.setdefault(drop, set()).add(keep)
    # follow chains to a gauge that is not itself dropped
    def final(g, seen=()):
        if g not in better or g in seen:
            return g
        return final(min(better[g], key=rank), seen + (g,))
    return att.gauge_id.map(lambda g: final(g) if g in better else np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--refresh", action="store_true", help="recompute record_stats.csv")
    args = ap.parse_args()
    cfg = load_config()
    out = cfg["paths"]["catalog"]
    out.mkdir(parents=True, exist_ok=True)

    att = load_attributes(cfg)
    print(f"{len(att)} catchments:", att.source.value_counts().to_dict())

    stats_file = out / "record_stats.csv"
    if stats_file.exists() and not args.refresh:
        stats = pd.read_csv(stats_file, dtype={"gauge_id": str})
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            stats = pd.DataFrame(ex.map(record_stats, [(cfg, g) for g in att.gauge_id], chunksize=50))
        stats.to_csv(stats_file, index=False)
    att = att.merge(stats, on="gauge_id", validate="1:1")

    pairs = candidate_pairs(att, cfg)
    with ProcessPoolExecutor(args.workers) as ex:
        chk = pd.DataFrame(ex.map(flow_check, [(cfg, a, b) for a, b in zip(pairs.a, pairs.b)], chunksize=20))
    pairs = pd.concat([pairs, chk], axis=1)
    d = cfg["dedup"]
    checkable = pairs.overlap_days >= d["min_overlap_days"]
    pairs["flow_agrees"] = (pairs.flow_r >= d["min_flow_r"]).where(checkable)
    pairs.to_csv(out / "duplicate_pairs.csv", index=False)

    att["duplicate_of"] = resolve(att, pairs, cfg)
    att.to_csv(out / "catchments.csv", index=False)

    print("candidate pairs:", pairs.groupby("found_by").size().to_dict())
    print("flow check (diagnostic): checkable", int(checkable.sum()),
          " r >= min_flow_r:", int((pairs.flow_agrees == True).sum()),
          " r < min_flow_r:", int((pairs.flow_agrees == False).sum()))
    dropped = att[att.duplicate_of.notna()]
    print("dropped as duplicates:", len(dropped), dropped.source.value_counts().to_dict())
    print("kept:", int(att.duplicate_of.isna().sum()))


if __name__ == "__main__":
    main()
