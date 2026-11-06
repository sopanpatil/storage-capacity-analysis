"""Select the pilot catchments, stratified by aridity (FAO-PM) x snow fraction.

Reads catalog/catchments.csv (from build_catalog.py) and applies the final pre-calibration
screen in config.yaml `screen`. Picks `per_cell` catchments per
aridity x snow cell at random (fixed seed), spreading picks across source datasets within a cell.

Writes catalog/pilot_catchments.csv, catalog/screened_catchments.csv (every catchment passing the
screen) and catalog/pilot_screening_by_cell.csv (catchments per cell before and after screening).

Usage: python src/select_pilot.py
"""
import numpy as np
import pandas as pd

from caravan_io import load_config


def main():
    cfg = load_config()
    p, sc = cfg["pilot"], cfg["screen"]
    out = cfg["paths"]["catalog"]
    cat = pd.read_csv(out / "catchments.csv", dtype={"gauge_id": str, "nat_id": str})
    cat = cat[cat.duplicate_of.isna()].copy()

    aridity = cat[cfg["forcing"]["aridity_attr"]]
    cat["aridity_bin"] = pd.cut(aridity, p["aridity_bins"], right=False)
    cat["snow_bin"] = pd.cut(cat.frac_snow, p["snow_bins"], right=False, include_lowest=True)
    # frac_snow == 1.0 would fall outside right=False bins; none exist (max 0.9), but check
    assert cat.snow_bin.notna().all() and cat.aridity_bin.notna().all()

    screens = {
        "area": cat.area.between(*sc["area_km2"]),
        "record": cat.complete_years >= sc["min_complete_years"],
        "grdc_quality": ~cat.grdc_quality.isin(sc["exclude_grdc_quality"]),
        "runoff_ratio": cat.runoff_ratio < sc["max_runoff_ratio"],
        "glacier": cat.gla_pc_sse.fillna(0) <= sc["max_glacier_pct"],
        "regulation": cat.dor_pc_pva.fillna(0) < sc["max_dor_x10"],
        "no_negative_flow": cat.q_negative_days == 0,
    }
    passed = np.logical_and.reduce(list(screens.values()))
    print(f"{len(cat)} catchments after deduplication; {int(passed.sum())} pass the screen")
    for k, v in screens.items():
        print(f"  fail {k}: {int((~v).sum())}")

    cells = ["aridity_bin", "snow_bin"]
    table = pd.DataFrame({"all": cat.groupby(cells, observed=False).size(),
                          "screened": cat[passed].groupby(cells, observed=False).size()})
    table["survival"] = (table.screened / table["all"]).round(2)
    table.to_csv(out / "pilot_screening_by_cell.csv")
    cat.loc[passed, ["gauge_id", "source", "aridity_bin", "snow_bin"]].to_csv(
        out / "screened_catchments.csv", index=False)
    print(table.to_string())

    rng = np.random.default_rng(p["seed"])
    pool = cat[passed].sample(frac=1, random_state=rng)
    # round-robin across sources within each cell: 1st of each source, then 2nd, ...
    pool["src_rank"] = pool.groupby(cells + ["source"], observed=True).cumcount()
    pool = pool.sort_values("src_rank", kind="stable")
    pilot = pool.groupby(cells, observed=True).head(p["per_cell"])

    cols = ["gauge_id", "source", "country", "gauge_name", "gauge_lat", "gauge_lon", "area",
            cfg["forcing"]["aridity_attr"], "frac_snow", "aridity_bin", "snow_bin",
            "complete_years", "q_first", "q_last", "runoff_ratio", "zero_flow_frac"]
    pilot = pilot.sort_values(cells + ["gauge_id"])[cols]
    pilot.to_csv(out / "pilot_catchments.csv", index=False)
    print(f"\n{len(pilot)} pilot catchments:", pilot.source.value_counts().to_dict())
    short = table.screened[table.screened < p["per_cell"]]
    if len(short):
        print("cells with fewer than per_cell candidates:", short.to_dict())


if __name__ == "__main__":
    main()
