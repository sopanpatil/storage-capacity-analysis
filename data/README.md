# data

Per-catchment tables behind the paper. Together with [`../robustness/`](../robustness/), these
are what every figure and Supporting Information table is built from, so a clone rebuilds them
without the raw data or the calibration runs. Every column is defined in
[`DATA_DICTIONARY.md`](DATA_DICTIONARY.md).

| Folder | Contents | Written by (`src/`) |
|---|---|---|
| `catalog/` | All Caravan and GRDC-Caravan catchments, duplicates, record statistics, and the 6,144 selected catchments | `build_catalog.py`, `record_coverage.py`, `select_pilot.py` |
| `benchmarks/` | Independent capacity estimates per catchment | `benchmark_*.py` |
| `calibration/` | Split, optimum parameters, KGE and behavioural count per catchment and model | `summarise_calibration.py` |
| `rq1/` | Identifiability of the capacity parameter | `identifiability.py` |
| `rq2/` | Calibrated capacity against the benchmarks | `rq2_parameter.py` |
| `rq3/` | Simulated deficit, storage-free controls, and their agreement with the benchmarks | `state_capacity.py`, `rq3_*.py` |
| `rq4/` | Variation with aridity, snow and water balance | `rq4_where.py` |
| `classification/` | Identifiability and consistency class of each catchment-model pair | `classify.py` |
| `sensitivity/` | Results under other KGE thresholds | `kge_sensitivity.py` |
| `prior_tests/` | Designs of earlier studies reproduced on this sample | `prior_agreement_tests.py` |

## Sources

The tables hold values derived from the following datasets, which are cited in the paper's Open
Research statement. The raw data are not redistributed here, and attribution is owed to their
providers:

- Caravan v1.6 (doi:10.5281/zenodo.15529786) and GRDC-Caravan v0.6
  (doi:10.5281/zenodo.15349031), including their national source collections. Observed
  streamflow appears here only as record statistics.
- MOD16A2GF Collection 6.1 and ERA5-Land evaporation and snowfall, via Google Earth Engine.
- SSEBop version 5, from the USGS FEWS NET archive.
- SoilGrids 2.0 water retention, and the effective rooting depth of Yang et al. (2016).
- The capacity estimates of Stocker et al. (2023; doi:10.5281/zenodo.10885724) and Zhao et al.
  (2025; doi:10.5281/zenodo.14970062).

## Checksums

[`SHA256SUMS`](SHA256SUMS) lists every table in `data/` and `robustness/`, with paths relative
to the repository root. To check them, run this from the root:

```
sha256sum -c data/SHA256SUMS
```

## Licence

The tables in `data/` and `robustness/` are released under Creative Commons Attribution 4.0
International (CC BY 4.0), not the MIT licence that covers the code. Attribute them to the
paper and to this archive.
