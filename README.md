# storage-capacity-analysis

Analysis code and per-catchment results for *"Can calibrated soil storage capacity in
conceptual rainfall-runoff models be treated as catchment storage? Evidence from 6,144
Caravan catchments"*.

The study asks whether the storage capacity parameter of three widely used lumped models
(HBV, GR6J and SAC-SMA), calibrated on streamflow, can be read as a physical catchment
storage. It tests this in three ways across 6,144 catchments from
[Caravan](https://doi.org/10.1038/s41597-023-01975-w) and its GRDC extension:

1. **Identifiability**: how far the capacity parameter ranges across each model's
   behavioural parameter sets.
2. **Parameter level**: whether calibrated capacities agree with four independent
   estimates (climate-derived, soil-derived, and the products of Stocker et al. 2023 and
   Zhao et al. 2025).
3. **State level**: whether the storage a model actually uses agrees better than its
   parameter, using the same mass-curve deficit calculation on the model's own fluxes as on
   the climate data.

The code names these steps RQ1, RQ2 and RQ3, and the variation of the results with aridity,
snow and water balance RQ4.

## What you can run, and what it needs

| Goal | Needs | Time |
|---|---|---|
| Rebuild every figure and Supporting Information table | This repository only | About a minute |
| Rerun the summary steps (RQ2 onwards) from the per-catchment tables | This repository only (see [Pipeline](#pipeline)) | Minutes |
| Rerun everything from raw data | Caravan (about 96 GB), benchmark inputs (about 30 GB), a cluster, `hbv-model` | About 18 CPU minutes per catchment for calibration alone; outputs about 150 GB |

Figure and table scripts read only `data/` and `robustness/`:

```
python figures/fig1_design_sample.py
python figures/fig2_parameter_vs_climate.py
python figures/fig3_three_levels.py
python figures/fig4_identifiability_where.py
python figures/figS_si.py       # Figure S1 is figS_strata
python figures/si_tables.py     # Tables S1-S10, written to figures/si_tables.tex
```

Figures 1 to 4 and Figure S1 are written to `figures/` under the names of the scripts.

## Requirements

Python 3.12 (the results used 3.12.11). Install the dependencies with:

```
pip install -r requirements.txt
```

`requirements.txt` pins the versions used. Three models are involved, and each runs with its
own standard setup:

| Model | Capacity compared | Code |
|---|---|---|
| HBV | FC | [`hbv-model`](https://github.com/sopanpatil/hbv-model) v1.2.0, doi:[10.5281/zenodo.23245621](https://doi.org/10.5281/zenodo.23245621). Not on PyPI; clone it next to this repository at tag `v1.2.0` |
| GR6J-CemaNeige | X1 | `grsuite` 1.1.1, from PyPI. An unchanged copy of the source is in [`third_party/`](third_party/) |
| SAC-SMA + SNOW-17 + gamma unit hydrograph | UZTWM + LZTWM | The NWS Fortran from `nwsrfsr` 1.0.3, in [`third_party/`](third_party/), called through [`models/sacsma/sacsnow.py`](models/sacsma/sacsnow.py). Build with `bash models/sacsma/build.sh` (needs `gfortran`) |

The robustness runs also use GR4J (`grsuite`) and a FLEX-type model
([`src/flex_model.py`](src/flex_model.py)).

## Configuration

Paths and settings are in [`config.yaml`](config.yaml). Relative paths are relative to the
repository root, and raw data defaults to `external/`, which git ignores. To keep data
elsewhere, put overrides in a `config.local.yaml` beside it, with the same structure:

```yaml
paths:
  caravan: /data/caravan_v1.6/Caravan-nc
  runs: /data/storage-capacity/runs
```

## Data

| Tier | Where it comes from |
|---|---|
| Raw Caravan v1.6 and GRDC-Caravan v0.6 | `src/download_caravan.sh`, then `src/extract_caravan.sh`. Not redistributed here |
| Benchmark inputs | `src/prepare_benchmarks.py`, `src/download_ssebop.sh`, `src/gee_export.py` and the SoilGrids URLs in `src/benchmark_soilgrids.py`. Not redistributed here |
| Calibration and Monte Carlo outputs | Written to `<runs>` by `src/calibrate.py`. Regenerable, so not carried here |
| **Per-catchment tables** | **Tracked, in [`data/`](data/) and [`robustness/`](robustness/).** Every result in the paper comes from these |

Every column is defined in [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md), and
[`data/README.md`](data/README.md) gives the sources and checksums.
[`data/calibration/calibration_summary.csv`](data/calibration/calibration_summary.csv) holds
the optimum parameter set of every catchment and model, so any optimum can be rerun without
calibrating again.

## Pipeline

Scripts are in `src/`, and each describes its inputs, outputs and options in its docstring.
Run them from the repository root. `slurm/` holds the job scripts used on the JASMIN LOTUS
cluster, which show the task layout; change the account and partition for another system.
"Raw" below means the step needs the raw data or the `<runs>` outputs.

**Sample**

| Script | Writes | Needs |
|---|---|---|
| `build_catalog.py` | `data/catalog/{catchments,duplicate_pairs,record_stats}.csv` | Raw |
| `record_coverage.py` | `data/catalog/annual_valid_fraction.csv` | Raw |
| `select_pilot.py` | `data/catalog/{screened_catchments,pilot_catchments,pilot_screening_by_cell}.csv` | Tables |

**Benchmarks**

| Script | Writes | Needs |
|---|---|---|
| `prepare_benchmarks.py`, `download_ssebop.sh`, `gee_extract.py`, `gee_export.py` | Input rasters and Earth Engine exports in `<benchmarks>` | Earth Engine project for the last two |
| `benchmark_ssebop.py` | Catchment means in `<benchmarks>` | Raw |
| `benchmark_soilgrids.py`, `benchmark_static.py` | `data/benchmarks/{soilgrids_whc,static}_*.csv` | Raw |
| `benchmark_climate.py` | `data/benchmarks/climate_*.csv` | Raw |
| `benchmark_water_balance.py` | `data/benchmarks/water_balance_*.csv` | Raw |

**Calibration**

| Script | Writes | Needs |
|---|---|---|
| `calibrate.py` (with `model_setups.py`) | `<runs>/calibration/<gauge_id>/` | Raw |
| `summarise_calibration.py` | `data/calibration/calibration_summary.csv` | Raw |

**Analysis**

| Script | Writes | Backs | Needs |
|---|---|---|---|
| `identifiability.py` | `data/rq1/` | Identifiability | Raw |
| `rq2_parameter.py` | `data/rq2/` | Parameter-level agreement, Figure 2 | Tables |
| `state_capacity.py` (with `model_fluxes.py`, `mass_curve.py`) | `data/rq3/state_capacity_*` | Simulated deficit | Raw |
| `rq3_control.py`, `rq3_forcing_control.py` | `data/rq3/{control,forcing_control}_*` | Storage-free controls | Raw |
| `rq3_benchmarks.py` | `data/rq3/state_vs_benchmarks_summary_*` | State-level agreement, Figure 3 | Tables |
| `rq3_coupling.py` | `data/rq3/coupling*` | Text S2, Table S4 | Tables |
| `classify.py` | `data/classification/` | Identifiability classes, Table S8 | Tables |
| `rq4_where.py` | `data/rq4/` | Figure 4 | Tables |
| `kge_sensitivity.py` | `data/sensitivity/` | Text S1, Tables S2 and S3 | Tables |
| `prior_agreement_tests.py` | `data/prior_tests/` | Text S3, Tables S5 to S7 | Raw |

**Robustness runs** (Text S5, Tables S9 and S10): the `robustness_*.py` scripts, described in
[`robustness/README.md`](robustness/README.md).

## Model verification

[`verification/`](verification/) compares each model implementation with a reference, and
its README gives the results and how to repeat them. GR6J, GR4J and SAC-SMA agree with their R
references to machine precision. The FLEX-type model has no reference implementation, so it
is tested against independent calculations.

## Licence and citation

Code: MIT (see [`LICENSE`](LICENSE)). Tables in `data/` and `robustness/`: CC BY 4.0 (see
[`data/README.md`](data/README.md)). `third_party/` keeps the licences of its authors (see
[`third_party/README.md`](third_party/README.md)), including GPL-2.0-or-later for `grsuite`.
If you use this work, cite the software and the paper, as [`CITATION.cff`](CITATION.cff) sets
out.
