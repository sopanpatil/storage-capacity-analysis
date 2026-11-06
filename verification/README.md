# Model implementation checks

Each model implementation used in this study, compared with a reference. The R references need
R with `airGR` 1.7.9 and `nwsrfsr` 1.0.3 installed from CRAN; the R scripts look for them in
`~/R/lib`, so edit `.libPaths` if yours are elsewhere. `nwsrfsr` can be installed from the copy in this repository with
`R CMD INSTALL -l ~/R/lib third_party/nwsrfsr-1.0.3`.

| Implementation | Reference | Largest difference | Result file |
|---|---|---|---|
| GR6J-CemaNeige, `grsuite` 1.1.1 | `airGR` 1.7.9 | Qsim 1e-13 mm; stores 1.5e-11 mm; SWE 1.6e-12 mm | `grsuite_vs_airgr.csv` |
| GR4J-CemaNeige, `grsuite` 1.1.1 | `airGR` 1.7.9 | Qsim 1.5e-13 mm; stores 7.8e-12 mm; SWE 1.1e-12 mm | `grsuite_gr4j_vs_airgr.csv` |
| SAC-SMA + SNOW-17, `models/sacsma/sacsnow.py` | `nwsrfsr` 1.0.3 | Channel inflow 5e-14 mm; AET 5e-15 mm; stores 5e-13 mm; SWE 5e-12 mm | `wrapper_vs_nwsrfsr.csv`, `wrapper_vs_nwsrfsr_jasmin.csv` |
| Gamma unit hydrograph, `sacsnow.uh()` | `nwsrfsr::uh()` | 5e-14 mm | `uh_wrapper_vs_nwsrfsr.csv` |
| FLEX-type model, `src/flex_model.py` | Independent calculations (below) | See below | Printed by the script |

All differences except the FLEX-type model's are at machine precision.

## GR6J and GR4J

`airGR`'s example catchment L0123002 (snow-influenced), 5 elevation layers, 1991-2001 after a
1990 warm-up, 25 random parameter sets spanning wide ranges.

```
Rscript run_airgr.R                                   # in an empty folder
python verification/run_grsuite.py <that folder>
Rscript run_airgr_gr4j.R                              # the same for GR4J
python verification/run_grsuite_gr4j.py <that folder>
```

## SAC-SMA + SNOW-17

The Fortran in `nwsrfsr` is compiled unchanged into a standalone library
(`models/sacsma/build.sh`) and called from Python (`models/sacsma/sacsnow.py`). The test uses
`nwsrfsr`'s NRKW1 example (2 zones, 1979-2022) at 6-hourly and daily steps, with the default
parameters plus 10 sets with the five capacity parameters scaled by 0.3 to 3 times.

```
Rscript run_nwsrfsr.R                                 # writes sac/ (174 MB, about 15 s)
python verification/run_wrapper.py verification models/sacsma
```

The NWS routines compute internally in single precision, and the wrapper reproduces that
exactly. The `nwsrfsr` wrapper spins up initial SAC-SMA states automatically. The two result
files come from the machine the code was first checked on and from JASMIN.

## Gamma unit hydrograph

`sacsnow.uh()` calls `nwsrfsr`'s DUAMEL Fortran (2-parameter gamma unit hydrograph, shape and
scale in days, single precision) from the same library, and reproduces the one-step shift of
R's `uh()` for start-of-step forcing. The reference covers 9 shape and scale sets, daily and
6-hourly, with and without the shift. It is tracked in `uh/`, so this check needs no R:

```
python verification/run_uh_wrapper.py verification models/sacsma
```

`Rscript run_nwsrfsr_uh.R` regenerates `uh/`.

## FLEX-type model

No package implements this exact structure (Liang et al. 2024, Table 2), so the model is tested
against independent calculations. The script reads one catchment's Caravan forcing:

```
python verification/run_flex_checks.py
```

- The Xinanjiang closed form against numerical integration of its capacity distribution, 2,000
  cases: worst error 1.7e-6 of Sumax, limited by the quadrature.
- Water balance, 500 parameter sets over 14,610 days: worst residual 7.4e-15 of total P.
- A separate plain-Python implementation written from the equations, 20 sets: worst daily
  difference in Q 8.9e-16 mm.
- Limits with known answers: bucket at beta = 0, full and empty store, recession, lag weights
  and degree-day melt.

## HBV

`hbv-model` has its own tests in its repository
([sopanpatil/hbv-model](https://github.com/sopanpatil/hbv-model)), including water-balance
closure.
