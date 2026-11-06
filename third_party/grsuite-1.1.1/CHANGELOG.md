# Changelog

All notable changes to GRsuite are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - 2026-09-05

### Fixed

- `series_aggreg` returned an empty result for a monthly or yearly series
  dated by day (the first of each month, as airGR's `POSIXct` dates, or a
  pandas `DatetimeIndex`): the regular grid was built with a 28- to 31-day
  step. Such series are now handled in their calendar unit. Timezone-aware
  dates are rejected with a clear message.
- The `SCA` and `SWE` criteria crashed with a `KeyError` on the coupled
  CemaNeige models. They now follow `CreateInputsCrit` / `.ErrorCrit`:
  `obs` is a list with one observed vector per elevation layer (an already
  averaged vector is accepted too), averaged and compared with the mean of
  the simulated layers' `Gratio` / `SnowPack`; `id_layer` selects the
  layers. Validated against airGR (`tests/data/snow_crits`).
- `RunOptions` applied airGR's default store levels (0.3, 0.5, ...) even
  when `ini_states` was given, so a run could not be resumed from a previous
  `StateEnd`. As in `CreateRunOptions`, the defaults now apply only when
  neither `ini_states` nor `ini_res_levels` is given; an explicit
  `ini_res_levels` still overrides the store levels of `ini_states`.
- Composite criteria followed a sign convention of their own: `error_crit`
  returned the weighted sum multiplied by the sign convention (a composite
  of NSE came out negative), with `multiplier = +1`, `crit_best_value = NaN`
  and the *weights* in `sub_crit_values`. It now returns what airGR's
  `ErrorCrit` returns: the weighted mean of the raw sub-criteria
  (`CritValue`), `multiplier = -1`, `crit_best_value = 1`, and, as
  `CritCompo$MultiCritValues` / `MultiCritNames` / `MultiCritWeights`, the
  sub-criteria values in `sub_crit_values`, their names, and the normalised
  weights in a new `weights` attribute. As `CreateInputsCrit`, a composite
  refuses the RMSE. Calibrations are unchanged (they minimise
  `crit_value * multiplier`); `Simulation.score` and `CalibratedModel.score`
  no longer need to flip the sign.
- `Catchment.period_index(None)` now skips one calendar year (twelve steps
  for GR2M, one for GR1A, 365 or 366 days, 8760 or 8784 hours) instead of
  365 days divided by the step, and works with `datetime64[M]` / `[Y]`
  dates.
- `Catchment.calibrate` and `Simulation.score` accept composite criteria
  given as `("NSE", "log")` pairs or bare names, not only 3-tuples.
- On an hourly catchment, a period end given as a date (`"2005-12-31"`)
  now covers the whole day instead of stopping at midnight.
- `run_model_cemaneige` (the snow module alone) used the daily state layout
  and the daily conversion factor with hourly run options; `hourly` is now
  inferred from the model of the run options. Its `StateEnd` has the full
  state-vector layout and can be passed back as `ini_states`.
- `pe_oudin` returned 0 for a missing temperature; like `PE_Oudin` it
  returns NaN and warns.
- EnKF: when the ensemble had collapsed and the observation error was zero
  (observed discharge of zero and a zero variance threshold, typical of dry
  catchments with many zero-flow days), the Kalman gain was 0 / 0 and the
  analysed states became NaN, silently reset to 0 at the next step. The
  update is now skipped at such steps and a warning reports how many there
  were (airGRdatassim stops on a singular system).
- `InputsPert` / `Catchment.assimilate(perturb=True)`: on the CemaNeige
  models the perturbed precipitation never reaches the snow module, whose
  elevation layers are computed from the unperturbed series, so only the
  potential evapotranspiration perturbation acts. This is airGRdatassim
  0.1.4's own behaviour (`CreateInputsPert` builds the layers once, and
  `RunModel_DA` only swaps `Precip` and `PotEvap` per member), so GRsuite
  keeps it, but `run_model_da` now warns about it, and
  `InputsPert(..., perturb_layers=True)` /
  `Catchment.assimilate(..., perturb_layers=True)` redo the elevation
  extrapolation member by member (`InputsPert.layer_precip_ens`): a GRsuite
  extension with no R reference. See docs/VALIDATION.md section 9.
- `CalibOptions(..., is_int_store=True)` ignored the interception store for
  `CemaNeigeGR5H` and `CemaNeigeGR5HHyst` and screened the grid without
  interception; the coupled models now get airGR's interception grid.
- `MeanAnSolidPrecip` was computed lazily at the first run, from whichever
  `InputsModel` that run received; `RunOptions` now computes it at creation
  from its own `inputs_model`, as `CreateRunOptions` does, and accepts it as
  an argument (`mean_an_solid_precip`).
- `InputsModelSD` rejected a one-dimensional `q_upstream` for a single
  upstream sub-catchment.
- `run_model_lag` took a daily time step whatever the series; the step is
  now inferred from the dates of `run_options.inputs_model` when
  `time_step_seconds` is not given, and a zero, negative or missing
  velocity is refused.
- `data_alti_extrapolation_valery` crashed or returned all-NaN series
  without hypsometry; as `DataAltiExtrapolation_Valery`, it now uses a
  single layer (with a warning when more were asked for) and passes the
  series through unchanged, and it rejects a malformed `hypso_data` (101
  quantiles).

### Added

- `Catchment.inputs(model)`: the `InputsModel` the high-level API feeds to
  a model, to reach the airGR-mirroring functions with the same inputs
  (examples 04, 05 and 07 used the private `_inputs`).
- `Catchment` refuses a model whose time step does not match the series
  (GR4H on daily data, GR4J on hourly data, GR2M on daily data) and a
  `warmup` period that does not end right before `period`.
- `Simulation.bias()` raises a clear error without observed discharge.
- Input validation, after `CreateInputsModel` / `CreateRunOptions`:
  `InputsModel` requires strictly increasing `datetime64` dates and warns,
  as airGR does, about missing or negative precipitation and potential
  evapotranspiration (a single NaN used to blank the rest of the
  simulation silently); `RunOptions` checks that the period indices are
  integers inside the series and continuous (airGR accepts both and fails
  later), warns as airGR when the warm-up does not end right before the
  run or when `ini_states` and `ini_res_levels` are both given, and checks
  the length of `ini_res_levels`; the model wrappers
  refuse a state vector shorter than the model's layout instead of letting
  the kernels read and write past its end; `Catchment` requires
  `pot_evap`, `datetime64` dates and a 101-value hypsometry.
- `tests/test_api.py`: the high-level API is compared to the airGR-mirroring
  API (simulation, calibration, criteria, periods, assimilation) on the
  daily, hourly, snow, monthly and annual catchments; tests for resuming a
  run from `StateEnd`, the composite criterion outputs, the snow criteria,
  the aggregation of monthly series, the degenerate EnKF step and the
  CemaNeige perturbation.
- airGR references for CemaNeige at the hourly time step (`CemaNeigeGR4H`,
  with and without hysteresis, `CemaNeigeGR5H` with and without the
  interception store, layer by layer for `CemaNeigeGR4H`), for `Imax()`
  (`imax_estimate` is now compared to airGR, not to a constant) and for the
  SCA / SWE criteria; `tools/oracle/export_fixtures.R` produces them, on
  L0123003 with a deterministic synthetic temperature since the dataset
  ships none. The existing 112 reference files were regenerated by the
  same run and came out identical.
- `MANIFEST.in`: the source distribution now ships the complete test suite
  (`conftest.py` and `tests/data`), the docs, the examples and the R
  exporters; it used to ship the test modules alone, which could not run.

### Changed

- The `Typing :: Typed` classifier is dropped: the package ships no
  `py.typed` marker and is not annotated.

## [1.1.0] - 2026-08-19

### Added

- Data assimilation for the daily models (GR4J, GR5J, GR6J, with or without
  CemaNeige), ported from airGRdatassim 0.1.4: perturbed forcing ensembles
  (`InputsPert`), the ensemble Kalman filter and the particle filter
  (`run_model_da`), a high-level `Catchment.assimilate()` wrapper and
  `examples/09_data_assimilation.py`. Validated value by value against R
  reference outputs (`tools/oracle/export_da_fixtures.R`), with the
  reference's own random draws replayed exactly.
- One deliberate deviation from airGRdatassim 0.1.4, documented in
  docs/VALIDATION.md section 9: the official package's EnKF is a silent
  no-op with the documented character `StateEnKF` (`which(StateEnKF == 1)`
  selects nothing, so the Kalman update never runs). GRsuite implements the
  documented behaviour — the names in `state_enkf` select the state
  variables to update.

## [1.0.0] - 2026-08-15

First public release.

### Added

- The full airGR 1.7.9 model suite, translated from the Fortran and R sources
  and validated bit-for-bit against airGR: GR4J, GR5J, GR6J (daily), GR4H and
  GR5H with or without the interception store (hourly), GR2M (monthly), GR1A
  (annual).
- The CemaNeige snow module, with or without linear hysteresis, coupled to
  every daily and hourly model, with Valéry elevation-band extrapolation and
  daily temperature gradients.
- Michel's calibration algorithm (grid screening then steepest descent),
  reproducing airGR's iteration counts exactly.
- Error criteria: NSE, KGE, KGE′, RMSE, weighted composites, and the
  `sqrt`, `log`, `inv`, `sort`, `boxcox` and power transformations.
- Utilities: Oudin potential evapotranspiration, interception capacity
  estimation, time-series aggregation, semi-distributed routing
  (`run_model_lag`).
- A high-level API (`Catchment`, `calibrate`, `split_sample`, `evaluate`)
  on top of the airGR-mirroring one.
- A test suite of 159 tests comparing GRsuite to reference outputs produced by
  airGR 1.7.9, shipped as gzipped fixtures under `tests/data`.
- Validation material: campaign over 100 CAMELS-FR catchments, reported in
  [docs/VALIDATION.md](docs/VALIDATION.md), and the numerical fidelity notes
  in [docs/FIDELITY.md](docs/FIDELITY.md).
- `docs/make_validation_tables.py`: recomputes the whole airGR comparison from the
  committed reference files and writes one row per compared quantity to
  `docs/data/series_deviations.csv` — 502 quantities, 1 230 966 values, worst
  deviation 2.0 × 10⁻¹³. Section 2.1 of the validation report is its output, so
  the figures quoted there can be audited rather than trusted.
- [docs/REFERENCES.md](docs/REFERENCES.md): the published source of every model,
  snow routine, criterion, transformation and algorithm implemented here, with
  DOIs, plus the reference data used for validation. The READMEs, `NOTICE`,
  `CITATION.cff` and the module docstrings point to it.
