# Data dictionary

Every file in `data/` and `robustness/`, and what its columns mean. Columns that recur are
defined once, under [Conventions](#conventions) and [Recurring columns](#recurring-columns).

## Conventions

- **File names.** `_screened_catchments` means the 6,144 selected catchments.
- **`gauge_id`** is `<source>_<native id>`, the join key in every file. **`source`** is one of
  `camels`, `camelsaus`, `camelsbr`, `camelscl`, `camelsgb`, `hysets`, `lamah` (Caravan) or
  `GRDC` (GRDC-Caravan).
- **`model`** is `HBV`, `GR6J` or `SACSMA`, and in `robustness/` also `GR4J` or `FLEX`.
- **Capacity** is the parameter compared: FC in HBV, X1 in GR6J and GR4J, UZTWM + LZTWM in
  SAC-SMA, Sumax in FLEX. Capacities, deficits and benchmarks are in **mm**.
- **A pair** is one catchment and one model. A pair **passes** when its optimum reaches
  KGE >= 0.5 in both calibration and validation.
- **A behavioural set** is a Monte Carlo parameter set with calibration KGE >= 0.5, unless a
  file gives another threshold.
- **S_R,T** is the mass-curve storage deficit at a return period of T years, from a Gumbel fit
  to annual maxima (`src/mass_curve.py`). 20 years is the main value, over 2003-2020.
- **Ranges.** `_lo` and `_hi` are a bootstrap 90% interval. `p5`, `p50` and `p95` are
  percentiles over the behavioural sets.
- **Empty means not available**, for example no benchmark for that catchment or too few
  behavioural sets. It never means zero.

## Recurring columns

**Benchmarks**, per catchment:

| Column | Meaning |
|---|---|
| `climate` | Climate-derived S_R,20: mean of the MOD16 and SSEBop evaporation products, where the catchment is eligible (snow fraction <= 0.2, aridity <= 2) and satellite E <= P |
| `climate_T10`, `climate_T40` | The same at 10 and 40 years |
| `climate_incl_EgtP` | The same, including catchments with E > P |
| `climate_mod16`, `climate_ssebop` | The same with one evaporation product |
| `soil` | Effective rooting depth x SoilGrids water-holding capacity |
| `stocker`, `zhao` | The estimates of Stocker et al. (2023) and Zhao et al. (2025) |
| `gao` | Water-balance benchmark as in Gao et al. (2014), which shares observed streamflow with the calibration |
| `env_lo`, `env_hi`, `n_benchmarks` | The benchmark range: minimum and maximum of `climate`, `soil`, `stocker` and `zhao`, and how many are available |
| `env_lo_nosoil`, `env_hi_nosoil`, `n_benchmarks_nosoil` | The same without `soil` |

**Identifiability**, per pair:

| Column | Meaning |
|---|---|
| `capacity_opt` | Capacity of the optimum |
| `opt_at_bound` | The optimum's capacity parameter or parameters within 1% of the prior range from a bound |
| `kge_cal`, `kge_val` | KGE of the optimum in calibration and validation |
| `passes` | See [Conventions](#conventions) |
| `n_behavioural`, `n_samples` | Behavioural sets, and Monte Carlo sets in total |
| `prior_version` | Label of the prior used for that model |
| `prior_p5`, `prior_p95` | Prior percentiles of capacity, from all Monte Carlo sets |
| `beh_p5`, `beh_p50`, `beh_p95` | Behavioural percentiles of capacity |
| `spread_abs`, `spread_abs_lo`, `spread_abs_hi` | log10(P95 / P5) of behavioural capacity, with its interval. Needs at least 10 behavioural sets |
| `spread_rel` | Behavioural (P95 − P5) / prior (P95 − P5) |
| `opt_position` | The optimum relative to the behavioural range: `below`, `inside` or `above` |

**Classes**: `class` is the main criterion, in which a pair is unidentifiable when `spread_abs`
exceeds `envelope_width`. The others are alternative criteria: `class_ci` (the lower bound of
the spread exceeds the width), `class_nosoil` (range without `soil`), `class_fixed10`
(P95 / P5 > 10) and `class_halfprior` (`spread_rel` > 0.5). Values are `fails_screen`,
`few_behavioural` (fewer than 10 sets), `not_classifiable` (fewer than 2 benchmarks),
`unidentifiable`, `consistent` and `inconsistent` (identifiable, with the behavioural range
overlapping the benchmark range or not).

**Agreement statistics**, in the summary tables:

| Column | Meaning |
|---|---|
| `level` | What is compared: `parameter` (capacity), `state` (simulated deficit) or `control` (storage-free deficit) |
| `benchmark` | A benchmark above, or `envelope` / `envelope_nosoil` for the benchmark range |
| `n` | Pairs compared |
| `spearman`, `spearman_lo`, `spearman_hi` | Rank correlation across catchments, with its interval |
| `bias_log10`, `bias_log10_q25`, `bias_log10_q75` | Median log10(level / benchmark), with its interquartile range |
| `n_coverage`, `coverage` | Pairs with at least 10 behavioural sets, and the share of them whose behavioural P5-P95 contains the benchmark or overlaps the range |
| `opt_inside` | Share of optima inside the benchmark range |

**`stratum`** is an aridity and snow-fraction cell, for example
`aridity [0.5, 1.0), snow [0.05, 0.2)`. **Class counts** (`fails_screen`, `few_behavioural`,
`not_classifiable`, `unidentifiable`, `consistent`, `inconsistent`) count or share pairs per
class, and `n` is the total.

## data/catalog

**`catchments.csv`**: all 21,655 catchments, one row each.

| Column | Meaning |
|---|---|
| `gauge_name`, `country`, `gauge_lat`, `gauge_lon`, `nat_id` | From the Caravan and GRDC attributes |
| `area` | Catchment area, km² |
| `aridity_FAO_PM`, `aridity_ERA5_LAND` | PET / P, with FAO Penman-Monteith or ERA5-Land PET. The study uses FAO-PM |
| `frac_snow`, `high_prec_dur`, `high_prec_freq`, `low_prec_dur`, `low_prec_freq`, `moisture_index_*`, `p_mean`, `pet_mean_*`, `seasonality_*` | Caravan climate attributes, with mean P and PET in mm/day |
| `grdc_quality`, `grdc_comment` | GRDC-Caravan quality flag and comment |
| `ele_mt_sav`, `gla_pc_sse`, `dor_pc_pva`, `for_pc_sse` | HydroATLAS elevation (m), glacier cover (%), degree of regulation (% x 10) and forest cover (%) |
| `q_days`, `q_first`, `q_last`, `complete_years` | Days of valid streamflow, record start and end, and complete years in 1981-2023 |
| `q_negative_days`, `zero_flow_frac`, `runoff_ratio` | Days with negative flow, share of days with no flow, and long-term Q / P |
| `p_mean_qdays`, `pet_mean_qdays` | Mean P and PET over the days with streamflow, mm/day |
| `duplicate_of` | The gauge this one is dropped in favour of; empty if kept |

| File | Contents |
|---|---|
| `record_stats.csv` | The `q_*`, `complete_years`, `zero_flow_frac`, `runoff_ratio` and `*_qdays` columns above |
| `annual_valid_fraction.csv` | Share of valid streamflow days per calendar year, one column per year from 1950 |
| `duplicate_pairs.csv` | Candidate duplicates: gauges `a` and `b`, distance `km`, `found_by` (`id`, `spatial` or `id+spatial`), `area_ratio`, and the flow check `overlap_days`, `flow_r`, `flow_mean_ratio`, `flow_agrees` |
| `screened_catchments.csv` | The 6,144 selected catchments: `gauge_id`, `source`, `aridity_bin`, `snow_bin` |
| `pilot_catchments.csv` | 48 catchments, stratified by aridity and snow, used to test the pipeline before the full run. They are part of the full sample |
| `pilot_screening_by_cell.csv` | Catchments per aridity and snow cell: `all`, `screened` and `survival` (their ratio) |

## data/benchmarks

**`climate_screened_catchments.csv`**:

| Column | Meaning |
|---|---|
| `eligible` | Snow fraction <= 0.2 and aridity <= 2 |
| `mod16_valid_frac`, `ssebop_valid_frac` | Mean share of valid pixels over the catchment |
| `products` | Evaporation products that count: valid fraction at least 0.5 |
| `p_mean`, `era5land_e_mean` | Mean ERA5-Land P and E, 2003-2020, mm/day |
| `mod16_e_mean`, `ssebop_e_mean`, `ens_e_mean` | Mean E per product and for their mean (`ens`), mm/day |
| `mod16_sr_*`, `ssebop_sr_*`, `ens_sr_*` | S_R at 10, 20 and 40 years |
| `ens_sr_max` | Largest annual maximum deficit in the period |
| `ens_lateral_inflow` | Mean E exceeds mean P, so a lateral inflow was added |
| `ens_start_month`, `ens_n_years` | Start of the hydrological year (month with the lowest mean deficit), and complete years |
| `ens_sr_20_calendar`, `ens_n_years_calendar` | The same with calendar years |
| `ens_snow_sr_*`, `snowfall_frac` | Snow-aware variant with rain + snowmelt as inflow, and the snowfall share of P |

| File | Columns |
|---|---|
| `static_screened_catchments.csv` | `rooting_depth_m`, `zhao_sr_mm`, `stocker_scwdx80_mm`, each with `*_valid_frac`; `whc_0-200cm_mm`; `soil_sr_mm`; `soil_sr_excluded` (North American gauges north of 60° N, where Yang et al. call the deep values unreasonable; the value is kept but not used) |
| `soilgrids_whc_screened_catchments.csv` | Water-holding capacity per SoilGrids layer and for 0-200 cm, mm; `min_valid_frac` |
| `water_balance_screened_catchments.csv` | `eligible`; `q_days` and `q_frac` (days and share of 2003-2020 with streamflow); `p_mean`; `runoff_ratio`; `e_wb` (P (1 − Q/P), mm/day); `ens_e_ratio`, `mod16_e_ratio`, `ssebop_e_ratio` (satellite E / `e_wb`) |

## data/calibration

**`calibration_summary.csv`**: one row per pair.

| Column | Meaning |
|---|---|
| `cal_start`, `cal_end`, `val_end` | Split dates: calibration is the first two thirds of valid days |
| `n_cal_days`, `n_val_days` | Valid days per period |
| `cal_aridity`, `val_aridity` | Aridity per period |
| `capacity`, `kge_cal`, `kge_val`, `n_behavioural` | As `capacity_opt` and the recurring columns |
| `seconds_optimise`, `seconds_monte_carlo` | Run time |
| `p_<NAME>` | The optimum's parameters under the model's own names. Empty for parameters of other models |

## data/rq1

**`identifiability_screened_catchments.csv`**: the identifiability columns, per pair.
`_beh0.3` and `_beh0.7` are the same with those behavioural thresholds.

## data/rq2

| File | Contents |
|---|---|
| `parameter_benchmarks_screened_catchments.csv` | Per passing pair: `capacity_opt`, behavioural percentiles, `opt_position`, the benchmarks, `stratum` |
| `parameter_summary_screened_catchments.csv` | Agreement statistics per `model`, `stratum` and `benchmark` |

## data/rq3

| File | Contents |
|---|---|
| `state_capacity_screened_catchments_summary.csv` | Per pair: for the optimum `opt_capacity`, `opt_kge_cal`, `opt_sr_*`, `opt_sr_max`, `opt_start_month`, `opt_lateral_inflow`, `opt_aet_mean` and `opt_soil_in_mean` (mm/day), and `opt_store_deficit_20` (the 20-year level of capacity minus storage, a diagnostic bounded by the parameter); behavioural percentiles `beh_capacity_*`, `beh_sr_*_p*`, `beh_store_deficit_20_*` |
| `control_screened_catchments.csv` | Storage-free deficit per pair: `ctrl_sr_*` uses the model's soil input with FAO-PM PET scaled to its mean AET. Also `state_sr_20`, `ctrl_lateral_inflow`, `aet_mean`, `pet_mean` |
| `forcing_control_screened_catchments.csv` | Forcing-only deficit per catchment: `fctrl_sr_*` uses P with PET scaled to the Budyko evaporation `e_budyko`. Also `p_mean`, `pet_mean`, `lateral_inflow`. No model, streamflow or satellite data |
| `state_vs_benchmarks_summary_screened_catchments.csv` | Agreement statistics per `level`, `model`, `stratum` and `benchmark`, on the same pairs for each level |
| `coupling_screened_catchments.csv` | Per model: `rho_state_param`, `rho_state_control`, `rho_param_control`; `share_state_gt_param`, `median_state_over_param`; a rank regression of the state on the parameter and the control, `beta_param`, `beta_control`, `r2`, `unique_r2_param`, `unique_r2_control` |
| `coupling_partial_screened_catchments.csv` | Per model and benchmark: Spearman of `state` and `control`, and partial rank correlations `state_given_control`, `state_given_param`, `control_given_state`, each with `_lo` and `_hi` |

## data/rq4

| File | Contents |
|---|---|
| `profiles_screened_catchments.csv` | Per `covariate` (`aridity`, `snow`, `wb_ratio`), `bin` and model: `passes` and `behavioural_10` (shares passing, and also having at least 10 sets), `n_classified`, `spread_median`, `share_*` (class shares), and for each level and for the climate and Stocker benchmarks `n_`, `rho_` with `_lo` and `_hi`, `bias_` and `abserr_` (median absolute log10 error) |
| `covariates_screened_catchments.csv` | Per model, `outcome`, `spec` (`base`, or `+runoff_ratio`) and `covariate`: `spearman` with the covariate alone, and `effect`, `effect_lo`, `effect_hi`, the change in the outcome across the covariate's range in a joint model with the others held fixed. `wb_ratio` is satellite E / (P (1 − Q/P)) |

## data/classification

| File | Contents |
|---|---|
| `classification_screened_catchments.csv` | Per pair: `stratum`, `pilot` (in `pilot_catchments.csv`), `passes`, `n_behavioural`, `capacity_opt`, `beh_p5`, `beh_p95`, the spreads, `opt_position`, `env_lo`, `env_hi`, `n_benchmarks`, `envelope_width` (log10(`env_hi` / `env_lo`)), and the classes |
| `classification_summary_screened_catchments.csv` | Class counts per `group` (`all`, `pilot` or a stratum), `model` and `variant` (the class column used) |

## data/sensitivity

| File | Contents |
|---|---|
| `kge_classification_screened_catchments.csv` | Class counts per `screen` and `behavioural` KGE threshold, `rule` (`envelope`, `fixed10`, `halfprior`) and model, with `unidentifiable_share` of the assessed pairs |
| `kge_agreement_screened_catchments.csv` | Agreement statistics per `screen`, `behavioural`, `level`, model and benchmark |

## data/prior_tests

| File | Contents |
|---|---|
| `gao_benchmark_screened_catchments.csv` | The `gao` benchmark: `eligible`, `q_days`, `pe_mean` (effective precipitation), `t_mean` (transpiration), `gao_sr_10`, `gao_sr_20`, `gao_sr_40`, `gao_lateral_inflow` |
| `gao_agreement_screened_catchments.csv` | Agreement statistics with the `climate` or `gao` benchmark, per `level`, model and `sample` (`common`: the same pairs for both; `all_gao`: every pair with `gao`) |
| `cluster_agreement_screened_catchments.csv` | Agreement of medians over groups, per `level`, model, benchmark and `grouping` (`catchment`, `stratum`, `kmeans10`, `kmeans50`): `n_groups`, Spearman with its interval, `abs_bias_T20` and `abs_bias_bestT` (median absolute log10 bias at 20 years and at the best of 10, 20 and 40 per group), `share_T10_best`, `share_T40_best` |
| `constrained_prior_screened_catchments.csv` | Monte Carlo sets restricted to the benchmark's range, per passing pair and `band`: `narrow` (S_R,10 to S_R,40) or `wide` (S_R,5 to S_R,80), with limits `band_lo`, `band_hi`. `n_in` and `n_beh_in` count sets and behavioural sets in the band. Estimates: `band_best`, the in-band set with the best calibration KGE (`kge_cal_band_best`, `kge_val_band_best`, `band_best_passes`); `band_beh_median`; `band_prior_median`; `free_opt`; `free_beh_median`. Also `climate`, `kge_cal_best_all` (best over all sets) and `dkge_cal` (the difference) |
| `constrained_prior_summary_screened_catchments.csv` | Per model, `band`, `sample` (`in_band`: at least one set in the band; `beh_ge10`: at least 10 behavioural sets) and `estimate`: `n_pairs`, `share_in`, `share_any_beh`, `share_beh_ge10`, `median_n_in`, `median_dkge_cal`, `share_band_best_passes`, `n`, Spearman with its interval, `pearson`, `pearson_log`, `bias_log10` |

## robustness

See [`../robustness/README.md`](../robustness/README.md) for what each item tests.

**`subset.csv`** lists the 397 catchments: `gauge_id`, `stratum`, `climate_eligible`.
**`subset_by_class.csv`** gives `candidates`, `climate_eligible` and `selected` per stratum.

**`AD_per_catchment.csv`** and **`J_per_catchment.csv`**: per catchment and `variant`, the
identifiability columns, and `sr_20` (the optimum's simulated deficit) with `kgelog_cal` and
`kgelog_val` where available. `J_per_catchment.csv` adds the benchmarks and classes. `item`
gives the test.

**`A_`, `B_`, `D_` and `J_` summary tables**: per `variant`:

| Column | Meaning |
|---|---|
| `n`, `passes`, `n_beh_ge10` | Pairs, passing pairs, and those with at least 10 behavioural sets |
| `median_n_behavioural`, `median_spread_abs` | Medians over the pairs |
| `assessed_<class column>`, `unidentifiable_<class column>`, `consistent_<class column>` | Assessed pairs, and shares unidentifiable and consistent, under each criterion |
| `rho_param_<benchmark>`, `rho_sr_20_<benchmark>` | Spearman of the optimum's capacity and simulated deficit with the benchmark, with `_lo`, `_hi` and `n_` |
| `median_kge_cal`, `median_kge_val`, `median_kgelog_cal` | In `A_hbv_multiobjective.csv` only |

Variants: `main` (main run), `so` (KGE(Q) with the full SCE-UA budget), `mo_both`, `mo_mean` and
`mo_both_150k` (two objectives, with behavioural sets from both KGEs >= 0.5, their mean >= 0.5,
or both >= 0.5 in 150,000 samples); `10k` to `150k` (Monte Carlo samples); `FLEX_so`, `FLEX_mo`,
`FLEX_liang` (Sumax 30-700 mm), `FLEX_so_10k`, `HBV_so`, `HBV_mo`.

| File | Contents |
|---|---|
| `C_gr6j_optimiser.csv`, `S_sacsma_optimum.csv` | One row comparing a reference optimum (`_ref`: the main run) with a new one (`_new`: full-budget SCE-UA): median KGE and its change, `share_new_better_by_0_01`, `share_new_worse_by_0_01`, `spearman_capacity`, `median_abs_log10_ratio`, `share_within_10pct`, `share_beyond_factor_2`, `median_log10_ratio`, and `rho_<optimum>_<benchmark>` |
| `E_sacsma_reweight.csv` | Per catchment, the classification columns unweighted and, with `w_`, weighted to a uniform prior on UZTWM + LZTWM; `ess`, the effective sample size of the weights |
| `E_sacsma_reweight_classes.csv` | Class counts and `unidentifiable_share` per `variant` (the class column) and `prior` (`unweighted`, `weighted`) |
| `E_sacsma_reweight_levels.csv` | Per benchmark, median log10 ratio of `opt`, `beh_p50_unweighted` and `beh_p50_weighted`, and the shares of optima with `opt_uztwm_at_800`, `opt_lztwm_at_800`, `opt_either_at_800` |
| `F_hbv_lp.csv` | Per `quantity` (`FC`, `FC*LP`, `FC*(1-LP)`) and benchmark: `n`, `n_zero`, Spearman with its interval, `bias_log10`, `inside` (share inside the benchmark) |
| `G_common_subset.parquet` | Per catchment, model, `variant` (`min`: the smallest behavioural count of the three models; or a fixed `10` or `100`) and repetition `rep`: subsample size `n`, recomputed percentiles and spreads, benchmark range, classes |
| `G_common_subset_classes.csv` | Per `variant`, model and `criterion`: `n_catchments`, `median_n`, unidentifiable and consistent shares with all sets (`full_`) and with subsampling (`subsampled_`, with `_min` and `_max` over repetitions) |
| `G_common_subset_n.csv` | Distribution of the common count for `min` |
| `H_deficit_excess.csv` | Per model, pairs where S_R,20 exceeds capacity: `exceed`, `exceed_share`, split into `gumbel` (only the extrapolated level exceeds it) and `other_stores` (the observed maximum exceeds it); `max_srmax_over_capacity`; ratios `gumbel_median_sr20_over_capacity`, `gumbel_median_sr20_over_srmax`, `all_median_sr20_over_srmax` |
| `H_sacsma_free_water.csv` | SAC-SMA pairs whose observed maximum exceeds capacity: `capacity`, `free_capacity`, `sr_max`, `sr_max_rerun`, `sr_20`, tension and free-water deficits on the day of the maximum, `excess`, `bound_holds` (`sr_max` <= their sum), `excess_share_of_free_deficit` |
| `I_uh_zero_flow.csv` | `item` and `value`: share of prior routing parameters giving no routed flow, by the routine and by the stated rule, their agreement, and the working scale range (days) per shape |
| `I_uh_fix_test.csv` | Three catchments, per `group` (`zero_flow`, `flowing`): `n_in_mc`, `n_rerun`, behavioural counts with the routine and with a unit hydrograph integrated over each day, `best_kge_integrated`, `median_kge_integrated`, `median_kge_change` |
| `J_flex_paired.csv` | Per `comparison` and benchmark, on the pairs where both optima pass: `rho_flex`, `rho_hbv` with intervals, `diff` with `diff_lo` and `diff_hi` (paired bootstrap), `median_ratio_flex`, `median_ratio_hbv`, and the decision-rule columns `rule_intervals_separate` and `rule_identifiable_in_most_pairs` |
