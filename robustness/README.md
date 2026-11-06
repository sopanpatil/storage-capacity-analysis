# Robustness runs and re-analyses

Results behind Text S5 and Tables S9 and S10 of the Supporting Information. Each test changes
one design choice. Items A, B, C, D, J and S are new calibrations on a fixed subset of 397
catchments; items E to I work from the main run and its parameter ranges.

The subset (`subset.csv`, `subset_by_class.csv`) comes from `src/robustness_subset.py`, and the
new calibrations from `src/robustness_calibrate.py` and `src/robustness_flex.py`, whose outputs
go to `<workspace>/robustness/` (not carried here).

| Item | Test | Script in `src/` | Files |
|---|---|---|---|
| A | HBV with two objectives, KGE(Q) and KGE(log Q), and with the full SCE-UA budget | `robustness_summary_ad.py` | `A_hbv_multiobjective.csv` |
| B | HBV with up to 150,000 Monte Carlo samples | `robustness_summary_ad.py` | `B_hbv_convergence.csv` |
| C | GR6J optimum from the Michel algorithm or SCE-UA | `robustness_summary_ad.py` | `C_gr6j_optimiser.csv` |
| D | GR4J in place of GR6J | `robustness_summary_ad.py` | `D_gr4j.csv` |
| S | SAC-SMA optimum with the full SCE-UA budget | `robustness_summary_ad.py` | `S_sacsma_optimum.csv` |
| A-D, S | The same runs per catchment | `robustness_summary_ad.py` | `AD_per_catchment.csv` |
| E | SAC-SMA behavioural capacity under a uniform prior on UZTWM + LZTWM (full sample) | `robustness_sacsma_reweight.py` | `E_sacsma_reweight*.csv` |
| F | HBV FC·LP and FC·(1 − LP) against the benchmarks (full sample) | `robustness_hbv_lp.py` | `F_hbv_lp.csv` |
| G | The same number of behavioural sets per model (full sample) | `robustness_common_subset.py` | `G_common_subset*` |
| H | Why the simulated deficit can exceed the calibrated capacity (full sample) | `robustness_deficit_excess.py` | `H_*.csv` |
| I | SAC-SMA routing parameters that give no routed flow | `robustness_uh_zero_flow.py` | `I_*.csv` |
| J | A FLEX-type model against HBV on the same catchments | `robustness_flex.py`, `robustness_summary_flex.py` | `J_*.csv` |

`src/flex_model.py` is the FLEX-type model; its checks are in `verification/`. Columns are
defined in [`../data/DATA_DICTIONARY.md`](../data/DATA_DICTIONARY.md), and the licence of these
tables is in [`../data/README.md`](../data/README.md).
