# Data

## Raw data
Open Power System Data (OPSD), *Data Package Time Series*, version 2020-10-06
https://doi.org/10.25832/time_series/2020-10-06 (file `time_series_60min_singleindex.csv`)

Targets: `DE_load_actual_entsoe_transparency` (load) and `DE_solar_generation_actual` (solar), hourly, January 2015 – September 2020.
Split: training 2015–2018, validation 2019, test January–September 2020 (6,576 hourly targets for every horizon).

Processed one-hour dataset (`frozen_168h_sequences.npz`, created by the notebooks) SHA-256:
`ed67feaed54517cac5f84d314caca75a7b3837ed9bb7f17c7613f0023bd69912`

## Saved predictions (`data/predictions/`)
| File | Content |
|---|---|
| `frozen_predictions_all_MW.npz` | Original-protocol test predictions for all models, horizons (1, 6, 24 h), and seeds (1, 7, 21, 42, 100); keys `h{h}__{model}__s{seed}` (6,576 × 2: load, solar), observed targets `y_true_h{h}`, and the one-hour input history `X_dyn_h1_MW` (6,576 × 168 × 2) |
| `stageC/*_test_predictions_MW.npz` | Corrected (uncapped) PatchTST and DLinear runs: `pred_uncapped`, the replayed best-within-50-epochs checkpoint `pred_within50`, and `y_true` |
| `stageB_degraded_predictions_MW.npz` | One-hour predictions under clean inputs, forward fill, and seasonal fill for gaps of 1, 3, 5, 6, 12, 23, 24 h; keys `{model}__s{seed}__{condition}` |
| `random_degradation_masks.npz` | The five nested random-masking replicates at 10%, 30%, 50% (True = masked) |

All values are in MW. Model names: `LSTM`, `ST-GAP` (= ST), `MTL-LSTA`, `PatchTST`, `DLinear`, `ST-LSTA`, `MTL-GAP`.
The arrays are derived from OPSD data, which aggregates sources such as the ENTSO-E Transparency Platform; please cite OPSD and respect the terms of the original sources.
