# Training data guide (`outputs/training_data.csv`)

684 events, one row each. Made by `src/event_labeling/make_training_data.py` from the labels in
`outputs/event_labels.csv`. UNKNOWN events are not in this file.
**The labels are rule-derived and unreviewed, not verified faults** (see `docs/event_label_rules.md`).
This file is only valid if the team accepts rule-derived labels (section 5 of that document).

## What to predict
`event_label`: 0 = NORMAL, 1 = ANOMALY.

## Fixed split (everyone must use it, do not make your own)
| Column | Meaning |
|---|---|
| `split` | `train` (513 rows) or `test` (171 rows). Train on `train`, report final scores on `test` only once. |
| `fold` | 1 to 5 for `train` rows only (0 for `test`). For cross-validation: train on four folds, check on the fifth. |

Counts: train 499 normal + 14 anomaly; test 166 normal + 5 anomaly. With 5 test anomalies the test
scores are noisy, so report cross-validation results as well.

## Columns
| Column | Meaning |
|---|---|
| `event_id` | Row number in `data/final_irrigation_events.csv`. Do not use as a feature. |
| `start` | Start date and time. Not a feature. |
| `sector_1` to `sector_5` | 1 for the sector the event was in, else 0 |
| `is_fertigation` | 1 = fertigation, 0 = standard irrigation |
| `start_hour`, `start_month` | Hour (0 to 23) and month (2 to 8) the event started |
| `initial_humidity`, `final_humidity` | Soil moisture at the start and end |
| `duration_hours` | How long the event ran |
| `humidity_change` | `final_humidity - initial_humidity` |
| `water_volume_l` | Litres applied = duration (minutes) x valve flow (1.1 L/min; 0.4 for sector 5). Derived from duration, so it adds no new information. |
| `temperature_c` | Air temperature at the start (Open-Meteo ERA5 weather model, not an on-site sensor) |
| `rain_prev_24h_mm` | Rain in the 24 hours before the start, in mm (hourly model values added up) |
| `stop_fertigation_timeout`, `stop_physical_valve_closure`, `stop_time_restriction`, `stop_humidity_threshold` | Why it stopped (one of the four is 1) |
| `event_label` | The target |
| `split`, `fold` | See above |

Weather comes from `data/zone_1_features_final.csv`; the dataset README says all sectors share
the same weather.

## Two feature sets (train and report both)
The anomaly rules use `duration_hours`, `initial_humidity`, `final_humidity` and `humidity_change`.
`water_volume_l` is duration times a constant. A model that sees these can partly copy the rule.

- **Set A (all features):** every column from `sector_1` to `stop_humidity_threshold`.
- **Set B (without the rule columns):** Set A minus `duration_hours`, `initial_humidity`,
  `final_humidity`, `humidity_change` and `water_volume_l`. It keeps sector, type, time of day,
  month, temperature, rain and stop reason.

A big drop from A to B shows how much of the score was the rule being copied. The `stop_*`
columns describe how the event ended; say in the report whether you kept them.

## Notes for each model
- **Decision Tree:** uses the numbers as they are.
- **KNN and Logistic Regression:** scale the numbers to a similar range. Work out the scaling from
  the `train` rows only, then apply it to `test`.
- **Class imbalance:** 19 anomalies in 684. Do not report accuracy alone. Report precision,
  recall, F1 and the confusion matrix, and say how you handled imbalance (for example class weights).
- **Interface:** `docs/model_interface.md` lists `final_humidity` and `stop_reason` as inputs.
  If the model uses Set B, the app does not need them. Agree on the input list before building
  the form. Temperature and 24-hour rain can be asked as plain numbers or looked up by date.
