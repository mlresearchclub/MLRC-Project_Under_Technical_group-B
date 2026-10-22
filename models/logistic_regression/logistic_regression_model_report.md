# Logistic Regression: results (Set A and Set B)

Labels are rule-derived (rules A1 to A3 in `event_label_rules.md`), not verified faults.

## Setup (same for both sets)
- Data: `training_data.csv`, 684 labelled events (665 normal, 19 anomaly). 105 UNKNOWN events left out.
- Split: train 513 rows (499 normal, 14 anomaly); test 171 rows (166 normal, 5 anomaly). 5 fixed cross-validation folds on the training rows.
- Pipeline: log(1 + x) on skewed columns, standard scaling (learned from training rows only), Logistic Regression
  (class_weight balanced). Settings and threshold chosen on the training folds, not on the test set.

## Set A: all 19 features
Sector (5 columns), is_fertigation, start_hour, start_month, initial_humidity, duration_hours, final_humidity,
humidity_change, water_volume_l, temperature_c, rain_prev_24h_mm, 4 stop-reason columns.

- Chosen settings: C = 10, L1 penalty, threshold 0.95. Cross-validated PR-AUC 0.691.
- Saved file: `logreg_set_A.joblib`

| | TN | FP | FN | TP | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Cross-validation, threshold 0.5 | 484 | 15 | 4 | 10 | 0.400 | 0.714 | 0.513 | 0.874 | 0.624 |
| Cross-validation, threshold 0.95 | 495 | 4 | 5 | 9 | 0.692 | 0.643 | 0.667 | 0.874 | 0.624 |
| **Test** | 166 | 0 | 0 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Set B: 14 features (Set A without duration_hours, initial_humidity, final_humidity, humidity_change, water_volume_l)
- Chosen settings: C = 1, L1 penalty, threshold 0.85. Cross-validated PR-AUC 0.207.
- Saved file: `logreg_set_B.joblib`

| | TN | FP | FN | TP | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Cross-validation, threshold 0.5 | 342 | 157 | 10 | 4 | 0.025 | 0.286 | 0.046 | 0.541 | 0.146 |
| Cross-validation, threshold 0.85 | 489 | 10 | 11 | 3 | 0.231 | 0.214 | 0.222 | 0.541 | 0.146 |
| **Test** | 163 | 3 | 5 | 0 | 0.000 | 0.000 | 0.000 | 0.775 | 0.105 |

Resampling comparison on Set B (6 resampling strategies x 2 feature sets, 12 configurations): best by
cross-validation was Set B with a 20-to-1 ratio, threshold 0.20. Test: TN 163, FP 3, FN 5, TP 0,
ROC-AUC 0.814, PR-AUC 0.093.

## Untuned baseline (scaler + balanced Logistic Regression, threshold 0.5, no tuning), test rows

| Feature set | TN | FP | FN | TP | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| A | 155 | 11 | 1 | 4 | 0.267 | 0.800 | 0.400 | 0.849 |
| B | n/a | n/a | n/a | n/a | 0.071 | 0.800 | 0.131 | n/a |

## Notes
- Only 19 anomalies, 5 of them in the test set, so scores can change a lot from a single event.
- Accuracy is not given: 97.2% of events are normal.
- Files: `logreg_set_A.joblib` and `logreg_set_B.joblib` (each holds the trained pipeline, the threshold and the feature list),
  `logreg_results.csv` (test scores for both sets).
