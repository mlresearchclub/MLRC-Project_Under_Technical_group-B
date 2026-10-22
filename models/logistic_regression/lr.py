"""Compare logistic-regression models with different normal-class sample sizes.

Place training_data.csv next to this script and run:
    python lr.py

The training folds choose the feature set and sampling strategy. The test set is
kept at its original class balance and evaluated only for the selected model.
Labels are rule-derived, not verified faults.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "event_label"

df = pd.read_csv(os.path.join(HERE, "training_data.csv"))
train = df.loc[df["split"] == "train"].reset_index(drop=True)
test = df.loc[df["split"] == "test"].reset_index(drop=True)

if train.empty or test.empty:
    raise ValueError("The CSV must contain rows with split='train' and split='test'.")
if set(train[TARGET].unique()) != {0, 1}:
    raise ValueError("Training data must contain both event_label classes 0 and 1.")
if set(test[TARGET].unique()) != {0, 1}:
    raise ValueError("Test data must contain both event_label classes 0 and 1.")

print("Training class counts:")
print(train[TARGET].value_counts().sort_index().rename(index={0: "normal", 1: "anomaly"}))
print("\nTest class counts (not resampled):")
print(test[TARGET].value_counts().sort_index().rename(index={0: "normal", 1: "anomaly"}))

not_features = {"event_id", "start", TARGET, "split", "fold"}
rule_columns = {
    "duration_hours",
    "initial_humidity",
    "final_humidity",
    "humidity_change",
    "water_volume_l",
}
all_features = [column for column in df.columns if column not in not_features]
feature_sets = {
    "A_all_features": all_features,
    "B_without_rule_columns": [column for column in all_features if column not in rule_columns],
}
skewed_columns = ["duration_hours", "water_volume_l", "rain_prev_24h_mm"]

# Compare the current weighted baseline with progressively smaller normal samples.
strategies = [
    ("all_weighted", None, "balanced"),
    ("20_to_1", 20, None),
    ("10_to_1", 10, None),
    ("5_to_1", 5, None),
    ("3_to_1", 3, None),
    ("1_to_1", 1, None),
]


def fold_splits(frame):
    folds = sorted(value for value in frame["fold"].unique() if value != 0)
    if len(folds) < 2:
        raise ValueError("Training data needs at least two nonzero fold values.")
    return [
        (np.flatnonzero(frame["fold"].to_numpy() != fold),
         np.flatnonzero(frame["fold"].to_numpy() == fold))
        for fold in folds
    ]


def build_pipeline(features, class_weight):
    skewed = [column for column in skewed_columns if column in features]
    prep = ColumnTransformer(
        [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), skewed)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("prep", prep),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight=class_weight,
                    solver="liblinear",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def undersample_indices(y, max_normal_ratio, seed):
    labels = np.asarray(y)
    anomaly_indices = np.flatnonzero(labels == 1)
    normal_indices = np.flatnonzero(labels == 0)
    if max_normal_ratio is None:
        return np.arange(len(labels))

    keep_count = min(len(normal_indices), max_normal_ratio * len(anomaly_indices))
    rng = np.random.default_rng(seed)
    kept_normals = rng.choice(normal_indices, size=keep_count, replace=False)
    return np.sort(np.concatenate((anomaly_indices, kept_normals)))


def choose_threshold(y, probabilities):
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.05, 1.0, 0.05):
        score = f1_score(y, probabilities >= threshold, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


splits = fold_splits(train)
results = []
oof_probabilities = {}

for feature_set_name, features in feature_sets.items():
    X = train[features]
    y = train[TARGET].to_numpy()

    for strategy_name, normal_ratio, class_weight in strategies:
        oof = np.full(len(train), np.nan)
        fold_average_precisions = []

        for fold_number, (fit_indices, validation_indices) in enumerate(splits):
            fold_y = y[fit_indices]
            selected_indices = undersample_indices(
                fold_y, normal_ratio, SEED + fold_number
            )
            sampled_fit_indices = fit_indices[selected_indices]

            model = build_pipeline(features, class_weight)
            model.fit(X.iloc[sampled_fit_indices], y[sampled_fit_indices])
            probabilities = model.predict_proba(X.iloc[validation_indices])[:, 1]
            oof[validation_indices] = probabilities
            fold_average_precisions.append(
                average_precision_score(y[validation_indices], probabilities)
            )

        if np.isnan(oof).any():
            raise RuntimeError("Cross-validation did not produce a prediction for every row.")

        mean_fold_ap = float(np.mean(fold_average_precisions))
        pooled_ap = average_precision_score(y, oof)
        key = (feature_set_name, strategy_name)
        oof_probabilities[key] = oof
        results.append(
            {
                "feature_set": feature_set_name,
                "strategy": strategy_name,
                "mean_fold_pr_auc": mean_fold_ap,
                "pooled_oof_pr_auc": pooled_ap,
            }
        )

results_df = pd.DataFrame(results).sort_values(
    ["mean_fold_pr_auc", "pooled_oof_pr_auc"], ascending=False
)
print("\nTraining cross-validation comparison (higher PR-AUC is better):")
print(results_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

# Set A includes inputs used to create the rule-derived labels, so treat it as
# a leakage diagnostic only. Select the deployable model from set B.
selection_results = results_df[
    results_df["feature_set"] == "B_without_rule_columns"
]
winner = selection_results.iloc[0]
best_feature_set = winner["feature_set"]
best_strategy = winner["strategy"]
best_features = feature_sets[best_feature_set]
strategy = next(item for item in strategies if item[0] == best_strategy)
_, best_normal_ratio, best_class_weight = strategy

winner_oof = oof_probabilities[(best_feature_set, best_strategy)]
threshold = choose_threshold(train[TARGET], winner_oof)
print(
    f"\nSelected by training CV: {best_feature_set}, {best_strategy}; "
    f"threshold={threshold:.2f}"
)

final_indices = undersample_indices(
    train[TARGET].to_numpy(), best_normal_ratio, SEED
)
final_model = build_pipeline(best_features, best_class_weight)
final_model.fit(train.iloc[final_indices][best_features], train.iloc[final_indices][TARGET])

test_probabilities = final_model.predict_proba(test[best_features])[:, 1]
test_predictions = (test_probabilities >= threshold).astype(int)
y_test = test[TARGET]

print("\nHeld-out test results (test data was not resampled):")
print(classification_report(y_test, test_predictions, labels=[0, 1], zero_division=0))
print("Confusion matrix (rows=true, columns=predicted; labels=[0, 1]):")
print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))
print(f"Accuracy: {accuracy_score(y_test, test_predictions):.3f}")
print(f"Precision: {precision_score(y_test, test_predictions, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, test_predictions, zero_division=0):.3f}")
print(f"F1: {f1_score(y_test, test_predictions, zero_division=0):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, test_probabilities):.3f}")
print(f"PR AUC: {average_precision_score(y_test, test_probabilities):.3f}")

results_df.to_csv(os.path.join(HERE, "lr_sampling_comparison.csv"), index=False)
joblib.dump(
    {
        "pipeline": final_model,
        "threshold": threshold,
        "features": best_features,
        "feature_set": best_feature_set,
        "sampling_strategy": best_strategy,
    },
    os.path.join(HERE, "lr_best_model.joblib"),
)
print("\nSaved: lr_sampling_comparison.csv, lr_best_model.joblib")
print("Reminder: anomaly labels are rule-derived, not verified faults.")
