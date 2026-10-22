"""k-Nearest Neighbors (KNN): NORMAL (0) vs ANOMALY (1) irrigation events.
Labels in the data are rule-derived, not verified faults. 
KNN is distance-based, so a StandardScaler is included in the pipeline.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                    # running inside a notebook
    HERE = os.getcwd()

# ------------------------------------------------------------------ 1. load and split
df = pd.read_csv(os.path.join(HERE, "training_data.csv"))
train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)
print("train rows:", len(train), " test rows:", len(test))      # expect 513 and 171

# ------------------------------------------------------------------ 2. feature sets
NOT_FEATURES = ["event_id", "start", "event_label", "split", "fold"]
RULE_COLUMNS = ["duration_hours", "initial_humidity", "final_humidity",
                "humidity_change", "water_volume_l"]            # the label rules use these
SET_A = [c for c in df.columns if c not in NOT_FEATURES]        # all features
SET_B = [c for c in SET_A if c not in RULE_COLUMNS]             # without the rule columns
FEATURE_SETS = {"A": SET_A, "B": SET_B}


# ------------------------------------------------------------------ 3. helpers
def fold_splits(frame):
    """Use the fold column (1 to 5) so every model is cross-validated on the same folds."""
    splits = []
    for k in range(1, 6):
        splits.append((np.where(frame["fold"] != k)[0], np.where(frame["fold"] == k)[0]))
    return splits


def best_threshold(y, p):
    """Pick the probability cut-off with the best F1 (plain loop, no library)."""
    best_t, best_f1 = 0.5, -1.0
    for step in range(5, 96, 5):              # 0.05, 0.10 ... 0.95
        t = step / 100
        tp = fp = fn = 0
        for yi, pi in zip(y, p):
            pred = 1 if pi >= t else 0
            if pred == 1 and yi == 1:
                tp += 1
            elif pred == 1 and yi == 0:
                fp += 1
            elif pred == 0 and yi == 1:
                fn += 1
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def scores(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, prob),
            "pr_auc": average_precision_score(y, prob)}


def show(title, s):
    print("  %-26s TN %d  FP %d  FN %d  TP %d | precision %.3f  recall %.3f  F1 %.3f"
          "  ROC-AUC %.3f  PR-AUC %.3f" % (title, s["tn"], s["fp"], s["fn"], s["tp"],
                                         s["precision"], s["recall"], s["f1"],
                                         s["roc_auc"], s["pr_auc"]))


# ------------------------------------------------------------------ 4. train each feature set
results = []
for name, features in FEATURE_SETS.items():
    print("\n================ Feature set %s (%d features) ================" % (name, len(features)))
    X_train, y_train = train[features], train["event_label"]
    X_test, y_test = test[features], test["event_label"]
    splits = fold_splits(train)

    # Create a pipeline with StandardScaler and KNeighborsClassifier
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    # Tune hyperparameters on the same 5 folds, scoring by PR-AUC
    grid = {
        "knn__n_neighbors": [3, 5, 7, 11, 15, 21],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2]  # 1 = Manhattan distance, 2 = Euclidean distance
    }
    
    search = GridSearchCV(pipeline, grid, scoring="average_precision", cv=splits, refit=True)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("best settings:", search.best_params_, " cross-validated PR-AUC: %.3f" % search.best_score_)

    # out-of-fold probabilities on the training rows
    oof = cross_val_predict(best_model, X_train, y_train, cv=splits, method="predict_proba")[:, 1]
    threshold = best_threshold(y_train.tolist(), oof.tolist())      # chosen on training folds only
    print("chosen probability threshold:", threshold)

    print("cross-validation (training rows, out-of-fold):")
    show("at threshold 0.5", scores(y_train, (oof >= 0.5).astype(int), oof))
    show("at chosen threshold", scores(y_train, (oof >= threshold).astype(int), oof))

    # final check on the test rows
    prob_test = best_model.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= threshold).astype(int)
    print("test rows (%d anomalies):" % int(y_test.sum()))
    test_scores = scores(y_test, pred_test, prob_test)
    show("test", test_scores)

    # Save model artifacts and collect results
    joblib.dump({"pipeline": best_model, "threshold": threshold, "features": features},
                os.path.join(HERE, "knn_set_%s.joblib" % name))
    
    row = {
        "feature_set": name, 
        "n_neighbors": search.best_params_["knn__n_neighbors"],
        "weights": search.best_params_["knn__weights"],
        "p": search.best_params_["knn__p"],
        "threshold": threshold
    }
    row.update({"test_" + k: v for k, v in test_scores.items()})
    results.append(row)

pd.DataFrame(results).to_csv(os.path.join(HERE, "knn_results.csv"), index=False)
print("\nSaved: knn_set_A.joblib, knn_set_B.joblib, knn_results.csv")