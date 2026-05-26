"""Run baseline models (Elastic Net, Random Forest, XGBoost) under LOSO-CV.

LOSO-CV = Leave-One-Subject-Out Cross-Validation.
This is the correct split for n=38 because it ensures no subject's data
appears in both the training and test set for any prediction.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

from utils import load_features_and_labels, loso_cv_splits, compute_auroc
from utils import balanced_accuracy, expected_calibration_error
from sklearn.metrics import brier_score_loss

np.random.seed(42)

RESULTS_DIR = Path("results/baseline_predictions")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


MODELS = {
    "elastic_net": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("clf",    LogisticRegression(
            penalty="elasticnet", solver="saga",
            l1_ratio=0.5, C=0.1,
            max_iter=2000, random_state=42
        )),
    ]),
    "random_forest": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    RandomForestClassifier(
            n_estimators=200, max_depth=4,
            random_state=42
        )),
    ]),
    "xgboost": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    xgb.XGBClassifier(
            n_estimators=100, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42, verbosity=0
        )),
    ]),
}


def run_loso(model_name, model, X, y):
    uids = X.index.tolist()
    records = []

    for fold, train_uids, test_uids in loso_cv_splits(uids):
        X_train = X.loc[train_uids]
        y_train = y.loc[train_uids]
        X_test  = X.loc[test_uids]
        y_test  = y.loc[test_uids]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        for uid, yt, yp, ypred in zip(test_uids, y_test, y_prob, y_pred):
            records.append({
                "fold": fold, "uid": uid,
                "y_true": int(yt), "y_prob": float(yp), "y_pred": int(ypred)
            })

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / f"predictions_{model_name}.csv", index=False)

    auc, ci_lo, ci_hi = compute_auroc(df["y_true"], df["y_prob"])
    bacc = balanced_accuracy(df["y_true"], df["y_pred"])
    brier = brier_score_loss(df["y_true"], df["y_prob"])
    ece = expected_calibration_error(df["y_true"], df["y_prob"])

    print(f"{model_name:20s}  AUROC={auc:.4f} [{ci_lo:.4f},{ci_hi:.4f}]  "
          f"BalAcc={bacc:.4f}  Brier={brier:.4f}  ECE={ece:.4f}")
    return {"model": model_name, "auroc": auc, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "balanced_acc": bacc, "brier": brier, "ece": ece}


def main():
    X, y = load_features_and_labels()
    print(f"Dataset: n={len(X)}, positives={y.sum()}, features={X.shape[1]}\n")

    summary = []
    for name, model in MODELS.items():
        result = run_loso(name, model, X, y)
        summary.append(result)

    pd.DataFrame(summary).to_csv("results/baseline_metrics.csv", index=False)
    print("\nSaved to results/baseline_metrics.csv")


if __name__ == "__main__":
    main()
