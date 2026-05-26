"""Shared utilities: data loading, LOSO-CV splitting, metric computation."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve


DATA_DIR = "data/processed"
RESULTS_DIR = "results"


def load_features_and_labels():
    """Load preprocessed feature matrix and binary endpoint labels."""
    X = pd.read_csv(f"{DATA_DIR}/features.csv", index_col="uid")
    y = pd.read_csv(f"{DATA_DIR}/labels.csv", index_col="uid")["elevated"]
    # Align indices
    shared = X.index.intersection(y.index)
    return X.loc[shared], y.loc[shared]


def loso_cv_splits(uids):
    """Generate leave-one-subject-out splits.

    Each split: train = all subjects except uid_i, test = uid_i.
    This prevents any subject-level data leakage.
    """
    uids = list(uids)
    for i, test_uid in enumerate(uids):
        train_uids = [u for u in uids if u != test_uid]
        yield i, train_uids, [test_uid]


def compute_auroc(y_true, y_prob):
    """AUROC with DeLong 95% CI via bootstrap (1000 resamples)."""
    base_auc = roc_auc_score(y_true, y_prob)

    rng = np.random.default_rng(42)
    n = len(y_true)
    boot_aucs = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        yt_b = np.array(y_true)[idx]
        yp_b = np.array(y_prob)[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yt_b, yp_b))

    ci_lo = float(np.percentile(boot_aucs, 2.5))
    ci_hi = float(np.percentile(boot_aucs, 97.5))
    return base_auc, ci_lo, ci_hi


def balanced_accuracy(y_true, y_pred):
    """Balanced accuracy = mean of sensitivity and specificity."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (sens + spec) / 2


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE)."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return ece
