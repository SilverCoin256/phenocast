"""Calibration analysis: Brier score, ECE, and calibration curves.

A model that discriminates well (high AUROC) can still be badly calibrated:
it might say 90% probability for something that's only 50/50 in reality.
Brier score and ECE measure this directly.

For clinical contexts, calibration matters more than AUROC alone.
This is why we report both.
"""

import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from utils import expected_calibration_error

RESULTS_DIR = Path("results/calibration")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def analyze_calibration(name, y_true, y_prob):
    brier = brier_score_loss(y_true, y_prob)
    ece   = expected_calibration_error(y_true, y_prob)

    # Calibration curve
    n_bins = 5  # small n=38, so keep bins coarse
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins,
                                                strategy="quantile")
    except ValueError:
        frac_pos, mean_pred = np.array([]), np.array([])

    print(f"  {name:<22}  Brier={brier:.4f}  ECE={ece:.4f}")
    return brier, ece, frac_pos, mean_pred


def plot_calibration_curves(results):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)

    for name, _, _, frac_pos, mean_pred in results:
        if len(frac_pos) > 0:
            ax.plot(mean_pred, frac_pos, marker="o", label=name)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration curves (n=38, 7 positives)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "calibration_curves.png", dpi=150)
    plt.close()
    print("  Saved calibration_curves.png")


def main():
    pred_files = {
        "PHENOCAST-CORE": Path("results/core_predictions.csv"),
        "Elastic Net":    Path("results/baseline_predictions/predictions_elastic_net.csv"),
        "Random Forest":  Path("results/baseline_predictions/predictions_random_forest.csv"),
        "XGBoost":        Path("results/baseline_predictions/predictions_xgboost.csv"),
    }

    print("Calibration analysis:")
    print("-" * 50)

    results = []
    records = []
    for name, path in pred_files.items():
        if not path.exists():
            print(f"  {name}: not found")
            continue
        rows = list(csv.DictReader(open(path)))
        y_true = [int(r["y_true"]) for r in rows]
        y_prob = [float(r["y_prob"]) for r in rows]

        brier, ece, frac_pos, mean_pred = analyze_calibration(name, y_true, y_prob)
        results.append((name, brier, ece, frac_pos, mean_pred))
        records.append({"model": name, "brier": brier, "ece": ece})

    plot_calibration_curves(results)

    if records:
        pd.DataFrame(records).to_csv(RESULTS_DIR / "calibration_metrics.csv", index=False)
        print("  Saved calibration_metrics.csv")


if __name__ == "__main__":
    main()
