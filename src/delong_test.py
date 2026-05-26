"""DeLong one-sided significance test: H0: AUC = 0.50.

Tests whether each model's AUROC is significantly greater than chance.
Uses the nonparametric DeLong method (DeLong et al., 1988; Hanley & McNeil, 1982).

Key insight: AUROC = P(score_positive > score_negative) over all positive/negative pairs.
This is the Mann-Whitney U statistic normalized to [0,1].
So the DeLong test is testing whether positive cases are consistently ranked higher.
"""

import math
import csv
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def delong_test(y_true, y_score):
    """Compute AUC and one-sided DeLong p-value vs. H0: AUC=0.50.

    Returns: (auc, se, z_stat, p_one_sided, ci_lo_95, ci_hi_95)
    """
    pos = [s for s, l in zip(y_score, y_true) if l == 1]
    neg = [s for s, l in zip(y_score, y_true) if l == 0]
    n1, n0 = len(pos), len(neg)

    if n1 == 0 or n0 == 0:
        raise ValueError("Need at least one positive and one negative case.")

    # AUC = P(score_pos > score_neg) [Mann-Whitney U]
    auc = sum(
        (1 if p > n else 0.5 if p == n else 0)
        for p in pos for n in neg
    ) / (n1 * n0)

    # Structural components for DeLong variance
    vp = [sum(1 if p > n else 0.5 if p == n else 0 for n in neg) / n0
          for p in pos]
    vn = [sum(1 if p > n else 0.5 if p == n else 0 for p in pos) / n1
          for n in neg]

    s10 = sum((v - auc) ** 2 for v in vp) / (n1 - 1) if n1 > 1 else 0.0
    s01 = sum((v - auc) ** 2 for v in vn) / (n0 - 1) if n0 > 1 else 0.0
    var = s10 / n1 + s01 / n0
    se  = math.sqrt(var) if var > 0 else 1e-9

    # One-sided z-test
    z = (auc - 0.50) / se

    # Normal survival function (Abramowitz & Stegun)
    def norm_sf(z_val):
        z_abs = abs(z_val)
        t = 1.0 / (1.0 + 0.2316419 * z_abs)
        poly = t * (0.319381530 + t * (-0.356563782 +
               t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        phi  = 0.3989422803 * math.exp(-0.5 * z_abs * z_abs)
        sf   = phi * poly
        return sf if z_val >= 0 else 1.0 - sf

    p = norm_sf(z)

    ci_lo = max(0.0, auc - 1.96 * se)
    ci_hi = min(1.0, auc + 1.96 * se)

    return auc, se, z, p, ci_lo, ci_hi


def main():
    # Find prediction files
    pred_files = {
        "PHENOCAST-CORE":         RESULTS_DIR / "core_predictions.csv",
        "Elastic Net":            RESULTS_DIR / "baseline_predictions/predictions_elastic_net.csv",
        "Random Forest":          RESULTS_DIR / "baseline_predictions/predictions_random_forest.csv",
        "XGBoost":                RESULTS_DIR / "baseline_predictions/predictions_xgboost.csv",
    }

    print(f"DeLong one-sided test: H0: AUC = 0.50")
    print(f"{'Model':<22} {'AUC':>6} {'SE':>6} {'z':>6} {'p(1-sided)':>12} {'95% CI'}")
    print("-" * 72)

    records = []
    for name, path in pred_files.items():
        if not path.exists():
            print(f"  {name}: predictions not found (run scripts first)")
            continue
        rows = list(csv.DictReader(open(path)))
        y_true = [int(r["y_true"]) for r in rows]
        y_prob = [float(r["y_prob"]) for r in rows]

        auc, se, z, p, lo, hi = delong_test(y_true, y_prob)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:<20} {auc:6.4f} {se:6.4f} {z:6.2f} {p:>12.5f} [{lo:.4f}, {hi:.4f}]  {sig}")
        records.append({"model": name, "auc": auc, "se": se, "z": z,
                        "p_one_sided": p, "ci_lo_95": lo, "ci_hi_95": hi})

    print("\nSignificance: *** p<0.001  ** p<0.01  * p<0.05")
    print("Note: all interpretations are exploratory given n=38, 7 positives.")

    if records:
        pd.DataFrame(records).to_csv(RESULTS_DIR / "delong_results.csv", index=False)
        print(f"\nSaved to results/delong_results.csv")


if __name__ == "__main__":
    main()
