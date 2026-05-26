# Reproducibility Guide

This document explains how to reproduce every result in the paper from scratch.

## Environment

```bash
git clone https://github.com/SilverCoin256/phenocast.git
cd phenocast
pip install -r requirements.txt
```

Tested on: Python 3.10.14, Python 3.11.9, macOS 14, Ubuntu 22.04.

## Step 1: Data Setup

Follow `data/README.md` to download StudentLife and place it correctly.

## Step 2: Feature Extraction

```bash
python src/feature_engineering.py
```

Outputs:
- `data/processed/features.csv` — daily behavioral summaries per subject
- `data/processed/labels.csv` — binarized PHQ-9 endpoint labels

Expected: 38 rows in labels.csv, 7 rows with label=1.

## Step 3: Run Baselines (LOSO-CV)

```bash
python src/run_loso_cv.py
```

Runs elastic net, random forest, and XGBoost under leave-one-subject-out CV.
Outputs to `results/baseline_predictions/` and `results/baseline_metrics.csv`.

Expected AUROCs (may vary slightly by sklearn version):
- Elastic Net: ~0.710
- Random Forest: ~0.788
- XGBoost: ~0.806

## Step 4: Run PHENOCAST-CORE

```bash
python src/run_phenocast_core.py
```

Outputs `results/core_predictions.csv` and prints AUROC, Brier, ECE.

Expected: AUROC ~0.820, Brier ~0.190, ECE ~0.126.

## Step 5: Run Ablations

```bash
python src/run_ablations.py
```

Tests: modality removal (one modality at a time), shuffled-day control, random 7-day window.
Outputs `results/ablation_results.csv`.

Key expected results:
- Remove conversation: AUROC drops to ~0.56
- Shuffled-day: AUROC drops to ~0.57
- Random window: AUROC drops to ~0.52

## Step 6: Statistical Tests

```bash
python src/delong_test.py
```

Computes DeLong one-sided test (H0: AUC=0.50) for each model.
Outputs `results/delong_results.csv`.

Expected for PHENOCAST-CORE: z ≈ 4.3, p < 0.001.

## Step 7: Calibration Analysis

```bash
python src/calibration_analysis.py
```

Outputs calibration curves, Brier scores, and ECE to `results/calibration/`.

## Randomness Control

All scripts use `random_state=42` and `numpy.random.seed(42)` at the top.
Results should be identical across runs with the same sklearn/numpy versions.

## Known Version Sensitivity

XGBoost AUROC may shift by ±0.005 across versions (2.0.x vs 2.1.x).
All reported numbers use the pinned versions in requirements.txt.
