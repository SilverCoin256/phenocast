"""Extract daily behavioral summaries from raw StudentLife sensing data.

For each subject and each study day, compute summary statistics per modality.
Output: data/processed/features.csv and data/processed/labels.csv

Study period: 2013-03-25 through 2013-06-01 (active term only).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

STUDY_START = pd.Timestamp("2013-03-25")
STUDY_END   = pd.Timestamp("2013-06-01")

RAW_DIR = Path("data/raw/studentlife")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# PHQ-9 threshold for binary label (Kroenke et al., 2001)
PHQ_THRESHOLD = 10

# Modalities to extract (passive only; EMA excluded)
MODALITIES = [
    "activity", "audio", "bluetooth", "conversation",
    "dark", "gps", "phonelock", "sms",
]


def load_phq(survey_dir):
    """Load PHQ-9 survey and return uid -> post-study score mapping."""
    phq_path = survey_dir / "PHQ.csv"
    df = pd.read_csv(phq_path)
    # Columns vary; try common formats
    uid_col = [c for c in df.columns if "uid" in c.lower() or "id" in c.lower()][0]
    # Post-study is the last PHQ score per subject
    score_col = [c for c in df.columns if "phq" in c.lower() or "score" in c.lower()][0]
    df[uid_col] = df[uid_col].astype(str)
    # Take last measurement per subject
    post = df.sort_values("study_week" if "study_week" in df.columns else uid_col)
    post = post.groupby(uid_col)[score_col].last()
    return post


def daily_summaries_for_modality(sensing_dir, modality, uids):
    """Compute per-subject per-day summary for one modality."""
    mod_dir = sensing_dir / modality
    records = []

    for uid in uids:
        uid_files = list(mod_dir.glob(f"*{uid}*"))
        if not uid_files:
            continue
        dfs = []
        for f in uid_files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception:
                continue
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True)

        # Find timestamp column
        ts_col = next((c for c in df.columns if "time" in c.lower()), None)
        if ts_col is None:
            continue
        df["date"] = pd.to_datetime(df[ts_col], unit="s", errors="coerce").dt.date
        df = df.dropna(subset=["date"])
        df = df[
            (df["date"] >= STUDY_START.date()) &
            (df["date"] <= STUDY_END.date())
        ]
        if df.empty:
            continue

        # Compute daily counts/durations as summary stats
        daily = df.groupby("date").size().reset_index(name=f"{modality}_count")
        daily["uid"] = uid
        records.append(daily)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def main():
    sensing_dir = RAW_DIR / "dataset" / "sensing"
    survey_dir  = RAW_DIR / "survey"

    if not sensing_dir.exists():
        print(f"ERROR: Sensing directory not found at {sensing_dir}")
        print("Please follow data/README.md to set up StudentLife data.")
        return

    phq = load_phq(survey_dir)
    uids = list(phq.index)
    print(f"Found {len(uids)} subjects with PHQ-9 data")

    # Build modality feature tables
    feature_frames = []
    for modality in MODALITIES:
        print(f"  Extracting {modality}...")
        daily = daily_summaries_for_modality(sensing_dir, modality, uids)
        if daily.empty:
            print(f"    Warning: no data for {modality}")
            continue
        # Aggregate to subject-level summaries
        summary = daily.groupby("uid")[f"{modality}_count"].agg(
            ["mean", "std", "min", "max", "median"]
        )
        summary.columns = [f"{modality}_{s}" for s in summary.columns]
        feature_frames.append(summary)

    if not feature_frames:
        print("ERROR: No features extracted. Check your data directory.")
        return

    features = pd.concat(feature_frames, axis=1).fillna(0)

    # Restrict to subjects with both features and PHQ label
    common = features.index.intersection(phq.index)
    features = features.loc[common]
    labels = (phq.loc[common] >= PHQ_THRESHOLD).astype(int).rename("elevated")

    features.to_csv(OUT_DIR / "features.csv")
    labels.to_frame().to_csv(OUT_DIR / "labels.csv")

    n_pos = labels.sum()
    print(f"\nSaved {len(features)} subjects: {n_pos} elevated, {len(features)-n_pos} non-elevated")
    print(f"  -> {OUT_DIR}/features.csv  ({features.shape[1]} features)")
    print(f"  -> {OUT_DIR}/labels.csv")


if __name__ == "__main__":
    main()
