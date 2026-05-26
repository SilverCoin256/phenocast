# Data Setup

## StudentLife Dataset

PHENOCAST uses the [StudentLife dataset](http://studentlife.cs.dartmouth.edu/) (Wang et al., 2014).

The dataset is publicly available for research. To get it:

1. Visit http://studentlife.cs.dartmouth.edu/
2. Download the sensing data archive and PHQ survey CSV
3. Place files as follows:

```
data/raw/studentlife/
├── dataset/
│   └── sensing/
│       ├── activity/
│       ├── audio/
│       ├── bluetooth/
│       ├── conversation/
│       ├── dark/
│       ├── gps/
│       ├── phonelock/
│       ├── phonecharge/
│       └── sms/
└── survey/
    └── PHQ.csv
```

4. Run preprocessing:
```bash
python src/feature_engineering.py
```
This writes `data/processed/features.csv` and `data/processed/labels.csv`.

## Features Used

Only passive telemetry modalities (no EMA/self-report features):

| Modality | What it measures |
|---|---|
| `conversation` | Daily conversation duration (seconds) |
| `activity` | Activity level |
| `bluetooth` | Nearby device scan counts (social proximity proxy) |
| `dark` | Screen-dark duration (rest proxy) |
| `gps` | Mobility entropy and variance |
| `phonelock` | Screen unlock frequency |
| `sms` | SMS message counts |
| `calls` | Call duration |

**Excluded:** EMA sleep_quality, sleep_hours, stress_level — these are self-reported, not passive.

## Outcome Label

Post-study PHQ-9 score binarized at ≥10 (moderate-to-severe), following Kroenke et al. (2001) and the StudentLife published protocol.

## Study Period

Restricted to: **2013-03-25 through 2013-06-01**.

## Final Dataset Size

- After validity filtering: **n=38 subjects**
- Elevated PHQ-9 (≥10): **7 subjects** (18.4%)
- Non-elevated: **31 subjects**
