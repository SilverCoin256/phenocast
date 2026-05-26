# PHENOCAST

**Can passive smartphone behavioral patterns predict endpoint depression severity in university students?**

This is my Regeneron ISEF 2026 research project in Behavioral and Social Sciences (BEHA). Short answer: there is a real but fragile signal — and making the model *simpler* actually worked better than making it complex.

---

## Research Question

> Can longitudinal passive smartphone behavioral representations predict endpoint PHQ-9 mental health status in a leakage-safe, calibrated manner in the StudentLife cohort?

I used the [StudentLife dataset](http://studentlife.cs.dartmouth.edu/) (Wang et al., 2014), which collected passive smartphone data from 48 Dartmouth students over one academic term (spring 2013). After preprocessing, I retained **n=38 subjects** with complete behavioral telemetry and a post-study PHQ-9 score.

---

## Key Findings

| Model | AUROC | 95% CI | Brier | ECE |
|---|---|---|---|---|
| PHENOCAST-CORE (~10 parameters) | **0.8203** | [0.664, 0.946] | 0.1898 | 0.1257 |
| Temporal MLP (flattened) | 0.8848 | [0.687, 1.000] | 0.0781 | 0.0930 |
| XGBoost (audited features) | 0.8065 | [0.669, 0.944] | 0.1604 | 0.1599 |
| Random Forest (audited features) | 0.7880 | [0.614, 0.963] | 0.1314 | 0.0748 |
| Elastic Net (audited features) | 0.7097 | [0.548, 0.872] | 0.2093 | 0.0796 |

All results use **leave-one-subject-out cross-validation (LOSO-CV)** — the only split strategy that fully prevents subject-level data leakage at n=38.

**DeLong one-sided test vs. chance (AUC = 0.50):** PHENOCAST-CORE z ≈ 4.3, p < 0.001. All primary models significantly exceed chance (RF: p=0.0006; XGBoost: p<0.0001; MLP: p=0.0001).

---

## What Failed — and Why That's the Point

I started by adding graph-based complexity (GNN layers connecting behavioral features across time). The result:

| Condition | AUROC | Interpretation |
|---|---|---|
| PHENOCAST-CORE (full) | 0.8203 | signal present |
| Graph-augmented | 0.5392 | near chance — complexity hurt |
| Remove conversation features | 0.5576 | conversation is the strongest signal |
| Shuffled-day controls | 0.5705 | temporal structure matters |
| Random 7-day window | 0.5161 | full-term context needed |

The graph model collapsing to near-chance AUROC is actually the most important result. With only 7 positive cases out of 38 subjects, adding parameters made things worse. The data forced me to make the model smaller — hence "evidence-forced simplification."

---

## Why This is a BEHA Project (Not Just a CS Project)

The features I used are behavioral proxies grounded in psychology literature:

| Feature family | Behavioral theory |
|---|---|
| Conversation duration | Social withdrawal (Beck, 1967; Coyne, 1976) |
| GPS mobility entropy | Routine fragmentation, behavioral activation |
| Screen unlock timing | Circadian rhythm disruption |
| Bluetooth proximity | Social exposure |
| Dark/rest proxy | Rest-activity rhythm theory |

I'm not claiming these features *cause* depression — I'm testing whether they *contain predictive information* consistent with these theories. That distinction matters.

---

## Important Limitations

- **n=38 with 7 elevated PHQ-9 cases** — this is a pilot study, not a validation
- **Observational, not experimental** — no manipulation of independent variables
- **Endpoint labels only** — I predict end-of-term PHQ-9 status, not weekly changes
- **No external validation** — tested within StudentLife only; generalizability is unknown
- **Potential confounders** — exam timing, year-in-program, and social calendar all correlate with both phone usage and PHQ-9, and I cannot fully control for them
- **Not a clinical tool** — this project does not diagnose, screen, or monitor mental health

---

## Ethics

No new human subjects were recruited. The StudentLife dataset is a de-identified public research dataset collected under IRB approval at Dartmouth College (Wang et al., 2014; ACM UbiComp). Secondary analysis of de-identified public data does not require additional IRB approval. No personally identifiable information was accessed or retained.

---

## Setup

```bash
git clone https://github.com/SilverCoin256/phenocast.git
cd phenocast
pip install -r requirements.txt
```

Python 3.10 or 3.11 recommended.

---

## Data

See `data/README.md` for step-by-step instructions on downloading the StudentLife dataset.

Raw data files go in `data/raw/studentlife/`. Nothing in that folder is committed to this repo.

---

## Reproducing Results

See `REPRODUCIBILITY.md` for the full walkthrough.

Quick version (after data setup):

```bash
python src/feature_engineering.py      # extract features from raw StudentLife
python src/run_loso_cv.py              # run all baselines under LOSO-CV
python src/run_phenocast_core.py       # run PHENOCAST-CORE
python src/run_ablations.py            # modality removal, shuffled-day, random-window
python src/delong_test.py              # DeLong significance tests vs. chance
python src/calibration_analysis.py    # Brier, ECE, calibration plots
```

All scripts write output to `results/`.

---

## Repository Structure

```
phenocast/
├── src/                        # all analysis code
│   ├── feature_engineering.py  # behavioral feature extraction from raw StudentLife
│   ├── run_loso_cv.py          # baseline LOSO-CV pipeline
│   ├── run_phenocast_core.py   # PHENOCAST-CORE model
│   ├── run_ablations.py        # ablation study
│   ├── delong_test.py          # DeLong AUC significance test
│   ├── calibration_analysis.py # Brier, ECE, calibration curves
│   └── utils.py                # shared utilities
├── data/
│   └── README.md               # how to download StudentLife
├── results/                    # saved predictions and metric tables
├── notebooks/
│   └── exploration.ipynb       # exploratory analysis notebook
├── requirements.txt
├── REPRODUCIBILITY.md
└── README.md
```

---

## Citation

```bibtex
@misc{gupta2026phenocast,
  title   = {{PHENOCAST}: Transportability-Aware Longitudinal Behavioral Representation
             for Endpoint Mental-Health Status Prediction from Passive Smartphone Telemetry},
  author  = {Gupta, Shaurya},
  year    = {2026},
  note    = {Regeneron ISEF 2026, BEHA category. Dataset: StudentLife (Wang et al., 2014).},
  url     = {https://github.com/SilverCoin256/phenocast}
}
```

Dataset citation:
```bibtex
@inproceedings{wang2014studentlife,
  title     = {StudentLife: assessing mental health, academic performance and behavioral
               trends of college students using smartphones},
  author    = {Wang, Rui and Chen, Fanglin and Chen, Zhenyu and Li, Tianxing and
               Harari, Gabriella and Tignor, Stefanie and Zhou, Xia and Ben-Zeev, Dror
               and Campbell, Andrew T},
  booktitle = {Proceedings of the ACM International Joint Conference on Pervasive
               and Ubiquitous Computing (UbiComp)},
  year      = {2014}
}
```

---

## License

MIT. See `LICENSE`.
