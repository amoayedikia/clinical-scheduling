# Prescriptive Analytics for MRI Appointment Scheduling in Alzheimer's Clinical Trials

This repository contains the code and reproducibility scripts for a two-phase
prescriptive-analytics framework that integrates machine-learning risk prediction
with stochastic job-shop scheduling for MRI scan appointments in Alzheimer's
disease clinical trials.

The framework couples:

- **Phase 1 — Prediction.** A gradient-boosting classifier trained on
  cognitive, demographic, and imaging-quality features derived from the
  National Alzheimer's Coordinating Center (NACC) Uniform Data Set,
  producing patient-specific probabilities of MRI scan failure.
- **Phase 2 — Optimization.** A stochastic mixed-integer linear program
  with Bernoulli service-time distributions (expected-value formulation),
  benchmarked against a risk-first Longest-Processing-Time (LPT) heuristic
  and five additional baselines (FCFS, SPT, deterministic, age-stratified,
  random).

The central empirical finding is that **prediction quality dominates
optimization sophistication**: once rich clinical features are incorporated,
a simple risk-first LPT heuristic matches the stochastic MILP, while age-based
heuristics counter-intuitively underperform a naive deterministic schedule.

---

## Repository structure

```
.
├── code/                         # Active pipeline
│   ├── baselines.py              #   Baseline scheduling methods + simulation harness
│   ├── experiments.py            #   Main experimental suite E1–E6 (--cohort synthetic|real)
│   ├── e7_scaling_sensitivity.py #   E7: scaling-factor sensitivity (--cohort synthetic|real)
│   ├── e8_shift_experiment.py    #   E8: center/era distribution-shift boundary conditions
│   ├── nacc74_cohort_builder.py  #   Real NACC74 cohort pool for the real-data arm
│   ├── a2_qc_validation.py       #   Risk predictions vs real SCAN/Clariti MRI QC signal
│   ├── phase2_ml_validation.py   #   ML cross-validation / calibration (Phase 1)
│   └── ejor_clinical_scheduling_v3_fixed.py  # Phase-1 reference pipeline
├── docs/                         # HANDOVER.md (project log) + UDS3→UDS4 crosswalk
├── manuscript/                   # LaTeX (main-v9.tex current; main-v8.tex as submitted)
├── data-nacc69/, data-nacc74/    # NACC data (gitignored under DUA)
├── results/                      # Experiment outputs (real-arm/, a2/, week1/, submitted-era/)
├── archive/                      # Superseded code & outputs (safe to delete)
└── README.md
```

Run scripts from the repository root, e.g. `python3 code/experiments.py --quick`
(data and results paths are resolved relative to the working directory).
The legacy NACC69 pipeline expects `investigator_*nacc69.csv` in the root;
recreate the symlinks if needed:
`for f in investigator_nacc69 investigator_fcsf_nacc69 investigator_scan_mriqc_nacc69 investigator_scan_mrisbm_nacc69; do ln -s data-nacc69/$f.csv $f.csv; done`


### Phase 1 — Prediction

- `ejor_clinical_scheduling_v3_fixed.py` — primary script. Trains the
  gradient-boosting model on the simulation-augmented cohort using
  cognitive predictors (MoCA, CDR sum-of-boxes, NPI-Q), demographics,
  and imaging-quality covariates. Outputs per-patient failure probabilities.
- `phase2_ml_validation.py` — k-fold cross-validation, AUC, calibration plots.
- `diagnostic_cognitive_predictors.py` — audit script confirming UDS3-era
  (2021–2025) MRI coverage and resolving the cognitive-predictor linkage.

### Phase 2 — Optimization

- `baselines.py` — implementations of FCFS, SPT, LPT, deterministic,
  age-stratified, and random schedules; the stochastic MILP solver; and the
  Monte-Carlo simulation evaluator.
- `experiments.py` — orchestrates experiments E1–E6 (main comparison, MILP
  vs LPT, prediction-quality ablation, etc.).
- `e7_scaling_sensitivity.py` — experiment E7: scalability analysis showing
  MILP becomes intractable at N ≥ 20 patients while LPT solves in ≈2 ms
  regardless of cohort size.

---

## Requirements

- Python ≥ 3.10
- pandas, numpy, scikit-learn, scipy
- pulp (with CBC, the default open-source solver, or Gurobi for larger instances)
- matplotlib (for figures)
- tqdm

Install with:

```bash
pip install pandas numpy scikit-learn scipy pulp matplotlib tqdm
```

---

## Data

This project uses features derived from the **NACC Uniform Data Set (UDS3)**
and NACC MRI imaging tables. Raw NACC data is **not redistributed** in this
repository per the NACC Data Use Agreement. Researchers wishing to reproduce
the prediction stage should:

1. Register and obtain a data use agreement at
   <https://naccdata.org/>.
2. Download the UDS, FCSF, MRI-QC, and MRI-SBM tables.
3. Place the CSV exports in the repository root using the filenames expected
   by the scripts (see `ejor_clinical_scheduling_v3_fixed.py` for the
   expected schema).

The optimization stage (`experiments.py`, `baselines.py`, `e7_scaling_sensitivity.py`)
runs on **simulation-augmented synthetic cohorts** generated from the feature
distributions and does **not** require raw NACC access; the trained model
artefacts and synthetic cohort suffice.

---

## Reproducing the experiments

Quick smoke test:

```bash
python3 code/experiments.py --quick
```

Full E1–E6 suite:

```bash
python3 code/experiments.py --experiments E1 E2 E3 E4 E5 E6
```

Scalability analysis (E7):

```bash
python3 code/e7_scaling_sensitivity.py
```

Phase-1 model validation:

```bash
python3 code/phase2_ml_validation.py
```

Outputs are written to `results/` with timestamped filenames.

---

## Manuscript

A manuscript describing this framework is under review. The repository will
be updated with a citation upon acceptance.

---

## License

Code: MIT License (see `LICENSE`).
Data derived from NACC is governed by the NACC Data Use Agreement and is
not covered by this license.

---

## Contact

For questions about the code or reproducibility, please open an issue on
this repository.
