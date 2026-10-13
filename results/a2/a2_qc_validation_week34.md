# A2 — Validation of Predicted Risk Against Real MRI QC Signal (Weeks 3–4)
*13 Oct 2026. Script: `a2_qc_validation.py`. Outputs: `results/a2/a2_qc_validation_summary.csv` (aggregate, committable), `results/a2/a2_qc_scored.csv` (participant-level, gitignored under DUA).*

## Design
- Outcome (per Week-1 gate): binary failure unusable (0.13%); SCAN/Clariti series-level reviewer flags give a graded signal. Study-level outcomes: **any-motion** (primary; prevalence 43.7%) and **flagged fraction**.
- Sample: 9,681 QC studies → 6,124 with a UDS visit within 90d → **5,790 studies / 4,515 participants** with non-missing age+MoCA+CDR (UDS3 5,183 / UDS4 607).
- Scores: Phase-1 GBM (paper spec: GB 100/4/20, median impute, trained on the simulation-augmented 2,000 cohort, scored on real covariates), calibrated mapping (u=0.5), age-only (E4 formula), MoCA-only. CIs: 1,000-rep cluster bootstrap by participant.

## Results
| Score | AUC any-motion [95% CI] | Spearman ρ flagged fraction [95% CI] |
|---|---|---|
| Phase-1 GBM | 0.538 [0.522, 0.553] | 0.069 [0.044, 0.095] |
| Calibrated mapping | 0.546 [0.530, 0.560] | 0.080 [0.052, 0.104] |
| Age-only | 0.538 [0.523, 0.553] | 0.066 [0.039, 0.089] |
| MoCA-only | **0.573 [0.559, 0.590]** | **0.110 [0.086, 0.137]** |

- GBM risk quintiles: any-motion 41.0% (Q1) → 50.9% (Q5); flagged fraction 0.238 → 0.306.
- MoCA≤20: 53.4% vs MoCA≥26: 37.9%. CDR≥1: 53.8% vs CDR<1: 42.8% (matches Week-1 gate).
- GBM − age-only AUC difference: −0.000 [−0.018, 0.017].

## Robustness
- Series-count confound ruled out: corr(n_series, score)=0.045; AUC holds within n_series strata (GBM 0.51–0.55, MoCA 0.54–0.57); logistic beta_gbm=0.137 with n_series controlled.
- One study per participant (n=4,515): GBM 0.531, MoCA 0.570. Era: UDS3 0.542, UDS4 0.523 (n=607). Window ≤30d: 0.535 (n=3,230).

## Interpretation (as written into the manuscript, §qc_validation)
1. **Risk ordering is real** — first direct evidence: all CIs exclude 0.5, all ρ>0 at p<1e-6, +9.9pp any-motion top-vs-bottom quintile.
2. **Magnitude modest (0.54–0.57), as the design anticipates**: pre-screened cohort, graded proxy on released scans (not operational failure), covariates up to 90d stale.
3. **Cognition, not age, carries the real signal**: MoCA 0.573 > age 0.538 — real-outcome confirmation of the paper's mechanism (age is a weak proxy).
4. **Honest boundary**: the synthetic-trained GBM ≈ age-only on this proxy and < MoCA alone — the model transfers its ordering imperfectly to an outcome it never saw. Constructive deployment implication (now in paper): retrain/recalibrate the predictive layer on locally observed QC outcomes; architecture unchanged.

## Manuscript
Written into `manuscript/main-v9.tex` (red): new §"Validation Against Real MRI Quality-Control Signal" (`sec:qc_validation`, Table `tab:qc_validation`), plus red updates to abstract, intro, real-arm closing, Limitation 1, future research, conclusion.
