# Real-Data Arm (NACC74) vs Submitted Synthetic Results — Track A1, Weeks 2–3
*13 Oct 2026. Run: overnight 12→13 Oct on Ali's Mac (`results/real_arm_run.log`), `--cohort real`, recalibrated p_fail (pool mean 0.212), NACC74 pool n=30,167 (natural case-mix, CDR≥1 = 18.8%), distinct cohort seed per replication (42+rep). Protocol otherwise identical to the submitted runs (same replication counts, 100 MC simulations, seeds, methods, solver timeouts incl. E2 max(60,6n)). "Synthetic" columns are the submitted values from `manuscript/main-v8.tex`.*

## Headline

**The paper's central finding replicates on real covariate heterogeneity — and gets stronger.**
1. **LPT ≈ Stochastic-MILP everywhere.** E1: identical to one decimal (4.4% vs 4.4% overtime; makespan 691.0 vs 691.1). E7: max |LPT − MILP| gap 0.24 pp across all nine scaling combinations (synthetic: 0.47 pp).
2. **The value of ML prediction nearly doubles on real data.** LPT vs Age-Stratified: 9.2 pp (synthetic) → **17.5 pp** (real). Real cohorts decouple age from cognition more than the synthetic generator assumed, so age-based buffers misallocate more severely. The submitted simulation, if anything, *understated* the case for prediction quality.
3. Overtime levels shift up ~1–10 pp across all methods on real cohorts (heavier-tailed real covariate joint distribution); orderings and gaps are preserved throughout — triangulation, not divergence.

## E1 — Baseline comparison (100 reps × 10 patients × 100 sims)

| Method | Synthetic mksp | Synthetic OT | Real mksp | Real OT |
|---|---|---|---|---|
| LPT | 689.0 ± 0.4 | 3.0% ± 0.4 | 691.0 ± 0.7 | 4.4% ± 0.5 |
| Stochastic-MILP | 689.1 ± 0.4 | 3.1% ± 0.4 | 691.1 ± 0.7 | 4.4% ± 0.5 |
| Deterministic | 689.0 ± 0.4 | 9.4% ± 0.8 | 691.0 ± 0.7 | 12.7% ± 1.2 |
| Random | 695.4 ± 0.7 | 6.2% ± 0.5 | 697.5 ± 1.3 | 8.1% ± 0.8 |
| FCFS | 698.1 ± 1.2 | 7.0% ± 0.7 | 697.4 ± 1.5 | 7.6% ± 0.9 |
| SPT | 715.0 ± 0.7 | 9.8% ± 0.7 | 712.9 ± 1.9 | 11.1% ± 1.1 |
| Age-Stratified | 692.4 ± 0.6 | 12.2% ± 1.0 | 698.1 ± 1.7 | **21.9% ± 2.8** |

Value decomposition (real): ML prediction (LPT vs Age-Strat) **17.5 pp**; risk adjustment (LPT vs Det) **8.3 pp**; optimization (MILP vs LPT) **0.0 pp**. (Synthetic: 9.2 / 6.4 / 0.1 pp.) MILP solve time real: mean 17.3 s, max 60.1 s (synthetic: 20.8 / 60.1) — same 4-orders-of-magnitude overhead for zero gain.

## E2 — Scalability (10 reps/size; MILP timeout max(60,6n); LPT supplement in `e2_scalability_real_LPT_supplement_20261013.csv`)

| n | Synth MILP time | Synth optimal | Synth MILP/LPT mksp | Real MILP time | Real optimal | Real MILP mksp | Real LPT mksp | Real LPT OT |
|---|---|---|---|---|---|---|---|---|
| 10 | 17.1 ± 17.3 | 9/10 | 687.1 / 687.1 | 15.4 ± 20.2 | 9/10 | 689.0 | 689.0 | 4.3% |
| 20 | 120.2 (timeout) | 0/10 | 1297.0 / 1297.0 | 120.1 (timeout) | 0/10 | 1300.0 | 1298.2 | 0.6% |
| 30 | 180.2 (timeout) | 0/10 | 1913.1 / 1905.7 | 180.1 (timeout) | 0/10 | **1916.4** | **1906.8** | 0.1% |
| 50 | — (none in 300 s) | — | — / 3124.8 | — (none in 300 s) | — | — | 3124.9 | 0.0% |

Replicates all three submitted patterns: intractability at n≥20; the **timeout inversion** (MILP *worse* than LPT at n=30: +9.6 min mean, 0.50% vs submitted 0.39%); LPT overtime falling monotonically with n (4.3 → 0.6 → 0.1 → 0.0%; law of large numbers).

## E3 — Lambda sensitivity (5 λ × 20 reps)

Real: makespan flat (688.3→688.5) across λ ∈ {0, 0.25, 0.5, 1, 2}; total wait collapses from 340.7 min (λ=0) to ≈0 for λ≥0.25. Same qualitative trade-off structure as submitted: tiny wait penalties suffice, at negligible makespan cost.

## E4 — Value of ML (50 reps)

| Predictor | Synthetic OT | Real OT |
|---|---|---|
| Full ML | 3.1% ± 0.6 | 4.3% ± 0.6 |
| No Prediction | 8.7% ± 1.1 | 12.6% ± 1.6 |
| Age-Stratified | 11.7% ± 1.4 | 21.9% ± 4.0 |
| Age-Only ML | 12.0% ± 1.6 | 21.0% ± 3.3 |

All three submitted findings replicate, amplified: (i) Age-Only ML ≈ Age-Stratified rules (ML on age features alone adds nothing); (ii) No-Prediction *beats* both age-based methods (age buffers actively misallocate — now by 8–9 pp rather than ~3 pp); (iii) Full-ML vs Age-Only gap widens from 8.9 to **16.7 pp**.

## E5 — Robustness to prediction error (4 σ × 30 reps)

| σ | Synthetic OT | Real OT |
|---|---|---|
| 0 (E1/E4 ref) | 3.1% | 4.4% |
| 0.05 | 5.7% | 7.2% |
| 0.10 | ~7.6% | 9.1% |
| 0.15 | ~7.6% | 10.3% |
| 0.20 | ~7.6% | 9.1% |

Graceful degradation with a plateau replicates (real plateau ≈9–10%). Even at σ=0.20, noisy-ML scheduling (9.1%) beats Age-Stratified (21.9%) by a wider margin than in the submitted runs.

## E6 — Cohort composition (CDR≥1-stratified real sampling)

Overtime %, Deterministic → Stochastic-MILP (improvement):

| High-risk frac | Synthetic | Real |
|---|---|---|
| 0% | 5.8 → 4.6 (1.2 pp) | 8.7 → 5.8 (2.9 pp) |
| 10% | — | 10.6 → 3.4 |
| 20% | — | 12.5 → 3.6 |
| 30% | — | 16.1 → 4.6 |
| 50% | 19.4 → 5.8 (13.6 pp) | 23.7 → 7.5 (**16.2 pp**) |

MILP stays in a 3.4–7.5% band while Deterministic degrades monotonically — the "highest value in high-risk clinics" conclusion replicates with larger benefits.

## E7 — Scaling-factor sensitivity (9 combos × 30 reps)

| | Synthetic | Real |
|---|---|---|
| LPT OT band (range) | 2.40–3.97% (1.57 pp) | 4.00–5.77% (1.77 pp) |
| MILP OT band (range) | 2.50–4.27% (1.77 pp) | 4.03–5.87% (1.84 pp) |
| Deterministic range | 11.23 pp | 14.37 pp |
| Age-Stratified range | 11.80 pp | 14.80 pp |
| Max \|LPT − MILP\| | 0.47 pp | **0.24 pp** |
| Baseline combo (0.30, 0.80) LPT / MILP | 3.03 / 3.23% | 4.37 / 4.40% |

Risk-aware sequencing is again both best and most robust to scaling-factor miscalibration; non-predictive baselines degrade faster on real cohorts.

## Notes & caveats for the write-up

- **Outcome mapping is still the calibrated published-rate formula** (by design — plan A1). Real *outcome* evidence is A2 (QC-signal validation, Weeks 3–4). What this arm establishes is that the findings survive the real joint covariate distribution and within-pool correlation structure that the synthetic generator could not capture.
- Primary p_fail = recalibrated (pool mean 0.212). Raw sensitivity (`--p-fail raw`, mean 0.199) not yet run; it's one overnight flag if a reviewer asks.
- The upward level shift in overtime (~1–10 pp) and the doubling of the age-heuristic penalty come from real age–cognition decoupling (e.g., old-but-cognitively-intact and young-but-impaired participants are common in NACC74); worth a paragraph, not just a table.
- E2 deviation: none in the native run (full max(60,6n) timeouts, incl. 300 s at n=50, where — as in the submitted run — CBC returned no solution). LPT rows computed separately (supplement CSV) to mirror the manuscript's scalability table.
- Raw files: `e1…e6_*_real_20261012_213030.csv`, `e7_scaling_sensitivity_real_20261013_020519.csv` (+ `_summary`), `e2_scalability_real_LPT_supplement_20261013.csv`.
