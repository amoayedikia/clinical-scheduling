# HCMS Extension — Handover (13 Oct 2026, 07:45)

**Continuation prompt suggestion:** *"Read docs/HANDOVER.md in my Healthcare-Management-Science folder (and the plan at project doc `claude/hcms-extension-plan.md`). Tracks A1 (Weeks 2–3) and A2 (Weeks 3–4) are COMPLETE and already written into manuscript/main-v9.tex in red (write-as-we-go). Continue with Weeks 4–5 / Track A3: new experiment E8 — center- and era-shift boundary conditions (train on UDS3 era / subset of centers, evaluate on UDS4 era / held-out centers; when does LPT≈MILP break?)."*

---

## 1. What this is

Extension of the HCMS submission **HCMS-D-26-00509** — *"Prediction Quality Versus Optimization Sophistication in Risk-Adjusted Clinical Scheduling"* (Moayedikia, Wiil, Fin; submitted 23 Jun 2026, under review). Two-phase prescriptive analytics: GBM predicts patient-specific MRI scan-failure probability (MoCA, CDR-SB, NPI-Q, demographics, imaging-QC features from NACC UDS3/NACC69), then stochastic job-shop MILP vs risk-first LPT heuristic + 5 baselines. Central finding: **prediction quality dominates optimization sophistication** — LPT ≈ MILP, age-based heuristics underperform even deterministic.

**Goal of the extension:** by the time the HCMS decision arrives, have real-data (NACC74 / June 2026 freeze) results ready — R&R ammunition if revised, upgraded manuscript if rejected.

## 2. Key decisions made

- **Add, don't replace.** All submitted simulation experiments (E1–E7) stay exactly as reviewed. Real-data arm runs alongside; results reported side by side (triangulation). Divergence = the boundary-conditions finding.
- **NACC cannot supply real binary failures** (~0.1% in matched records; pre-screened research cohort — see manuscript Limitation 1). So the real-data arm uses: (A1) real covariate heterogeneity for cohorts, (A2) graded SCAN/Clariti MRI QC metrics as continuous real outcome proxy, (A3) center/era shift for boundary conditions (new E8).
- **Keep calibrated failure mapping** (10–30% published range, 21.2% base) unchanged in A1.
- **Scope guard:** no-shows, multi-day scheduling, cross-domain → separate future papers, NOT this revision.
- **Resubmission ladder if rejected** (Archive shows prior EJOR + INFORMS versions — confirm actual rejection history before using): Omega (A) → Annals of OR (A) → Computers & OR (A).
- Full plan: project doc `claude/hcms-extension-plan.md` (MRI Scheduling project).

## 3. Folder layout (reorganised again 13 Oct 2026, midday)

`~/Documents/Research-MRI/Healthcare-Management-Science/`
- **Repo root:** `README.md` + `.gitignore` only (per Ali: no loose files).
- **`code/`** — all active pipeline code: `baselines.py`, `experiments.py` (E1–E6), `e7_scaling_sensitivity.py`, `e8_shift_experiment.py`, `nacc74_cohort_builder.py`, `a2_qc_validation.py`, `phase2_ml_validation.py` (imported by a2), `ejor_clinical_scheduling_v3_fixed.py` (Phase-1 reference). **Run from the repo root** (`python3 code/experiments.py ...`) — data/results paths resolve against the working directory.
- The NACC69 root symlinks were archived (`archive/nacc69-symlinks/`; they break when moved). Only the legacy NACC69 pipeline needs them; recreate from repo root with: `for f in investigator_nacc69 investigator_fcsf_nacc69 investigator_scan_mriqc_nacc69 investigator_scan_mrisbm_nacc69; do ln -s data-nacc69/$f.csv $f.csv; done`
- **`docs/`** — this `HANDOVER.md` + `crosswalk_uds3_uds4.csv` (UDS3→UDS4 variable crosswalk; tracked deliverables)
- **`data-nacc69/`** — NACC69 CSVs behind the submitted results (gitignored)
- **`data-nacc74/`** — NACC74 parquets + `scan_clariti_mri/` QC series + `uds4-rdd.csv`, `rdd-gen.csv` (gitignored). Canonical freeze stays in `Research-MRI/NACC-Dataset/NACC-June2026-DataFreeze/` — don't move it.
- **`manuscript/`** — `main-v9.tex` (current, red-highlighted extension), `main-v8.tex` (as submitted — untouched), `references.bib`, `cover.tex`, figures; `Archive/` (v5/v6 + EJOR/INFORMS-era versions)
- **`results/`** — root is the landing zone for new runs (currently `e8_run.log` + incoming `e8_shift_*.csv`); organised subfolders:
  - `real-arm/` — A1 real E1–E7 CSVs, LPT supplement, `real_arm_comparison_week23.md`, `real_arm_run.log`
  - `a2/` — A2 outputs (`a2_qc_validation_week34.md`, summary CSV; `a2_qc_scored.csv` gitignored)
  - `week1/` — Week-1 gate outputs (`week1_a2_*` gitignored)
  - `submitted-era/` — pre-extension synthetic outputs (May/June 2026 e2, e7)
- **`archive/`** — superseded material, safe to Trash entirely if desired: `code/` (`ejor_clinical_scheduling.py` legacy, `diagnostic_cognitive_predictors.py`, `real_arm_driver.py`, Week-1 one-shot scripts `build_crosswalk.py`/`finalize_crosswalk.py`/`week1_go_nogo.py`), `old-results/` (smoke/quick-test CSVs, cohort test samples, `optimal_schedule*.csv`, driver state), `results_phase_2_old/`

## 4. Environment notes

- Claude's shell on the Mac is a sandboxed Linux VM (not the Mac's conda env). Already installed there this session: pyarrow, pandas, scikit-learn, scipy, pulp, tqdm. Reinstall with `pip3 install --break-system-packages ...` if a fresh session's VM lacks them.
- **UDS4 gotchas found so far:** `VISITYR` no longer exists in NACC74 — derive year from `VISITDATE` (string). Expect renames for cognitive instruments; map via `data-nacc74/uds4-rdd.csv`. NACCUDSD distribution in NACC74: NC 108,107 / impaired-not-MCI 9,769 / MCI 38,366 / dementia 61,326.
- Background processes do NOT survive between Claude's shell calls (each call is its own PID namespace, killed at call end, ~2 min cap) — long runs go on Ali's own terminal; `real_arm_driver.py` is the resumable in-sandbox fallback.
- Ali's conda pandas hits a copy-on-write `replace()` bug (IndexError) — fixed in `nacc74_cohort_builder.py` by using `.mask(.isin())`.
- Git from Claude's shell: tags/branches work, but `git commit` leaves lock files (no delete permission) — commit from Ali's own terminal.

## 5. Status: Week 1 COMPLETE (12 Oct 2026)

**Day 1:** Tag `hcms-submitted` + branch `nacc74-extension` created (still on `main`; housekeeping commit below still pending). Smoke test passed: `python3 code/experiments.py --quick` reproduces LPT 0.038 ~ MILP 0.046, age-stratified worst.
- `.gitignore` extended: `data-nacc74/`, `*.parquet` (DUA). **Uncommitted** - run `git add -A && git commit -m "Reorganise + Week 1: crosswalk, go/no-go gates"` from terminal.

**Days 2-3 - crosswalk DONE:** `crosswalk_uds3_uds4.csv` (41 rows; built by `build_crosswalk.py` + `finalize_crosswalk.py`). Result: 32/39 variables identical UDS3->UDS4 (all NPI-Q, CDR, clinician-judgment, NACCAGE, MoCA names). Only real changes: VISITYR/MO/DAY -> single `VISITDATE` string 'YYYY-MM-DD'; STUDYDATE/STUDYQC same schema but moved to `data-nacc74/scan_clariti_mri/investigator_scan_clariti_mriqc_nacc74.parquet`; MOCATOTS/NACCMOCA flagged "same-harmonize" (UDS4 C2 administration changes - treat as same scale, era flag in sensitivity analysis). NACCMMSE dead (0% UDS4). Era split variable: `FORMVER` (UDS3 = {3.0, 3.2}: 106,876 visits; UDS4 = 4.0: 6,597; UDS1-2: 104,125).

**Days 3-4 - go/no-go gates DONE (`week1_go_nogo.py`, outputs in `results/week1_*.csv`):**
- **A1 GO.** Feature missingness essentially identical across eras (NPI-Q ~92-93%, CDR ~100% both eras); MoCA *better* in UDS4 era (89.8% vs 72.9%). Cohort pool = UDS3+UDS4 visits.
- **A2 GO - with modified outcome.** `STUDYQC` binary failure has no variance (8 fails / 6,124 matched studies = 0.13%, as expected) and `SERIESQC`=1 for all released series. BUT `SERIESQCREASONS` carries a graded signal: 44.6% of the 9,681 studies have >=1 series flagged (motion 6,696 series, motion_blurring, snr,motion, hyperintense_csf...). Overlap: 6,124 studies / 4,762 participants have a UDS visit within 90d (5,498 UDS3-era, 626 UDS4-era matches), 95% with MoCA. Sanity gradient already visible: any-motion-flag rate 53.4% (MoCA<=20) vs 38.1% (MoCA>=26); AGIT+ 46.6% vs 40.2%. **A2 outcome = per-study motion/QC-flag signal (any-motion, or flagged-series fraction), not binary failure.**

## 6. Week 1 Day 5 DONE + next steps

**Day 5 — cohort builder DONE:** `nacc74_cohort_builder.py` — loads NACC74 UDS parquet, applies crosswalk (parses `VISITDATE`, era via `FORMVER`), cleans missing codes, pools one latest eligible visit per participant (needs non-missing NACCAGE+MOCATOTS+CDRGLOB), maps real covariates → p_fail with the SAME formula as `generate_synthetic_cohort` (CDR>=1 = high-risk branch), emits List[PatientData].
- Pool: 30,167 participants (24,542 UDS3-era / 5,625 UDS4-era); p_fail mean 0.199 raw, 0.211 with `--recalibrate` (target 0.212). CDR>=1: 18.8% (vs 25% synthetic high-risk fraction).
- **E1 proof run passed (n=10 real cohort, 100 sims): LPT = MILP exactly (E[makespan] 689.4, OT prob 0.030; LPT ~0s vs MILP 5.7s); Age-Stratified worst OT prob (0.150). The paper's central pattern replicates on first contact with real covariates.**
- Era-shift preview for A3: UDS4-era pool is much healthier (CDR>=1 6.2%, p_fail mean 0.156 vs 0.199 overall) — real case-mix shift to exploit in E8.
- CLI: `--n --seed --era {both,UDS3,UDS4} --recalibrate --out --no-e1`. Test outputs in `results/nacc74_cohort_*.csv`.

**Weeks 2–3 — Track A1 COMPLETE (13 Oct 2026):**
- `experiments.py` + `e7_scaling_sensitivity.py` take `--cohort {synthetic,real} --era {both,UDS3,UDS4} --p-fail {recalibrated,raw}`. Real mode: NACC74 pool loaded once, fresh real cohort per replication (seed 42+rep), natural case-mix; E6 stratifies by CDRGLOB>=1. Synthetic default verified bit-for-bit identical. Output files tagged `_real`.
- **Full real E1–E7 run executed overnight (12→13 Oct) on Ali's own terminal** (~4.5 h; log `results/real_arm_run.log`). Outputs: `e1…e6_*_real_20261012_213030.csv`, `e7_scaling_sensitivity_real_20261013_020519.csv` (+ `_summary`), `e2_scalability_real_LPT_supplement_20261013.csv`.
- **Full side-by-side vs submitted synthetic tables: `results/real-arm/real_arm_comparison_week23.md`** (also in the MRI Scheduling project as `claude/hcms-real-arm-comparison.md`). Headline: **everything replicates, and gets stronger.** LPT = MILP (E1: 4.4% vs 4.4% OT, makespan 691.0 vs 691.1; E7 max gap 0.24 pp vs 0.47 synthetic); value of ML prediction 9.2 pp → **17.5 pp** (Age-Stratified hits 21.9% OT on real covariates vs 12.2% synthetic); No-Prediction beats age-based methods by 8–9 pp; E2 intractability at n≥20, the n=30 timeout inversion (MILP 1916.4 vs LPT 1906.8) and n=50 no-MILP-solution all replicate; E5 graceful degradation (plateau ≈9–10%, still beating Age-Strat at σ=0.20); E6 improvement up to 16.2 pp at 50% high-risk; E7 LPT range 1.77 pp vs Deterministic 14.37 pp. Overtime levels shift up ~1–10 pp across all methods (real age–cognition decoupling) with orderings/gaps preserved — triangulation, and the amplified age-heuristic penalty is new revision ammunition.
- Decisions: recalibrated p_fail (mean 0.212) = primary; raw (`--p-fail raw`) is a one-flag overnight sensitivity if a reviewer asks.
- Fix: `nacc74_cohort_builder.py` missing-code cleanup switched `.replace()` → `.mask(.isin())` (pandas copy-on-write IndexError on Ali's conda pandas); pool verified identical (30,167 / raw 0.199 / recal 0.211 / CDR≥1 18.8%).
- `real_arm_driver.py` + `archive/old-results/real_arm_driver_state/` = resumable chunked fallback for Claude's sandboxed shell (processes die per ~2-min call); superseded by the native run — safe to delete after commit.
- Housekeeping (Ali's terminal): git commit now also covers Week 2–3 (`git add -A && git commit -m "Weeks 2-3: real-cohort mode, E1-E7 real arm, side-by-side comparison"`); `_to_delete_crosswalk_draft.csv` still to delete.
- **Write-up policy (Ali, 13 Oct): write up and revise the manuscript as we go, not batched at Week 6.** A1 is written into `manuscript/main-v9.tex` (v8 = as-submitted, untouched): new \\subsection `sec:real_data_arm` (design + side-by-side Table tab:real_baseline + replication-summary Table tab:real_replication), plus red-highlighted edits to abstract, intro, experimental setup, discussion, Limitation 1, future research, conclusion, and the Data Availability Statement (now discloses sampling real UDS covariate vectors). ALL new/changed text wrapped in red (`\\rev{}` macro / `{\\color{red}...}` blocks) for Ali's review; strip by deleting the colour wrappers once approved. Not yet compiled (Overleaf workflow). A2 and A3 results get written into the manuscript immediately when done.
**Weeks 3–4 — Track A2 COMPLETE (13 Oct 2026):** `a2_qc_validation.py`; full write-up `results/a2/a2_qc_validation_week34.md` (also in project as `claude/hcms-a2-qc-validation.md`). 5,790 matched studies / 4,515 participants. Headline: predicted risk ordering is REAL (all AUC CIs exclude 0.5; top-vs-bottom GBM quintile any-motion 50.9% vs 41.0%); magnitude modest (AUC 0.54–0.57, anticipated: pre-screened cohort + graded proxy); **MoCA alone (0.573) beats age-only (0.538)** — real-outcome confirmation that cognition, not age, carries the signal; honest boundary: synthetic-trained GBM ≈ age-only on the real proxy (diff −0.000 [−0.018, 0.017]) — written into the paper with the retrain-on-QC-outcomes deployment implication. Robust to series-count, dedup, era, 30d window. `results/a2/a2_qc_scored.csv` is participant-level → added to .gitignore (DUA); summary CSV is committable. Manuscript updated same day (write-as-we-go): new §`sec:qc_validation` + Table `tab:qc_validation` + red edits to abstract/intro/Limitation 1/future research/conclusion in `main-v9.tex`.
- **Next (Weeks 4–5, Track A3):** new E8 — train/evaluate under center- and era-shift (UDS4-era pool is much healthier: CDR≥1 6.2%, p_fail mean 0.156 vs 0.199); when does LPT≈MILP break under shift? Then Week-6 supplement/response-to-reviewer blocks (manuscript already being updated as we go).

## 7. Weeks 2–10 (from the plan)

| Week | Task |
|---|---|
| 2–3 | A1: E1–E7 re-runs on real NACC74 cohorts (side by side with synthetic) — **DONE 13 Oct** |
| 3–4 | A2: validate risk predictions against real QC degradation (AUC/rank corr.) — **DONE 13 Oct** |
| 4–5 | A3: new E8 — train on subset of centers/UDS3 era, test held-out centers/UDS4 era; when does LPT≈MILP break under shift? |
| 6 | Write-up: standalone supplement + response-to-reviewer blocks |
| 6–10 | Track B (contingency): integrate into manuscript as new Experiments subsections; rewrite Limitations 1 & 6; update cover letter; same contribution claim |

**Week-10 end state:** same paper, same central claim, substantially harder evidence (real heterogeneity + real QC signal) plus one new bounded result (E8 boundary conditions). ~80% deeper, ~20% broader.

## 8. Git policy

No commits needed during the work beyond the housekeeping one above; commit at milestones (end of A1, A2, A3) from Ali's terminal, and tag `nacc74-extension-complete` at week 10. The `hcms-submitted` tag guarantees the reviewed state stays reproducible.
