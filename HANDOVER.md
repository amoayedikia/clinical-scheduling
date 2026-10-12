# HCMS Extension — Handover (12 Oct 2026)

**Continuation prompt suggestion:** *"Read HANDOVER.md in my Healthcare-Management-Science folder (and the plan at project doc `claude/hcms-extension-plan.md`). Week 1 is COMPLETE (crosswalk, gates both GO, cohort builder proven on a real E1 iteration). Continue with Week 2–3 / Track A1: full E1–E7 re-runs on real NACC74 cohorts via nacc74_cohort_builder.py, reported side by side with the submitted synthetic results."*

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

## 3. Folder layout (reorganised 12 Oct 2026)

`~/Documents/Research-MRI/Healthcare-Management-Science/`
- **Code (repo root):** `ejor_clinical_scheduling_v3_fixed.py` (primary Phase-1), `baselines.py`, `experiments.py` (E1–E6), `e7_scaling_sensitivity.py`, `phase2_ml_validation.py`, `diagnostic_cognitive_predictors.py`, `ejor_clinical_scheduling.py` (legacy)
- **`data-nacc69/`** — NACC69 (June 2025) CSVs behind the submitted results. Symlinks in repo root preserve the filenames the scripts expect — do not break them.
- **`data-nacc74/`** — NACC74 (June 2026) parquets: `investigator_nacc74.parquet` (main UDS, 217,598 visits / 57,038 participants / 46 centers / 2005–2026), `investigator_mri_nacc74.parquet`, `investigator_fcsf_nacc74.parquet`, `scan_clariti_mri/` (QC series), `uds4-rdd.csv` (UDS4 data dictionary — crosswalk source), `rdd-gen.csv`
- **`manuscript/`** — `main-v8.tex` (current), `main-v5/v6.tex`, `references.bib`, `cover.tex`, figures, `Archive/` (EJOR/INFORMS-era versions)
- **`results/`, `results_phase_2_old/`** — experiment outputs
- Canonical NACC74 freeze stays in `Research-MRI/NACC-Dataset/NACC-June2026-DataFreeze/` (also feeds the separate MCCF project — don't move it)

## 4. Environment notes

- Claude's shell on the Mac is a sandboxed Linux VM (not the Mac's conda env). Already installed there this session: pyarrow, pandas, scikit-learn, scipy, pulp, tqdm. Reinstall with `pip3 install --break-system-packages ...` if a fresh session's VM lacks them.
- **UDS4 gotchas found so far:** `VISITYR` no longer exists in NACC74 — derive year from `VISITDATE` (string). Expect renames for cognitive instruments; map via `data-nacc74/uds4-rdd.csv`. NACCUDSD distribution in NACC74: NC 108,107 / impaired-not-MCI 9,769 / MCI 38,366 / dementia 61,326.
- Git from Claude's shell: tags/branches work, but `git commit` leaves lock files (no delete permission) — commit from Ali's own terminal.

## 5. Status: Week 1 COMPLETE (12 Oct 2026)

**Day 1:** Tag `hcms-submitted` + branch `nacc74-extension` created (still on `main`; housekeeping commit below still pending). Smoke test passed: `python3 experiments.py --quick` reproduces LPT 0.038 ~ MILP 0.046, age-stratified worst.
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

**Next (Week 2–3, new chat):** wire cohort builder into `experiments.py` (add `--cohort real` or equivalent: swap `generate_synthetic_cohort` → `sample_cohort` per replication with distinct seeds); run full E1–E7 real-arm; report side by side with submitted synthetic tables. Decide raw vs `--recalibrate` p_fail as primary (suggest: recalibrated primary, raw as sensitivity).
- Housekeeping: `_to_delete_crosswalk_draft.csv` in repo root is a leftover draft — delete from Ali's terminal (Claude's shell can't delete). Git commit still pending (see section 5).

## 7. Weeks 2–10 (from the plan)

| Week | Task |
|---|---|
| 2–3 | A1: E1–E7 re-runs on real NACC74 cohorts (side by side with synthetic) |
| 3–4 | A2: validate risk predictions against real QC degradation (AUC/rank corr.) |
| 4–5 | A3: new E8 — train on subset of centers/UDS3 era, test held-out centers/UDS4 era; when does LPT≈MILP break under shift? |
| 6 | Write-up: standalone supplement + response-to-reviewer blocks |
| 6–10 | Track B (contingency): integrate into manuscript as new Experiments subsections; rewrite Limitations 1 & 6; update cover letter; same contribution claim |

**Week-10 end state:** same paper, same central claim, substantially harder evidence (real heterogeneity + real QC signal) plus one new bounded result (E8 boundary conditions). ~80% deeper, ~20% broader.

## 8. Git policy

No commits needed during the work beyond the housekeeping one above; commit at milestones (end of A1, A2, A3) from Ali's terminal, and tag `nacc74-extension-complete` at week 10. The `hcms-submitted` tag guarantees the reviewed state stays reproducible.
