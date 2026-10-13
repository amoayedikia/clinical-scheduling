"""
A2: VALIDATE PHASE-1 RISK PREDICTIONS AGAINST REAL MRI QC SIGNAL (Weeks 3-4)

Outcome (per Week-1 gate): binary scan failure has no variance in NACC
(pre-screened research setting), but SCAN/Clariti SERIESQCREASONS carries a
graded signal. Study-level outcomes:
  any_motion  - >=1 series in the study flagged for motion (PRIMARY)
  any_flag    - >=1 series flagged for any QC reason
  frac_flag   - fraction of the study's series flagged

Predictors validated:
  gbm       - Phase-1 GradientBoosting model (identical spec to
              ejor_clinical_scheduling_v3_fixed.build_predictive_model),
              trained on the paper's simulation-augmented 2,000-patient cohort
              (phase2_ml_validation.generate_synthetic_dataset, seed 42),
              scored on real matched covariates.
  calib     - the published-rate-calibrated p_fail mapping (as used in the
              scheduling experiments; stochastic term at its mean, u=0.5).
  age_only  - the E4 age-only predictor: clip(0.05+max(0,age-65)*.005,.03,.40)
  moca_only - 30 - MoCA (single-instrument reference)

Metrics: AUC vs any_motion / any_flag (cluster bootstrap CI by participant),
Spearman rho vs frac_flag, GBM-vs-age-only AUC difference (paired cluster
bootstrap), risk-quintile gradient, era and match-window sensitivity.

Usage: python3 a2_qc_validation.py [--n-boot 1000] [--window 90]
Outputs: results/a2_qc_scored.csv (participant-level; NOT for git),
         results/a2_qc_validation_summary.csv, console report.
"""
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr

from nacc74_cohort_builder import UDS_PARQUET, MISSING, FEATURES

RNG = np.random.default_rng(42)

# NACC column -> synthetic-training feature name (phase2 generate_synthetic_dataset)
NACC2SYN = {'AGIT': 'agit', 'MOT': 'mot', 'ANX': 'anx',
            'AGITSEV': 'agit_sev', 'MOTSEV': 'mot_sev', 'ANXSEV': 'anx_sev',
            'MOCATOTS': 'moca', 'CDRSUM': 'cdr_sum', 'CDRGLOB': 'cdr',
            'NACCAGE': 'age', 'DECCLCOG': 'decclcog',
            'IRR': 'irr', 'DEL': 'del', 'HALL': 'hall', 'DEPD': 'dep'}
BIN_COLS = ['agit', 'mot', 'anx', 'irr', 'del', 'hall', 'dep', 'apa', 'disn']


def build_matched(window=90):
    qc = pd.read_csv('results/week1/week1_a2_study_level_qc.csv')
    qc['MRI_DATE'] = pd.to_datetime(qc['STUDYDATE'])
    cols = ['NACCID', 'VISITDATE', 'FORMVER'] + FEATURES
    uds = pq.ParquetFile(UDS_PARQUET).read(columns=cols).to_pandas()
    uds['UDS_DATE'] = pd.to_datetime(uds['VISITDATE'], errors='coerce')
    uds['ERA'] = np.select([uds.FORMVER.isin([3.0, 3.2]), uds.FORMVER == 4.0],
                           ['UDS3', 'UDS4'], default='UDS1-2')
    uds = uds[uds.ERA.isin(['UDS3', 'UDS4'])].dropna(subset=['UDS_DATE'])
    uds[FEATURES] = uds[FEATURES].mask(uds[FEATURES].isin(MISSING))

    m = qc.merge(uds, on='NACCID', how='inner')
    m['DD'] = (m.MRI_DATE - m.UDS_DATE).dt.days.abs()
    m = m[m.DD <= window]
    m = m.sort_values('DD').drop_duplicates(['NACCID', 'STUDYDATE'], keep='first')
    print(f"Matched within {window}d: {len(m)} studies / {m.NACCID.nunique()} participants "
          f"(era mix: {dict(m.ERA.value_counts())})")
    m = m.dropna(subset=['NACCAGE', 'MOCATOTS', 'CDRGLOB']).reset_index(drop=True)
    print(f"Analysis set (non-missing age/MoCA/CDR-global): {len(m)} studies / "
          f"{m.NACCID.nunique()} participants | any_motion rate {m.any_motion.mean():.3f}")
    return m


def engineer(m):
    d = pd.DataFrame({syn: m[nacc] for nacc, syn in NACC2SYN.items()})
    d['apa'], d['disn'] = m['APA'], m['DISN']
    for c in BIN_COLS:                       # NPI-Q binaries: absent/missing -> 0
        d[c] = d[c].fillna(0).clip(0, 1)
    for c in ['agit_sev', 'mot_sev', 'anx_sev', 'dep_sev']:
        base = {'agit_sev': 'agit', 'mot_sev': 'mot', 'anx_sev': 'anx', 'dep_sev': 'dep'}[c]
        if c == 'dep_sev':
            d[c] = m['DEPDSEV'].fillna(0)
        d[c] = d[c].fillna(0) * d[base]
    d['decclcog'] = d['decclcog'].fillna(0)
    d['cdr_sum'] = d['cdr_sum'].fillna(d['cdr_sum'].median())
    d['COGNITIVE_IMPAIRMENT'] = 30 - d['moca']
    d['AGE_RISK'] = (d['age'] - 65).clip(lower=0) / 20
    d['CDR_IMPAIRED'] = (d['cdr'] >= 1).astype(int)
    d['CDR_SEVERITY'] = d['cdr_sum'].clip(lower=0)
    d['BEHAVIORAL_RISK'] = d[BIN_COLS].sum(axis=1)
    d['SEVERITY_WEIGHTED_RISK'] = (d['agit'] * d['agit_sev'] + d['anx'] * d['anx_sev'] +
                                   d['mot'] * d['mot_sev'] + d['dep'] * d['dep_sev'])
    d['MRI_SPECIFIC_RISK'] = d['agit'] + d['mot'] + d['anx']
    return d


def train_phase1_gbm():
    import importlib.util as iu
    import os as _os
    spec = iu.spec_from_file_location('p2', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'phase2_ml_validation.py'))
    p2 = iu.module_from_spec(spec); spec.loader.exec_module(p2)
    X, y, feats, _ = p2.generate_synthetic_dataset(n_patients=2000, seed=42, verbose=False)
    imp = SimpleImputer(strategy='median')
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                       min_samples_leaf=20, random_state=42)
    model.fit(imp.fit_transform(X), y)
    return model, imp, feats


def fast_auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # midranks for ties
    sr = pd.Series(s).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return (sr[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cluster_boot(df, stat_fn, n_boot, seed=42):
    rng = np.random.default_rng(seed)
    ids = df['NACCID'].values
    uniq = pd.unique(ids)
    groups = {u: np.flatnonzero(ids == u) for u in uniq}
    stats = []
    for _ in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[u] for u in take])
        b = df.iloc[idx]
        v = stat_fn(b)
        if v is not None and np.isfinite(v):
            stats.append(v)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--window', type=int, default=90)
    a = ap.parse_args()

    m = build_matched(a.window)
    d = engineer(m)

    model, imp, feats = train_phase1_gbm()
    m['score_gbm'] = model.predict_proba(imp.transform(d[feats].values))[:, 1]
    hi = d['cdr'] >= 1
    m['score_calib'] = np.where(
        hi, np.clip(0.25 + 0.25 * 0.5 + (d['age'] - 75) * 0.003 + (26 - d['moca']) * 0.008, 0.25, 0.55),
        np.clip(0.05 + 0.15 * 0.5 + np.maximum(0, d['age'] - 70) * 0.002, 0.03, 0.24))
    m['score_age'] = np.clip(0.05 + np.maximum(0, d['age'] - 65) * 0.005, 0.03, 0.40)
    m['score_moca'] = 30 - d['moca']
    m['CDR_IMPAIRED'] = d['CDR_IMPAIRED']

    m.to_csv('results/a2/a2_qc_scored.csv', index=False)

    rows = []
    print(f"\n=== A2 DISCRIMINATION (n={len(m)} studies, {m.NACCID.nunique()} participants, "
          f"window {a.window}d, {a.n_boot} cluster-bootstrap reps) ===")
    for outc in ['any_motion', 'any_flag']:
        y = m[outc].astype(int).values
        for sc in ['score_gbm', 'score_calib', 'score_age', 'score_moca']:
            auc = fast_auc(y, m[sc].values)
            lo, hi_ = cluster_boot(m, lambda b, s=sc, o=outc: fast_auc(b[o].astype(int).values, b[s].values)
                                   if 0 < b[o].sum() < len(b) else None, a.n_boot)
            rows.append({'analysis': 'AUC', 'outcome': outc, 'predictor': sc,
                         'value': round(auc, 4), 'ci_lo': round(lo, 4), 'ci_hi': round(hi_, 4), 'n': len(m)})
            print(f"AUC {outc:<11} {sc:<12} {auc:.3f} [{lo:.3f}, {hi_:.3f}]")

    # paired AUC difference, gbm - age_only, primary outcome
    def auc_diff(b):
        y = b['any_motion'].astype(int).values
        if not 0 < y.sum() < len(y): return None
        return fast_auc(y, b['score_gbm'].values) - fast_auc(y, b['score_age'].values)
    dv = auc_diff(m); lo, hi_ = cluster_boot(m, auc_diff, a.n_boot)
    rows.append({'analysis': 'AUC_diff_gbm_minus_age', 'outcome': 'any_motion', 'predictor': 'gbm-age',
                 'value': round(dv, 4), 'ci_lo': round(lo, 4), 'ci_hi': round(hi_, 4), 'n': len(m)})
    print(f"\nAUC difference (GBM - age-only), any_motion: {dv:.3f} [{lo:.3f}, {hi_:.3f}]")

    # Spearman vs frac_flag
    for sc in ['score_gbm', 'score_calib', 'score_age', 'score_moca']:
        rho, p = spearmanr(m[sc], m['frac_flag'])
        lo, hi_ = cluster_boot(m, lambda b, s=sc: spearmanr(b[s], b['frac_flag'])[0], a.n_boot)
        rows.append({'analysis': 'spearman_frac_flag', 'outcome': 'frac_flag', 'predictor': sc,
                     'value': round(rho, 4), 'ci_lo': round(lo, 4), 'ci_hi': round(hi_, 4), 'n': len(m)})
        print(f"Spearman frac_flag {sc:<12} rho={rho:.3f} [{lo:.3f}, {hi_:.3f}] (p={p:.1e})")

    # risk-quintile gradient (GBM)
    m['q'] = pd.qcut(m['score_gbm'], 5, labels=False, duplicates='drop') + 1
    grad = m.groupby('q').agg(n=('any_motion', 'size'), any_motion_rate=('any_motion', 'mean'),
                              frac_flag_mean=('frac_flag', 'mean'), score_mean=('score_gbm', 'mean')).round(3)
    print("\nGBM risk-quintile gradient:")
    print(grad.to_string())
    for q, r in grad.iterrows():
        rows.append({'analysis': 'quintile_gradient', 'outcome': 'any_motion', 'predictor': f'gbm_q{q}',
                     'value': r['any_motion_rate'], 'ci_lo': np.nan, 'ci_hi': np.nan, 'n': int(r['n'])})

    # subgroups / sensitivity (point estimates)
    for era in ['UDS3', 'UDS4']:
        s = m[m.ERA == era]
        if s.any_motion.nunique() == 2:
            v = fast_auc(s.any_motion.astype(int).values, s.score_gbm.values)
            rows.append({'analysis': 'AUC_era', 'outcome': 'any_motion', 'predictor': f'gbm_{era}',
                         'value': round(v, 4), 'ci_lo': np.nan, 'ci_hi': np.nan, 'n': len(s)})
            print(f"Era {era}: GBM AUC {v:.3f} (n={len(s)})")
    s = m[m.DD <= 30]
    v = fast_auc(s.any_motion.astype(int).values, s.score_gbm.values)
    rows.append({'analysis': 'AUC_window30', 'outcome': 'any_motion', 'predictor': 'gbm_dd30',
                 'value': round(v, 4), 'ci_lo': np.nan, 'ci_hi': np.nan, 'n': len(s)})
    print(f"Match window <=30d: GBM AUC {v:.3f} (n={len(s)})")
    # CDR-impaired subgroup rates (sanity tie to Week-1 gradient)
    print(f"\nSanity: any_motion rate CDR>=1 {m[m.CDR_IMPAIRED==1].any_motion.mean():.3f} "
          f"vs CDR<1 {m[m.CDR_IMPAIRED==0].any_motion.mean():.3f}")

    pd.DataFrame(rows).to_csv('results/a2/a2_qc_validation_summary.csv', index=False)
    print("\nSaved: results/a2/ CSVs")


if __name__ == '__main__':
    main()
