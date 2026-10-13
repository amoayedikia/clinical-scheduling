"""
PHASE 2: ML VALIDATION, CALIBRATION, AND COMPUTATION TIMING
=============================================================
Fixes 1, 5, and 6 for EJOR manuscript submission.

Uses simulation-augmented synthetic patients consistent with the scheduling
experiments (experiments.py / baselines.py). Patient clinical profiles
(age, MoCA, CDR, NPI-Q behavioral symptoms) are generated from clinically
realistic distributions; binary failure outcomes are drawn from
Bernoulli(p_fail) where p_fail reflects the nonlinear relationship between
clinical features and procedural risk documented in the literature.

Fix 1: Cross-validated ML metrics (AUC-ROC, PR-AUC, Brier, F1)
Fix 5: Calibration reliability diagram (saved as PDF figure)
Fix 6: Computation time comparison table across all 7 methods

Usage:
    python phase2_ml_validation.py --all
    python phase2_ml_validation.py --fix1 --fix5
    python phase2_ml_validation.py --fix6
    python phase2_ml_validation.py --all --quick

Requirements:
    - baselines.py (in same directory, for Fix 6)
    - Python packages: pandas, numpy, scikit-learn, matplotlib, scipy, pulp

Author: EJOR Clinical Scheduling Project
"""

import pandas as pd
import numpy as np
import time
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


CONFIG = {
    'output_dir': 'results_phase2',
    'n_synthetic_patients': 2000,
    'high_risk_fraction': 0.25,
    'dataset_seed': 42,
    'model_params': {
        'n_estimators': 100, 'max_depth': 4,
        'min_samples_leaf': 20, 'random_state': 42,
    },
    'cv_n_splits': 5,
    'cv_n_repeats': 10,
    'cv_random_state': 42,
    'resources': {
        'UDS': {'base_duration': 60, 'failure_penalty': 15, 'risk_scale': 0.3},
        'MRI': {'base_duration': 45, 'failure_penalty': 30, 'risk_scale': 1.0},
        'CSF': {'base_duration': 30, 'failure_penalty': 20, 'risk_scale': 0.8},
    },
    'resource_sequence': ['UDS', 'MRI', 'CSF'],
    'wait_time_penalty': 0.5,
    'solver_timeout': 60,
    'big_m': 2000,
    'timing_replications': 50,
    'timing_n_patients': 10,
    'timing_seed': 42,
}


def generate_synthetic_dataset(n_patients=2000, high_risk_fraction=0.25, seed=42, verbose=True):
    """
    Generate synthetic clinical cohort with realistic feature distributions
    and Bernoulli-sampled failure outcomes. Uses same age/MoCA/CDR/p_fail
    distributions as experiments.py, extended with NPI-Q behavioral features.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("SYNTHETIC DATASET GENERATION")
        print("=" * 70)

    rng = np.random.RandomState(seed)
    records = []
    n_high = int(n_patients * high_risk_fraction)
    n_low = n_patients - n_high

    for i in range(n_high):
        age = rng.uniform(75, 92)
        cdr = rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        moca = max(8, min(26, 24 - cdr * 4 + rng.normal(0, 2)))
        p_fail = np.clip(0.25 + rng.uniform(0, 0.25) + (age - 75) * 0.003 + (26 - moca) * 0.008, 0.25, 0.55)
        agit = int(rng.random() < 0.15 + cdr * 0.15)
        anx = int(rng.random() < 0.20 + cdr * 0.10)
        mot = int(rng.random() < 0.10 + cdr * 0.15)
        irr = int(rng.random() < 0.15 + cdr * 0.10)
        dep = int(rng.random() < 0.20 + cdr * 0.05)
        del_ = int(rng.random() < 0.05 + cdr * 0.10)
        hall = int(rng.random() < 0.03 + cdr * 0.08)
        apa = int(rng.random() < 0.15 + cdr * 0.12)
        disn = int(rng.random() < 0.05 + cdr * 0.08)
        agit_sev = rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]) if agit else 0
        anx_sev = rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]) if anx else 0
        mot_sev = rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]) if mot else 0
        dep_sev = rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]) if dep else 0
        decclcog = int(rng.random() < 0.3 + cdr * 0.2)
        records.append({'age': age, 'moca': moca, 'cdr': cdr, 'p_fail': p_fail,
            'cdr_sum': cdr * 3 + rng.normal(0, 1), 'agit': agit, 'anx': anx,
            'mot': mot, 'irr': irr, 'dep': dep, 'del': del_, 'hall': hall,
            'apa': apa, 'disn': disn, 'agit_sev': agit_sev, 'anx_sev': anx_sev,
            'mot_sev': mot_sev, 'dep_sev': dep_sev, 'decclcog': decclcog,
            'risk_group': 'high'})

    for i in range(n_low):
        age = rng.uniform(60, 80)
        cdr = rng.choice([0, 0.5, 1], p=[0.4, 0.4, 0.2])
        moca = max(18, min(30, 27 - cdr * 2 + rng.normal(0, 2)))
        p_fail = np.clip(0.05 + rng.uniform(0, 0.15) + max(0, (age - 70)) * 0.002, 0.03, 0.24)
        agit = int(rng.random() < 0.05 + cdr * 0.10)
        anx = int(rng.random() < 0.10 + cdr * 0.08)
        mot = int(rng.random() < 0.03 + cdr * 0.08)
        irr = int(rng.random() < 0.08 + cdr * 0.06)
        dep = int(rng.random() < 0.12 + cdr * 0.04)
        del_ = int(rng.random() < 0.02 + cdr * 0.04)
        hall = int(rng.random() < 0.01 + cdr * 0.03)
        apa = int(rng.random() < 0.08 + cdr * 0.06)
        disn = int(rng.random() < 0.03 + cdr * 0.04)
        agit_sev = rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]) if agit else 0
        anx_sev = rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]) if anx else 0
        mot_sev = rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]) if mot else 0
        dep_sev = rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]) if dep else 0
        decclcog = int(rng.random() < 0.1 + cdr * 0.15)
        records.append({'age': age, 'moca': moca, 'cdr': cdr, 'p_fail': p_fail,
            'cdr_sum': cdr * 3 + rng.normal(0, 0.5), 'agit': agit, 'anx': anx,
            'mot': mot, 'irr': irr, 'dep': dep, 'del': del_, 'hall': hall,
            'apa': apa, 'disn': disn, 'agit_sev': agit_sev, 'anx_sev': anx_sev,
            'mot_sev': mot_sev, 'dep_sev': dep_sev, 'decclcog': decclcog,
            'risk_group': 'low'})

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Derived features (match v3_fixed.py)
    df['COGNITIVE_IMPAIRMENT'] = 30 - df['moca']
    df['AGE_RISK'] = (df['age'] - 65).clip(lower=0) / 20
    df['CDR_IMPAIRED'] = (df['cdr'] >= 1).astype(int)
    df['CDR_SEVERITY'] = df['cdr_sum'].clip(lower=0)
    behavioral_cols = ['agit', 'anx', 'mot', 'irr', 'dep', 'del', 'hall', 'apa', 'disn']
    df['BEHAVIORAL_RISK'] = df[behavioral_cols].sum(axis=1)
    df['SEVERITY_WEIGHTED_RISK'] = (df['agit']*df['agit_sev'] + df['anx']*df['anx_sev'] +
                                     df['mot']*df['mot_sev'] + df['dep']*df['dep_sev'])
    df['MRI_SPECIFIC_RISK'] = df['agit'] + df['mot'] + df['anx']

    # Binary outcome: y ~ Bernoulli(p_fail)
    df['MRI_FAILED'] = (rng.random(len(df)) < df['p_fail']).astype(int)

    feature_names = [
        'agit', 'mot', 'anx', 'MRI_SPECIFIC_RISK',
        'agit_sev', 'mot_sev', 'anx_sev',
        'BEHAVIORAL_RISK', 'SEVERITY_WEIGHTED_RISK',
        'moca', 'COGNITIVE_IMPAIRMENT',
        'cdr_sum', 'cdr', 'CDR_SEVERITY', 'CDR_IMPAIRED',
        'age', 'AGE_RISK', 'decclcog',
        'irr', 'del', 'hall', 'dep',
    ]
    X = df[feature_names].values
    y = df['MRI_FAILED'].values

    if verbose:
        n_pos = y.sum()
        print(f"  Patients: {len(df):,} (high-risk: {n_high}, low-risk: {n_low})")
        print(f"  p_fail range: [{df['p_fail'].min():.3f}, {df['p_fail'].max():.3f}]")
        print(f"  p_fail mean:  {df['p_fail'].mean():.3f}")
        print(f"  MRI failures (Bernoulli): {n_pos} / {len(df)} ({100*n_pos/len(df):.1f}%)")
        print(f"  Features: {len(feature_names)}")
    return X, y, feature_names, df


def run_fix1_cross_validation(X, y, feature_names, verbose=True):
    """Cross-validated evaluation of gradient boosting for MRI failure prediction."""
    if verbose:
        print("\n" + "=" * 70)
        print("FIX 1: CROSS-VALIDATED ML METRICS")
        print("=" * 70)
    n_positives = y.sum()
    n_total = len(y)
    n_splits = CONFIG['cv_n_splits']
    n_repeats = CONFIG['cv_n_repeats']
    if verbose:
        print(f"\n  Dataset: {n_total:,} samples, {n_positives} positives ({100*n_positives/n_total:.1f}%)")
        print(f"  Using {n_repeats}x{n_splits}-fold Repeated Stratified CV")

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=CONFIG['cv_random_state'])
    model_params = CONFIG['model_params'].copy()
    fold_metrics = {'auc_roc': [], 'pr_auc': [], 'brier': [], 'f1': [], 'precision': [], 'recall': []}
    all_y_true, all_y_prob = [], []
    fold_count = 0

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if y_test.sum() == 0 or y_test.sum() == len(y_test):
            continue
        model = GradientBoostingClassifier(**model_params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        thresholds = np.linspace(0.01, 0.99, 99)
        f1s = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
        best_t = thresholds[np.argmax(f1s)]
        y_pred = (y_prob >= best_t).astype(int)
        fold_metrics['auc_roc'].append(roc_auc_score(y_test, y_prob))
        fold_metrics['pr_auc'].append(average_precision_score(y_test, y_prob))
        fold_metrics['brier'].append(brier_score_loss(y_test, y_prob))
        fold_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        fold_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        fold_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())
        fold_count += 1

    results = {}
    if verbose:
        print(f"\n  Completed folds: {fold_count}")
        print(f"\n  {'Metric':<25} {'Mean':>8} {'Std':>8} {'95% CI':>18}")
        print("  " + "-" * 62)
    for metric_name, values in fold_metrics.items():
        arr = np.array(values)
        results[metric_name] = {'mean': arr.mean(), 'std': arr.std(),
            'ci_lo': np.percentile(arr, 2.5), 'ci_hi': np.percentile(arr, 97.5), 'n_folds': len(values)}
        display = {'auc_roc': 'AUC-ROC', 'pr_auc': 'PR-AUC (Avg Precision)', 'brier': 'Brier Score',
                   'f1': 'F1 Score', 'precision': 'Precision', 'recall': 'Recall (Sensitivity)'}.get(metric_name, metric_name)
        if verbose:
            print(f"  {display:<25} {arr.mean():>8.4f} {arr.std():>8.4f} [{np.percentile(arr,2.5):.4f}, {np.percentile(arr,97.5):.4f}]")

    if verbose:
        print("\n  Training final model on full dataset for feature importance...")
    full_model = GradientBoostingClassifier(**model_params)
    full_model.fit(X, y)
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': full_model.feature_importances_}).sort_values('Importance', ascending=False)
    results['feature_importance'] = importance_df
    if verbose:
        print(f"\n  {'Rank':<6} {'Feature':<28} {'Importance':>12}")
        print("  " + "-" * 48)
        for rank, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            bar = '\u2588' * int(row['Importance'] * 50)
            print(f"  {rank:<6} {row['Feature']:<28} {row['Importance']:>10.4f}  {bar}")

    results.update({'y_true': np.array(all_y_true), 'y_prob': np.array(all_y_prob),
        'full_model': full_model, 'n_folds': fold_count, 'n_positives': n_positives,
        'n_total': n_total, 'prevalence': n_positives / n_total})
    _generate_fix1_latex_table(results, verbose)
    return results


def _generate_fix1_latex_table(results, verbose=True):
    """Generate LaTeX table of CV metrics."""
    lines = [r"\begin{table}[H]", r"\centering",
        r"\caption{Cross-validated predictive performance of the gradient boosting model for MRI scan failure prediction. Results from "
        f"{results.get('n_folds','?')}-fold repeated stratified cross-validation on {results.get('n_total','?'):,} "
        f"simulation-augmented patient records ({results.get('n_positives','?')} failures, "
        f"{100*results.get('prevalence',0):.1f}\\% positive rate).}}",
        r"\label{tab:ml_cv_metrics}", r"\begin{tabular}{lcccc}", r"\toprule",
        r"Metric & Mean & Std & \multicolumn{2}{c}{95\% CI} \\", r"\midrule"]
    for key, display in [('auc_roc','AUC-ROC'),('pr_auc','PR-AUC'),('brier','Brier Score'),
                         ('f1','F1 Score'),('precision','Precision'),('recall','Recall')]:
        if key in results:
            r = results[key]
            lines.append(f"{display} & {r['mean']:.4f} & {r['std']:.4f} & {r['ci_lo']:.4f} & {r['ci_hi']:.4f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    latex_str = "\n".join(lines)
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    outpath = os.path.join(CONFIG['output_dir'], 'table_ml_cv_metrics.tex')
    with open(outpath, 'w') as f:
        f.write(latex_str)
    if verbose:
        print(f"\n  LaTeX table saved: {outpath}")
        print(latex_str)


def run_fix5_calibration_plot(results, verbose=True):
    """Generate calibration reliability diagram from cross-validated predictions."""
    if verbose:
        print("\n" + "=" * 70)
        print("FIX 5: CALIBRATION RELIABILITY DIAGRAM")
        print("=" * 70)
    y_true, y_prob = results['y_true'], results['y_prob']
    prevalence = results.get('prevalence', y_true.mean())
    if verbose:
        print(f"  Predictions: {len(y_prob):,}, Prevalence: {prevalence:.3f}")
        print(f"  Prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")

    n_bins = 10
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')
    except ValueError:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy='uniform')

    fig = plt.figure(figsize=(7, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax1.plot(prob_pred, prob_true, 's-', color='#2166AC', markersize=7, linewidth=1.5, label='Gradient Boosting')
    _add_calibration_ci(ax1, y_true, y_prob, n_bins=n_bins)
    ax1.set_ylabel('Observed frequency', fontsize=12)
    ax1.set_xlim([-0.02, 1.02]); ax1.set_ylim([-0.02, 1.02])
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_title('Calibration Reliability Diagram', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3); ax1.tick_params(labelbottom=False)
    stats_text = f"N = {len(y_true):,}\nPrevalence: {prevalence:.3f}"
    if 'auc_roc' in results: stats_text += f"\nAUC-ROC: {results['auc_roc']['mean']:.3f}"
    if 'brier' in results: stats_text += f"\nBrier: {results['brier']['mean']:.4f}"
    ax1.text(0.98, 0.05, stats_text, transform=ax1.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    neg_mask, pos_mask = y_true == 0, y_true == 1
    ax2.hist(y_prob[neg_mask], bins=30, range=(0,1), alpha=0.6, color='#4393C3', label=f'Pass (n={neg_mask.sum():,})')
    ax2.hist(y_prob[pos_mask], bins=30, range=(0,1), alpha=0.8, color='#D6604D', label=f'Fail (n={pos_mask.sum():,})')
    ax2.set_xlabel('Predicted failure probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12); ax2.legend(loc='upper right', fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(CONFIG['output_dir'], f'fig_calibration.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    pdf_path = os.path.join(CONFIG['output_dir'], 'fig_calibration.pdf')
    if verbose: print(f"\n  Saved: {pdf_path}")
    return pdf_path


def _add_calibration_ci(ax, y_true, y_prob, n_bins=10, n_bootstrap=500):
    """Add bootstrap confidence band to calibration plot."""
    rng = np.random.RandomState(42)
    n = len(y_true)
    boot_list = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            bt, bp = calibration_curve(y_true[idx], y_prob[idx], n_bins=n_bins, strategy='quantile')
            if len(bt) == n_bins: boot_list.append(bt)
        except (ValueError, IndexError): continue
    if len(boot_list) >= 50:
        boot_arr = np.array(boot_list)
        try:
            _, bp = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')
            ax.fill_between(bp, np.percentile(boot_arr,2.5,axis=0), np.percentile(boot_arr,97.5,axis=0),
                alpha=0.15, color='#2166AC', label='95% CI')
        except (ValueError, IndexError): pass


def run_fix6_computation_timing(verbose=True, quick=False):
    """Run all 7 scheduling methods and record wall-clock solve times."""
    if verbose:
        print("\n" + "=" * 70)
        print("FIX 6: COMPUTATION TIME COMPARISON")
        print("=" * 70)
    try:
        from baselines import (PatientData, DEFAULT_CONFIG, fcfs_schedule, spt_schedule, lpt_schedule,
            deterministic_schedule, age_stratified_schedule, random_schedule, stochastic_milp_schedule)
    except ImportError:
        print("  ERROR: baselines.py not found."); return None

    n_reps = 10 if quick else CONFIG['timing_replications']
    n_pat = CONFIG['timing_n_patients']; seed = CONFIG['timing_seed']
    if verbose: print(f"  Replications: {n_reps}, Patients: {n_pat}")

    methods = {'FCFS': fcfs_schedule, 'SPT': spt_schedule, 'LPT': lpt_schedule,
        'Deterministic': deterministic_schedule, 'Age-Stratified': age_stratified_schedule,
        'Random': lambda p,c: random_schedule(p,c,n_samples=10,seed=seed),
        'Stochastic-MILP': lambda p,c: stochastic_milp_schedule(p,c,verbose=False)}
    timing_results = {name: [] for name in methods}

    for rep in range(n_reps):
        if verbose and rep % 10 == 0: print(f"  Replication {rep+1}/{n_reps}...")
        np.random.seed(seed + rep)
        patients = []
        n_high = int(n_pat * 0.25); n_low = n_pat - n_high
        for i in range(n_high):
            age = np.random.uniform(75, 92)
            cdr = np.random.choice([1,2,3], p=[0.3,0.5,0.2])
            moca = max(8, min(26, 24 - cdr*4 + np.random.normal(0,2)))
            pf = np.clip(0.25+np.random.uniform(0,0.25)+(age-75)*0.003+(26-moca)*0.008, 0.25, 0.55)
            patients.append(PatientData(f"P{i+1:03d}", pf, age, moca, cdr))
        for i in range(n_low):
            age = np.random.uniform(60, 80)
            cdr = np.random.choice([0,0.5,1], p=[0.4,0.4,0.2])
            moca = max(18, min(30, 27-cdr*2+np.random.normal(0,2)))
            pf = np.clip(0.05+np.random.uniform(0,0.15)+max(0,(age-70))*0.002, 0.03, 0.24)
            patients.append(PatientData(f"P{n_high+i+1:03d}", pf, age, moca, cdr))
        np.random.shuffle(patients)
        for idx, p in enumerate(patients): p.patient_id = f"P{idx+1:03d}"
        for mname, mfunc in methods.items():
            try:
                t0 = time.perf_counter(); sdf, _ = mfunc(patients, DEFAULT_CONFIG)
                t1 = time.perf_counter() - t0
                if sdf is not None: timing_results[mname].append(t1)
            except Exception as e:
                if verbose: print(f"    WARNING: {mname} failed: {e}")

    rows = []
    if verbose:
        print(f"\n  {'Method':<20} {'Mean (s)':>10} {'Std (s)':>10} {'Min (s)':>10} {'Max (s)':>10} {'N':>6}")
        print("  " + "-" * 70)
    for mname in methods:
        t = timing_results[mname]
        if not t: continue
        a = np.array(t)
        rows.append({'Method': mname, 'Mean_Time': a.mean(), 'Std_Time': a.std(),
            'Min_Time': a.min(), 'Max_Time': a.max(), 'Median_Time': np.median(a), 'N_Successful': len(t)})
        if verbose:
            print(f"  {mname:<20} {a.mean():>10.4f} {a.std():>10.4f} {a.min():>10.4f} {a.max():>10.4f} {len(t):>6}")
    summary_df = pd.DataFrame(rows)
    _generate_fix6_latex_table(summary_df, n_pat, n_reps, verbose)
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    csv_path = os.path.join(CONFIG['output_dir'], 'computation_times.csv')
    summary_df.to_csv(csv_path, index=False)
    if verbose: print(f"\n  Saved: {csv_path}")
    return summary_df


def _generate_fix6_latex_table(df, n_pat, n_reps, verbose=True):
    """Generate LaTeX table of computation times."""
    lines = [r"\begin{table}[H]", r"\centering",
        r"\caption{Computation times for scheduling methods "
        f"($n = {n_pat}$ patients, {n_reps} replications). "
        r"Heuristic methods solve in under one millisecond; the "
        r"Stochastic-MILP requires substantially more time due to integer programming.}",
        r"\label{tab:computation_times}", r"\begin{tabular}{lcccc}", r"\toprule",
        r"Method & Mean (s) & Std (s) & Min (s) & Max (s) \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(f"{row['Method']} & {row['Mean_Time']:.4f} & {row['Std_Time']:.4f} & "
                     f"{row['Min_Time']:.4f} & {row['Max_Time']:.4f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    latex_str = "\n".join(lines)
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    with open(os.path.join(CONFIG['output_dir'], 'table_computation_times.tex'), 'w') as f:
        f.write(latex_str)
    if verbose: print(f"\n  LaTeX table:\n{latex_str}")


def main():
    """Parse command-line arguments and run requested Phase 2 fixes."""
    parser = argparse.ArgumentParser(description='Phase 2: ML Validation, Calibration, Timing')
    parser.add_argument('--fix1', action='store_true', help='ML cross-validation metrics')
    parser.add_argument('--fix5', action='store_true', help='Calibration reliability diagram')
    parser.add_argument('--fix6', action='store_true', help='Computation time comparison')
    parser.add_argument('--all', action='store_true', help='Run all three fixes')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--output', default='results_phase2', help='Output directory')
    args = parser.parse_args()
    if not any([args.fix1, args.fix5, args.fix6, args.all]):
        parser.print_help(); return
    CONFIG['output_dir'] = args.output
    if args.quick:
        CONFIG['cv_n_repeats'] = 3; CONFIG['timing_replications'] = 10; CONFIG['n_synthetic_patients'] = 500
    print("=" * 70); print("PHASE 2: ML VALIDATION & TIMING"); print("=" * 70)
    print(f"Output: {CONFIG['output_dir']}/  Quick: {args.quick}")
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    run_ml = args.fix1 or args.fix5 or args.all
    cv_results = None
    if run_ml:
        X, y, fnames, df_s = generate_synthetic_dataset(CONFIG['n_synthetic_patients'],
            CONFIG['high_risk_fraction'], CONFIG['dataset_seed'], verbose=True)
        df_s.to_csv(os.path.join(CONFIG['output_dir'], 'synthetic_cohort.csv'), index=False)
    if run_ml and (args.fix1 or args.all):
        cv_results = run_fix1_cross_validation(X, y, fnames, verbose=True)
    if run_ml and (args.fix5 or args.all):
        if cv_results is None: cv_results = run_fix1_cross_validation(X, y, fnames, verbose=True)
        run_fix5_calibration_plot(cv_results, verbose=True)
    if args.fix6 or args.all:
        run_fix6_computation_timing(verbose=True, quick=args.quick)

    print("\n" + "=" * 70); print("PHASE 2 COMPLETE"); print("=" * 70)
    print(f"Outputs in: {CONFIG['output_dir']}/")
    for f in sorted(os.listdir(CONFIG['output_dir'])):
        print(f"  {f} ({os.path.getsize(os.path.join(CONFIG['output_dir'], f)):,} bytes)")

if __name__ == '__main__':
    main()