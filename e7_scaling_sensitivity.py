"""
E7: SCALING FACTOR SENSITIVITY ANALYSIS

Tests robustness of the framework's qualitative findings (LPT-MILP equivalence,
LPT outperforming Age-Stratified and Deterministic) to perturbations in the
resource-specific risk scaling factors (UDS=0.3, MRI=1.0, CSF=0.8).

Usage:
    python e7_scaling_sensitivity.py
    python e7_scaling_sensitivity.py --quick   # ~3 min instead of ~20 min

Produces:
    - results/e7_scaling_sensitivity_<timestamp>.csv
    - Console summary table

Save in same directory as experiments.py and baselines.py.

Author: Clinical Scheduling Project
"""

import argparse
import copy
import os
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# Reuse the existing scheduling infrastructure
from baselines import (
    DEFAULT_CONFIG,
    deterministic_schedule,
    age_stratified_schedule,
    lpt_schedule,
    stochastic_milp_schedule,
    evaluate_schedule_with_simulation,
)
from experiments import generate_synthetic_cohort


# =============================================================================
# E7 CONFIGURATION
# =============================================================================
# Grid: each scaling factor perturbed -50%, baseline, +50%
# Baseline values (from DEFAULT_CONFIG): UDS=0.3, CSF=0.8
# MRI stays at 1.0 by construction (it defines the reference probability)
UDS_FACTORS = [0.15, 0.30, 0.45]   # -50%, baseline, +50%
CSF_FACTORS = [0.40, 0.80, 1.20]   # -50%, baseline, +50%

# Methods to compare (subset of E1; we keep the most informative ones)
METHODS = [
    ('Stochastic-MILP', lambda p, c: stochastic_milp_schedule(p, c, verbose=False)),
    ('LPT',             lpt_schedule),
    ('Deterministic',   deterministic_schedule),
    ('Age-Stratified',  age_stratified_schedule),
]


def make_config_with_scaling(uds_scale: float, csf_scale: float) -> dict:
    """Return a deep copy of DEFAULT_CONFIG with overridden risk_scale values."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg['resources']['UDS']['risk_scale'] = uds_scale
    cfg['resources']['CSF']['risk_scale'] = csf_scale
    # MRI risk_scale stays at 1.0 (the reference)
    return cfg


def run_experiment_e7(
    uds_factors=UDS_FACTORS,
    csf_factors=CSF_FACTORS,
    n_patients: int = 10,
    n_replications: int = 30,
    n_simulations: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For every (UDS, CSF) scaling combination, run all four methods over
    n_replications cohorts of n_patients each, evaluate each schedule with
    n_simulations Monte Carlo draws, return per-(combination, method, rep)
    overtime probability and expected makespan.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT E7: Scaling Factor Sensitivity")
        print(f"  UDS scaling grid: {uds_factors}")
        print(f"  CSF scaling grid: {csf_factors}")
        print(f"  Combinations:     {len(uds_factors) * len(csf_factors)}")
        print(f"  Methods:          {[m[0] for m in METHODS]}")
        print(f"  Replications:     {n_replications}")
        print(f"  Simulations/sch:  {n_simulations}")
        print(f"  Patients/cohort:  {n_patients}")
        print("=" * 70)

    rows = []
    combo_idx = 0
    total_combos = len(uds_factors) * len(csf_factors)

    for uds_scale in uds_factors:
        for csf_scale in csf_factors:
            combo_idx += 1
            cfg = make_config_with_scaling(uds_scale, csf_scale)
            is_baseline = (abs(uds_scale - 0.30) < 1e-9 and abs(csf_scale - 0.80) < 1e-9)

            if verbose:
                tag = " (BASELINE)" if is_baseline else ""
                print(f"\n  Combo {combo_idx}/{total_combos}: UDS={uds_scale}, CSF={csf_scale}{tag}")

            for rep in tqdm(range(n_replications), desc="reps", disable=not verbose):
                patients = generate_synthetic_cohort(n_patients, seed=seed + rep)

                for method_name, method_func in METHODS:
                    try:
                        schedule_df, _ = method_func(patients, cfg)
                    except Exception as err:
                        # Don't let one solver hiccup kill the whole run
                        rows.append({
                            'UDS_Scale': uds_scale,
                            'CSF_Scale': csf_scale,
                            'Is_Baseline': is_baseline,
                            'Replication': rep + 1,
                            'Method': method_name,
                            'Expected_Makespan': np.nan,
                            'Overtime_Prob': np.nan,
                            'Error': str(err)[:80],
                        })
                        continue

                    if schedule_df is None:
                        continue

                    sim = evaluate_schedule_with_simulation(
                        schedule_df, patients, cfg, n_simulations, seed + rep)

                    rows.append({
                        'UDS_Scale': uds_scale,
                        'CSF_Scale': csf_scale,
                        'Is_Baseline': is_baseline,
                        'Replication': rep + 1,
                        'Method': method_name,
                        'Expected_Makespan': sim['expected_makespan'],
                        'Overtime_Prob': sim['overtime_probability'],
                        'Error': '',
                    })

    return pd.DataFrame(rows)


def summarize_e7(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (UDS_Scale, CSF_Scale, Method): mean overtime probability,
    mean expected makespan, and 95% CI on overtime probability.
    """
    grp = df.groupby(['UDS_Scale', 'CSF_Scale', 'Method'])
    summary = grp.agg(
        OT_Mean=('Overtime_Prob', 'mean'),
        OT_Std=('Overtime_Prob', 'std'),
        Makespan_Mean=('Expected_Makespan', 'mean'),
        Makespan_Std=('Expected_Makespan', 'std'),
        N=('Overtime_Prob', 'count'),
    ).reset_index()
    summary['OT_CI95'] = (1.96 * summary['OT_Std'] / np.sqrt(summary['N'])).round(4)
    summary['OT_Mean'] = summary['OT_Mean'].round(4)
    summary['Makespan_Mean'] = summary['Makespan_Mean'].round(2)
    summary['Is_Baseline'] = (
        (summary['UDS_Scale'].sub(0.30).abs() < 1e-9)
        & (summary['CSF_Scale'].sub(0.80).abs() < 1e-9)
    )
    return summary


def print_summary_for_claude(summary: pd.DataFrame) -> None:
    """
    Print a compact table suitable for pasting into chat. Two views:
    (a) per-method overtime probability across all nine (UDS, CSF) combinations,
    (b) range and mean overtime across combinations, per method.
    """
    print("\n" + "=" * 70)
    print("E7 RESULTS SUMMARY  (paste this entire block back to Claude)")
    print("=" * 70)

    # View A: full grid
    pivot_ot = summary.pivot_table(
        index=['UDS_Scale', 'CSF_Scale'],
        columns='Method',
        values='OT_Mean',
    )
    pivot_mk = summary.pivot_table(
        index=['UDS_Scale', 'CSF_Scale'],
        columns='Method',
        values='Makespan_Mean',
    )
    print("\n[A] Overtime probability per (UDS, CSF) combo and method:")
    print(pivot_ot.to_string())

    print("\n[B] Expected makespan per (UDS, CSF) combo and method:")
    print(pivot_mk.to_string())

    # View C: per-method robustness summary
    print("\n[C] Per-method robustness across the 9 combinations:")
    by_method = summary.groupby('Method').agg(
        OT_Min=('OT_Mean', 'min'),
        OT_Max=('OT_Mean', 'max'),
        OT_Range=('OT_Mean', lambda s: s.max() - s.min()),
        OT_BaselineValue=('OT_Mean', lambda s: s[summary.loc[s.index, 'Is_Baseline']].mean()),
    ).round(4)
    print(by_method.to_string())

    # View D: LPT vs MILP gap across combinations
    print("\n[D] LPT vs Stochastic-MILP overtime gap across combinations:")
    if 'LPT' in pivot_ot.columns and 'Stochastic-MILP' in pivot_ot.columns:
        gap = (pivot_ot['LPT'] - pivot_ot['Stochastic-MILP']).round(4)
        print(gap.to_string())
        print(f"\n  Max |LPT - MILP| gap across all combinations: {gap.abs().max():.4f}")
    print("\n" + "=" * 70)


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='E7: Scaling-factor sensitivity.')
    parser.add_argument('--quick', action='store_true',
                        help='Reduced settings for fast smoke test (~3 min).')
    parser.add_argument('--output-dir', default='results',
                        help='Directory for the output CSV.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        n_replications, n_simulations = 10, 50
    else:
        n_replications, n_simulations = 30, 100

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    df = run_experiment_e7(
        uds_factors=UDS_FACTORS,
        csf_factors=CSF_FACTORS,
        n_patients=10,
        n_replications=n_replications,
        n_simulations=n_simulations,
        seed=args.seed,
        verbose=True,
    )

    raw_path = os.path.join(args.output_dir, f'e7_scaling_sensitivity_{timestamp}.csv')
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results written to: {raw_path}")

    summary = summarize_e7(df)
    summary_path = os.path.join(args.output_dir, f'e7_scaling_sensitivity_summary_{timestamp}.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Summary written to:     {summary_path}")

    print_summary_for_claude(summary)


if __name__ == '__main__':
    main()
