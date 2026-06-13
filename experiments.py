"""
EXPERIMENTAL SUITE FOR EJOR PAPER
Computational Experiments E1-E6

Run with:
    python experiments.py --experiments E1 E2 E3 E4 E5 E6
    python experiments.py --quick  # Fast mode for testing

Author: EJOR Clinical Scheduling Project
"""

import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from baselines import (
    PatientData, DEFAULT_CONFIG,
    fcfs_schedule, spt_schedule, lpt_schedule,
    deterministic_schedule, age_stratified_schedule, random_schedule,
    stochastic_milp_schedule, calculate_metrics,
    evaluate_schedule_with_simulation, calculate_improvement_pct
)


# =============================================================================
# CONFIGURATION
# =============================================================================
EXPERIMENT_CONFIG = {
    'output_dir': 'results',
    'seed': 42,
    'n_replications': 100,
    'n_simulations': 100,
    'patient_sizes': [10, 20, 30, 50],
    'lambda_values': [0.0, 0.25, 0.5, 1.0, 2.0],
    'prediction_errors': [0.05, 0.10, 0.15, 0.20],
    'high_risk_fractions': [0.0, 0.1, 0.2, 0.3, 0.5],
}


# =============================================================================
# DATA GENERATION
# =============================================================================
def generate_synthetic_cohort(
    n_patients: int,
    high_risk_fraction: float = 0.25,
    seed: int = None
) -> List[PatientData]:
    """Generate synthetic patient cohort with realistic characteristics."""
    if seed is not None:
        np.random.seed(seed)
    
    patients = []
    n_high_risk = int(n_patients * high_risk_fraction)
    n_low_risk = n_patients - n_high_risk
    
    for i in range(n_high_risk):
        age = np.random.uniform(75, 92)
        cdr = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        moca = max(8, min(26, 24 - cdr * 4 + np.random.normal(0, 2)))
        p_fail = np.clip(0.25 + np.random.uniform(0, 0.25) + (age - 75) * 0.003 + (26 - moca) * 0.008, 0.25, 0.55)
        patients.append(PatientData(f"P{i+1:03d}", p_fail, age, moca, cdr))
    
    for i in range(n_low_risk):
        age = np.random.uniform(60, 80)
        cdr = np.random.choice([0, 0.5, 1], p=[0.4, 0.4, 0.2])
        moca = max(18, min(30, 27 - cdr * 2 + np.random.normal(0, 2)))
        p_fail = np.clip(0.05 + np.random.uniform(0, 0.15) + max(0, (age - 70)) * 0.002, 0.03, 0.24)
        patients.append(PatientData(f"P{n_high_risk + i + 1:03d}", p_fail, age, moca, cdr))
    
    np.random.shuffle(patients)
    for i, p in enumerate(patients):
        p.patient_id = f"P{i+1:03d}"
    
    return patients


# =============================================================================
# EXPERIMENT E1: BASELINE COMPARISON
# =============================================================================
def run_experiment_e1(n_replications=100, n_patients=10, n_simulations=100, seed=42, verbose=True):
    """E1: Compare all methods across multiple random cohorts."""
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT E1: Baseline Comparison")
        print(f"  Replications: {n_replications}, Patients: {n_patients}, Simulations: {n_simulations}")
        print("="*70)
    
    methods = {
        'FCFS': fcfs_schedule,
        'SPT': spt_schedule,
        'LPT': lpt_schedule,
        'Deterministic': deterministic_schedule,
        'Age-Stratified': age_stratified_schedule,
        'Random': lambda p, c: random_schedule(p, c, n_samples=10, seed=seed),
        'Stochastic-MILP': lambda p, c: stochastic_milp_schedule(p, c, verbose=False),
    }
    
    results = []
    iterator = tqdm(range(n_replications), desc="E1") if verbose else range(n_replications)
    
    for rep in iterator:
        patients = generate_synthetic_cohort(n_patients, seed=seed + rep)
        
        for method_name, method_func in methods.items():
            schedule_df, solve_time = method_func(patients, DEFAULT_CONFIG)
            if schedule_df is None:
                continue
            
            sim_metrics = evaluate_schedule_with_simulation(
                schedule_df, patients, DEFAULT_CONFIG, n_simulations, seed + rep
            )
            planned_metrics = calculate_metrics(schedule_df, DEFAULT_CONFIG, solve_time)
            
            results.append({
                'Replication': rep + 1,
                'Method': method_name,
                'Planned_Makespan': planned_metrics.makespan,
                'Expected_Makespan': sim_metrics['expected_makespan'],
                'Makespan_Std': sim_metrics['makespan_std'],
                'Makespan_95pct': sim_metrics['makespan_95pct'],
                'Overtime_Prob': sim_metrics['overtime_probability'],
                'Solve_Time': solve_time,
            })
    
    return pd.DataFrame(results)


# =============================================================================
# EXPERIMENT E2: SCALABILITY
# =============================================================================
def run_experiment_e2(patient_sizes=[10, 20, 30, 50], n_replications=10, seed=42, verbose=True):
    """E2: Test computational tractability across problem sizes."""
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT E2: Scalability Analysis")
        print(f"  Patient sizes: {patient_sizes}")
        print("="*70)
    
    results = []
    
    for n_patients in patient_sizes:
        if verbose:
            print(f"\n  Testing n = {n_patients}...")
        
        for rep in tqdm(range(n_replications), desc=f"n={n_patients}", disable=not verbose):
            patients = generate_synthetic_cohort(n_patients, seed=seed + rep)
            
            config = DEFAULT_CONFIG.copy()
            config['solver_timeout'] = max(60, n_patients * 6)
            
            schedule_df, solve_time = stochastic_milp_schedule(patients, config, verbose=False)
            
            if schedule_df is not None:
                metrics = calculate_metrics(schedule_df, config, solve_time)
                n_vars = 3 * n_patients + n_patients * (n_patients - 1) // 2 * 3 + 1 + n_patients
                n_constraints = n_patients * 2 + n_patients * (n_patients - 1) * 3 + n_patients * 3
                
                results.append({
                    'N_Patients': n_patients,
                    'Replication': rep + 1,
                    'Solve_Time': solve_time,
                    'Makespan': metrics.makespan,
                    'N_Variables': n_vars,
                    'N_Constraints': n_constraints,
                    'Status': 'Optimal' if solve_time < config['solver_timeout'] - 1 else 'Timeout',
                })
    
    return pd.DataFrame(results)


# =============================================================================
# EXPERIMENT E3: LAMBDA SENSITIVITY
# =============================================================================
def run_experiment_e3(lambda_values=[0.0, 0.25, 0.5, 1.0, 2.0], n_patients=10, n_replications=20, seed=42, verbose=True):
    """E3: Explore makespan vs waiting time trade-off."""
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT E3: Lambda Sensitivity")
        print(f"  Lambda values: {lambda_values}")
        print("="*70)
    
    results = []
    
    for lambda_val in lambda_values:
        if verbose:
            print(f"\n  Testing λ = {lambda_val}...")
        
        for rep in tqdm(range(n_replications), desc=f"λ={lambda_val}", disable=not verbose):
            patients = generate_synthetic_cohort(n_patients, seed=seed + rep)
            
            config = DEFAULT_CONFIG.copy()
            config['wait_time_penalty'] = lambda_val
            
            schedule_df, solve_time = stochastic_milp_schedule(patients, config, verbose=False)
            
            if schedule_df is not None:
                metrics = calculate_metrics(schedule_df, config, solve_time)
                results.append({
                    'Lambda': lambda_val,
                    'Replication': rep + 1,
                    'Makespan': metrics.makespan,
                    'Total_Wait': metrics.total_wait,
                    'Avg_Wait': metrics.avg_wait,
                    'Objective': metrics.objective_value,
                })
    
    return pd.DataFrame(results)


# =============================================================================
# EXPERIMENT E4: VALUE OF ML
# =============================================================================
def run_experiment_e4(n_patients=10, n_replications=50, n_simulations=100, seed=42, verbose=True):
    """E4: Compare Full ML vs Age-Only vs Stratification."""
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT E4: Value of ML")
        print("="*70)
    
    results = []
    iterator = tqdm(range(n_replications), desc="E4") if verbose else range(n_replications)
    
    for rep in iterator:
        patients = generate_synthetic_cohort(n_patients, seed=seed + rep)
        original_p_fail = {p.patient_id: p.p_fail_mri for p in patients}
        
        # Full ML
        schedule_df, _ = stochastic_milp_schedule(patients, DEFAULT_CONFIG, verbose=False)
        if schedule_df is not None:
            sim = evaluate_schedule_with_simulation(schedule_df, patients, DEFAULT_CONFIG, n_simulations, seed + rep)
            results.append({'Replication': rep + 1, 'Predictor': 'Full_ML', 
                          'Expected_Makespan': sim['expected_makespan'], 'Overtime_Prob': sim['overtime_probability']})
        
        # Age-Only
        for p in patients:
            p.p_fail_mri = np.clip(0.05 + max(0, (p.age - 65)) * 0.005, 0.03, 0.40)
        schedule_df, _ = stochastic_milp_schedule(patients, DEFAULT_CONFIG, verbose=False)
        if schedule_df is not None:
            for p in patients:
                p.p_fail_mri = original_p_fail[p.patient_id]
            sim = evaluate_schedule_with_simulation(schedule_df, patients, DEFAULT_CONFIG, n_simulations, seed + rep)
            results.append({'Replication': rep + 1, 'Predictor': 'Age_Only',
                          'Expected_Makespan': sim['expected_makespan'], 'Overtime_Prob': sim['overtime_probability']})
        
        # Age Stratified
        for p in patients:
            p.p_fail_mri = original_p_fail[p.patient_id]
        schedule_df, _ = age_stratified_schedule(patients, DEFAULT_CONFIG)
        if schedule_df is not None:
            sim = evaluate_schedule_with_simulation(schedule_df, patients, DEFAULT_CONFIG, n_simulations, seed + rep)
            results.append({'Replication': rep + 1, 'Predictor': 'Age_Stratified',
                          'Expected_Makespan': sim['expected_makespan'], 'Overtime_Prob': sim['overtime_probability']})
        
        # Deterministic
        schedule_df, _ = deterministic_schedule(patients, DEFAULT_CONFIG)
        if schedule_df is not None:
            sim = evaluate_schedule_with_simulation(schedule_df, patients, DEFAULT_CONFIG, n_simulations, seed + rep)
            results.append({'Replication': rep + 1, 'Predictor': 'No_Prediction',
                          'Expected_Makespan': sim['expected_makespan'], 'Overtime_Prob': sim['overtime_probability']})
    
    return pd.DataFrame(results)


# =============================================================================
# EXPERIMENT E5: ROBUSTNESS
# =============================================================================
def run_experiment_e5(prediction_errors=[0.05, 0.10, 0.15, 0.20], n_patients=10, n_replications=30, n_simulations=100, seed=42, verbose=True):
    """E5: Test robustness to prediction error."""
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT E5: Robustness Analysis")
        print(f"  Error levels: {prediction_errors}")
        print("="*70)
    
    results = []
    
    for error_std in prediction_errors:
        if verbose:
            print(f"\n  Testing σ = {error_std}...")
        
        for rep in tqdm(range(n_replications), desc=f"σ={error_std}", disable=not verbose):
            patients = generate_synthetic_cohort(n_patients, seed=seed + rep)
            true_p_fail = {p.patient_id: p.p_fail_mri for p in patients}
            
            np.random.seed(seed + rep + int(error_std * 1000))
            for p in patients:
                p.p_fail_mri = np.clip(p.p_fail_mri + np.random.normal(0, error_std), 0.01, 0.80)
            
            schedule_df, _ = stochastic_milp_schedule(patients, DEFAULT_CONFIG, verbose=False)
            
            if schedule_df is not None:
                for p in patients:
                    p.p_fail_mri = true_p_fail[p.patient_id]
                sim = evaluate_schedule_with_simulation(schedule_df, patients, DEFAULT_CONFIG, n_simulations, seed + rep)
                results.append({
                    'Error_Std': error_std,
                    'Replication': rep + 1,
                    'Expected_Makespan': sim['expected_makespan'],
                    'Overtime_Prob': sim['overtime_probability'],
                    'Makespan_95pct': sim['makespan_95pct'],
                })
    
    return pd.DataFrame(results)


# =============================================================================
# EXPERIMENT E6: COHORT COMPOSITION
# =============================================================================
def run_experiment_e6(high_risk_fractions=[0.0, 0.1, 0.2, 0.3, 0.5], n_patients=10, n_replications=30, n_simulations=100, seed=42, verbose=True):
    """E6: When does risk-adjusted scheduling help most?"""
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT E6: Cohort Composition")
        print(f"  High-risk fractions: {high_risk_fractions}")
        print("="*70)
    
    results = []
    
    for hr_frac in high_risk_fractions:
        if verbose:
            print(f"\n  Testing {int(hr_frac*100)}% high-risk...")
        
        for rep in tqdm(range(n_replications), desc=f"{int(hr_frac*100)}%", disable=not verbose):
            patients = generate_synthetic_cohort(n_patients, high_risk_fraction=hr_frac, seed=seed + rep)
            
            for method_name, method_func in [
                ('Stochastic-MILP', lambda p, c: stochastic_milp_schedule(p, c, verbose=False)),
                ('Deterministic', deterministic_schedule),
            ]:
                schedule_df, _ = method_func(patients, DEFAULT_CONFIG)
                if schedule_df is not None:
                    sim = evaluate_schedule_with_simulation(schedule_df, patients, DEFAULT_CONFIG, n_simulations, seed + rep)
                    results.append({
                        'High_Risk_Fraction': hr_frac,
                        'Replication': rep + 1,
                        'Method': method_name,
                        'Expected_Makespan': sim['expected_makespan'],
                        'Overtime_Prob': sim['overtime_probability'],
                    })
    
    return pd.DataFrame(results)


# =============================================================================
# RESULTS SUMMARY
# =============================================================================
def summarize_e1(df):
    """Summarize E1 results with confidence intervals."""
    summary = df.groupby('Method').agg({
        'Expected_Makespan': ['mean', 'std'],
        'Overtime_Prob': ['mean', 'std'],
        'Solve_Time': ['mean', 'max'],
    }).round(3)
    summary.columns = ['Makespan_Mean', 'Makespan_Std', 'OT_Mean', 'OT_Std', 'Time_Mean', 'Time_Max']
    n = df.groupby('Method').size()
    summary['Makespan_CI'] = (1.96 * summary['Makespan_Std'] / np.sqrt(n)).round(1)
    summary['OT_CI'] = (1.96 * summary['OT_Std'] / np.sqrt(n)).round(3)
    return summary.sort_values('OT_Mean')


# =============================================================================
# MAIN
# =============================================================================
def run_all_experiments(experiments=['E1', 'E2', 'E3', 'E4', 'E5', 'E6'], output_dir='results', quick=False, verbose=True):
    """Run all experiments and save results."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if quick:
        EXPERIMENT_CONFIG['n_replications'] = 10
        EXPERIMENT_CONFIG['n_simulations'] = 50
    
    results = {}
    
    print("\n" + "="*70)
    print("EJOR EXPERIMENTAL SUITE")
    print("="*70)
    print(f"Experiments: {experiments}")
    print(f"Quick mode: {quick}")
    print(f"Output: {output_dir}/")
    
    if 'E1' in experiments:
        df = run_experiment_e1(
            n_replications=EXPERIMENT_CONFIG['n_replications'],
            n_simulations=EXPERIMENT_CONFIG['n_simulations'],
            seed=EXPERIMENT_CONFIG['seed'], verbose=verbose)
        df.to_csv(f'{output_dir}/e1_baseline_{timestamp}.csv', index=False)
        results['E1'] = df
        print("\nE1 Summary:")
        print(summarize_e1(df).to_string())
    
    if 'E2' in experiments:
        df = run_experiment_e2(
            patient_sizes=EXPERIMENT_CONFIG['patient_sizes'],
            n_replications=10 if quick else 10,
            seed=EXPERIMENT_CONFIG['seed'], verbose=verbose)
        df.to_csv(f'{output_dir}/e2_scalability_{timestamp}.csv', index=False)
        results['E2'] = df
    
    if 'E3' in experiments:
        df = run_experiment_e3(
            lambda_values=EXPERIMENT_CONFIG['lambda_values'],
            n_replications=20 if not quick else 5,
            seed=EXPERIMENT_CONFIG['seed'], verbose=verbose)
        df.to_csv(f'{output_dir}/e3_lambda_{timestamp}.csv', index=False)
        results['E3'] = df
    
    if 'E4' in experiments:
        df = run_experiment_e4(
            n_replications=50 if not quick else 10,
            n_simulations=EXPERIMENT_CONFIG['n_simulations'],
            seed=EXPERIMENT_CONFIG['seed'], verbose=verbose)
        df.to_csv(f'{output_dir}/e4_ml_value_{timestamp}.csv', index=False)
        results['E4'] = df
    
    if 'E5' in experiments:
        df = run_experiment_e5(
            prediction_errors=EXPERIMENT_CONFIG['prediction_errors'],
            n_replications=30 if not quick else 10,
            seed=EXPERIMENT_CONFIG['seed'], verbose=verbose)
        df.to_csv(f'{output_dir}/e5_robustness_{timestamp}.csv', index=False)
        results['E5'] = df
    
    if 'E6' in experiments:
        df = run_experiment_e6(
            high_risk_fractions=EXPERIMENT_CONFIG['high_risk_fractions'],
            n_replications=30 if not quick else 10,
            seed=EXPERIMENT_CONFIG['seed'], verbose=verbose)
        df.to_csv(f'{output_dir}/e6_cohort_{timestamp}.csv', index=False)
        results['E6'] = df
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print("="*70)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run EJOR experiments')
    parser.add_argument('--experiments', nargs='+', default=['E1'], help='Experiments to run')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer replications)')
    args = parser.parse_args()
    
    run_all_experiments(experiments=args.experiments, output_dir=args.output, quick=args.quick)
