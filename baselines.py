"""
BASELINE SCHEDULING METHODS FOR EJOR PAPER
Computational Experiments: Comparison Framework

This module implements six baseline scheduling methods for comparison
against the risk-adjusted stochastic scheduling framework.

Baselines:
    1. FCFS - First-Come-First-Served
    2. SPT  - Shortest Processing Time first
    3. LPT  - Longest Processing Time first
    4. DET  - Deterministic (ignore risk, use base durations)
    5. AGE  - Age-Stratified (simple rule-based buffers)
    6. RND  - Random ordering (lower bound, averaged)

Author: EJOR Clinical Scheduling Project
"""

import pandas as pd
import numpy as np
import pulp
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_CONFIG = {
    'resources': {
        'UDS': {'base_duration': 60, 'failure_penalty': 15, 'risk_scale': 0.3},
        'MRI': {'base_duration': 45, 'failure_penalty': 30, 'risk_scale': 1.0},
        'CSF': {'base_duration': 30, 'failure_penalty': 20, 'risk_scale': 0.8},
    },
    'resource_sequence': ['UDS', 'MRI', 'CSF'],
    'wait_time_penalty': 0.5,  # λ in objective
    'solver_timeout': 60,
    'big_m': 2000,
}


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class ScheduleMetrics:
    """Container for schedule performance metrics."""
    makespan: float
    total_wait: float
    avg_wait: float
    max_wait: float
    solve_time: float
    utilization: Dict[str, float]
    objective_value: float
    
    def to_dict(self) -> dict:
        return {
            'makespan': self.makespan,
            'total_wait': self.total_wait,
            'avg_wait': self.avg_wait,
            'max_wait': self.max_wait,
            'solve_time': self.solve_time,
            'objective_value': self.objective_value,
            **{f'{k}_util': v for k, v in self.utilization.items()}
        }


@dataclass
class PatientData:
    """Patient information for scheduling."""
    patient_id: str
    p_fail_mri: float
    age: float
    moca: Optional[float] = None
    cdr: Optional[float] = None
    
    def get_risk_adjusted_durations(self, config: dict) -> Dict[str, float]:
        """Calculate expected durations for each resource."""
        durations = {}
        for res, params in config['resources'].items():
            base = params['base_duration']
            penalty = params['failure_penalty']
            risk_scale = params.get('risk_scale', 1.0)
            p_fail = self.p_fail_mri * risk_scale
            durations[res] = base + p_fail * penalty
        return durations
    
    def get_base_durations(self, config: dict) -> Dict[str, float]:
        """Get nominal durations (ignoring risk)."""
        return {res: params['base_duration'] 
                for res, params in config['resources'].items()}


# =============================================================================
# SCHEDULE BUILDER (Heuristic-based)
# =============================================================================
def build_heuristic_schedule(
    patient_order: List[str],
    durations: Dict[Tuple[str, str], float],
    resources: List[str]
) -> pd.DataFrame:
    """
    Build a schedule given a patient ordering using a greedy forward pass.
    
    This schedules patients in the given order, respecting:
    - Precedence constraints (UDS → MRI → CSF)
    - Resource capacity (one patient per resource at a time)
    
    Parameters:
        patient_order: List of patient IDs in scheduling order
        durations: Dict mapping (patient_id, resource) to duration
        resources: List of resources in precedence order
    
    Returns:
        DataFrame with schedule (start/end times for each resource)
    """
    # Track when each resource becomes available
    resource_available = {r: 0.0 for r in resources}
    
    schedule_rows = []
    
    for pid in patient_order:
        row = {'Patient': pid}
        patient_available = 0.0  # When patient finishes previous operation
        
        for res in resources:
            # Start time = max(resource available, patient available)
            start = max(resource_available[res], patient_available)
            duration = durations[(pid, res)]
            end = start + duration
            
            row[f'{res}_Start'] = start
            row[f'{res}_End'] = end
            
            # Update tracking
            resource_available[res] = end
            patient_available = end
        
        schedule_rows.append(row)
    
    df = pd.DataFrame(schedule_rows)
    
    # Calculate wait times
    df['Wait'] = 0.0
    for i in range(len(resources) - 1):
        curr_res = resources[i]
        next_res = resources[i + 1]
        gap = df[f'{next_res}_Start'] - df[f'{curr_res}_End']
        df['Wait'] += gap.clip(lower=0)
    
    return df


# =============================================================================
# BASELINE 1: FCFS (First-Come-First-Served)
# =============================================================================
def fcfs_schedule(
    patients: List[PatientData],
    config: dict = None
) -> Tuple[pd.DataFrame, float]:
    """
    First-Come-First-Served scheduling.
    Patients are scheduled in their original (arrival) order.
    
    Parameters:
        patients: List of PatientData objects
        config: Configuration dict (uses DEFAULT_CONFIG if None)
    
    Returns:
        (schedule_df, solve_time)
    """
    config = config or DEFAULT_CONFIG
    resources = config['resource_sequence']
    
    start_time = time.time()
    
    # Build duration dictionary using risk-adjusted durations
    durations = {}
    for p in patients:
        d = p.get_risk_adjusted_durations(config)
        for res in resources:
            durations[(p.patient_id, res)] = d[res]
    
    # Schedule in original order
    patient_order = [p.patient_id for p in patients]
    schedule_df = build_heuristic_schedule(patient_order, durations, resources)
    
    # Add failure probabilities for reference
    p_fail_map = {p.patient_id: p.p_fail_mri for p in patients}
    schedule_df['p_fail'] = schedule_df['Patient'].map(p_fail_map)
    
    solve_time = time.time() - start_time
    
    return schedule_df, solve_time


# =============================================================================
# BASELINE 2: SPT (Shortest Processing Time)
# =============================================================================
def spt_schedule(
    patients: List[PatientData],
    config: dict = None
) -> Tuple[pd.DataFrame, float]:
    """
    Shortest Processing Time first scheduling.
    Patients sorted by total expected duration (ascending).
    
    Classical scheduling heuristic that minimizes mean flow time
    under deterministic assumptions.
    """
    config = config or DEFAULT_CONFIG
    resources = config['resource_sequence']
    
    start_time = time.time()
    
    # Calculate total duration for each patient
    total_durations = {}
    durations = {}
    for p in patients:
        d = p.get_risk_adjusted_durations(config)
        total_durations[p.patient_id] = sum(d.values())
        for res in resources:
            durations[(p.patient_id, res)] = d[res]
    
    # Sort by total duration (ascending = shortest first)
    sorted_patients = sorted(patients, key=lambda p: total_durations[p.patient_id])
    patient_order = [p.patient_id for p in sorted_patients]
    
    schedule_df = build_heuristic_schedule(patient_order, durations, resources)
    
    p_fail_map = {p.patient_id: p.p_fail_mri for p in patients}
    schedule_df['p_fail'] = schedule_df['Patient'].map(p_fail_map)
    
    solve_time = time.time() - start_time
    
    return schedule_df, solve_time


# =============================================================================
# BASELINE 3: LPT (Longest Processing Time / Risk-First)
# =============================================================================
def lpt_schedule(
    patients: List[PatientData],
    config: dict = None
) -> Tuple[pd.DataFrame, float]:
    """
    Longest Processing Time first scheduling (Risk-First heuristic).
    Patients sorted by total expected duration (descending).
    
    Schedules high-risk patients early to provide buffer time
    for potential delays without cascading effects.
    """
    config = config or DEFAULT_CONFIG
    resources = config['resource_sequence']
    
    start_time = time.time()
    
    # Calculate total duration for each patient
    total_durations = {}
    durations = {}
    for p in patients:
        d = p.get_risk_adjusted_durations(config)
        total_durations[p.patient_id] = sum(d.values())
        for res in resources:
            durations[(p.patient_id, res)] = d[res]
    
    # Sort by total duration (descending = longest/highest-risk first)
    sorted_patients = sorted(patients, key=lambda p: total_durations[p.patient_id], 
                            reverse=True)
    patient_order = [p.patient_id for p in sorted_patients]
    
    schedule_df = build_heuristic_schedule(patient_order, durations, resources)
    
    p_fail_map = {p.patient_id: p.p_fail_mri for p in patients}
    schedule_df['p_fail'] = schedule_df['Patient'].map(p_fail_map)
    
    solve_time = time.time() - start_time
    
    return schedule_df, solve_time


# =============================================================================
# BASELINE 4: DETERMINISTIC (Ignore Risk)
# =============================================================================
def deterministic_schedule(
    patients: List[PatientData],
    config: dict = None
) -> Tuple[pd.DataFrame, float]:
    """
    Deterministic scheduling ignoring failure risk.
    Uses nominal/base durations only (no buffer for predicted failures).
    
    This baseline tests the value of incorporating risk predictions.
    """
    config = config or DEFAULT_CONFIG
    resources = config['resource_sequence']
    
    start_time = time.time()
    
    # Use BASE durations (no risk adjustment)
    durations = {}
    for p in patients:
        d = p.get_base_durations(config)
        for res in resources:
            durations[(p.patient_id, res)] = d[res]
    
    # Use LPT ordering based on actual risk (but deterministic durations)
    sorted_patients = sorted(patients, key=lambda p: p.p_fail_mri, reverse=True)
    patient_order = [p.patient_id for p in sorted_patients]
    
    schedule_df = build_heuristic_schedule(patient_order, durations, resources)
    
    p_fail_map = {p.patient_id: p.p_fail_mri for p in patients}
    schedule_df['p_fail'] = schedule_df['Patient'].map(p_fail_map)
    
    solve_time = time.time() - start_time
    
    return schedule_df, solve_time


# =============================================================================
# BASELINE 5: AGE-STRATIFIED (Simple Rule-Based)
# =============================================================================
def age_stratified_schedule(
    patients: List[PatientData],
    config: dict = None
) -> Tuple[pd.DataFrame, float]:
    """
    Age-stratified scheduling using simple rule-based buffers.
    
    Buffer rules (no ML):
        Age < 65:  0% buffer
        Age 65-74: 5% of penalty as buffer
        Age 75-84: 15% of penalty as buffer  
        Age >= 85: 25% of penalty as buffer
    
    This baseline tests the value of ML over simple heuristics.
    """
    config = config or DEFAULT_CONFIG
    resources = config['resource_sequence']
    
    start_time = time.time()
    
    # Calculate age-based buffer percentages
    def get_age_buffer(age: float) -> float:
        if age >= 85:
            return 0.25
        elif age >= 75:
            return 0.15
        elif age >= 65:
            return 0.05
        else:
            return 0.0
    
    # Build durations with age-based buffers
    durations = {}
    for p in patients:
        buffer_pct = get_age_buffer(p.age)
        for res, params in config['resources'].items():
            base = params['base_duration']
            penalty = params['failure_penalty']
            risk_scale = params.get('risk_scale', 1.0)
            # Age-based buffer instead of ML-predicted buffer
            durations[(p.patient_id, res)] = base + buffer_pct * penalty * risk_scale
    
    # Sort by age (oldest first = highest buffer first)
    sorted_patients = sorted(patients, key=lambda p: p.age, reverse=True)
    patient_order = [p.patient_id for p in sorted_patients]
    
    schedule_df = build_heuristic_schedule(patient_order, durations, resources)
    
    p_fail_map = {p.patient_id: p.p_fail_mri for p in patients}
    schedule_df['p_fail'] = schedule_df['Patient'].map(p_fail_map)
    
    solve_time = time.time() - start_time
    
    return schedule_df, solve_time


# =============================================================================
# BASELINE 6: RANDOM (Lower Bound)
# =============================================================================
def random_schedule(
    patients: List[PatientData],
    config: dict = None,
    n_samples: int = 10,
    seed: int = None
) -> Tuple[pd.DataFrame, float]:
    """
    Random patient ordering (averaged over multiple samples).
    
    This provides a lower-bound baseline showing the value of
    ANY intelligent scheduling approach.
    
    Parameters:
        n_samples: Number of random orderings to average
        seed: Random seed for reproducibility
    """
    config = config or DEFAULT_CONFIG
    resources = config['resource_sequence']
    
    if seed is not None:
        np.random.seed(seed)
    
    start_time = time.time()
    
    # Build duration dictionary
    durations = {}
    for p in patients:
        d = p.get_risk_adjusted_durations(config)
        for res in resources:
            durations[(p.patient_id, res)] = d[res]
    
    # Generate multiple random schedules
    all_schedules = []
    for _ in range(n_samples):
        random_order = np.random.permutation([p.patient_id for p in patients]).tolist()
        schedule_df = build_heuristic_schedule(random_order, durations, resources)
        all_schedules.append(schedule_df)
    
    # Return the median-performing schedule
    makespans = []
    for sched in all_schedules:
        end_cols = [f'{res}_End' for res in resources]
        makespans.append(sched[end_cols].max().max())
    
    median_idx = np.argsort(makespans)[len(makespans) // 2]
    schedule_df = all_schedules[median_idx]
    
    p_fail_map = {p.patient_id: p.p_fail_mri for p in patients}
    schedule_df['p_fail'] = schedule_df['Patient'].map(p_fail_map)
    
    solve_time = time.time() - start_time
    
    return schedule_df, solve_time


# =============================================================================
# STOCHASTIC MILP (Your Method)
# =============================================================================
def stochastic_milp_schedule(
    patients: List[PatientData],
    config: dict = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, float]:
    """
    Risk-adjusted stochastic MILP scheduling.
    
    This is your proposed method: uses ML-predicted failure probabilities
    to compute expected durations, then optimizes via MILP.
    """
    config = config or DEFAULT_CONFIG
    resources = config['resource_sequence']
    lambda_weight = config.get('wait_time_penalty', 0.5)
    timeout = config.get('solver_timeout', 60)
    M = config.get('big_m', 2000)
    
    start_time = time.time()
    
    # Build duration and p_fail dictionaries
    patient_ids = [p.patient_id for p in patients]
    n = len(patient_ids)
    
    d_tilde = {}
    p_fail = {}
    for p in patients:
        d = p.get_risk_adjusted_durations(config)
        for res in resources:
            d_tilde[(p.patient_id, res)] = d[res]
            risk_scale = config['resources'][res].get('risk_scale', 1.0)
            p_fail[(p.patient_id, res)] = p.p_fail_mri * risk_scale
    
    # Build MILP
    prob = pulp.LpProblem("Stochastic_Scheduling", pulp.LpMinimize)
    
    # Decision variables
    x = pulp.LpVariable.dicts("x",
        ((i, j) for i in patient_ids for j in resources),
        lowBound=0, cat='Continuous')
    
    C_max = pulp.LpVariable("C_max", lowBound=0)
    W = pulp.LpVariable.dicts("W", patient_ids, lowBound=0)
    
    # Binary ordering variables (only unique pairs i < j)
    patient_pairs = [(patient_ids[i], patient_ids[j])
                     for i in range(n) for j in range(i + 1, n)]
    
    y = {}
    for j in resources:
        for (i1, i2) in patient_pairs:
            y[(i1, i2, j)] = pulp.LpVariable(f"y_{i1}_{i2}_{j}", cat='Binary')
    
    # Objective: min C_max + λ * Σ W_i
    prob += C_max + lambda_weight * pulp.lpSum(W[i] for i in patient_ids)
    
    # Precedence constraints: x[i,j+1] >= x[i,j] + d[i,j]
    for pid in patient_ids:
        for s in range(len(resources) - 1):
            j_curr = resources[s]
            j_next = resources[s + 1]
            prob += x[(pid, j_next)] >= x[(pid, j_curr)] + d_tilde[(pid, j_curr)]
    
    # Wait time definition
    for pid in patient_ids:
        wait_terms = []
        for s in range(len(resources) - 1):
            j_curr = resources[s]
            j_next = resources[s + 1]
            wait_terms.append(x[(pid, j_next)] - x[(pid, j_curr)] - d_tilde[(pid, j_curr)])
        prob += W[pid] == pulp.lpSum(wait_terms)
    
    # Disjunctive constraints with symmetry breaking
    for j in resources:
        for (i1, i2) in patient_pairs:
            prob += x[(i2, j)] >= x[(i1, j)] + d_tilde[(i1, j)] - M * (1 - y[(i1, i2, j)])
            prob += x[(i1, j)] >= x[(i2, j)] + d_tilde[(i2, j)] - M * y[(i1, i2, j)]
            
            # Symmetry breaking: higher risk patients first
            if p_fail[(i1, 'MRI')] > p_fail[(i2, 'MRI')] + 0.01:
                prob += y[(i1, i2, j)] == 1
    
    # Makespan constraints
    for pid in patient_ids:
        for res in resources:
            prob += C_max >= x[(pid, res)] + d_tilde[(pid, res)]
    
    # Solve
    solver = pulp.PULP_CBC_CMD(msg=verbose, timeLimit=timeout)
    prob.solve(solver)
    
    solve_time = time.time() - start_time
    
    if prob.status not in [pulp.constants.LpStatusOptimal, 1]:
        # Return None if no solution found
        return None, solve_time
    
    # Build schedule dataframe
    schedule_rows = []
    for pid in patient_ids:
        row = {'Patient': pid, 'p_fail': p_fail[(pid, 'MRI')]}
        for res in resources:
            start = pulp.value(x[(pid, res)])
            row[f'{res}_Start'] = start
            row[f'{res}_End'] = start + d_tilde[(pid, res)]
        row['Wait'] = pulp.value(W[pid])
        schedule_rows.append(row)
    
    schedule_df = pd.DataFrame(schedule_rows).sort_values('UDS_Start')
    
    return schedule_df, solve_time


# =============================================================================
# METRICS CALCULATION
# =============================================================================
def calculate_metrics(
    schedule_df: pd.DataFrame,
    config: dict = None,
    solve_time: float = 0.0
) -> ScheduleMetrics:
    """
    Calculate comprehensive metrics from a schedule.
    
    Parameters:
        schedule_df: Schedule DataFrame with Start/End columns
        config: Configuration dict
        solve_time: Time taken to generate schedule
    
    Returns:
        ScheduleMetrics dataclass
    """
    config = config or DEFAULT_CONFIG
    resources = config['resource_sequence']
    lambda_weight = config.get('wait_time_penalty', 0.5)
    
    # Makespan
    end_cols = [f'{res}_End' for res in resources]
    makespan = schedule_df[end_cols].max().max()
    
    # Wait times
    if 'Wait' in schedule_df.columns:
        total_wait = schedule_df['Wait'].sum()
        avg_wait = schedule_df['Wait'].mean()
        max_wait = schedule_df['Wait'].max()
    else:
        # Calculate if not present
        total_wait = 0.0
        for _, row in schedule_df.iterrows():
            for i in range(len(resources) - 1):
                curr = resources[i]
                next_r = resources[i + 1]
                gap = row[f'{next_r}_Start'] - row[f'{curr}_End']
                total_wait += max(0, gap)
        avg_wait = total_wait / len(schedule_df)
        max_wait = None  # Not calculable without per-patient data
    
    # Resource utilization
    utilization = {}
    for res in resources:
        total_proc = (schedule_df[f'{res}_End'] - schedule_df[f'{res}_Start']).sum()
        utilization[res] = total_proc / makespan if makespan > 0 else 0.0
    
    # Objective value
    objective_value = makespan + lambda_weight * total_wait
    
    return ScheduleMetrics(
        makespan=makespan,
        total_wait=total_wait,
        avg_wait=avg_wait,
        max_wait=max_wait if max_wait is not None else 0.0,
        solve_time=solve_time,
        utilization=utilization,
        objective_value=objective_value
    )


# =============================================================================
# SIMULATION: GENERATE REALISTIC FAILURE PROBABILITIES
# =============================================================================
def simulate_failure_probabilities(
    df_patients: pd.DataFrame,
    base_rate: float = 0.10,
    seed: int = 42
) -> List[float]:
    """
    Generate realistic failure probabilities based on patient profiles.
    
    Calibrated to literature: 5-15% failure rate in dementia populations
    (Porembka et al. 2025).
    
    Parameters:
        df_patients: DataFrame with patient characteristics
        base_rate: Baseline failure probability
        seed: Random seed
    
    Returns:
        List of failure probabilities
    """
    np.random.seed(seed)
    
    probs = []
    for _, row in df_patients.iterrows():
        age = row.get('NACCAGE', 70)
        moca = row.get('MOCATOTS', 25)
        cdr = row.get('CDRGLOB', 0.5)
        agit = row.get('AGIT', 0)
        mot = row.get('MOT', 0)
        anx = row.get('ANX', 0)
        
        # Base probability
        p = base_rate
        
        # Age effect: +0.5% per year over 65
        p += max(0, (age - 65)) * 0.005
        
        # Cognitive effect: +1.5% per MoCA point below 26
        if pd.notna(moca):
            p += max(0, (26 - moca)) * 0.015
        
        # CDR effect: +10% per 0.5 CDR increment
        if pd.notna(cdr):
            p += cdr * 0.10
        
        # Behavioral effects
        p += (agit * 0.10) if pd.notna(agit) else 0
        p += (mot * 0.08) if pd.notna(mot) else 0
        p += (anx * 0.05) if pd.notna(anx) else 0
        
        # Add noise and clip
        p += np.random.normal(0, 0.03)
        p = np.clip(p, 0.02, 0.60)
        probs.append(p)
    
    return probs


# =============================================================================
# COMPARISON RUNNER
# =============================================================================
def run_baseline_comparison(
    patients: List[PatientData],
    config: dict = None,
    include_milp: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run all baseline methods on a patient cohort and compare.
    
    Parameters:
        patients: List of PatientData objects
        config: Configuration dict
        include_milp: Whether to include MILP (slower)
        verbose: Print progress
    
    Returns:
        DataFrame with metrics for each method
    """
    config = config or DEFAULT_CONFIG
    
    methods = {
        'FCFS': fcfs_schedule,
        'SPT': spt_schedule,
        'LPT': lpt_schedule,
        'Deterministic': deterministic_schedule,
        'Age-Stratified': age_stratified_schedule,
        'Random': lambda p, c: random_schedule(p, c, n_samples=10, seed=42),
    }
    
    if include_milp:
        methods['Stochastic-MILP'] = lambda p, c: stochastic_milp_schedule(p, c, verbose=False)
    
    results = []
    
    for method_name, method_func in methods.items():
        if verbose:
            print(f"  Running {method_name}...")
        
        schedule_df, solve_time = method_func(patients, config)
        
        if schedule_df is not None:
            metrics = calculate_metrics(schedule_df, config, solve_time)
            result = {'Method': method_name, **metrics.to_dict()}
            results.append(result)
        else:
            if verbose:
                print(f"    WARNING: {method_name} failed to find solution")
    
    return pd.DataFrame(results)


# =============================================================================
# VALUE CALCULATIONS
# =============================================================================
def calculate_value_of_stochastic_solution(
    stochastic_obj: float,
    deterministic_obj: float
) -> float:
    """
    Value of Stochastic Solution (VSS).
    
    VSS = EV solution cost - Stochastic solution cost
    Positive value indicates stochastic approach is better.
    """
    return deterministic_obj - stochastic_obj


def calculate_value_of_ml(
    ml_obj: float,
    stratified_obj: float
) -> float:
    """
    Value of ML over simple stratification.
    
    Positive value indicates ML approach is better.
    """
    return stratified_obj - ml_obj


def calculate_improvement_pct(
    baseline_value: float,
    improved_value: float
) -> float:
    """Calculate percentage improvement (lower is better for costs)."""
    if baseline_value == 0:
        return 0.0
    return (baseline_value - improved_value) / baseline_value * 100


# =============================================================================
# STATISTICAL TESTS
# =============================================================================
def paired_comparison_test(
    method1_values: List[float],
    method2_values: List[float],
    alternative: str = 'less'
) -> Tuple[float, float]:
    """
    Perform paired statistical test between two methods.
    
    Parameters:
        method1_values: Metric values for method 1 (e.g., your method)
        method2_values: Metric values for method 2 (e.g., baseline)
        alternative: 'less' if method1 < method2 is better
    
    Returns:
        (t_statistic, p_value)
    """
    # Wilcoxon signed-rank test (non-parametric)
    stat, p_value = stats.wilcoxon(method1_values, method2_values, 
                                    alternative=alternative)
    return stat, p_value


# =============================================================================
# SIMULATION EVALUATION: Test schedules against actual outcomes
# =============================================================================
def evaluate_schedule_with_simulation(
    schedule_df: pd.DataFrame,
    patients: List[PatientData],
    config: dict = None,
    n_simulations: int = 100,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate a schedule by simulating actual failure outcomes.
    
    For each simulation:
    1. Draw Bernoulli outcomes for each patient/resource based on p_fail
    2. Compute actual durations (base + penalty if failure)
    3. Re-compute schedule metrics with actual durations
    
    Returns expected (averaged) metrics across simulations.
    """
    config = config or DEFAULT_CONFIG
    resources = config['resource_sequence']
    
    np.random.seed(seed)
    
    # Build patient lookup
    p_fail_lookup = {p.patient_id: p.p_fail_mri for p in patients}
    
    # Get the patient ordering from the schedule
    patient_order = schedule_df['Patient'].tolist()
    
    simulated_makespans = []
    simulated_total_waits = []
    simulated_overtimes = []  # Count of simulations exceeding target
    
    target_makespan = schedule_df[[f'{r}_End' for r in resources]].max().max()
    
    for sim in range(n_simulations):
        # Simulate actual durations
        actual_durations = {}
        for pid in patient_order:
            p_mri = p_fail_lookup[pid]
            for res, params in config['resources'].items():
                base = params['base_duration']
                penalty = params['failure_penalty']
                risk_scale = params.get('risk_scale', 1.0)
                
                # Bernoulli draw: did failure occur?
                p_res = p_mri * risk_scale
                failed = np.random.random() < p_res
                
                actual_durations[(pid, res)] = base + (penalty if failed else 0)
        
        # Re-build schedule with actual durations (greedy forward pass)
        sim_schedule = build_heuristic_schedule(patient_order, actual_durations, resources)
        
        # Compute metrics
        end_cols = [f'{res}_End' for res in resources]
        sim_makespan = sim_schedule[end_cols].max().max()
        
        sim_wait = 0.0
        for _, row in sim_schedule.iterrows():
            for i in range(len(resources) - 1):
                curr = resources[i]
                next_r = resources[i + 1]
                gap = row[f'{next_r}_Start'] - row[f'{curr}_End']
                sim_wait += max(0, gap)
        
        simulated_makespans.append(sim_makespan)
        simulated_total_waits.append(sim_wait)
        
        # Check if this exceeded planned schedule
        if sim_makespan > target_makespan * 1.05:  # 5% tolerance
            simulated_overtimes.append(1)
        else:
            simulated_overtimes.append(0)
    
    return {
        'expected_makespan': np.mean(simulated_makespans),
        'makespan_std': np.std(simulated_makespans),
        'makespan_95pct': np.percentile(simulated_makespans, 95),
        'expected_wait': np.mean(simulated_total_waits),
        'overtime_probability': np.mean(simulated_overtimes),
        'planned_makespan': target_makespan,
    }


def run_full_comparison_with_simulation(
    patients: List[PatientData],
    config: dict = None,
    n_simulations: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run all baselines and evaluate with simulation.
    
    This is the key experiment: compare planned schedules against
    simulated actual outcomes to measure robustness.
    """
    config = config or DEFAULT_CONFIG
    
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
    
    for method_name, method_func in methods.items():
        if verbose:
            print(f"  {method_name}...", end=" ", flush=True)
        
        # Generate schedule
        schedule_df, solve_time = method_func(patients, config)
        
        if schedule_df is None:
            if verbose:
                print("FAILED")
            continue
        
        # Calculate planned metrics
        planned_metrics = calculate_metrics(schedule_df, config, solve_time)
        
        # Evaluate with simulation
        sim_metrics = evaluate_schedule_with_simulation(
            schedule_df, patients, config, n_simulations, seed
        )
        
        result = {
            'Method': method_name,
            'Planned_Makespan': planned_metrics.makespan,
            'Expected_Makespan': sim_metrics['expected_makespan'],
            'Makespan_Std': sim_metrics['makespan_std'],
            'Makespan_95pct': sim_metrics['makespan_95pct'],
            'Overtime_Prob': sim_metrics['overtime_probability'],
            'Solve_Time': solve_time,
        }
        results.append(result)
        
        if verbose:
            print(f"Plan: {planned_metrics.makespan:.0f}, "
                  f"Exp: {sim_metrics['expected_makespan']:.0f}, "
                  f"OT: {sim_metrics['overtime_probability']*100:.0f}%")
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN TESTING
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("BASELINE SCHEDULING METHODS - Test Run")
    print("=" * 70)
    
    # Create test patients with realistic heterogeneity
    np.random.seed(42)
    test_patients = []
    for i in range(10):
        # Create realistic patient profiles
        age = np.random.uniform(65, 90)
        cdr = np.random.choice([0, 0.5, 1, 2], p=[0.2, 0.3, 0.35, 0.15])
        moca = max(10, min(30, 28 - cdr * 5 + np.random.normal(0, 3)))
        
        # Calculate realistic failure probability
        p_fail = 0.08  # base rate
        p_fail += (age - 65) * 0.004  # age effect
        p_fail += (26 - moca) * 0.012 if moca < 26 else 0  # cognitive effect
        p_fail += cdr * 0.08  # dementia severity
        p_fail = np.clip(p_fail + np.random.normal(0, 0.02), 0.03, 0.50)
        
        p = PatientData(
            patient_id=f"P{i+1:03d}",
            p_fail_mri=p_fail,
            age=age,
            moca=moca,
            cdr=cdr
        )
        test_patients.append(p)
    
    print(f"\nTest cohort: {len(test_patients)} patients")
    print(f"{'Patient':<10} {'Age':>6} {'p_fail':>8} {'MoCA':>6} {'CDR':>5}")
    print("-" * 40)
    for p in sorted(test_patients, key=lambda x: x.p_fail_mri, reverse=True)[:5]:
        print(f"{p.patient_id:<10} {p.age:>6.1f} {p.p_fail_mri:>8.3f} {p.moca:>6.1f} {p.cdr:>5.1f}")
    print(f"... and {len(test_patients) - 5} more\n")
    
    # Run comparison WITH simulation evaluation
    print("Running full comparison with simulation (100 scenarios)...")
    print("-" * 50)
    results_df = run_full_comparison_with_simulation(
        test_patients, 
        n_simulations=100,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Method':<18} {'Planned':>10} {'Expected':>10} {'95pct':>10} {'OT%':>8}")
    print("-" * 60)
    for _, row in results_df.sort_values('Expected_Makespan').iterrows():
        print(f"{row['Method']:<18} {row['Planned_Makespan']:>10.1f} "
              f"{row['Expected_Makespan']:>10.1f} {row['Makespan_95pct']:>10.1f} "
              f"{row['Overtime_Prob']*100:>7.1f}%")
    
    # Calculate key comparisons
    print("\n" + "=" * 70)
    print("VALUE ANALYSIS")
    print("=" * 70)
    
    if 'Stochastic-MILP' in results_df['Method'].values:
        milp_row = results_df[results_df['Method'] == 'Stochastic-MILP'].iloc[0]
        
        for baseline in ['FCFS', 'Deterministic', 'Age-Stratified']:
            if baseline in results_df['Method'].values:
                base_row = results_df[results_df['Method'] == baseline].iloc[0]
                
                exp_improvement = calculate_improvement_pct(
                    base_row['Expected_Makespan'], milp_row['Expected_Makespan']
                )
                ot_reduction = base_row['Overtime_Prob'] - milp_row['Overtime_Prob']
                
                print(f"\nvs {baseline}:")
                print(f"  Expected Makespan Improvement: {exp_improvement:+.1f}%")
                print(f"  Overtime Probability Reduction: {ot_reduction*100:+.1f} pp")
    
    print("\n✓ baselines.py test complete!")
