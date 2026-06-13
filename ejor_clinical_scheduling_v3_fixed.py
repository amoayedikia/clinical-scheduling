"""
STOCHASTIC MULTI-MODAL CLINICAL TRIAL SCHEDULING
Predictive-Prescriptive Analytics for NACC Alzheimer's Research

Version 3.0 - With Correct Cognitive Predictors (MoCA, CDR, NPI-Q)
Based on diagnostic findings: MRI scans are 2021-2025 (UDS3 era)

Author: [Your Name]
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score
import pulp
import warnings
import time
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'uds_file': 'investigator_nacc69.csv',
    'mri_file': 'investigator_scan_mriqc_nacc69.csv',
    
    # Scheduling parameters
    'resources': {
        'UDS': {'base_duration': 60, 'failure_penalty': 15},
        'MRI': {'base_duration': 45, 'failure_penalty': 30},
        'CSF': {'base_duration': 30, 'failure_penalty': 20},
    },
    
    'n_patients_daily': 10,
    'use_simulation': False,  # Set True if real failure rate too low
    'wait_time_penalty': 0.5,  # λ in objective
    
    # NACC missing codes
    'missing_codes': [88, 95, 96, 97, 98, 99, 888, 999, -4, -1],
}


# =============================================================================
# PHASE 1: DATA LOADING WITH CORRECT COGNITIVE PREDICTORS
# =============================================================================
def load_and_merge_data():
    """Load UDS and MRI data with MoCA, CDR, and NPI-Q behavioral symptoms."""
    
    print("\n" + "=" * 70)
    print("PHASE 1: Data Loading & Merging")
    print("=" * 70)
    
    # Updated columns based on diagnostic findings
    uds_cols = [
        # Identifiers & Demographics
        'NACCID', 'VISITYR', 'VISITMO', 'VISITDAY', 'NACCAGE',
        
        # MoCA (UDS3 standard - 93.8% coverage in merged data!)
        'MOCATOTS',      # MoCA Total Raw Score (0-30)
        'NACCMOCA',      # MoCA corrected for education
        'MOCAREAS',      # Reason MoCA not completed
        
        # CDR (100% coverage!)
        'CDRSUM',        # CDR Sum of Boxes (0-18)
        'CDRGLOB',       # Global CDR (0, 0.5, 1, 2, 3)
        'MEMORY',        # CDR Memory domain
        'ORIENT',        # CDR Orientation domain
        'JUDGMENT',      # CDR Judgment domain
        
        # NPI-Q Behavioral Symptoms (~94% coverage!)
        'NPIQINF',       # NPI-Q informant present
        'AGIT',          # Agitation/aggression (0/1)
        'AGITSEV',       # Agitation severity (1-3)
        'ANX',           # Anxiety (0/1)
        'ANXSEV',        # Anxiety severity (1-3)
        'MOT',           # Motor disturbance (0/1)
        'MOTSEV',        # Motor disturbance severity
        'IRR',           # Irritability (0/1)
        'DEL',           # Delusions (0/1)
        'HALL',          # Hallucinations (0/1)
        'DEPD',          # Depression (0/1)
        'DEPDSEV',       # Depression severity
        'ELAT',          # Elation/euphoria
        'APA',           # Apathy
        'DISN',          # Disinhibition
        'NITE',          # Nighttime behavior
        'APP',           # Appetite changes
        
        # Clinician Judgment
        'DECCLCOG',      # Clinician: meaningful cognitive decline
        'COGMEM',        # Clinician: memory concern
        'BEAPATHY',      # Clinician: behavioral apathy
        'BEDEP',         # Clinician: behavioral depression
        'BEAGIT',        # Clinician: behavioral agitation
        'BEANX',         # Clinician: behavioral anxiety
        
        # MMSE (backup, low coverage in UDS3)
        'NACCMMSE',
    ]
    
    mri_cols = ['NACCID', 'STUDYDATE', 'STUDYQC']
    
    print(f"Loading {CONFIG['uds_file']}...")
    try:
        uds_df = pd.read_csv(CONFIG['uds_file'], 
                            usecols=lambda c: c in uds_cols, 
                            low_memory=False)
        mri_df = pd.read_csv(CONFIG['mri_file'], 
                            usecols=lambda c: c in mri_cols, 
                            low_memory=False)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    print(f"  UDS records: {len(uds_df)}")
    print(f"  MRI records: {len(mri_df)}")

    # Clean dates
    if 'VISITDAY' in uds_df.columns:
        uds_df['VISITDAY'] = pd.to_numeric(uds_df['VISITDAY'], errors='coerce')
        uds_df['VISITDAY'] = uds_df['VISITDAY'].apply(
            lambda x: 15 if pd.isna(x) or x > 31 or x < 1 else x
        )
    else:
        uds_df['VISITDAY'] = 15

    uds_df['UDS_DATE'] = pd.to_datetime(
        uds_df['VISITYR'].astype(str) + '-' + 
        uds_df['VISITMO'].astype(str) + '-' + 
        uds_df['VISITDAY'].astype(int).astype(str), 
        errors='coerce'
    )
    mri_df['MRI_DATE'] = pd.to_datetime(mri_df['STUDYDATE'], errors='coerce')
    
    uds_df = uds_df.dropna(subset=['UDS_DATE'])
    mri_df = mri_df.dropna(subset=['MRI_DATE'])

    # Merge
    print("Merging on NACCID with 90-day temporal proximity...")
    df = pd.merge(uds_df, mri_df, on='NACCID', how='inner')
    df['DAYS_DIFF'] = (df['MRI_DATE'] - df['UDS_DATE']).dt.days.abs()
    df = df[df['DAYS_DIFF'] <= 90]
    df = df.sort_values('DAYS_DIFF').drop_duplicates(subset=['NACCID', 'STUDYDATE'])
    
    print(f"  Matched records: {len(df)}")

    # Replace missing codes
    df.replace(CONFIG['missing_codes'], np.nan, inplace=True)
    
    # Create target: MRI Failure
    df['MRI_FAILED'] = df['STUDYQC'].apply(
        lambda x: 1 if pd.to_numeric(x, errors='coerce') > 1 else 0
    )
    
    return df


# =============================================================================
# PHASE 2: FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    """Create derived features from cognitive and behavioral measures."""
    
    print("\n" + "=" * 70)
    print("PHASE 2: Feature Engineering")
    print("=" * 70)
    
    df = df.copy()
    
    # 1. Cognitive severity from MoCA (inverse - higher = worse)
    if 'MOCATOTS' in df.columns:
        df['COGNITIVE_IMPAIRMENT'] = 30 - df['MOCATOTS'].fillna(30)
        print("  Created COGNITIVE_IMPAIRMENT = 30 - MOCATOTS")
    
    # 2. Behavioral risk score (sum of NPI-Q symptoms)
    behavioral_cols = ['AGIT', 'ANX', 'MOT', 'IRR', 'DEL', 'HALL', 'DEPD', 'APA', 'DISN']
    available = [c for c in behavioral_cols if c in df.columns]
    if available:
        df['BEHAVIORAL_RISK'] = df[available].sum(axis=1, skipna=True)
        print(f"  Created BEHAVIORAL_RISK from {len(available)} NPI-Q symptoms")
    
    # 3. Severity-weighted behavioral score
    severity_pairs = [
        ('AGIT', 'AGITSEV'), ('ANX', 'ANXSEV'), 
        ('MOT', 'MOTSEV'), ('DEPD', 'DEPDSEV')
    ]
    severity_score = 0
    count = 0
    for symptom, severity in severity_pairs:
        if symptom in df.columns and severity in df.columns:
            weighted = df[symptom].fillna(0) * df[severity].fillna(1)
            severity_score = severity_score + weighted
            count += 1
    if count > 0:
        df['SEVERITY_WEIGHTED_RISK'] = severity_score
        print(f"  Created SEVERITY_WEIGHTED_RISK from {count} severity pairs")
    
    # 4. CDR-based impairment level
    if 'CDRGLOB' in df.columns:
        df['CDR_IMPAIRED'] = (df['CDRGLOB'] >= 1).astype(int)
        print("  Created CDR_IMPAIRED (CDR >= 1)")
    
    if 'CDRSUM' in df.columns:
        df['CDR_SEVERITY'] = df['CDRSUM'].fillna(0)
        print("  Created CDR_SEVERITY from CDRSUM")
    
    # 5. Age risk factor
    if 'NACCAGE' in df.columns:
        df['AGE_RISK'] = (df['NACCAGE'] - 65).clip(lower=0) / 20
        print("  Created AGE_RISK (normalized age > 65)")
    
    # 6. Clinician judgment composite
    clinician_cols = ['DECCLCOG', 'BEAGIT', 'BEANX']
    available_clin = [c for c in clinician_cols if c in df.columns]
    if available_clin:
        df['CLINICIAN_CONCERN'] = df[available_clin].sum(axis=1, skipna=True)
        print(f"  Created CLINICIAN_CONCERN from {len(available_clin)} flags")
    
    # 7. MRI-specific risk composite (most predictive for scan compliance)
    # Agitation + Motor + Anxiety = direct predictors of movement
    mri_risk_cols = ['AGIT', 'MOT', 'ANX']
    available_mri = [c for c in mri_risk_cols if c in df.columns]
    if available_mri:
        df['MRI_SPECIFIC_RISK'] = df[available_mri].sum(axis=1, skipna=True)
        print(f"  Created MRI_SPECIFIC_RISK from AGIT + MOT + ANX")
    
    return df


# =============================================================================
# PHASE 3: PREDICTIVE ML MODEL
# =============================================================================
def build_predictive_model(df):
    """Build ML model with correct cognitive predictors."""
    
    print("\n" + "=" * 70)
    print("PHASE 3: Predictive ML Model")
    print("=" * 70)
    
    # Feature priority (based on clinical relevance and availability)
    feature_priority = [
        # Tier 1: Direct behavioral predictors of scan compliance
        'AGIT', 'MOT', 'ANX', 'MRI_SPECIFIC_RISK',
        
        # Tier 2: Severity-weighted behavioral
        'AGITSEV', 'MOTSEV', 'ANXSEV',
        'BEHAVIORAL_RISK', 'SEVERITY_WEIGHTED_RISK',
        
        # Tier 3: Cognitive measures (MoCA replaces MMSE)
        'MOCATOTS', 'NACCMOCA', 'COGNITIVE_IMPAIRMENT',
        
        # Tier 4: CDR staging
        'CDRSUM', 'CDRGLOB', 'CDR_SEVERITY', 'CDR_IMPAIRED',
        
        # Tier 5: Demographics & clinician judgment
        'NACCAGE', 'AGE_RISK',
        'CLINICIAN_CONCERN', 'DECCLCOG', 'BEAGIT',
        
        # Tier 6: Other behavioral
        'IRR', 'DEL', 'HALL', 'DEPD',
    ]
    
    # Select available features
    available_features = [f for f in feature_priority if f in df.columns]
    
    print(f"\nAvailable features ({len(available_features)}):")
    for f in available_features:
        valid = df[f].notna().sum()
        pct = 100 * valid / len(df)
        print(f"  {f}: {valid}/{len(df)} ({pct:.1f}%)")
    
    # Filter to rows with valid age and target
    df_model = df.dropna(subset=['NACCAGE', 'MRI_FAILED']).copy()
    
    if len(df_model) == 0:
        print("ERROR: No valid data!")
        return None, None, None
    
    X = df_model[available_features]
    y = df_model['MRI_FAILED']
    
    # Impute missing
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Class distribution
    print(f"\n[Target Distribution]")
    print(f"  Total samples: {len(y)}")
    print(f"  MRI Failures: {y.sum()} ({100*y.mean():.2f}%)")
    print(f"  MRI Passes: {len(y) - y.sum()} ({100*(1-y.mean()):.2f}%)")
    
    # Handle class imbalance
    if y.sum() > 0 and y.sum() < len(y):
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"  Class weights: {class_weight_dict}")
    else:
        class_weight_dict = None
        print("  WARNING: Extreme class imbalance!")
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X_imputed, y)
    
    # Feature importance
    print(f"\n[Feature Importances - Top 10]")
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.head(10).iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"  {row['feature']:<25} {row['importance']:.4f} {bar}")
    
    # Cross-validation (if enough positive cases)
    if y.sum() >= 5:
        cv_scores = cross_val_score(model, X_imputed, y, cv=5, scoring='roc_auc')
        print(f"\n[Model Performance]")
        print(f"  CV AUC-ROC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # Predict probabilities
    df_model['p_fail_mri'] = model.predict_proba(X_imputed)[:, 1]
    
    print(f"\n[Predicted Failure Probabilities]")
    print(f"  Range: {df_model['p_fail_mri'].min():.4f} - {df_model['p_fail_mri'].max():.4f}")
    print(f"  Mean: {df_model['p_fail_mri'].mean():.4f}")
    print(f"  Median: {df_model['p_fail_mri'].median():.4f}")
    print(f"  Patients with p > 0.10: {(df_model['p_fail_mri'] > 0.10).sum()}")
    print(f"  Patients with p > 0.20: {(df_model['p_fail_mri'] > 0.20).sum()}")
    print(f"  Patients with p > 0.30: {(df_model['p_fail_mri'] > 0.30).sum()}")
    
    return df_model, model, (imputer, available_features)


# =============================================================================
# PHASE 4: MULTI-RESOURCE SCHEDULING
# =============================================================================
def build_schedule(df_patients, lambda_weight=0.5):
    """Build optimized multi-resource schedule."""
    
    print("\n" + "=" * 70)
    print("PHASE 4: Multi-Resource Scheduling (UDS → MRI → CSF)")
    print("=" * 70)
    
    patients = df_patients['NACCID'].astype(str).tolist()
    n_patients = len(patients)
    resources = ['UDS', 'MRI', 'CSF']
    
    # Get failure probabilities
    p_fail = {}
    for _, row in df_patients.iterrows():
        pid = str(row['NACCID'])
        p_mri = row.get('p_fail_mri', 0.05)
        p_fail[(pid, 'UDS')] = p_mri * 0.3  # Lower risk for interview
        p_fail[(pid, 'MRI')] = p_mri
        p_fail[(pid, 'CSF')] = p_mri * 0.8
    
    # Expected durations
    d_tilde = {}
    for i in patients:
        for j in resources:
            base = CONFIG['resources'][j]['base_duration']
            penalty = CONFIG['resources'][j]['failure_penalty']
            d_tilde[(i, j)] = base + p_fail[(i, j)] * penalty
    
    print(f"\nPatients: {n_patients}")
    print(f"Resources: {resources}")
    
    # Patient summary
    print(f"\n{'Patient':<15} {'MoCA':>6} {'CDR':>5} {'p_MRI':>8} {'Exp.Total':>10}")
    print("-" * 50)
    for pid in patients[:5]:
        row = df_patients[df_patients['NACCID'].astype(str) == pid].iloc[0]
        moca = row.get('MOCATOTS', 'N/A')
        cdr = row.get('CDRGLOB', 'N/A')
        p = p_fail[(pid, 'MRI')]
        total = sum(d_tilde[(pid, j)] for j in resources)
        moca_str = f"{moca:.0f}" if pd.notna(moca) else "N/A"
        cdr_str = f"{cdr:.1f}" if pd.notna(cdr) else "N/A"
        print(f"{pid:<15} {moca_str:>6} {cdr_str:>5} {p:>8.3f} {total:>8.1f}m")
    if n_patients > 5:
        print(f"... and {n_patients - 5} more")
    
    # =========================================================================
    # MILP
    # =========================================================================
    print("\nBuilding MILP...")
    start_time = time.time()
    
    prob = pulp.LpProblem("MultiResource_Scheduling", pulp.LpMinimize)
    
    # Variables
    x = pulp.LpVariable.dicts("x", 
        ((i, j) for i in patients for j in resources), 
        lowBound=0, cat='Continuous')
    C_max = pulp.LpVariable("C_max", lowBound=0)
    W = pulp.LpVariable.dicts("W", patients, lowBound=0)
    
    # Binary ordering - only unique pairs
    patient_pairs = [(patients[i], patients[j]) 
                     for i in range(len(patients)) 
                     for j in range(i+1, len(patients))]
    
    y = {}
    for j in resources:
        for (i1, i2) in patient_pairs:
            y[(i1, i2, j)] = pulp.LpVariable(f"y_{i1}_{i2}_{j}", cat='Binary')
    
    M = 10000
    
    # Objective
    prob += C_max + lambda_weight * pulp.lpSum(W[i] for i in patients)
    
    # Precedence constraints
    for i in patients:
        for s in range(len(resources) - 1):
            j_curr = resources[s]
            j_next = resources[s + 1]
            prob += x[(i, j_next)] >= x[(i, j_curr)] + d_tilde[(i, j_curr)]
    
    # Wait time
    for i in patients:
        wait_terms = []
        for s in range(len(resources) - 1):
            j_curr = resources[s]
            j_next = resources[s + 1]
            wait_terms.append(x[(i, j_next)] - x[(i, j_curr)] - d_tilde[(i, j_curr)])
        prob += W[i] == pulp.lpSum(wait_terms)
    
    # Disjunctive (with symmetry breaking)
    for j in resources:
        for (i1, i2) in patient_pairs:
            prob += x[(i2, j)] >= x[(i1, j)] + d_tilde[(i1, j)] - M * (1 - y[(i1, i2, j)])
            prob += x[(i1, j)] >= x[(i2, j)] + d_tilde[(i2, j)] - M * y[(i1, i2, j)]
            
            # Symmetry breaking: higher risk patients first
            if p_fail[(i1, 'MRI')] > p_fail[(i2, 'MRI')] + 0.01:
                prob += y[(i1, i2, j)] == 1
    
    # Makespan
    for i in patients:
        for j in resources:
            prob += C_max >= x[(i, j)] + d_tilde[(i, j)]
    
    print(f"  Constraints: {len(prob.constraints)}")
    print(f"  Variables: {len(prob.variables())}")
    
    # Solve
    print("\nSolving (timeout: 60s)...")
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=60)
    prob.solve(solver)
    
    solve_time = time.time() - start_time
    print(f"\nSolve time: {solve_time:.1f}s")
    print(f"Status: {pulp.LpStatus[prob.status]}")
    
    if prob.status not in [pulp.constants.LpStatusOptimal, 1]:
        print("WARNING: No optimal solution found")
        return None
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n{'='*70}")
    print("OPTIMAL SCHEDULE")
    print(f"{'='*70}")
    
    makespan = pulp.value(C_max)
    total_wait = sum(pulp.value(W[i]) for i in patients)
    
    print(f"\nMakespan: {makespan:.0f} min ({makespan/60:.1f} hrs)")
    print(f"Total Wait: {total_wait:.0f} min")
    print(f"Avg Wait/Patient: {total_wait/n_patients:.1f} min")
    
    # Build schedule dataframe
    schedule = []
    for i in patients:
        row = {'Patient': i, 'p_fail': p_fail[(i, 'MRI')]}
        for j in resources:
            start = pulp.value(x[(i, j)])
            row[f'{j}_Start'] = start
            row[f'{j}_End'] = start + d_tilde[(i, j)]
        row['Wait'] = pulp.value(W[i])
        schedule.append(row)
    
    schedule_df = pd.DataFrame(schedule).sort_values('UDS_Start')
    
    # Print schedule
    def fmt(mins):
        return f"{int(mins//60):02d}:{int(mins%60):02d}"
    
    print(f"\n{'#':<3} {'Patient':<12} {'UDS':<13} {'MRI':<13} {'CSF':<13} {'Wait':>6} {'Risk':>6}")
    print("-" * 75)
    
    for idx, row in schedule_df.iterrows():
        seq = list(schedule_df.index).index(idx) + 1
        uds = f"{fmt(row['UDS_Start'])}-{fmt(row['UDS_End'])}"
        mri = f"{fmt(row['MRI_Start'])}-{fmt(row['MRI_End'])}"
        csf = f"{fmt(row['CSF_Start'])}-{fmt(row['CSF_End'])}"
        print(f"{seq:<3} {row['Patient']:<12} {uds:<13} {mri:<13} {csf:<13} {row['Wait']:>5.0f}m {row['p_fail']:>5.2f}")
    
    return schedule_df


# =============================================================================
# PHASE 5: ANALYSIS
# =============================================================================
def analyze_results(schedule_df, n_patients):
    """Compare stochastic vs deterministic."""
    
    print(f"\n{'='*70}")
    print("COMPARISON: Stochastic vs Deterministic")
    print(f"{'='*70}")
    
    # Deterministic (ignore risk)
    det_per_patient = sum(r['base_duration'] for r in CONFIG['resources'].values())
    det_makespan = n_patients * det_per_patient  # Sequential
    
    # Stochastic
    stoch_makespan = schedule_df[['UDS_End', 'MRI_End', 'CSF_End']].max().max()
    
    print(f"\nDeterministic Makespan: {det_makespan} min (no parallelism)")
    print(f"Stochastic Makespan:    {stoch_makespan:.0f} min")
    print(f"Total Wait Time:        {schedule_df['Wait'].sum():.0f} min")
    print(f"\nBuffer Built In:        {stoch_makespan - det_makespan:.0f} min")
    
    # Risk distribution
    high = (schedule_df['p_fail'] > 0.15).sum()
    med = ((schedule_df['p_fail'] > 0.05) & (schedule_df['p_fail'] <= 0.15)).sum()
    low = (schedule_df['p_fail'] <= 0.05).sum()
    
    print(f"\n[Risk Distribution]")
    print(f"  High risk (p > 0.15):    {high}")
    print(f"  Medium (0.05 < p ≤ 0.15): {med}")
    print(f"  Low risk (p ≤ 0.05):     {low}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("STOCHASTIC CLINICAL TRIAL SCHEDULING - V3")
    print("With Correct Cognitive Predictors (MoCA, CDR, NPI-Q)")
    print("=" * 70)
    
    # Phase 1
    df = load_and_merge_data()
    if df is None:
        return
    
    # Phase 2
    df = engineer_features(df)
    
    # Phase 3
    df_model, model, artifacts = build_predictive_model(df)
    if df_model is None:
        return
    
    # Select diverse patients
    df_sorted = df_model.sort_values('p_fail_mri')
    df_unique = df_sorted.drop_duplicates(subset=['NACCID'])
    
    n = CONFIG['n_patients_daily']
    if len(df_unique) >= n:
        indices = np.linspace(0, len(df_unique)-1, n, dtype=int)
        df_patients = df_unique.iloc[indices].copy()
    else:
        df_patients = df_unique.head(n).copy()
    
    # Phase 4
    schedule_df = build_schedule(df_patients, CONFIG['wait_time_penalty'])
    
    if schedule_df is not None:
        # Phase 5
        analyze_results(schedule_df, len(df_patients))
        
        # Save
        schedule_df.to_csv('optimal_schedule_v3.csv', index=False)
        print(f"\nSaved to: optimal_schedule_v3.csv")
        
        return schedule_df
    
    return None


if __name__ == "__main__":
    schedule = main()
