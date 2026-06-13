import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
import pulp
import warnings
import time
warnings.filterwarnings('ignore')

def main():
    print("=" * 70)
    print("STOCHASTIC MULTI-MODAL CLINICAL TRIAL SCHEDULING")
    print("Predictive-Prescriptive Analytics for NACC Alzheimer's Research")
    print("=" * 70)
    
    # =========================================================================
    # PHASE 1: DATA LOADING & DATE-BASED MERGING
    # =========================================================================
    print("\n=== Phase 1: Data Loading & Date-Based Merging ===")
    
    uds_file = 'investigator_nacc69.csv'
    mriqc_file = 'investigator_scan_mriqc_nacc69.csv'
    
    uds_cols = ['NACCID', 'VISITYR', 'VISITMO', 'VISITDAY', 'NACCAGE', 'NACCMMSE']
    mri_cols = ['NACCID', 'STUDYDATE', 'STUDYQC']
    
    print(f"Loading {uds_file}...")
    try:
        uds_df = pd.read_csv(uds_file, usecols=lambda c: c in uds_cols, low_memory=False)
        mri_df = pd.read_csv(mriqc_file, usecols=lambda c: c in mri_cols, low_memory=False)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"  UDS records loaded: {len(uds_df)}")
    print(f"  MRI records loaded: {len(mri_df)}")

    # Clean UDS Dates (NACC uses 88, 99 for missing days)
    if 'VISITDAY' in uds_df.columns:
        uds_df['VISITDAY'] = pd.to_numeric(uds_df['VISITDAY'], errors='coerce')
        uds_df['VISITDAY'] = uds_df['VISITDAY'].apply(lambda x: 15 if pd.isna(x) or x > 31 or x < 1 else x)
    else:
        uds_df['VISITDAY'] = 15

    print("Aligning Clinical and MRI timelines...")
    uds_df['UDS_DATE'] = pd.to_datetime(
        uds_df['VISITYR'].astype(str) + '-' + 
        uds_df['VISITMO'].astype(str) + '-' + 
        uds_df['VISITDAY'].astype(int).astype(str), 
        errors='coerce'
    )
    
    mri_df['MRI_DATE'] = pd.to_datetime(mri_df['STUDYDATE'], errors='coerce')
    
    # Drop rows without valid dates
    uds_df = uds_df.dropna(subset=['UDS_DATE'])
    mri_df = mri_df.dropna(subset=['MRI_DATE'])

    # Merge on Patient ID
    print("Merging on NACCID...")
    df = pd.merge(uds_df, mri_df, on='NACCID', how='inner')
    
    # Calculate the absolute difference in days
    df['DAYS_DIFF'] = (df['MRI_DATE'] - df['UDS_DATE']).dt.days.abs()
    
    # Keep only clinical visits within 90 days of MRI scan
    df = df[df['DAYS_DIFF'] <= 90]
    
    # Sort by time difference and drop duplicates
    df = df.sort_values('DAYS_DIFF').drop_duplicates(subset=['NACCID', 'STUDYDATE'])
    
    print(f"Successfully matched {len(df)} MRI scans to proximate clinical visits.")

    # Clean missing cognitive values (NACC uses codes like 88, 99, -4)
    missing_codes = [88, 99, 888, 999, -4, -1]
    df.replace(missing_codes, np.nan, inplace=True)
    
    # Create Target Variable: MRI Failure (SCAN Dictionary: 1 = Pass, >1 = Fail)
    df['MRI_FAILED'] = df['STUDYQC'].apply(lambda x: 1 if pd.to_numeric(x, errors='coerce') > 1 else 0) 
    
    # Drop rows missing the primary predictor
    df_clean = df.dropna(subset=['NACCAGE'])
    if len(df_clean) == 0:
        print("Error: No data left after dropping missing age values.")
        return

    # =========================================================================
    # DIAGNOSTIC: DATA QUALITY ASSESSMENT
    # =========================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Data Quality Assessment")
    print("=" * 70)
    
    print(f"\n[Class Distribution]")
    print(f"  Total records: {len(df_clean)}")
    print(f"  MRI Failures (STUDYQC > 1): {df_clean['MRI_FAILED'].sum()}")
    print(f"  MRI Passes (STUDYQC = 1): {len(df_clean) - df_clean['MRI_FAILED'].sum()}")
    print(f"  Failure Rate: {df_clean['MRI_FAILED'].mean()*100:.2f}%")
    
    print(f"\n[Age Distribution]")
    print(f"  Age range: {df_clean['NACCAGE'].min():.0f} - {df_clean['NACCAGE'].max():.0f} years")
    print(f"  Mean age: {df_clean['NACCAGE'].mean():.1f} years")
    
    print(f"\n[MMSE Distribution]")
    if 'NACCMMSE' in df_clean.columns:
        valid_mmse = df_clean['NACCMMSE'].dropna()
        print(f"  Valid MMSE scores: {len(valid_mmse)} / {len(df_clean)} ({100*len(valid_mmse)/len(df_clean):.1f}%)")
        if len(valid_mmse) > 0:
            print(f"  MMSE range: {valid_mmse.min():.0f} - {valid_mmse.max():.0f}")
            print(f"  Mean MMSE: {valid_mmse.mean():.1f}")
            # Breakdown by cognitive status
            severe_impairment = (valid_mmse < 18).sum()
            mild_impairment = ((valid_mmse >= 18) & (valid_mmse < 24)).sum()
            normal = (valid_mmse >= 24).sum()
            print(f"  Severe impairment (MMSE < 18): {severe_impairment}")
            print(f"  Mild impairment (18 <= MMSE < 24): {mild_impairment}")
            print(f"  Normal cognition (MMSE >= 24): {normal}")
    else:
        print("  NACCMMSE column not in dataset!")

    # =========================================================================
    # PHASE 2: PREDICTIVE ML MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 2: Predictive ML Model")
    print("=" * 70)
    
    features = ['NACCAGE']
    if 'NACCMMSE' in df_clean.columns:
        features.append('NACCMMSE')
    print(f"Features used: {features}")
        
    X = df_clean[features]
    y = df_clean['MRI_FAILED']
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Handle class imbalance with class weights
    if y.sum() > 0:
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"Class weights (to handle imbalance): {class_weight_dict}")
    else:
        class_weight_dict = None
        print("WARNING: No positive class (MRI failures) found in training data!")
    
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight=class_weight_dict
    )
    rf_model.fit(X_imputed, y)
    
    # Feature importance
    print(f"\nFeature Importances:")
    for feat, imp in zip(features, rf_model.feature_importances_):
        print(f"  {feat}: {imp:.3f}")
    
    # Probability distribution across ALL patients
    X_all = imputer.transform(df_clean[features])
    all_probs = rf_model.predict_proba(X_all)[:, 1]
    
    print(f"\n[Predicted Failure Probability Distribution - All Patients]")
    print(f"  p_fail range: {all_probs.min():.4f} - {all_probs.max():.4f}")
    print(f"  p_fail mean: {all_probs.mean():.4f}")
    print(f"  p_fail median: {np.median(all_probs):.4f}")
    print(f"  Patients with p_fail > 0.05: {(all_probs > 0.05).sum()}")
    print(f"  Patients with p_fail > 0.10: {(all_probs > 0.10).sum()}")
    print(f"  Patients with p_fail > 0.20: {(all_probs > 0.20).sum()}")

    # =========================================================================
    # PHASE 3: SELECT PATIENTS FOR SCHEDULING
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 3: Patient Selection for Daily Schedule")
    print("=" * 70)
    
    # Add predicted probabilities to all patients
    df_clean = df_clean.copy()
    df_clean['p_fail'] = all_probs
    
    # OPTION A: Select patients with DIVERSE risk profiles for demonstration
    # Sort by failure probability and sample across the distribution
    df_sorted = df_clean.sort_values('p_fail')
    n_patients = 10
    
    # Remove exact duplicates (same patient scheduled twice)
    df_unique = df_sorted.drop_duplicates(subset=['NACCID'])
    
    # Sample from different quantiles to get diverse risk profiles
    if len(df_unique) >= n_patients:
        indices = np.linspace(0, len(df_unique)-1, n_patients, dtype=int)
        daily_patients = df_unique.iloc[indices].copy()
    else:
        daily_patients = df_unique.head(n_patients).copy()
    
    print(f"Selected {len(daily_patients)} unique patients with diverse risk profiles:")
    print(f"  p_fail range in selection: {daily_patients['p_fail'].min():.4f} - {daily_patients['p_fail'].max():.4f}")
    
    # =========================================================================
    # OPTION B: SIMULATION MODE (uncomment to use synthetic probabilities)
    # =========================================================================
    USE_SIMULATION = False  # Set to True for paper demonstrations
    
    if USE_SIMULATION:
        print("\n*** SIMULATION MODE ENABLED ***")
        print("Using synthetic failure probabilities for demonstration")
        
        # Generate realistic failure probabilities based on cognitive profiles
        np.random.seed(42)
        synthetic_probs = []
        for _, row in daily_patients.iterrows():
            age = row['NACCAGE']
            mmse = row.get('NACCMMSE', 25)
            if pd.isna(mmse):
                mmse = 25
            
            # Base probability increases with age and decreases with MMSE
            base_prob = 0.05 + (age - 60) * 0.005 + (30 - mmse) * 0.02
            base_prob = np.clip(base_prob, 0.02, 0.50)
            # Add some noise
            prob = base_prob + np.random.normal(0, 0.05)
            prob = np.clip(prob, 0.01, 0.60)
            synthetic_probs.append(prob)
        
        daily_patients['p_fail'] = synthetic_probs
    
    # =========================================================================
    # PHASE 4: PRESCRIPTIVE OR SCHEDULING MODEL (FIXED VERSION)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 4: Prescriptive OR Scheduling Model")
    print("=" * 70)
    
    # Ensure unique patient IDs for scheduling
    daily_patients = daily_patients.drop_duplicates(subset=['NACCID'])
    patients = daily_patients['NACCID'].astype(str).tolist()
    p_fail = dict(zip(patients, daily_patients['p_fail']))
    
    # MRI parameters (from paper formulation)
    d_mri = 45      # Base MRI duration (minutes)
    delta_mri = 30  # Additional time if failure/rescan needed (minutes)
    
    # Calculate expected duration: d̃_ij = d_ij + p_ij * δ_ij
    expected_duration = {i: int(d_mri + (p_fail[i] * delta_mri)) for i in patients}
    
    print("\nPatient Risk Profiles and Expected MRI Durations:")
    print("-" * 65)
    print(f"{'Patient':<15} {'Age':>6} {'MMSE':>6} {'p_fail':>10} {'Expected Time':>15}")
    print("-" * 65)
    
    for i, patient_id in enumerate(patients):
        row = daily_patients[daily_patients['NACCID'].astype(str) == patient_id].iloc[0]
        age = row['NACCAGE']
        mmse = row.get('NACCMMSE', 'N/A')
        if pd.notna(mmse):
            mmse_str = f"{mmse:.0f}"
        else:
            mmse_str = "N/A"
        print(f"{patient_id:<15} {age:>6.0f} {mmse_str:>6} {p_fail[patient_id]:>10.4f} {expected_duration[patient_id]:>12} mins")
    
    print("-" * 65)
    
    # =========================================================================
    # MILP FORMULATION (FIXED - Reduced redundancy, proper Big-M)
    # =========================================================================
    print("\nBuilding MILP model...")
    start_time = time.time()
    
    prob = pulp.LpProblem("MRI_Scheduling_Min_Makespan", pulp.LpMinimize)
    
    # Decision Variables
    start_times = pulp.LpVariable.dicts("Start", patients, lowBound=0, cat='Continuous')
    C_max = pulp.LpVariable("Makespan", lowBound=0, cat='Continuous')
    
    # Binary variables for ordering - only for unique pairs (i < j)
    # y[i,j] = 1 means patient i is scheduled BEFORE patient j
    patient_pairs = [(patients[i], patients[j]) for i in range(len(patients)) for j in range(i+1, len(patients))]
    y_var = pulp.LpVariable.dicts("y", patient_pairs, cat='Binary')
    
    # Big-M: should be larger than max possible makespan
    # With 10 patients at max 75 mins each = 750 mins max
    M = 1000
    
    # Objective: Minimize makespan
    prob += C_max
    
    # Constraints
    print(f"  Adding {len(patients)} makespan constraints...")
    for i in patients:
        # Makespan constraint: C_max >= completion time of each patient
        prob += C_max >= start_times[i] + expected_duration[i]
    
    # Disjunctive constraints: no two patients overlap on the MRI scanner
    # FIXED: Only add constraints once per pair
    print(f"  Adding {len(patient_pairs)} disjunctive constraint pairs...")
    for (i, j) in patient_pairs:
        # If y[i,j] = 1: i before j, so start_j >= end_i
        # If y[i,j] = 0: j before i, so start_i >= end_j
        prob += start_times[j] >= start_times[i] + expected_duration[i] - M * (1 - y_var[(i, j)])
        prob += start_times[i] >= start_times[j] + expected_duration[j] - M * y_var[(i, j)]

    print(f"  Total constraints: {len(prob.constraints)}")
    print(f"  Total variables: {len(prob.variables())} ({len(patient_pairs)} binary)")
    
    # Solve with timeout
    print("\nSolving MILP (timeout: 60 seconds)...")
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=60)
    prob.solve(solver)
    
    solve_time = time.time() - start_time
    print(f"\nSolve time: {solve_time:.2f} seconds")
    print(f"Optimization Status: {pulp.LpStatus[prob.status]}")
    
    if prob.status != pulp.constants.LpStatusOptimal:
        print("WARNING: Optimal solution not found!")
        if prob.status == pulp.constants.LpStatusNotSolved:
            print("  The solver did not complete. Try reducing the number of patients.")
        return
    
    print(f"Optimal Clinic Makespan: {pulp.value(C_max):.0f} minutes ({pulp.value(C_max)/60:.1f} hours)")
    
    # =========================================================================
    # RESULTS: OPTIMAL SCHEDULE
    # =========================================================================
    print("\n" + "=" * 70)
    print("OPTIMAL PATIENT SCHEDULE")
    print("=" * 70)
    
    schedule = []
    for i in patients:
        start = pulp.value(start_times[i])
        end = start + expected_duration[i]
        schedule.append({
            'Patient': i,
            'Start_Time': start,
            'End_Time': end,
            'Duration': expected_duration[i],
            'p_fail': p_fail[i]
        })
    
    schedule_df = pd.DataFrame(schedule).sort_values(by='Start_Time')
    
    print(f"\n{'Seq':<4} {'Patient':<15} {'Start':>10} {'End':>10} {'Duration':>10} {'p_fail':>10}")
    print("-" * 65)
    
    for idx, (_, row) in enumerate(schedule_df.iterrows(), 1):
        start_hr = int(row['Start_Time'] // 60)
        start_min = int(row['Start_Time'] % 60)
        end_hr = int(row['End_Time'] // 60)
        end_min = int(row['End_Time'] % 60)
        print(f"{idx:<4} {row['Patient']:<15} {start_hr:02d}:{start_min:02d}     {end_hr:02d}:{end_min:02d}     {row['Duration']:>6} mins  {row['p_fail']:>8.4f}")
    
    print("-" * 65)
    print(f"Total Makespan: {pulp.value(C_max):.0f} minutes")
    
    # =========================================================================
    # COMPARISON: Stochastic vs Deterministic Scheduling
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Stochastic vs Deterministic Approach")
    print("=" * 70)
    
    # Deterministic: All patients get base duration (ignoring risk)
    deterministic_makespan = len(patients) * d_mri
    stochastic_makespan = pulp.value(C_max)
    
    # Expected actual completion (accounting for failures)
    expected_actual = sum(d_mri + p_fail[i] * delta_mri for i in patients)
    
    print(f"\nDeterministic Makespan (ignoring risk): {deterministic_makespan} minutes")
    print(f"Stochastic Makespan (risk-adjusted):    {stochastic_makespan:.0f} minutes")
    print(f"Expected Actual Duration:               {expected_actual:.0f} minutes")
    print(f"\nBuffer time built in: {stochastic_makespan - deterministic_makespan:.0f} minutes")
    print(f"Risk premium: {(stochastic_makespan/deterministic_makespan - 1)*100:.1f}%")
    
    # =========================================================================
    # SCHEDULING INSIGHT
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCHEDULING INSIGHT")
    print("=" * 70)
    
    # Check if high-risk patients are scheduled strategically
    schedule_df = schedule_df.reset_index(drop=True)
    schedule_df['Sequence'] = range(1, len(schedule_df) + 1)
    
    high_risk = schedule_df[schedule_df['p_fail'] > 0.3]
    low_risk = schedule_df[schedule_df['p_fail'] < 0.1]
    
    print(f"\nHigh-risk patients (p_fail > 0.30): {len(high_risk)}")
    if len(high_risk) > 0:
        print(f"  Scheduled at positions: {high_risk['Sequence'].tolist()}")
        
    print(f"\nLow-risk patients (p_fail < 0.10): {len(low_risk)}")
    if len(low_risk) > 0:
        print(f"  Scheduled at positions: {low_risk['Sequence'].tolist()}")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    schedule_df.to_csv('optimal_schedule.csv', index=False)
    print(f"\nSchedule saved to: optimal_schedule.csv")
    
    return schedule_df

if __name__ == "__main__":
    schedule = main()