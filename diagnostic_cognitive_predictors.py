"""
DIAGNOSTIC: Why Are Cognitive Predictors Missing?

This script investigates the data linkage issue between UDS clinical data
and MRI imaging data to understand why MMSE and other cognitive measures
are not available in the merged dataset.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 70)
    print("DIAGNOSTIC: Cognitive Predictor Availability")
    print("=" * 70)
    
    # =========================================================================
    # LOAD RAW DATA
    # =========================================================================
    uds_file = 'investigator_nacc69.csv'
    mriqc_file = 'investigator_scan_mriqc_nacc69.csv'
    
    # Load ALL potentially relevant cognitive columns
    cognitive_cols = [
        'NACCID', 'VISITYR', 'VISITMO', 'VISITDAY', 'NACCAGE',
        # MMSE
        'NACCMMSE', 'MMSECOMP', 'MMSEREAS',
        # MoCA (UDS3)
        'MOCATOTS', 'NACCMOCA', 'MOCAREAS',
        # CDR
        'CDRSUM', 'CDRGLOB', 'MEMORY', 'ORIENT', 'JUDGMENT',
        # NPI-Q Behavioral
        'NPIQINF', 'DEL', 'HALL', 'AGIT', 'DEPD', 'ANX', 'ELAT', 
        'APA', 'DISN', 'IRR', 'MOT', 'NITE', 'APP',
        # Severity scores
        'DELSEV', 'HALLSEV', 'AGITSEV', 'DEPDSEV', 'ANXSEV', 'MOTSEV',
        # Clinician judgment
        'DECCLCOG', 'COGMEM', 'BEAPATHY', 'BEAGIT', 'BEDEP', 'BEANX',
        # UDS version indicator
        'FORMVER', 'PACKET',
    ]
    
    print(f"\nLoading {uds_file}...")
    uds_df = pd.read_csv(uds_file, usecols=lambda c: c in cognitive_cols, low_memory=False)
    print(f"  UDS records: {len(uds_df)}")
    print(f"  UDS columns found: {list(uds_df.columns)}")
    
    print(f"\nLoading {mriqc_file}...")
    mri_df = pd.read_csv(mriqc_file, low_memory=False)
    print(f"  MRI records: {len(mri_df)}")
    print(f"  MRI columns: {list(mri_df.columns)}")
    
    # =========================================================================
    # PART 1: CHECK COGNITIVE VARIABLE AVAILABILITY IN RAW UDS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Cognitive Variable Availability in RAW UDS Data")
    print("=" * 70)
    
    # NACC missing codes
    missing_codes = [88, 95, 96, 97, 98, 99, 888, 999, -4, -1]
    
    cognitive_vars = ['NACCMMSE', 'MOCATOTS', 'NACCMOCA', 'CDRSUM', 'CDRGLOB',
                      'AGIT', 'ANX', 'MOT', 'DEL', 'HALL', 'DEPD']
    
    print(f"\n{'Variable':<15} {'In Data':<10} {'Non-Missing':<15} {'% Valid':<10} {'Sample Values'}")
    print("-" * 80)
    
    for var in cognitive_vars:
        if var in uds_df.columns:
            # Count values not in missing codes
            col = pd.to_numeric(uds_df[var], errors='coerce')
            valid = col[~col.isin(missing_codes) & col.notna()]
            pct = 100 * len(valid) / len(uds_df)
            sample = valid.head(5).tolist() if len(valid) > 0 else []
            print(f"{var:<15} {'Yes':<10} {len(valid):<15} {pct:<10.1f} {sample}")
        else:
            print(f"{var:<15} {'NO':<10} {'-':<15} {'-':<10}")
    
    # =========================================================================
    # PART 2: CHECK MRI PATIENTS IN UDS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: Do MRI Patients Have Cognitive Data?")
    print("=" * 70)
    
    mri_patients = set(mri_df['NACCID'].unique())
    uds_patients = set(uds_df['NACCID'].unique())
    
    overlap = mri_patients & uds_patients
    
    print(f"\n  Unique patients in MRI data: {len(mri_patients)}")
    print(f"  Unique patients in UDS data: {len(uds_patients)}")
    print(f"  Patients in BOTH datasets: {len(overlap)}")
    
    # For overlapping patients, check cognitive data
    uds_overlap = uds_df[uds_df['NACCID'].isin(overlap)]
    
    print(f"\n  For the {len(overlap)} patients with MRI scans:")
    for var in ['NACCMMSE', 'MOCATOTS', 'CDRSUM', 'AGIT']:
        if var in uds_overlap.columns:
            col = pd.to_numeric(uds_overlap[var], errors='coerce')
            valid = col[~col.isin(missing_codes) & col.notna()]
            print(f"    {var}: {len(valid)} valid values ({100*len(valid)/len(uds_overlap):.1f}%)")
    
    # =========================================================================
    # PART 3: TEMPORAL ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: Temporal Alignment Between UDS and MRI")
    print("=" * 70)
    
    # Parse dates
    if 'VISITDAY' in uds_df.columns:
        uds_df['VISITDAY'] = pd.to_numeric(uds_df['VISITDAY'], errors='coerce')
        uds_df['VISITDAY'] = uds_df['VISITDAY'].apply(lambda x: 15 if pd.isna(x) or x > 31 or x < 1 else x)
    else:
        uds_df['VISITDAY'] = 15
    
    uds_df['UDS_DATE'] = pd.to_datetime(
        uds_df['VISITYR'].astype(str) + '-' + 
        uds_df['VISITMO'].astype(str) + '-' + 
        uds_df['VISITDAY'].astype(int).astype(str),
        errors='coerce'
    )
    
    mri_df['MRI_DATE'] = pd.to_datetime(mri_df['STUDYDATE'], errors='coerce')
    
    uds_years = uds_df['VISITYR'].value_counts().sort_index()
    mri_years = mri_df['MRI_DATE'].dt.year.value_counts().sort_index()
    
    print(f"\n  UDS Visit Years (sample):")
    for yr in sorted(uds_years.index)[-5:]:
        print(f"    {yr}: {uds_years[yr]} visits")
    
    print(f"\n  MRI Scan Years (sample):")
    for yr in sorted(mri_years.index)[-5:]:
        print(f"    {int(yr)}: {mri_years[yr]} scans")
    
    # =========================================================================
    # PART 4: CHECK MMSE vs MoCA BY YEAR
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: MMSE vs MoCA by Year (UDS2 → UDS3 Transition)")
    print("=" * 70)
    
    print("\n  MMSE was standard in UDS2 (pre-2015)")
    print("  MoCA became standard in UDS3 (post-2015)")
    
    if 'NACCMMSE' in uds_df.columns and 'MOCATOTS' in uds_df.columns:
        for year in [2012, 2014, 2016, 2018, 2020, 2022]:
            year_data = uds_df[uds_df['VISITYR'] == year]
            if len(year_data) > 0:
                mmse_col = pd.to_numeric(year_data['NACCMMSE'], errors='coerce')
                moca_col = pd.to_numeric(year_data['MOCATOTS'], errors='coerce')
                
                mmse_valid = mmse_col[~mmse_col.isin(missing_codes) & mmse_col.notna()]
                moca_valid = moca_col[~moca_col.isin(missing_codes) & moca_col.notna()]
                
                print(f"    {year}: MMSE={len(mmse_valid)}, MoCA={len(moca_valid)} (n={len(year_data)})")
    
    # =========================================================================
    # PART 5: ALTERNATIVE MERGE STRATEGY
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 5: Try Different Merge Strategies")
    print("=" * 70)
    
    # Strategy A: Merge on NACCID only (ignore date proximity)
    merged_any = pd.merge(uds_df, mri_df[['NACCID', 'STUDYDATE', 'STUDYQC']], 
                          on='NACCID', how='inner')
    print(f"\n  Strategy A (merge on NACCID only): {len(merged_any)} records")
    
    if 'NACCMMSE' in merged_any.columns:
        col = pd.to_numeric(merged_any['NACCMMSE'], errors='coerce')
        valid = col[~col.isin(missing_codes) & col.notna()]
        print(f"    NACCMMSE valid: {len(valid)} ({100*len(valid)/len(merged_any):.1f}%)")
    
    if 'MOCATOTS' in merged_any.columns:
        col = pd.to_numeric(merged_any['MOCATOTS'], errors='coerce')
        valid = col[~col.isin(missing_codes) & col.notna()]
        print(f"    MOCATOTS valid: {len(valid)} ({100*len(valid)/len(merged_any):.1f}%)")
    
    if 'CDRSUM' in merged_any.columns:
        col = pd.to_numeric(merged_any['CDRSUM'], errors='coerce')
        valid = col[~col.isin(missing_codes) & col.notna()]
        print(f"    CDRSUM valid: {len(valid)} ({100*len(valid)/len(merged_any):.1f}%)")
    
    # Strategy B: Keep CLOSEST UDS visit to each MRI (current approach)
    merged_any['MRI_DATE'] = pd.to_datetime(merged_any['STUDYDATE'], errors='coerce')
    merged_any['DAYS_DIFF'] = (merged_any['MRI_DATE'] - merged_any['UDS_DATE']).dt.days.abs()
    
    # Within 90 days
    merged_90 = merged_any[merged_any['DAYS_DIFF'] <= 90]
    merged_90 = merged_90.sort_values('DAYS_DIFF').drop_duplicates(subset=['NACCID', 'STUDYDATE'])
    
    print(f"\n  Strategy B (closest UDS within 90 days): {len(merged_90)} records")
    
    if 'NACCMMSE' in merged_90.columns:
        col = pd.to_numeric(merged_90['NACCMMSE'], errors='coerce')
        valid = col[~col.isin(missing_codes) & col.notna()]
        print(f"    NACCMMSE valid: {len(valid)} ({100*len(valid)/len(merged_90):.1f}%)")
    
    if 'MOCATOTS' in merged_90.columns:
        col = pd.to_numeric(merged_90['MOCATOTS'], errors='coerce')
        valid = col[~col.isin(missing_codes) & col.notna()]
        print(f"    MOCATOTS valid: {len(valid)} ({100*len(valid)/len(merged_90):.1f}%)")
    
    # Strategy C: Wider window (365 days)
    merged_365 = merged_any[merged_any['DAYS_DIFF'] <= 365]
    merged_365 = merged_365.sort_values('DAYS_DIFF').drop_duplicates(subset=['NACCID', 'STUDYDATE'])
    
    print(f"\n  Strategy C (closest UDS within 365 days): {len(merged_365)} records")
    
    if 'NACCMMSE' in merged_365.columns:
        col = pd.to_numeric(merged_365['NACCMMSE'], errors='coerce')
        valid = col[~col.isin(missing_codes) & col.notna()]
        print(f"    NACCMMSE valid: {len(valid)} ({100*len(valid)/len(merged_365):.1f}%)")
    
    if 'MOCATOTS' in merged_365.columns:
        col = pd.to_numeric(merged_365['MOCATOTS'], errors='coerce')
        valid = col[~col.isin(missing_codes) & col.notna()]
        print(f"    MOCATOTS valid: {len(valid)} ({100*len(valid)/len(merged_365):.1f}%)")
    
    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    print("""
Based on this diagnostic, consider:

1. USE MoCA INSTEAD OF MMSE
   - If MRI scans are from 2015+, MoCA is the standard cognitive test
   - Check MOCATOTS availability above

2. USE CDR (CDRSUM, CDRGLOB)
   - Often more complete than MMSE/MoCA
   - Directly measures dementia severity

3. USE BEHAVIORAL SYMPTOMS (NPI-Q)
   - AGIT, ANX, MOT are direct predictors of scan compliance
   - May have better coverage

4. EXPAND TIME WINDOW
   - Current: 90 days between UDS and MRI
   - Try 180 or 365 days — cognitive status is relatively stable

5. USE MOST RECENT UDS BEFORE MRI
   - Instead of closest, use most recent PRIOR visit
   - More clinically meaningful
""")
    
    return merged_90

if __name__ == "__main__":
    result = main()
