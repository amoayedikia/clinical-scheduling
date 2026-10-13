"""Week 1 Days 3-4 go/no-go gates (see HANDOVER.md section 6).
Gate 1 (A2): QC coverage - participants with SCAN/Clariti QC AND contemporaneous
             cognitive data; QC variance.
Gate 2 (A1): feature missingness in NACC74 by era (UDS3 vs UDS4 visits).
"""
import pandas as pd, numpy as np
import pyarrow.parquet as pq

MISSING = [88, 95, 96, 97, 98, 99, 888, 999, -4, -1]
FEATURES = ['AGIT','MOT','ANX','AGITSEV','MOTSEV','ANXSEV','MOCATOTS','NACCMOCA',
            'CDRSUM','CDRGLOB','NACCAGE','DECCLCOG','BEAGIT','BEANX',
            'IRR','DEL','HALL','DEPD','DEPDSEV','APA','DISN','NACCMMSE']

uds = pq.ParquetFile('data-nacc74/investigator_nacc74.parquet').read(
    columns=['NACCID','VISITDATE','FORMVER','NACCADC']+FEATURES).to_pandas()
uds['UDS_DATE'] = pd.to_datetime(uds['VISITDATE'], errors='coerce')
uds['ERA'] = np.select([uds.FORMVER.isin([3.0,3.2]), uds.FORMVER==4.0],
                       ['UDS3','UDS4'], default='UDS1-2')
print(f"UDS visits: {len(uds)}, participants: {uds.NACCID.nunique()}")
print(uds.ERA.value_counts().to_string())

# ---------- Gate 2 (A1): missingness by era ----------
print("\n" + "="*78)
print("GATE A1: feature non-missing %% by era (NACC missing codes -> NaN)")
print("="*78)
u = uds.copy()
u[FEATURES] = u[FEATURES].replace(MISSING, np.nan)
rep = {}
for era in ['UDS3','UDS4']:
    sub = u[u.ERA==era]
    rep[era] = sub[FEATURES].notna().mean()*100
rep = pd.DataFrame(rep).round(1)
rep['delta'] = (rep['UDS4']-rep['UDS3']).round(1)
print(rep.to_string())
rep.to_csv('results/week1_a1_missingness_by_era.csv')

# ---------- Gate 1 (A2): QC coverage ----------
print("\n" + "="*78)
print("GATE A2: SCAN/Clariti QC coverage + variance")
print("="*78)
qc = pq.ParquetFile('data-nacc74/scan_clariti_mri/investigator_scan_clariti_mriqc_nacc74.parquet').read(
    columns=['NACCID','STUDYDATE','STUDYQC','SERIESQC','SERIESQCREASONS','STUDYRESCANREQUESTED','SERIESTYPE']).to_pandas()
print(f"QC rows (series-level): {len(qc)}, participants: {qc.NACCID.nunique()}")
print("\nSTUDYQC distribution:"); print(qc.drop_duplicates(['NACCID','STUDYDATE']).STUDYQC.value_counts(dropna=False).to_string())
print("\nSERIESQC distribution:"); print(qc.SERIESQC.value_counts(dropna=False).head(10).to_string())
print("\nSTUDYRESCANREQUESTED:"); print(qc.drop_duplicates(['NACCID','STUDYDATE']).STUDYRESCANREQUESTED.value_counts(dropna=False).to_string())

# study-level table
st = qc.drop_duplicates(['NACCID','STUDYDATE']).copy()
st['MRI_DATE'] = pd.to_datetime(st['STUDYDATE'], errors='coerce')
st = st.dropna(subset=['MRI_DATE'])
print(f"\nUnique studies: {len(st)}, participants with >=1 study: {st.NACCID.nunique()}")

# contemporaneous (<=90d) cognitive visit, per pipeline rule
m = st.merge(uds[['NACCID','UDS_DATE','ERA','MOCATOTS','CDRGLOB','AGIT']].dropna(subset=['UDS_DATE']), on='NACCID', how='inner')
m['DD'] = (m.MRI_DATE - m.UDS_DATE).dt.days.abs()
m90 = m[m.DD<=90].sort_values('DD').drop_duplicates(['NACCID','STUDYDATE'])
print(f"Studies with UDS visit within 90d: {len(m90)} ({m90.NACCID.nunique()} participants)")
print("  by era of matched visit:"); print(m90.ERA.value_counts().to_string())
# with usable cognition on matched visit (raw codes still in uds here; clean)
mm = m90.replace(MISSING, np.nan)
for v in ['MOCATOTS','CDRGLOB','AGIT']:
    print(f"  matched studies with non-missing {v}: {mm[v].notna().sum()} ({100*mm[v].notna().mean():.1f}%)")
# QC variance within matched set
print("\nSTUDYQC in matched set:"); print(m90.STUDYQC.value_counts(dropna=False).to_string())
m90.to_csv('results/week1_a2_qc_matched.csv', index=False)
print("\nwrote results/week1_a1_missingness_by_era.csv and results/week1_a2_qc_matched.csv")
