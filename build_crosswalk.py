"""
Build UDS3 -> UDS4 crosswalk for the ~20 Phase-1 features used by
ejor_clinical_scheduling_v3_fixed.py, checked against:
  (1) data-nacc74/uds4-rdd.csv  (UDS4 researcher data dictionary)
  (2) investigator_nacc74.parquet actual columns (ground truth)
Output: crosswalk_uds3_uds4.csv
"""
import csv, re
import pyarrow.parquet as pq

# --- UDS3 variables used by the v3_fixed pipeline ---
# (role, uds3_name, used_in_model_tiers_or_purpose)
VARS = [
    ("id/date",  "NACCID",   "merge key"),
    ("id/date",  "VISITYR",  "visit date build"),
    ("id/date",  "VISITMO",  "visit date build"),
    ("id/date",  "VISITDAY", "visit date build"),
    ("demog",    "NACCAGE",  "Tier 5 model feature"),
    ("moca",     "MOCATOTS", "Tier 3 model feature"),
    ("moca",     "NACCMOCA", "Tier 3 model feature"),
    ("moca",     "MOCAREAS", "QC/missingness reason"),
    ("cdr",      "CDRSUM",   "Tier 4 model feature"),
    ("cdr",      "CDRGLOB",  "Tier 4 model feature"),
    ("cdr",      "MEMORY",   "loaded, not in model"),
    ("cdr",      "ORIENT",   "loaded, not in model"),
    ("cdr",      "JUDGMENT", "loaded, not in model"),
    ("npiq",     "NPIQINF",  "informant flag"),
    ("npiq",     "AGIT",     "Tier 1 model feature"),
    ("npiq",     "AGITSEV",  "Tier 2 model feature"),
    ("npiq",     "ANX",      "Tier 1 model feature"),
    ("npiq",     "ANXSEV",   "Tier 2 model feature"),
    ("npiq",     "MOT",      "Tier 1 model feature"),
    ("npiq",     "MOTSEV",   "Tier 2 model feature"),
    ("npiq",     "IRR",      "Tier 6 model feature"),
    ("npiq",     "DEL",      "Tier 6 model feature"),
    ("npiq",     "HALL",     "Tier 6 model feature"),
    ("npiq",     "DEPD",     "Tier 6 model feature"),
    ("npiq",     "DEPDSEV",  "SEVERITY_WEIGHTED_RISK input"),
    ("npiq",     "ELAT",     "loaded, not in model"),
    ("npiq",     "APA",      "BEHAVIORAL_RISK input"),
    ("npiq",     "DISN",     "BEHAVIORAL_RISK input"),
    ("npiq",     "NITE",     "loaded, not in model"),
    ("npiq",     "APP",      "loaded, not in model"),
    ("clin",     "DECCLCOG", "Tier 5 model feature"),
    ("clin",     "COGMEM",   "loaded, not in model"),
    ("clin",     "BEAPATHY", "loaded, not in model"),
    ("clin",     "BEDEP",    "loaded, not in model"),
    ("clin",     "BEAGIT",   "Tier 5 model feature; CLINICIAN_CONCERN input"),
    ("clin",     "BEANX",    "CLINICIAN_CONCERN input"),
    ("mmse",     "NACCMMSE", "backup, low UDS3 coverage"),
    ("mri",      "STUDYDATE","MRI merge (NACC69 mriqc table)"),
    ("mri",      "STUDYQC",  "target: MRI_FAILED (NACC69 mriqc table)"),
]

# load UDS4 RDD
rdd = {}
with open("data-nacc74/uds4-rdd.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        rdd[row["variable_name"].strip().upper()] = row

# actual NACC74 investigator parquet columns
cols74 = set(c.upper() for c in pq.ParquetFile("data-nacc74/investigator_nacc74.parquet").schema_arrow.names)
# MRI parquet columns
colsmri = set(c.upper() for c in pq.ParquetFile("data-nacc74/investigator_mri_nacc74.parquet").schema_arrow.names)

def find_candidates(name):
    """Search RDD + parquet for plausible renames."""
    pat = re.compile(name[:4], re.I)
    hits = sorted(set([v for v in rdd if name in v or (len(name)>=4 and v.startswith(name[:4]))] +
                      [c for c in cols74 if name in c or (len(name)>=4 and c.startswith(name[:4]))]))
    return [h for h in hits if h != name][:6]

rows = []
for grp, v, role in VARS:
    in_rdd = v in rdd
    in_pq  = v in cols74 or (grp == "mri" and v in colsmri)
    desc   = rdd[v]["short_descriptor"] if in_rdd else ""
    form   = rdd[v]["form"] if in_rdd else ""
    if in_pq and in_rdd:
        status = "same"
    elif in_pq and not in_rdd:
        status = "same-in-data (not in UDS4 RDD)"
    else:
        status = "CHECK"
    cand = "" if in_pq else ";".join(find_candidates(v))
    rows.append(dict(group=grp, uds3_variable=v, role=role,
                     in_uds4_rdd=in_rdd, in_nacc74_parquet=in_pq,
                     uds4_form=form, uds4_descriptor=desc,
                     status=status, candidates=cand, uds4_variable="", notes=""))
    print(f"{grp:8s} {v:10s} rdd={str(in_rdd):5s} pq={str(in_pq):5s} {status:12s} {cand}")

with open("crosswalk_uds3_uds4_draft.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("\nwrote crosswalk_uds3_uds4_draft.csv")
print("\nNACC74 investigator parquet: %d cols; MRI parquet: %d cols" % (len(cols74), len(colsmri)))
