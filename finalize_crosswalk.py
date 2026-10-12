import csv

NOTES = {
 "NACCID":   ("NACCID", "same", "Merge key unchanged."),
 "VISITYR":  ("VISITDATE", "restructured", "UDS3 VISITYR/VISITMO/VISITDAY replaced by single VISITDATE string 'YYYY-MM-DD' in NACC74; derive components or parse directly with pd.to_datetime."),
 "VISITMO":  ("VISITDATE", "restructured", "See VISITYR."),
 "VISITDAY": ("VISITDATE", "restructured", "See VISITYR. Note v3_fixed imputes VISITDAY=15 when missing; unnecessary with VISITDATE."),
 "NACCAGE":  ("NACCAGE", "same", "NACC-derived; computed identically across eras."),
 "MOCATOTS": ("MOCATOTS", "same-harmonize", "Name/range unchanged (0-30). HARMONIZATION: UDS4 C2 changed MoCA administration (e.g., telephone/remote variants, item substitutions); treat UDS3 vs UDS4 MoCA as same scale but flag era in sensitivity analysis."),
 "NACCMOCA": ("NACCMOCA", "same-harmonize", "Education-corrected MoCA; same derivation, same caveat as MOCATOTS."),
 "MOCAREAS": ("MOCAREAS", "same", "Reason MoCA not completed."),
 "CDRSUM":   ("CDRSUM", "same", "CDR unchanged UDS3->UDS4 (Form B4 stable)."),
 "CDRGLOB":  ("CDRGLOB", "same", "CDR unchanged."),
 "MEMORY":   ("MEMORY", "same", "CDR domain; loaded but not a model feature."),
 "ORIENT":   ("ORIENT", "same", "CDR domain."),
 "JUDGMENT": ("JUDGMENT", "same", "CDR domain."),
 "NPIQINF":  ("NPIQINF", "same", "NPI-Q informant flag."),
 "AGIT":     ("AGIT", "same", "NPI-Q item names unchanged in UDS4 RDD and present in NACC74 data."),
 "AGITSEV":  ("AGITSEV", "same", "Severity 1-3 unchanged."),
 "ANX":      ("ANX", "same", ""), "ANXSEV": ("ANXSEV", "same", ""),
 "MOT":      ("MOT", "same", ""), "MOTSEV": ("MOTSEV", "same", ""),
 "IRR":      ("IRR", "same", ""), "DEL": ("DEL", "same", ""),
 "HALL":     ("HALL", "same", ""), "DEPD": ("DEPD", "same", ""),
 "DEPDSEV":  ("DEPDSEV", "same", ""), "ELAT": ("ELAT", "same", ""),
 "APA":      ("APA", "same", ""), "DISN": ("DISN", "same", ""),
 "NITE":     ("NITE", "same", ""), "APP": ("APP", "same", ""),
 "DECCLCOG": ("DECCLCOG", "same", "Clinician judgment of meaningful cognitive decline."),
 "COGMEM":   ("COGMEM", "same", ""),
 "BEAPATHY": ("BEAPATHY", "same", ""), "BEDEP": ("BEDEP", "same", ""),
 "BEAGIT":   ("BEAGIT", "same", ""), "BEANX": ("BEANX", "same", ""),
 "NACCMMSE": ("NACCMMSE", "same", "MMSE retired after UDS2; expect near-zero coverage in UDS3/UDS4-era visits (backup only)."),
 "STUDYDATE":("STUDYDATE", "same-new-file", "Identical schema; now in data-nacc74/scan_clariti_mri/investigator_scan_clariti_mriqc_nacc74.parquet (was investigator_scan_mriqc_nacc69.csv). NACC74 file adds FUNDINGSOURCE, QCUPDATESTAMP, PROJECT columns."),
 "STUDYQC":  ("STUDYQC", "same-new-file", "Target variable for MRI_FAILED; same file move as STUDYDATE."),
}

rows = list(csv.DictReader(open("crosswalk_uds3_uds4_draft.csv")))
for r in rows:
    v = r["uds3_variable"]
    uds4, status, note = NOTES[v]
    r["uds4_variable"] = uds4
    r["status"] = status
    r["notes"] = note

# extra rows: era variable + QC continuous candidates for A2
rows.append(dict(group="era", uds3_variable="(FORMVER)", role="era split for A3/E8",
    in_uds4_rdd=True, in_nacc74_parquet=True, uds4_form="header",
    uds4_descriptor="UDS form version", status="new-use",
    candidates="", uds4_variable="FORMVER",
    notes="Era definition: UDS3 = FORMVER in {3.0, 3.2} (106,876 visits); UDS4 = 4.0 (6,597 visits); legacy 1.0/2.0 = 104,125. PACKET/NACCVNUM also available."))
rows.append(dict(group="mri", uds3_variable="(SERIESQC)", role="A2 continuous QC proxy",
    in_uds4_rdd=False, in_nacc74_parquet=True, uds4_form="SCAN/Clariti",
    uds4_descriptor="Series-level QC", status="new-use",
    candidates="", uds4_variable="SERIESQC/SERIESQCREASONS/STUDYRESCANREQUESTED",
    notes="Graded QC signal for A2: SERIESQC + SERIESQCREASONS + STUDYRESCANREQUESTED in scan_clariti_mriqc parquet."))

with open("crosswalk_uds3_uds4.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("wrote crosswalk_uds3_uds4.csv with", len(rows), "rows")
from collections import Counter
print(Counter(r["status"] for r in rows))
