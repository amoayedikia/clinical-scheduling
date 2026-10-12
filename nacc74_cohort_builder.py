"""
NACC74 REAL-COHORT BUILDER (Track A1 — Week 1 Day 5 skeleton)

Samples real patient feature vectors (age, MoCA, CDR + NPI-Q context) from the
NACC74 freeze and emits cohorts in the exact format experiments.py/baselines.py
expect (List[PatientData]).

Failure-outcome mapping (per plan: "keep the published-rate calibration,
unchanged"): the SAME functional form as experiments.generate_synthetic_cohort,
but driven by REAL covariates instead of synthetic draws:
  high-risk  (CDRGLOB >= 1): p = clip(0.25 + U(0,0.25)*u + (age-75)*0.003
                                       + (26-moca)*0.008, 0.25, 0.55)
  low-risk   (CDRGLOB < 1):  p = clip(0.05 + U(0,0.15)*u + max(0,age-70)*0.002,
                                       0.03, 0.24)
where u is a deterministic per-patient uniform (seeded), so the only thing that
changes vs the submitted experiments is WHERE the covariates come from.
Optional --recalibrate shifts the pool mean to 0.212 (21.2% published base).

Usage:
  python3 nacc74_cohort_builder.py                 # build pool, emit one n=10 cohort, run one E1 iteration
  python3 nacc74_cohort_builder.py --era UDS4 --n 10 --seed 7 --no-e1
"""
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from baselines import PatientData, DEFAULT_CONFIG

UDS_PARQUET = 'data-nacc74/investigator_nacc74.parquet'
MISSING = [88, 95, 96, 97, 98, 99, 888, 999, -4, -1]
FEATURES = ['NACCAGE', 'MOCATOTS', 'NACCMOCA', 'CDRSUM', 'CDRGLOB',
            'AGIT', 'AGITSEV', 'ANX', 'ANXSEV', 'MOT', 'MOTSEV',
            'IRR', 'DEL', 'HALL', 'DEPD', 'DEPDSEV', 'APA', 'DISN',
            'DECCLCOG', 'BEAGIT', 'BEANX']
ESSENTIAL = ['NACCAGE', 'MOCATOTS', 'CDRGLOB']   # must be non-missing to enter pool


def load_pool(era='both'):
    """Load NACC74 visits, apply crosswalk changes, return one row per participant.

    Crosswalk (crosswalk_uds3_uds4.csv): all feature names unchanged; the only
    material change is VISITYR/MO/DAY -> VISITDATE string, parsed here.
    Era via FORMVER: UDS3 = {3.0, 3.2}, UDS4 = 4.0.
    """
    cols = ['NACCID', 'NACCADC', 'VISITDATE', 'FORMVER'] + FEATURES
    df = pq.ParquetFile(UDS_PARQUET).read(columns=cols).to_pandas()
    df['UDS_DATE'] = pd.to_datetime(df['VISITDATE'], errors='coerce')
    df['ERA'] = np.select([df.FORMVER.isin([3.0, 3.2]), df.FORMVER == 4.0],
                          ['UDS3', 'UDS4'], default='UDS1-2')
    df = df[df.ERA.isin(['UDS3', 'UDS4'])]
    if era != 'both':
        df = df[df.ERA == era]
    df[FEATURES] = df[FEATURES].replace(MISSING, np.nan)
    df = df.dropna(subset=ESSENTIAL + ['UDS_DATE'])
    # one row per participant: latest eligible visit (real covariate vector, no dupes)
    df = df.sort_values('UDS_DATE').drop_duplicates('NACCID', keep='last')
    return df.reset_index(drop=True)


def assign_p_fail(pool, seed=42, recalibrate=False):
    """Map real covariates -> p_fail with the submitted synthetic formula."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, len(pool))          # deterministic per-pool noise
    age, moca, cdr = pool.NACCAGE.values, pool.MOCATOTS.values, pool.CDRGLOB.values
    hi = cdr >= 1
    p = np.where(
        hi,
        np.clip(0.25 + 0.25 * u + (age - 75) * 0.003 + (26 - moca) * 0.008, 0.25, 0.55),
        np.clip(0.05 + 0.15 * u + np.maximum(0, age - 70) * 0.002, 0.03, 0.24),
    )
    if recalibrate:                            # shift pool mean to published 21.2% base
        p = np.clip(p + (0.212 - p.mean()), 0.03, 0.55)
    pool = pool.copy()
    pool['P_FAIL'] = p
    return pool


def sample_cohort(pool, n=10, seed=42):
    """Sample n distinct real participants -> List[PatientData]."""
    rows = pool.sample(n=n, random_state=seed)
    return [PatientData(f"P{i+1:03d}", float(r.P_FAIL), float(r.NACCAGE),
                        float(r.MOCATOTS), float(r.CDRGLOB))
            for i, (_, r) in enumerate(rows.iterrows())], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--era', choices=['both', 'UDS3', 'UDS4'], default='both')
    ap.add_argument('--recalibrate', action='store_true')
    ap.add_argument('--out', default='results/nacc74_cohort_sample.csv')
    ap.add_argument('--no-e1', action='store_true')
    a = ap.parse_args()

    pool = load_pool(a.era)
    pool = assign_p_fail(pool, seed=a.seed, recalibrate=a.recalibrate)
    print(f"Pool ({a.era}): {len(pool)} participants | p_fail mean {pool.P_FAIL.mean():.3f} "
          f"| CDR>=1 {100*(pool.CDRGLOB>=1).mean():.1f}% | era mix: {dict(pool.ERA.value_counts())}")

    patients, rows = sample_cohort(pool, a.n, a.seed)
    rows.assign(PATIENT_ID=[p.patient_id for p in patients]).to_csv(a.out, index=False)
    print(f"\nCohort n={a.n} (seed {a.seed}) -> {a.out}")
    for p in patients:
        print(f"  {p.patient_id}: p_fail={p.p_fail_mri:.3f} age={p.age:.0f} moca={p.moca:.0f} cdr={p.cdr}")

    if not a.no_e1:
        from baselines import (fcfs_schedule, spt_schedule, lpt_schedule,
                               deterministic_schedule, age_stratified_schedule,
                               random_schedule, stochastic_milp_schedule,
                               evaluate_schedule_with_simulation)
        methods = {
            'FCFS': fcfs_schedule, 'SPT': spt_schedule, 'LPT': lpt_schedule,
            'Deterministic': deterministic_schedule,
            'Age-Stratified': age_stratified_schedule,
            'Random': lambda p, c: random_schedule(p, c, n_samples=10, seed=a.seed),
            'Stochastic-MILP': lambda p, c: stochastic_milp_schedule(p, c, verbose=False),
        }
        print("\nONE E1 ITERATION ON REAL COHORT (n_simulations=100):")
        print(f"  {'method':<16}{'planned':>9}{'E[mksp]':>9}{'95pct':>9}{'OT_prob':>9}{'solve_s':>9}")
        for name, fn in methods.items():
            schedule_df, solve_time = fn(patients, DEFAULT_CONFIG)
            if schedule_df is None:
                print(f"  {name:<16}  (no solution)"); continue
            sim = evaluate_schedule_with_simulation(schedule_df, patients, DEFAULT_CONFIG,
                                                    n_simulations=100, seed=a.seed)
            print(f"  {name:<16}{sim['planned_makespan']:>9.1f}{sim['expected_makespan']:>9.1f}"
                  f"{sim['makespan_95pct']:>9.1f}{sim['overtime_probability']:>9.3f}{solve_time:>9.2f}")


if __name__ == '__main__':
    main()
