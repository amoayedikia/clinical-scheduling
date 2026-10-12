"""
REAL-ARM DRIVER (Track A1, Weeks 2-3)

Resumable, chunked runner for the E1-E7 real-data arm on NACC74 cohorts.
Claude's sandboxed shell on the Mac kills all processes when each tool call
ends (~3 min), so the full arm cannot run as a single process. This driver
executes the SAME per-replication logic as experiments.py (E1-E6) and
e7_scaling_sensitivity.py (E7), one work unit at a time, appending finished
units to results/real_arm/<exp>_rows.csv and resuming where it left off.

Usage:  python3 real_arm_driver.py --exp E1 [--budget 145] [--era both]
                                   [--p-fail recalibrated]

Equivalence with the submitted protocol:
- Cohorts come from experiments.get_cohort in real mode: one NACC74
  participant pool (recalibrated p_fail, mean 0.212), cohort seed = 42 + rep
  (distinct per replication), natural case-mix except E6 (CDR-stratified).
- Replication counts, n_simulations=100, seeds, and method set are identical
  to the submitted runs.
- E2 deviations (documented in the comparison report): solver timeout capped
  at 95 s for n>=20 (call-length limit; submitted used max(60,6n) and all
  n>=20 instances timed out regardless), and LPT rows + Monte Carlo
  evaluation added for both methods to mirror the manuscript's Table
  'scalability'.
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

import experiments as X
import e7_scaling_sensitivity as E7MOD
from baselines import (
    DEFAULT_CONFIG, fcfs_schedule, spt_schedule, lpt_schedule,
    deterministic_schedule, age_stratified_schedule, random_schedule,
    stochastic_milp_schedule, calculate_metrics,
    evaluate_schedule_with_simulation)

SEED = 42
OUTDIR = 'results/real_arm'

E1_METHODS = {
    'FCFS': fcfs_schedule,
    'SPT': spt_schedule,
    'LPT': lpt_schedule,
    'Deterministic': deterministic_schedule,
    'Age-Stratified': age_stratified_schedule,
    'Random': lambda p, c: random_schedule(p, c, n_samples=10, seed=SEED),
    'Stochastic-MILP': lambda p, c: stochastic_milp_schedule(p, c, verbose=False),
}


def setup_pool(era, recalibrate):
    tagp = 'recal' if recalibrate else 'raw'
    pkl = os.path.expanduser(f'~/nacc74_pool_{era}_{tagp}.pkl')
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            pool = pickle.load(f)
    else:
        from nacc74_cohort_builder import load_pool, assign_p_fail
        pool = assign_p_fail(load_pool(era), seed=SEED, recalibrate=recalibrate)
        with open(pkl, 'wb') as f:
            pickle.dump(pool, f)
    X.COHORT_SOURCE.update(mode='real', pool=pool, era=era, recalibrate=recalibrate)
    return pool


# ---------------------------------------------------------------- unit fns
def unit_e1(rep):
    patients = X.get_cohort(10, seed=SEED + rep)
    rows = []
    for name, fn in E1_METHODS.items():
        sdf, st = fn(patients, DEFAULT_CONFIG)
        if sdf is None:
            continue
        sim = evaluate_schedule_with_simulation(sdf, patients, DEFAULT_CONFIG, 100, SEED + rep)
        pm = calculate_metrics(sdf, DEFAULT_CONFIG, st)
        rows.append({'Replication': rep + 1, 'Method': name,
                     'Planned_Makespan': pm.makespan,
                     'Expected_Makespan': sim['expected_makespan'],
                     'Makespan_Std': sim['makespan_std'],
                     'Makespan_95pct': sim['makespan_95pct'],
                     'Overtime_Prob': sim['overtime_probability'],
                     'Solve_Time': st})
    return rows


def unit_e2(n, rep):
    nominal = max(60, n * 6)
    cap = min(nominal, 95)   # call-length limit is 120 s
    cfg = DEFAULT_CONFIG.copy()
    cfg['solver_timeout'] = cap
    patients = X.get_cohort(n, seed=SEED + rep)
    n_vars = 3 * n + n * (n - 1) // 2 * 3 + 1 + n
    n_cons = n * 2 + n * (n - 1) * 3 + n * 3
    rows = []
    sdf, st = stochastic_milp_schedule(patients, cfg, verbose=False)
    if sdf is not None:
        m = calculate_metrics(sdf, cfg, st)
        sim = evaluate_schedule_with_simulation(sdf, patients, cfg, 100, SEED + rep)
        rows.append({'N_Patients': n, 'Replication': rep + 1, 'Method': 'Stochastic-MILP',
                     'Solve_Time': st, 'Makespan': m.makespan,
                     'Expected_Makespan': sim['expected_makespan'],
                     'Overtime_Prob': sim['overtime_probability'],
                     'N_Variables': n_vars, 'N_Constraints': n_cons,
                     'Status': 'Optimal' if st < cap - 1 else 'Timeout',
                     'Timeout_Used': cap, 'Nominal_Timeout': nominal})
    ldf, lt = lpt_schedule(patients, DEFAULT_CONFIG)
    if ldf is not None:
        lm = calculate_metrics(ldf, DEFAULT_CONFIG, lt)
        lsim = evaluate_schedule_with_simulation(ldf, patients, DEFAULT_CONFIG, 100, SEED + rep)
        rows.append({'N_Patients': n, 'Replication': rep + 1, 'Method': 'LPT',
                     'Solve_Time': lt, 'Makespan': lm.makespan,
                     'Expected_Makespan': lsim['expected_makespan'],
                     'Overtime_Prob': lsim['overtime_probability'],
                     'N_Variables': np.nan, 'N_Constraints': np.nan,
                     'Status': 'Heuristic', 'Timeout_Used': np.nan, 'Nominal_Timeout': np.nan})
    return rows


def unit_e3(lam, rep):
    patients = X.get_cohort(10, seed=SEED + rep)
    cfg = DEFAULT_CONFIG.copy()
    cfg['wait_time_penalty'] = lam
    sdf, st = stochastic_milp_schedule(patients, cfg, verbose=False)
    if sdf is None:
        return []
    m = calculate_metrics(sdf, cfg, st)
    return [{'Lambda': lam, 'Replication': rep + 1, 'Makespan': m.makespan,
             'Total_Wait': m.total_wait, 'Avg_Wait': m.avg_wait,
             'Objective': m.objective_value}]


def unit_e4(rep, predictor):
    patients = X.get_cohort(10, seed=SEED + rep)
    original = {p.patient_id: p.p_fail_mri for p in patients}
    if predictor == 'Full_ML':
        sdf, _ = stochastic_milp_schedule(patients, DEFAULT_CONFIG, verbose=False)
    elif predictor == 'Age_Only':
        for p in patients:
            p.p_fail_mri = np.clip(0.05 + max(0, (p.age - 65)) * 0.005, 0.03, 0.40)
        sdf, _ = stochastic_milp_schedule(patients, DEFAULT_CONFIG, verbose=False)
        for p in patients:
            p.p_fail_mri = original[p.patient_id]
    elif predictor == 'Age_Stratified':
        sdf, _ = age_stratified_schedule(patients, DEFAULT_CONFIG)
    else:  # No_Prediction
        sdf, _ = deterministic_schedule(patients, DEFAULT_CONFIG)
    if sdf is None:
        return []
    sim = evaluate_schedule_with_simulation(sdf, patients, DEFAULT_CONFIG, 100, SEED + rep)
    return [{'Replication': rep + 1, 'Predictor': predictor,
             'Expected_Makespan': sim['expected_makespan'],
             'Overtime_Prob': sim['overtime_probability']}]


def unit_e5(err, rep):
    patients = X.get_cohort(10, seed=SEED + rep)
    true_p = {p.patient_id: p.p_fail_mri for p in patients}
    np.random.seed(SEED + rep + int(err * 1000))
    for p in patients:
        p.p_fail_mri = np.clip(p.p_fail_mri + np.random.normal(0, err), 0.01, 0.80)
    sdf, _ = stochastic_milp_schedule(patients, DEFAULT_CONFIG, verbose=False)
    if sdf is None:
        return []
    for p in patients:
        p.p_fail_mri = true_p[p.patient_id]
    sim = evaluate_schedule_with_simulation(sdf, patients, DEFAULT_CONFIG, 100, SEED + rep)
    return [{'Error_Std': err, 'Replication': rep + 1,
             'Expected_Makespan': sim['expected_makespan'],
             'Overtime_Prob': sim['overtime_probability'],
             'Makespan_95pct': sim['makespan_95pct']}]


def unit_e6(frac, rep):
    patients = X.get_cohort(10, high_risk_fraction=frac, seed=SEED + rep)
    rows = []
    for name, fn in [('Stochastic-MILP', lambda p, c: stochastic_milp_schedule(p, c, verbose=False)),
                     ('Deterministic', deterministic_schedule)]:
        sdf, _ = fn(patients, DEFAULT_CONFIG)
        if sdf is None:
            continue
        sim = evaluate_schedule_with_simulation(sdf, patients, DEFAULT_CONFIG, 100, SEED + rep)
        rows.append({'High_Risk_Fraction': frac, 'Replication': rep + 1, 'Method': name,
                     'Expected_Makespan': sim['expected_makespan'],
                     'Overtime_Prob': sim['overtime_probability']})
    return rows


def unit_e7(uds, csf, rep):
    cfg = E7MOD.make_config_with_scaling(uds, csf)
    is_base = (abs(uds - 0.30) < 1e-9 and abs(csf - 0.80) < 1e-9)
    patients = X.get_cohort(10, seed=SEED + rep)
    rows = []
    for name, fn in E7MOD.METHODS:
        try:
            sdf, _ = fn(patients, cfg)
        except Exception as err:
            rows.append({'UDS_Scale': uds, 'CSF_Scale': csf, 'Is_Baseline': is_base,
                         'Replication': rep + 1, 'Method': name,
                         'Expected_Makespan': np.nan, 'Overtime_Prob': np.nan,
                         'Error': str(err)[:80]})
            continue
        if sdf is None:
            continue
        sim = evaluate_schedule_with_simulation(sdf, patients, cfg, 100, SEED + rep)
        rows.append({'UDS_Scale': uds, 'CSF_Scale': csf, 'Is_Baseline': is_base,
                     'Replication': rep + 1, 'Method': name,
                     'Expected_Makespan': sim['expected_makespan'],
                     'Overtime_Prob': sim['overtime_probability'], 'Error': ''})
    return rows


# ---------------------------------------------------------------- unit lists
def build_units(exp):
    units = []  # (unit_id, worst_seconds, callable)
    if exp == 'E1':
        for rep in range(100):
            units.append((f'rep{rep}', 30, lambda r=rep: unit_e1(r)))
    elif exp == 'E2':
        for n in [10, 20, 30, 50]:
            worst = {10: 72, 20: 105, 30: 105, 50: 110}[n]
            for rep in range(10):
                units.append((f'n{n}_rep{rep}', worst, lambda nn=n, r=rep: unit_e2(nn, r)))
    elif exp == 'E3':
        for lam in [0.0, 0.25, 0.5, 1.0, 2.0]:
            for rep in range(20):
                units.append((f'lam{lam}_rep{rep}', 30, lambda l=lam, r=rep: unit_e3(l, r)))
    elif exp == 'E4':
        for rep in range(50):
            for pred in ['Full_ML', 'Age_Only', 'Age_Stratified', 'No_Prediction']:
                w = 30 if pred in ('Full_ML', 'Age_Only') else 5
                units.append((f'rep{rep}_{pred}', w, lambda r=rep, p=pred: unit_e4(r, p)))
    elif exp == 'E5':
        for err in [0.05, 0.10, 0.15, 0.20]:
            for rep in range(30):
                units.append((f'err{err}_rep{rep}', 30, lambda e=err, r=rep: unit_e5(e, r)))
    elif exp == 'E6':
        for frac in [0.0, 0.1, 0.2, 0.3, 0.5]:
            for rep in range(30):
                units.append((f'frac{frac}_rep{rep}', 30, lambda f=frac, r=rep: unit_e6(f, r)))
    elif exp == 'E7':
        for uds in E7MOD.UDS_FACTORS:
            for csf in E7MOD.CSF_FACTORS:
                for rep in range(30):
                    units.append((f'u{uds}_c{csf}_rep{rep}', 40,
                                  lambda u=uds, c=csf, r=rep: unit_e7(u, c, r)))
    else:
        raise SystemExit(f'unknown experiment {exp}')
    return units


FINAL_NAME = {'E1': 'e1_baseline', 'E2': 'e2_scalability', 'E3': 'e3_lambda',
              'E4': 'e4_ml_value', 'E5': 'e5_robustness', 'E6': 'e6_cohort',
              'E7': 'e7_scaling_sensitivity'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--budget', type=float, default=103.0)
    ap.add_argument('--era', choices=['both', 'UDS3', 'UDS4'], default='both')
    ap.add_argument('--p-fail', choices=['recalibrated', 'raw'], default='recalibrated')
    a = ap.parse_args()
    t0 = time.time()

    os.makedirs(OUTDIR, exist_ok=True)
    suffix = '' if a.p_fail == 'recalibrated' else '_rawp'
    suffix += '' if a.era == 'both' else f'_{a.era}'
    rows_path = f'{OUTDIR}/{a.exp}{suffix}_rows.csv'
    done_path = f'{OUTDIR}/{a.exp}{suffix}_done.txt'

    setup_pool(a.era, a.p_fail == 'recalibrated')
    units = build_units(a.exp)
    done = set()
    if os.path.exists(done_path):
        done = set(x.strip() for x in open(done_path) if x.strip())

    ran, broke = 0, False
    for uid, worst, fn in units:
        if uid in done:
            continue
        if time.time() - t0 + worst > a.budget:
            broke = True
            break
        rows = fn()
        if rows:
            pd.DataFrame(rows).to_csv(rows_path, mode='a', index=False,
                                      header=not os.path.exists(rows_path))
        with open(done_path, 'a') as f:
            f.write(uid + '\n')
        done.add(uid)
        ran += 1

    print(f'{a.exp}: ran {ran} units this call; {len(done)}/{len(units)} done; '
          f'{time.time() - t0:.0f}s elapsed')
    if len(done) == len(units):
        df = pd.read_csv(rows_path)
        final = f"results/{FINAL_NAME[a.exp]}_real{suffix}_week23.csv"
        df.to_csv(final, index=False)
        print(f'{a.exp} COMPLETE -> {final} ({len(df)} rows)')


if __name__ == '__main__':
    main()
