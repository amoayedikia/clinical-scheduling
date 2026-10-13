"""
E8: BOUNDARY CONDITIONS UNDER CENTER- AND ERA-SHIFT (Track A3, Weeks 4-5)

Question: does "prediction quality dominates optimization sophistication"
survive when predictions are degraded by REAL distribution shift, rather than
by synthetic Gaussian noise (E5)?

Design. Real NACC74 pool (as in the A1 real arm; recalibrated true p_fail,
mean 0.212). In each scenario the Phase-1 GBM (paper spec) is trained on
cohorts from a TRAINING stratum -- real covariates, outcomes simulated once
as y ~ Bernoulli(p_true) -- and used to schedule cohorts sampled from a
disjoint EVALUATION stratum. Schedules are built from the model's predicted
p-hat (exactly as E4/E5 build from corrupted predictions) and evaluated by
Monte Carlo against true p. An ORACLE arm (schedule from p_true) on the same
evaluation cohorts isolates the shift-induced loss.

Scenarios:
  split      random 50/50 participant split (no distribution shift; reference)
  era34      train UDS3-era participants -> evaluate UDS4-era cohorts
  era43      train UDS4-era participants -> evaluate UDS3-era cohorts
  centers    train on a random half of ADRCs -> evaluate held-out ADRCs

Methods: Stochastic-MILP, LPT (both under model + oracle arms);
Deterministic, Age-Stratified (prediction-free, once per replication).

Usage:  python3 e8_shift_experiment.py [--reps 30] [--quick]
Output: results/e8_shift_<timestamp>.csv (+ _predquality) + console summary.
"""
import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr

from nacc74_cohort_builder import load_pool, assign_p_fail
from a2_qc_validation import engineer
from baselines import (DEFAULT_CONFIG, PatientData, lpt_schedule,
                       deterministic_schedule, age_stratified_schedule,
                       stochastic_milp_schedule, evaluate_schedule_with_simulation)

SEED = 42
N_PATIENTS = 10
N_SIMS = 100

FEATS = ['agit', 'mot', 'anx', 'MRI_SPECIFIC_RISK',
         'agit_sev', 'mot_sev', 'anx_sev',
         'BEHAVIORAL_RISK', 'SEVERITY_WEIGHTED_RISK',
         'moca', 'COGNITIVE_IMPAIRMENT',
         'cdr_sum', 'cdr', 'CDR_SEVERITY', 'CDR_IMPAIRED',
         'age', 'AGE_RISK', 'decclcog',
         'irr', 'del', 'hall', 'dep']


def make_scenarios(pool, rng):
    sc = {}
    idx = rng.permutation(len(pool))
    half = len(pool) // 2
    sc['split'] = (pool.iloc[idx[:half]], pool.iloc[idx[half:]])
    sc['era34'] = (pool[pool.ERA == 'UDS3'], pool[pool.ERA == 'UDS4'])
    sc['era43'] = (pool[pool.ERA == 'UDS4'], pool[pool.ERA == 'UDS3'])
    centers = rng.permutation(pool.NACCADC.unique())
    a = set(centers[:len(centers) // 2])
    sc['centers'] = (pool[pool.NACCADC.isin(a)], pool[~pool.NACCADC.isin(a)])
    return sc


def train_model(train_pool, seed):
    rng = np.random.default_rng(seed)
    n_train = min(2000, len(train_pool))          # paper-consistent training size
    tr = train_pool.sample(n=n_train, random_state=seed)
    y = (rng.uniform(size=n_train) < tr.P_FAIL.values).astype(int)
    X = engineer(tr)[FEATS].values
    imp = SimpleImputer(strategy='median')
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                       min_samples_leaf=20, random_state=42)
    model.fit(imp.fit_transform(X), y)
    return model, imp, n_train, y.mean()


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--quick', action='store_true', help='1 rep, 30 sims (smoke test)')
    ap.add_argument('--scenarios', default='split,era34,era43,centers',
                    help='comma-separated subset of scenarios to run')
    a = ap.parse_args()
    reps, n_sims = (1, 30) if a.quick else (a.reps, N_SIMS)

    pool = assign_p_fail(load_pool('both'), seed=SEED, recalibrate=True)
    rng = np.random.default_rng(SEED)
    scenarios = make_scenarios(pool, rng)
    keep = [x.strip() for x in a.scenarios.split(',')]
    scenarios = {k: v for k, v in scenarios.items() if k in keep}

    rows, pq_rows = [], []
    for name, (train_pool, eval_pool) in scenarios.items():
        t0 = time.time()
        model, imp, n_train, event_rate = train_model(train_pool, SEED)
        ev = eval_pool.copy()
        ev['P_HAT'] = model.predict_proba(imp.transform(engineer(ev)[FEATS].values))[:, 1]
        rho = spearmanr(ev.P_HAT, ev.P_FAIL)[0]
        mae = float(np.abs(ev.P_HAT - ev.P_FAIL).mean())
        pq_rows.append({'Scenario': name, 'N_Train': n_train, 'Train_Event_Rate': round(event_rate, 3),
                        'N_Eval_Pool': len(ev), 'Spearman_phat_ptrue': round(rho, 4),
                        'MAE_phat_ptrue': round(mae, 4),
                        'Eval_p_true_mean': round(float(ev.P_FAIL.mean()), 4),
                        'Eval_p_hat_mean': round(float(ev.P_HAT.mean()), 4),
                        'Eval_CDR1_pct': round(100 * float((ev.CDRGLOB >= 1).mean()), 1)})
        print(f"[{name}] train n={n_train} (events {event_rate:.1%}) | eval pool {len(ev)} | "
              f"rho(p_hat,p_true)={rho:.3f} | MAE={mae:.3f} | "
              f"p_true mean {ev.P_FAIL.mean():.3f} vs p_hat mean {ev.P_HAT.mean():.3f}")

        for rep in tqdm(range(reps), desc=f"E8 {name}"):
            cohort_rows = ev.sample(n=N_PATIENTS, random_state=SEED + rep)
            def mk(pcol):
                return [PatientData(f"P{i+1:03d}", float(r[pcol]), float(r.NACCAGE),
                                    float(r.MOCATOTS), float(r.CDRGLOB))
                        for i, (_, r) in enumerate(cohort_rows.iterrows())]
            true_p = cohort_rows.P_FAIL.values

            for arm, pcol in [('model', 'P_HAT'), ('oracle', 'P_FAIL')]:
                for mname, fn in [('Stochastic-MILP', lambda p, c: stochastic_milp_schedule(p, c, verbose=False)),
                                  ('LPT', lpt_schedule)]:
                    patients = mk(pcol)
                    sdf, _ = fn(patients, DEFAULT_CONFIG)
                    if sdf is None:
                        continue
                    for p, tp in zip(patients, true_p):     # evaluate against TRUE p
                        p.p_fail_mri = float(tp)
                    sim = evaluate_schedule_with_simulation(sdf, patients, DEFAULT_CONFIG, n_sims, SEED + rep)
                    rows.append({'Scenario': name, 'Arm': arm, 'Replication': rep + 1, 'Method': mname,
                                 'Expected_Makespan': sim['expected_makespan'],
                                 'Overtime_Prob': sim['overtime_probability']})
            # prediction-free baselines once per replication
            for mname, fn in [('Deterministic', deterministic_schedule),
                              ('Age-Stratified', age_stratified_schedule)]:
                patients = mk('P_FAIL')
                sdf, _ = fn(patients, DEFAULT_CONFIG)
                if sdf is None:
                    continue
                sim = evaluate_schedule_with_simulation(sdf, patients, DEFAULT_CONFIG, n_sims, SEED + rep)
                rows.append({'Scenario': name, 'Arm': 'n/a', 'Replication': rep + 1, 'Method': mname,
                             'Expected_Makespan': sim['expected_makespan'],
                             'Overtime_Prob': sim['overtime_probability']})
        print(f"[{name}] done in {time.time() - t0:.0f}s")

    os.makedirs('results', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    df, pq = pd.DataFrame(rows), pd.DataFrame(pq_rows)
    df.to_csv(f'results/e8_shift_{ts}.csv', index=False)
    pq.to_csv(f'results/e8_shift_predquality_{ts}.csv', index=False)

    print("\n=== E8 SUMMARY: overtime probability (%) ===")
    piv = df.assign(key=df.Method + '|' + df.Arm).pivot_table(
        index='Scenario', columns='key', values='Overtime_Prob', aggfunc='mean') * 100
    print(piv.round(2).to_string())
    print("\nPrediction quality per scenario:")
    print(pq.to_string(index=False))
    print(f"\nSaved: results/e8_shift_{ts}.csv, results/e8_shift_predquality_{ts}.csv")


if __name__ == '__main__':
    run()
