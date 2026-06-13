"""
Experiment: Compare model performance across feature pruning strategies.

A: Current model (75 features — aggressive VIF/collinearity pruning)
B: Relaxed pruning (only remove true duplicates + metadata — keep correlated features)
C: Relaxed + new repair pattern features engineered from repair data

XGBoost is a tree model — collinearity doesn't affect it. This experiment tests
whether the aggressive pruning hurt performance.
"""
import json
import pickle
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# ======================================================================
# Load data
# ======================================================================
print('Loading data...')
df = pd.read_csv('data/processed/training/trip_features_v3.csv')
# Merge in columns from trip_features_final.csv (hwy_*, pm_cum_overdue_days)
df_final = pd.read_csv('data/processed/training/trip_features_final.csv',
                        usecols=['reefer_unit', 'trip_start',
                                 'hwy_trip_count', 'hwy_total_miles', 'hwy_gap_days',
                                 'hwy_avg_miles_per_trip', 'hwy_trips_per_day', 'pm_cum_overdue_days'])
df = df.merge(df_final, on=['reefer_unit', 'trip_start'], how='left', suffixes=('', '_final'))
print(f'  {len(df)} trips, {len(df.columns)} columns')

# Target
y = df['shutdown'].values

# Meta columns to always exclude
META = ['reefer_unit', 'trip_start', 'trip_end', 'shutdown', 'leg', 'manifest', 'trailer']

# ======================================================================
# Feature Set A: Current 75 features
# ======================================================================
with open('models/risk_engine/feature_meta.json') as f:
    current_features = json.load(f)['feature_cols']

# ======================================================================
# Feature Set B: Relaxed pruning — only remove TRUE duplicates & metadata
# ======================================================================
all_features = [c for c in df.columns if c not in META]

# Only prune genuinely identical/useless columns
minimal_prune = {
    'total_miles',               # exact copy of leg_miles
    'pm_policy_days',            # exact copy of is_slxi encoding
    'days_since_any_shop_visit', # exact duplicate
    'transit_temp_swing',        # = transit_max_temp - transit_min_temp (linear combo)
}

relaxed_features = [c for c in all_features if c not in minimal_prune]

# ======================================================================
# Feature Set C: Relaxed + new repair pattern features
# ======================================================================
# Engineer new features from existing data
print('Engineering repair pattern features...')

# Load repair data
repairs = pd.read_excel('data/raw/repairs/Updated Reefer Unit Repairs 2024-2025.xlsx')
repairs['OPENED'] = pd.to_datetime(repairs['OPENED'], errors='coerce')
repairs['UNITNUMBER'] = repairs['UNITNUMBER'].astype(str).str.strip()
repairs['COMPCODE'] = repairs['COMPCODE'].astype(str).str.strip()

# Component group mapping (simplified)
def comp_group(code):
    prefix = code.split('-')[0] if '-' in code else code[:3]
    groups = {
        '000': 'PM', '001': 'Compressor', '002': 'Condenser', '003': 'Evaporator',
        '035': 'Electrical', 'DOR': 'Door', 'DRY': 'DryRun', 'UNR': 'Recovery',
        '082': 'Diagnosis',
    }
    return groups.get(prefix, 'Other')

repairs['comp_group'] = repairs['COMPCODE'].apply(comp_group)

# Build work orders (group by unit + date)
wo = repairs.groupby(['UNITNUMBER', repairs['OPENED'].dt.date]).agg(
    comp_groups=('comp_group', lambda x: list(set(x))),
    comp_codes=('COMPCODE', lambda x: list(set(x))),
    n_codes=('COMPCODE', 'nunique'),
    has_diag=('comp_group', lambda x: 'Diagnosis' in set(x)),
    has_dry_run=('comp_group', lambda x: 'DryRun' in set(x)),
    has_door=('comp_group', lambda x: 'Door' in set(x)),
    has_recovery=('comp_group', lambda x: 'Recovery' in set(x)),
).reset_index()
wo.columns = ['unit', 'wo_date', 'comp_groups', 'comp_codes', 'n_codes',
              'has_diag', 'has_dry_run', 'has_door', 'has_recovery']
wo['wo_date'] = pd.to_datetime(wo['wo_date'])

# For each trip, compute new repair pattern features
df['trip_start_dt'] = pd.to_datetime(df['trip_start'])

new_features = []
for idx, trip in df.iterrows():
    unit = str(trip['reefer_unit']).strip()
    trip_date = trip['trip_start_dt']

    # Work orders in windows before this trip
    unit_wo = wo[wo['unit'] == unit].copy()
    unit_wo['days_before'] = (trip_date - unit_wo['wo_date']).dt.days

    wo_7d = unit_wo[(unit_wo['days_before'] >= 0) & (unit_wo['days_before'] <= 7)]
    wo_14d = unit_wo[(unit_wo['days_before'] >= 0) & (unit_wo['days_before'] <= 14)]
    wo_30d = unit_wo[(unit_wo['days_before'] >= 0) & (unit_wo['days_before'] <= 30)]

    # Feature: had dry run before trip
    dry_run_7d = int(wo_7d['has_dry_run'].any()) if len(wo_7d) else 0
    dry_run_14d = int(wo_14d['has_dry_run'].any()) if len(wo_14d) else 0

    # Feature: diagnosis without component fix (diag-only)
    diag_no_fix_14d = 0
    if len(wo_14d) > 0:
        last_wo = wo_14d.sort_values('wo_date').iloc[-1]
        has_fix = any(g not in ('Diagnosis', 'PM', 'Other') for g in last_wo['comp_groups'])
        if last_wo['has_diag'] and not has_fix:
            diag_no_fix_14d = 1

    # Feature: door repair before trip
    door_repair_7d = int(wo_7d['has_door'].any()) if len(wo_7d) else 0
    door_repair_14d = int(wo_14d['has_door'].any()) if len(wo_14d) else 0

    # Feature: unit recovery/tow before trip
    recovery_14d = int(wo_14d['has_recovery'].any()) if len(wo_14d) else 0

    # Feature: number of distinct work orders in window (repair intensity)
    n_work_orders_7d = len(wo_7d)
    n_work_orders_14d = len(wo_14d)
    n_work_orders_30d = len(wo_30d)

    # Feature: number of distinct component groups repaired in window
    groups_14d = set()
    for _, w in wo_14d.iterrows():
        groups_14d.update(w['comp_groups'])
    n_groups_14d = len(groups_14d)

    # Feature: repeat repair (same group in 2+ work orders in 30d)
    has_repeat_30d = 0
    if len(wo_30d) >= 2:
        all_groups = []
        for _, w in wo_30d.iterrows():
            all_groups.extend(w['comp_groups'])
        from collections import Counter
        counts = Counter(all_groups)
        has_repeat_30d = int(any(c >= 2 for g, c in counts.items() if g != 'PM'))

    # Feature: days since last repair (turnaround)
    turnaround_days = -1
    if len(wo_30d) > 0:
        turnaround_days = int(wo_30d['days_before'].min())

    # Feature: electrical repair before trip
    electrical_14d = 0
    for _, w in wo_14d.iterrows():
        if 'Electrical' in w['comp_groups']:
            electrical_14d = 1
            break

    new_features.append({
        'dry_run_7d': dry_run_7d,
        'dry_run_14d': dry_run_14d,
        'diag_no_fix_14d': diag_no_fix_14d,
        'door_repair_7d': door_repair_7d,
        'door_repair_14d': door_repair_14d,
        'recovery_14d': recovery_14d,
        'electrical_14d': electrical_14d,
        'n_work_orders_7d': n_work_orders_7d,
        'n_work_orders_14d': n_work_orders_14d,
        'n_work_orders_30d': n_work_orders_30d,
        'n_groups_14d': n_groups_14d,
        'has_repeat_repair_30d': has_repeat_30d,
        'turnaround_days': turnaround_days,
    })

    if idx % 5000 == 0:
        print(f'  Engineered {idx}/{len(df)} trips...')

new_feat_df = pd.DataFrame(new_features)
print(f'  New features: {list(new_feat_df.columns)}')

# Show prevalence
print('\nNew feature prevalence:')
for col in new_feat_df.columns:
    nonzero = (new_feat_df[col] > 0).sum()
    if new_feat_df[col].dtype in [int, float]:
        print(f'  {col}: {nonzero} trips ({nonzero/len(df)*100:.1f}%), mean={new_feat_df[col].mean():.3f}')

# Combine
df_c = pd.concat([df, new_feat_df], axis=1)
enhanced_features = relaxed_features + list(new_feat_df.columns)

# ======================================================================
# Train-test split (same as build_risk_engine: month-stratified 80/20)
# ======================================================================
df['month'] = pd.to_datetime(df['trip_start']).dt.month
from sklearn.model_selection import StratifiedShuffleSplit

# Stratify by shutdown x month
strat_key = df['shutdown'].astype(str) + '_' + df['month'].astype(str)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, strat_key))

print(f'\nTrain: {len(train_idx)}, Test: {len(test_idx)}')
print(f'Train shutdown rate: {y[train_idx].mean()*100:.2f}%')
print(f'Test shutdown rate: {y[test_idx].mean()*100:.2f}%')

# ======================================================================
# Train & evaluate each feature set
# ======================================================================
results = {}

for name, features, source_df in [
    ('A: Current 75 features', current_features, df),
    ('B: Relaxed pruning (all features)', relaxed_features, df),
    ('C: Relaxed + repair patterns', enhanced_features, df_c),
]:
    print(f'\n{"="*60}')
    print(f'Training: {name}')
    print(f'  Features: {len(features)}')
    print(f'{"="*60}')

    # Prepare data — encode categoricals
    X_df = source_df[features].copy()
    for col in X_df.select_dtypes(include=['object', 'category']).columns:
        X_df[col] = X_df[col].astype('category').cat.codes
    X = X_df.values.astype(np.float32)

    # Impute
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train XGBoost (same hyperparams as build_risk_engine)
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=235,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric='auc',
        random_state=42,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    train_time = time.time() - t0

    # Calibrate
    cal_model = CalibratedClassifierCV(model, cv=3, method='isotonic')
    cal_model.fit(X_train, y_train)

    # Evaluate
    raw_proba = model.predict_proba(X_test)[:, 1]
    cal_proba = cal_model.predict_proba(X_test)[:, 1]

    raw_auc = roc_auc_score(y_test, raw_proba)
    cal_auc = roc_auc_score(y_test, cal_proba)
    cal_ap = average_precision_score(y_test, cal_proba)
    cal_brier = brier_score_loss(y_test, cal_proba)

    results[name] = {
        'n_features': len(features),
        'raw_auc': round(raw_auc, 4),
        'cal_auc': round(cal_auc, 4),
        'cal_ap': round(cal_ap, 4),
        'cal_brier': round(cal_brier, 4),
        'train_time': round(train_time, 1),
    }

    print(f'  Raw AUC:       {raw_auc:.4f}')
    print(f'  Calibrated AUC:{cal_auc:.4f}')
    print(f'  Avg Precision: {cal_ap:.4f}')
    print(f'  Brier Score:   {cal_brier:.4f}')
    print(f'  Train time:    {train_time:.1f}s')

# ======================================================================
# Comparison
# ======================================================================
print(f'\n\n{"="*70}')
print('RESULTS COMPARISON')
print(f'{"="*70}')
print(f'{"Model":<35} {"Features":>8} {"Raw AUC":>9} {"Cal AUC":>9} {"Avg Prec":>9} {"Brier":>8}')
print('-' * 80)
for name, r in results.items():
    print(f'{name:<35} {r["n_features"]:>8} {r["raw_auc"]:>9.4f} {r["cal_auc"]:>9.4f} {r["cal_ap"]:>9.4f} {r["cal_brier"]:>8.4f}')

# Highlight winner
best = max(results.items(), key=lambda x: x[1]['cal_auc'])
print(f'\nBest: {best[0]} (AUC = {best[1]["cal_auc"]})')

# Delta
if len(results) >= 2:
    keys = list(results.keys())
    for i in range(1, len(keys)):
        delta = results[keys[i]]['cal_auc'] - results[keys[0]]['cal_auc']
        print(f'  {keys[i]} vs {keys[0]}: AUC {delta:+.4f}')
