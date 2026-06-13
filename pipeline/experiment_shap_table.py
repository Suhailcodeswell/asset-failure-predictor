"""
Build SHAP importance table for model with repair pattern features.
Outputs a formatted table like the client presentation slide.
"""
import json, numpy as np, pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from collections import Counter
import shap

# ======================================================================
# Load data
# ======================================================================
print('Loading data...')
df = pd.read_csv('data/processed/training/trip_features_v3.csv')
df_final = pd.read_csv('data/processed/training/trip_features_final.csv',
                        usecols=['reefer_unit', 'trip_start',
                                 'hwy_trip_count', 'hwy_total_miles', 'hwy_gap_days',
                                 'hwy_avg_miles_per_trip', 'hwy_trips_per_day', 'pm_cum_overdue_days'])
df = df.merge(df_final, on=['reefer_unit', 'trip_start'], how='left', suffixes=('', '_final'))

meta = json.load(open('models/risk_engine/feature_meta.json'))
feature_labels = meta['feature_labels']
feature_categories = meta['feature_categories']

# ======================================================================
# Engineer repair pattern features
# ======================================================================
print('Engineering repair pattern features...')
repairs = pd.read_excel('data/raw/repairs/Updated Reefer Unit Repairs 2024-2025.xlsx')
repairs['OPENED'] = pd.to_datetime(repairs['OPENED'], errors='coerce')
repairs['UNITNUMBER'] = repairs['UNITNUMBER'].astype(str).str.strip()
repairs['COMPCODE'] = repairs['COMPCODE'].astype(str).str.strip()

def comp_group(code):
    prefix = code.split('-')[0] if '-' in code else code[:3]
    groups = {'000': 'PM', '001': 'Compressor', '002': 'Condenser', '003': 'Evaporator',
              '035': 'Electrical', 'DOR': 'Door', 'DRY': 'DryRun', 'UNR': 'Recovery', '082': 'Diagnosis'}
    return groups.get(prefix, 'Other')

repairs['comp_group'] = repairs['COMPCODE'].apply(comp_group)
wo = repairs.groupby(['UNITNUMBER', repairs['OPENED'].dt.date]).agg(
    comp_groups=('comp_group', lambda x: list(set(x))),
    has_diag=('comp_group', lambda x: 'Diagnosis' in set(x)),
    has_dry_run=('comp_group', lambda x: 'DryRun' in set(x)),
    has_door=('comp_group', lambda x: 'Door' in set(x)),
    has_recovery=('comp_group', lambda x: 'Recovery' in set(x)),
).reset_index()
wo.columns = ['unit', 'wo_date', 'comp_groups', 'has_diag', 'has_dry_run', 'has_door', 'has_recovery']
wo['wo_date'] = pd.to_datetime(wo['wo_date'])

df['trip_start_dt'] = pd.to_datetime(df['trip_start'])

new_features = []
for idx, trip in df.iterrows():
    unit = str(trip['reefer_unit']).strip()
    trip_date = trip['trip_start_dt']
    unit_wo = wo[wo['unit'] == unit].copy()
    unit_wo['days_before'] = (trip_date - unit_wo['wo_date']).dt.days
    wo_7d = unit_wo[(unit_wo['days_before'] >= 0) & (unit_wo['days_before'] <= 7)]
    wo_14d = unit_wo[(unit_wo['days_before'] >= 0) & (unit_wo['days_before'] <= 14)]
    wo_30d = unit_wo[(unit_wo['days_before'] >= 0) & (unit_wo['days_before'] <= 30)]

    dry_run_14d = int(wo_14d['has_dry_run'].any()) if len(wo_14d) else 0

    diag_no_fix_14d = 0
    if len(wo_14d) > 0:
        last_wo = wo_14d.sort_values('wo_date').iloc[-1]
        has_fix = any(g not in ('Diagnosis', 'PM', 'Other') for g in last_wo['comp_groups'])
        if last_wo['has_diag'] and not has_fix:
            diag_no_fix_14d = 1

    door_repair_14d = int(wo_14d['has_door'].any()) if len(wo_14d) else 0
    recovery_14d = int(wo_14d['has_recovery'].any()) if len(wo_14d) else 0

    electrical_14d = 0
    for _, w in wo_14d.iterrows():
        if 'Electrical' in w['comp_groups']:
            electrical_14d = 1
            break

    n_work_orders_14d = len(wo_14d)
    n_work_orders_30d = len(wo_30d)

    groups_14d = set()
    for _, w in wo_14d.iterrows():
        groups_14d.update(w['comp_groups'])
    n_groups_14d = len(groups_14d)

    has_repeat_30d = 0
    if len(wo_30d) >= 2:
        all_g = []
        for _, w in wo_30d.iterrows():
            all_g.extend(w['comp_groups'])
        counts = Counter(all_g)
        has_repeat_30d = int(any(c >= 2 for g, c in counts.items() if g != 'PM'))

    turnaround_days = int(wo_30d['days_before'].min()) if len(wo_30d) > 0 else -1

    new_features.append({
        'dry_run_14d': dry_run_14d,
        'diag_no_fix_14d': diag_no_fix_14d,
        'door_repair_14d': door_repair_14d,
        'recovery_14d': recovery_14d,
        'electrical_14d': electrical_14d,
        'n_work_orders_14d': n_work_orders_14d,
        'n_work_orders_30d': n_work_orders_30d,
        'n_groups_14d': n_groups_14d,
        'has_repeat_repair_30d': has_repeat_30d,
        'turnaround_days': turnaround_days,
    })
    if idx % 5000 == 0:
        print(f'  {idx}/{len(df)}...')

new_feat_df = pd.DataFrame(new_features)
df_c = pd.concat([df, new_feat_df], axis=1)

# Labels for new features
new_labels = {
    'dry_run_14d': 'Dry Run Test (14d)',
    'diag_no_fix_14d': 'Diagnosis Without Fix (14d)',
    'door_repair_14d': 'Door Repair (14d)',
    'recovery_14d': 'Unit Recovery/Tow (14d)',
    'electrical_14d': 'Electrical Repair (14d)',
    'n_work_orders_14d': 'Work Orders (14d)',
    'n_work_orders_30d': 'Work Orders (30d)',
    'n_groups_14d': 'Component Groups Repaired (14d)',
    'has_repeat_repair_30d': 'Repeat Repair Same Component (30d)',
    'turnaround_days': 'Days Since Last Repair',
}
new_categories = {k: 'Pre-Trip Repair Pattern' for k in new_labels}

# ======================================================================
# Build feature set (current 75 + new repair patterns)
# ======================================================================
current_features = meta['feature_cols']
# Only add features that don't already exist in the current model or the source DataFrame
truly_new = [f for f in new_labels.keys()
             if f not in current_features and f not in df.columns]
combined_features = current_features + truly_new
print(f'\nFeature set: {len(current_features)} current + {len(truly_new)} truly new = {len(combined_features)}')
print(f'  Already in model: {[f for f in new_labels if f in current_features]}')
print(f'  Already in CSV (pruned): {[f for f in new_labels if f in df.columns and f not in current_features]}')
print(f'  Truly new (engineered): {truly_new}')

all_labels_map = {**feature_labels, **new_labels}
all_categories_map = {**feature_categories, **new_categories}

# Drop duplicate columns from df_c before selecting
df_c = df_c.loc[:, ~df_c.columns.duplicated()]

# ======================================================================
# Train
# ======================================================================
y = df_c['shutdown'].values
df_c['month'] = pd.to_datetime(df_c['trip_start']).dt.month
strat_key = df_c['shutdown'].astype(str) + '_' + df_c['month'].astype(str)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df_c, strat_key))

# Encode categoricals with label encoders from training
label_encoders = json.load(open('models/risk_engine/label_encoders.json'))
X_df = df_c[combined_features].copy()
for col in X_df.select_dtypes(include=['object', 'category']).columns:
    if col in label_encoders:
        le = label_encoders[col]
        X_df[col] = X_df[col].map(lambda v: le.get(str(v), -1)).astype(float)
    else:
        X_df[col] = X_df[col].astype('category').cat.codes.astype(float)

# Use DMatrix with feature names to avoid XGBoost expanding features
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X_df.values.astype(np.float32))

# Track actual feature names used
actual_feature_names = list(X_df.columns)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f'Train: {len(train_idx)}, Test: {len(test_idx)}')

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=actual_feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=actual_feature_names)

params = {
    'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'min_child_weight': 5, 'gamma': 0.1,
    'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'scale_pos_weight': (y_train == 0).sum() / max((y_train == 1).sum(), 1),
    'eval_metric': 'auc', 'objective': 'binary:logistic', 'seed': 42,
}
model = xgb.train(params, dtrain, num_boost_round=235, evals=[(dtest, 'test')], verbose_eval=False)

preds = model.predict(dtest)
auc = roc_auc_score(y_test, preds)
print(f'AUC: {auc:.4f}')

# ======================================================================
# SHAP importance
# ======================================================================
print('Computing SHAP on 500 test samples...')
explainer = shap.TreeExplainer(model)
dtest_500 = xgb.DMatrix(X_test[:500], feature_names=actual_feature_names)
shap_values = explainer.shap_values(dtest_500)
print(f'  SHAP shape: {shap_values.shape}, features: {len(combined_features)}')
mean_abs_shap = np.abs(shap_values).mean(axis=0)
assert len(mean_abs_shap) == len(combined_features), f'SHAP {len(mean_abs_shap)} != features {len(combined_features)}'

all_labels = {**feature_labels, **new_labels, **all_labels_map}
all_categories = {**feature_categories, **new_categories, **all_categories_map}

importance = pd.DataFrame({
    'Feature': combined_features,
    'Category': [all_categories.get(f, 'Other') for f in combined_features],
    'Mean |SHAP|': mean_abs_shap,
})
importance = importance.sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)
importance.index = importance.index + 1
importance.index.name = 'Rank'

total_shap = importance['Mean |SHAP|'].sum()
importance['% of Total'] = (importance['Mean |SHAP|'] / total_shap * 100).round(2)
importance['Cumulative %'] = importance['% of Total'].cumsum().round(2)
importance['Label'] = [all_labels.get(f, f.replace('_', ' ').title()) for f in importance['Feature']]

# ======================================================================
# Print table
# ======================================================================
print('\n')
print('=' * 130)
print('Top 25 Risk Factors (SHAP Importance) — 75 features + 10 repair pattern features')
print('=' * 130)
header = f"{'Rank':<5} {'Feature':<35} {'Category':<25} {'Mean |SHAP|':>12} {'% Total':>9} {'Cumul %':>9}  {'Label'}"
print(header)
print('-' * 130)
for rank, row in importance.head(25).iterrows():
    marker = '  <<<NEW' if row['Feature'] in new_labels else ''
    print(f"{rank:<5} {row['Feature']:<35} {row['Category']:<25} {row['Mean |SHAP|']:>12.4f} {row['% of Total']:>8.2f}% {row['Cumulative %']:>8.2f}%  {row['Label']}{marker}")

print('\n')
print('=' * 80)
print('NEW REPAIR PATTERN FEATURES — Where They Ranked')
print('=' * 80)
for rank, row in importance.iterrows():
    if row['Feature'] in new_labels:
        print(f"  Rank {rank:>3}: {row['Label']:<40} SHAP={row['Mean |SHAP|']:.4f}  ({row['% of Total']:.2f}%)")

# Save CSV
importance.to_csv('failure/output/shap_importance_with_repair_patterns.csv')
print('\nSaved: output/shap_importance_with_repair_patterns.csv')
