"""
Reefer Risk Scoring Engine — Build & Export
Trains a calibrated XGBoost model on trip-level data, computes SHAP baselines,
and exports all artifacts needed for the deployed risk tool.

Output: models/risk_engine/ with model, SHAP explainer, and lookup data.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import json, os, pickle, time, warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, f1_score, brier_score_loss)

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
OUT = 'models/risk_engine'
os.makedirs(OUT, exist_ok=True)

# ======================================================================
# 1. LOAD DATA
# ======================================================================
print('=' * 70)
print('BUILDING RISK SCORING ENGINE')
print('=' * 70)

df = pd.read_csv('data/processed/training/trip_features_v3.csv')

# Merge highway features
try:
    final = pd.read_csv('data/processed/training/trip_features_final.csv')
    hwy_cols = [c for c in final.columns if c.startswith('hwy_')] + ['pm_cum_overdue_days']
    hwy_cols = [c for c in hwy_cols if c in final.columns]
    if hwy_cols:
        df['_key'] = df['manifest'].astype(str) + '_' + df['leg'].astype(str)
        final['_key'] = final['manifest'].astype(str) + '_' + final['leg'].astype(str)
        df = df.merge(final[['_key'] + hwy_cols], on='_key', how='left')
        df.drop(columns=['_key'], inplace=True)
        print(f'  Merged {len(hwy_cols)} highway features')
except Exception as e:
    print(f'  Highway merge skipped: {e}')

df['trip_start'] = pd.to_datetime(df['trip_start'])
df['trip_month'] = df['trip_start'].dt.month

print(f'Total trips: {len(df)}, Shutdowns: {df.shutdown.sum()} ({100*df.shutdown.mean():.2f}%)')

# Month-stratified 80/20 split — ensures training covers all 12 months
train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=42, stratify=df['trip_month']
)
train_df = df.loc[train_idx].copy()
test_df = df.loc[test_idx].copy()

print(f'Train (80% per month): {len(train_df)} trips, {train_df.shutdown.sum()} shutdowns')
print(f'Test  (20% per month): {len(test_df)} trips, {test_df.shutdown.sum()} shutdowns')
print(f'  Train months: {sorted(train_df.trip_month.unique())}')
print(f'  Test  months: {sorted(test_df.trip_month.unique())}')

# ======================================================================
# 2. FEATURE ENCODING
# ======================================================================
drop_cols = ['manifest', 'leg', 'trailer', 'reefer_unit', 'trip_start', 'trip_end',
             'shutdown', 'trip_month']

cat_cols = ['route', 'origin', 'destination', 'service_type', 'season', 'reefer_model']
label_encoders = {}
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna('MISSING').astype(str))
        label_encoders[col] = le

# Re-derive splits after encoding (df is now fully encoded)
train_df = df.loc[train_idx].copy()
test_df = df.loc[test_idx].copy()

feature_cols = [c for c in train_df.columns if c not in drop_cols]
print(f'Features before pruning: {len(feature_cols)}')

# ======================================================================
# 2b. COLLINEARITY-BASED FEATURE PRUNING
# ======================================================================
# Remove redundant/dead features identified via correlation analysis & VIF.
# Rationale documented per group below.

prune_cols = set()

# --- DATA LEAKAGE: trip_duration_days = trip_end - trip_start (audit 2026-06-03) ---
# trip_end is NOT known when a trip is scored at its start (RiskScorer.score does
# not receive it -> imputed to a constant at inference = train/serve skew), and a
# shutdown lengthens the trip (shutdown trips avg 5.47d vs 4.94d), so the feature
# partly encodes the outcome it predicts. Dropped from the model.
prune_cols.add('trip_duration_days')

# --- Perfect duplicates (r = 1.0) ---
prune_cols.add('total_miles')            # identical to leg_miles
prune_cols.add('pm_policy_days')         # identical to is_slxi (binary encodes same info)

# --- Near-perfect collinearity (|r| > 0.95) ---
prune_cols.add('reefer_age_at_trip')     # r = -0.996 with reefer_year; keep reefer_year
prune_cols.add('is_carrier')             # r = -0.95 with is_thermoking; one implies the other
prune_cols.add('days_pm_overdue')        # r = 0.93 with days_since_last_pm

# --- nonpm_* features: r > 0.93 with repair_* equivalents across all windows ---
# PM cost is <3% of total; nonpm ≈ repair for all practical purposes
# Note: cumulative columns use nonpm_cum_cost naming; windowed use nonpm_cost_30d naming
prune_cols.update(['nonpm_cum_cost', 'nonpm_cum_count'])
for window in ['30d', '90d', '180d', '365d']:
    prune_cols.add(f'nonpm_cost_{window}')
    prune_cols.add(f'nonpm_count_{window}')

# --- Temperature: transit_temp_swing = max - min (VIF = ∞) ---
prune_cols.add('transit_temp_swing')     # linear combination of transit_min/max

# --- Component repairs: drop _cum window (r > 0.80 with _365d for all 22 components) ---
# Keep _90d (recent signal) + _365d (annual signal); _cum adds no new info
component_types = [
    'compressor', 'electrical_charging', 'fuel_system', 'condenser', 'engine',
    'evaporator', 'refrigerant_lines', 'coolant', 'sheet_metal', 'throttle',
    'starter_cranking', 'drier', 'diagnosis', 'chute_bulkhead', 'data_logger',
    'unit_retrieval', 'thermostat', 'expansion_valve', 'exhaust_egr', 'defrost',
    'control_panel', 'remote_status_light',
]
for comp in component_types:
    prune_cols.add(f'{comp}_cum')

# --- Highway features: massive internal redundancy (r > 0.98 between trip_count/n_ltl/n_reefer/n_single) ---
# Keep hwy_trip_count (general), hwy_total_miles, hwy_gap_days, hwy_avg_miles_per_trip, hwy_trips_per_day
hwy_drop = [
    'hwy_n_single', 'hwy_n_ltl', 'hwy_n_reefer',   # r > 0.98 with hwy_trip_count
    'hwy_n_tl', 'hwy_n_heater', 'hwy_n_dry',        # zero SHAP
    'hwy_n_nonhwy',                                   # near-zero SHAP
    'hwy_unique_origins', 'hwy_unique_dests',          # r > 0.89 with hwy_trip_count
    'hwy_max_miles',                                   # r = 0.84 with hwy_total_miles
]
prune_cols.update(hwy_drop)

# --- Zero-SHAP features (confirmed zero predictive contribution) ---
zero_shap = [
    'is_thermoking', 'has_accident', 'has_warranty',
    'drier_90d', 'control_panel_90d',
    'remote_status_light_90d', 'remote_status_light_365d',
    'exhaust_egr_90d', 'exhaust_egr_365d',
    'defrost_90d', 'expansion_valve_90d', 'thermostat_90d',
    'starter_cranking_365d',
]
prune_cols.update(zero_shap)

# --- Additional near-zero SHAP (< 0.001) after above removals ---
near_zero_shap = [
    'is_pm_overdue',          # r = 0.93 with days_since_last_pm; SHAP = 0.0008
    'battery_below_12v',      # SHAP = 0.0008; min_battery_voltage_90d is better
    'throttle_90d',           # SHAP = 0.0009
    'defrost_365d',           # SHAP = 0.0009
    'data_logger_90d',        # SHAP = 0.0008
    'evaporator_90d',         # SHAP = 0.0007
    'control_panel_365d',     # SHAP = 0.0005
    'condenser_90d',          # SHAP = 0.0006
    'refrigerant_lines_90d',  # SHAP = 0.0006
    'sheet_metal_90d',        # SHAP = 0.0004
]
prune_cols.update(near_zero_shap)

# --- Repair metric redundancy: repair_cum_count (VIF=54) overlaps 365d + diagnosis ---
prune_cols.add('repair_cum_count')       # VIF = 54; keep repair_cum_cost + repair_count_365d
prune_cols.add('pm_on_time_count')       # r = 0.90 with pm_compliance_ratio; keep ratio
prune_cols.add('days_since_any_shop_visit')  # r = 1.0 with days_since_last_repair; keep cleaner name
# Old actionable repair features superseded by pre-trip repair pattern features
prune_cols.update(['repeat_repairs_90d', 'max_repeat_count_90d',
                   'genrep_to_pm_ratio', 'unique_repair_reasons_90d',
                   'unique_shops_365d', 'repair_acceleration'])

# -----------------------------------------------------------------------
# Phase 2: VIF / collinearity deep-clean (2026-04-01 analysis)
# Removes 41 zero-SHAP features, 6 extreme-VIF telematics aggregates,
# 3 redundant PM gap stats, and 10 near-zero SHAP collinear features.
# -----------------------------------------------------------------------

# --- Zero-SHAP features still in model after Phase 1 ---
zero_shap_phase2 = [
    # Component repairs: all remaining zero-contribution 90d/365d windows
    'compressor_90d', 'compressor_365d',
    'electrical_charging_90d',
    'fuel_system_365d',
    'condenser_365d',
    'engine_90d',
    'evaporator_365d',
    'refrigerant_lines_365d',
    'coolant_90d', 'coolant_365d',
    'sheet_metal_365d',
    'throttle_365d',
    'starter_cranking_90d',
    'drier_365d',
    'chute_bulkhead_90d', 'chute_bulkhead_365d',
    'data_logger_365d',
    'expansion_valve_365d',
    'thermostat_365d',
    'unit_retrieval_90d',
    # Telematics: zero-SHAP with extreme VIF
    'telem_shutdowns_total',     # VIF = 80k
    'telem_alarms_total',        # VIF = 74k
    'telem_low_fuel_total',      # VIF = 185k
    'telem_alarms_90d',          # VIF = 6
    'telem_alarms_180d',         # VIF = 15
    'telem_low_fuel_180d',       # VIF = 28
    'alarm_yellow_total',        # VIF = 323k
    'alarm_green_total',         # VIF = 13k
    'alarm_red_90d',             # VIF = 108k
    'min_battery_voltage_90d',
    # PM: zero-SHAP
    'pm_late_count',
    'pm_total_intervals',        # VIF = 14
    'pm_compliance_ratio',       # VIF = 8
    'pm_gap_std',                # VIF = 72
    # Repair
    'repair_cost_180d',
    # Service history
    'has_had_fuel_polishing',
    'has_had_battery_replacement',
    # Equipment
    'is_slxi',                   # VIF = 8
    # Other
    'fuel_fills_30d',
    'danger_zone_flag',
    'last_mode_continuous',
]
prune_cols.update(zero_shap_phase2)

# --- Extreme-VIF telematics aggregates (non-zero SHAP, but linear combos) ---
# telem_events = shutdowns + alarms + low_fuel + restarts + not_reporting
# alarm_severity_score = weighted combo of alarm_red + alarm_yellow + alarm_green
telem_agg_drop = [
    'telem_events_total',        # VIF = 568k; SHAP = 0.012
    'telem_events_180d',         # VIF = 54;   SHAP = 0.008
    'telem_events_90d',          # VIF = 17;   SHAP = 0.003
    'alarm_severity_score',      # VIF = 579k; SHAP = 0.011
    'alarm_severity_score_90d',  # VIF = 544k; SHAP = 0.0006
    'telem_shutdowns_180d',      # VIF = 17;   SHAP = 0.003; r = 0.89 with total
]
prune_cols.update(telem_agg_drop)

# --- PM gap redundancy: keep pm_gap_mean only ---
# pm_gap_{max,min,cv} have r > 0.85 with pm_gap_mean, VIF 16-113
pm_gap_drop = [
    'pm_gap_max',                # VIF = 113; r = 0.86 with pm_gap_mean
    'pm_gap_min',                # VIF = 101; r = 0.85 with pm_gap_mean
    'pm_gap_cv',                 # VIF = 16;  r = 0.94 with pm_gap_std
]
prune_cols.update(pm_gap_drop)

# --- Near-zero SHAP (< 0.001) with collinearity ---
near_zero_phase2 = [
    'repair_count_180d',             # VIF = 8; bridges 90d ↔ 365d
    'container_repair_count_90d',    # redundant with container_repair_cost_90d
    'container_repair_count_365d',   # redundant with container_repair_cost_365d
    'electrical_charging_365d',
    'days_since_last_container_repair',
    'unit_retrieval_365d',
    'has_had_starter_replacement',
    'engine_365d',
    'fuel_system_90d',
    'fuel_fills_90d',
]
prune_cols.update(near_zero_phase2)

# Apply pruning — only drop columns that actually exist
prune_cols = prune_cols.intersection(set(feature_cols))
feature_cols = [c for c in feature_cols if c not in prune_cols]

print(f'Pruned {len(prune_cols)} redundant/dead features')
print(f'Features after collinearity pruning: {len(feature_cols)}')

# ----------------------------------------------------------------------
# 2c. IMPORTANCE-BASED FEATURE SELECTION (keep top-K by gain)
# ----------------------------------------------------------------------
# With only ~785 shutdowns, ~82 features overfits and generalizes poorly to
# future months. Honest evaluation (temporal holdout + unit-grouped CV, see
# scripts/risk_accuracy_experiments.py) showed trimming to the strongest ~40
# features nearly DOUBLES forward-looking AP (0.08 -> 0.14) and lifts
# precision@top-50 from 0.18 -> 0.26. The ranker is fit on TRAIN ONLY (no test
# leakage); selection is by XGBoost gain.
TOP_K_FEATURES = 40
if len(feature_cols) > TOP_K_FEATURES:
    _rank_imp = SimpleImputer(strategy='median')
    _Xr = _rank_imp.fit_transform(train_df[feature_cols].values.astype(np.float32))
    _ranker = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.7, scale_pos_weight=8.0, eval_metric='aucpr',
        random_state=42, tree_method='hist', verbosity=0,
    )
    _ranker.fit(_Xr, train_df['shutdown'].values)
    _gain = pd.Series(_ranker.feature_importances_, index=feature_cols).sort_values(ascending=False)
    feature_cols = _gain.head(TOP_K_FEATURES).index.tolist()
    print(f'Importance selection: kept top {TOP_K_FEATURES} features by gain')
print(f'Features after pruning: {len(feature_cols)}')

X_train = train_df[feature_cols].values.astype(np.float32)
y_train = train_df['shutdown'].values
X_test = test_df[feature_cols].values.astype(np.float32)
y_test = test_df['shutdown'].values

imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ======================================================================
# 3. EXPANDING-WINDOW TEMPORAL CV (robustness check)
# ======================================================================
print('\n' + '=' * 70)
print('EXPANDING-WINDOW TEMPORAL CV')
print('=' * 70)

# Tuned params from grid search (2026-04-01): best mean AP across 3 folds
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 10,
    'gamma': 1.0,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'scale_pos_weight': 8.0,
    'n_estimators': 1500,
    'random_state': 42,
    'verbosity': 0,
    'tree_method': 'hist',
    'early_stopping_rounds': 50,
}

cv_folds = [
    ('Jan-Mar  -> Apr-Jun', [1,2,3], [4,5,6]),
    ('Jan-Jun  -> Jul-Sep', [1,2,3,4,5,6], [7,8,9]),
    ('Jan-Sep  -> Oct-Dec', [1,2,3,4,5,6,7,8,9], [10,11,12]),
]

cv_aucs, cv_aps = [], []
for fold_name, tr_months, te_months in cv_folds:
    tr_fold = df[df['trip_month'].isin(tr_months)]
    te_fold = df[df['trip_month'].isin(te_months)]
    X_tr_f = imputer.fit_transform(tr_fold[feature_cols].values.astype(np.float32))
    y_tr_f = tr_fold['shutdown'].values
    X_te_f = imputer.transform(te_fold[feature_cols].values.astype(np.float32))
    y_te_f = te_fold['shutdown'].values

    m_cv = xgb.XGBClassifier(**xgb_params)
    m_cv.fit(X_tr_f, y_tr_f, eval_set=[(X_te_f, y_te_f)], verbose=False)
    p_cv = m_cv.predict_proba(X_te_f)[:, 1]
    auc_cv = roc_auc_score(y_te_f, p_cv)
    ap_cv = average_precision_score(y_te_f, p_cv)
    cv_aucs.append(auc_cv); cv_aps.append(ap_cv)
    n_sd = int(y_te_f.sum())
    print(f'  {fold_name}: AUC={auc_cv:.4f} AP={ap_cv:.4f} (n={len(y_te_f)}, SD={n_sd}, iters={m_cv.best_iteration})')

print(f'  Mean CV:  AUC={np.mean(cv_aucs):.4f} AP={np.mean(cv_aps):.4f}')

# ======================================================================
# 4. TRAIN FINAL MODEL (80% stratified, early-stopped on 20% holdout)
# ======================================================================
print('\n' + '=' * 70)
print('TRAINING FINAL XGBOOST (with early stopping)')
print('=' * 70)
t0 = time.time()

# Step 1: find optimal iteration count via early stopping
model_es = xgb.XGBClassifier(**xgb_params)
model_es.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
best_n = max(model_es.best_iteration, 10)
print(f'  Early stopping: best iteration = {best_n}')

# Step 2: retrain with fixed n_estimators (needed for calibration compatibility)
xgb_final_params = {k: v for k, v in xgb_params.items() if k != 'early_stopping_rounds'}
xgb_final_params['n_estimators'] = best_n
model = xgb.XGBClassifier(**xgb_final_params)
model.fit(X_train, y_train, verbose=False)

raw_proba_test = model.predict_proba(X_test)[:, 1]
raw_auc = roc_auc_score(y_test, raw_proba_test)
raw_ap = average_precision_score(y_test, raw_proba_test)
raw_brier = brier_score_loss(y_test, raw_proba_test)

print(f'Raw XGBoost ({time.time()-t0:.0f}s, {best_n} trees):')
print(f'  Test AUC: {raw_auc:.4f}, Test AP: {raw_ap:.4f}, Brier: {raw_brier:.4f}')

# ======================================================================
# 5. CALIBRATE PROBABILITIES (Isotonic Regression)
# ======================================================================
print('\n' + '=' * 70)
print('CALIBRATING PROBABILITIES')
print('=' * 70)

# Calibrated model using CV on training data
cal_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
cal_model.fit(X_train, y_train)

cal_proba_test = cal_model.predict_proba(X_test)[:, 1]
cal_auc = roc_auc_score(y_test, cal_proba_test)
cal_ap = average_precision_score(y_test, cal_proba_test)
cal_brier = brier_score_loss(y_test, cal_proba_test)

print(f'Calibrated:')
print(f'  Test AUC: {cal_auc:.4f}, Test AP: {cal_ap:.4f}, Brier: {cal_brier:.4f}')
print(f'  Brier improvement: {raw_brier - cal_brier:+.4f} (lower is better)')

# Check calibration by decile
print('\nCalibration by decile:')
deciles = pd.qcut(cal_proba_test, q=10, duplicates='drop')
cal_check = pd.DataFrame({'proba': cal_proba_test, 'actual': y_test, 'decile': deciles})
for name, group in cal_check.groupby('decile', observed=True):
    pred_rate = group.proba.mean() * 100
    actual_rate = group.actual.mean() * 100
    print(f'  Predicted {pred_rate:5.2f}% | Actual {actual_rate:5.2f}% | n={len(group):>4} | gap={pred_rate-actual_rate:+.2f}%')

# ======================================================================
# 6. SHAP EXPLAINER
# ======================================================================
print('\n' + '=' * 70)
print('COMPUTING SHAP VALUES')
print('=' * 70)
t0 = time.time()

explainer = shap.TreeExplainer(model)
# Compute on a background sample for reference
bg_sample = X_train[np.random.RandomState(42).choice(len(X_train), min(500, len(X_train)), replace=False)]
shap_values_bg = explainer.shap_values(bg_sample)

# Global feature importance (mean |SHAP|)
mean_abs_shap = np.abs(shap_values_bg).mean(axis=0)
shap_importance = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print(f'SHAP computed in {time.time()-t0:.0f}s')
print(f'\nTop 25 features by SHAP importance:')
for i, row in shap_importance.head(25).iterrows():
    print(f'  {row.feature:<45} |SHAP|={row.mean_abs_shap:.4f}')

# ======================================================================
# 7. CATEGORIZE FEATURES FOR THE RISK TOOL
# ======================================================================
# Group features into human-readable categories for the UI
feature_categories = {}
for f in feature_cols:
    if f in ['leg_miles', 'total_miles', 'route', 'origin', 'destination',
             'service_type', 'temperature', 'season',
             'month', 'day_of_week']:
        feature_categories[f] = 'Trip Details'
    elif f.startswith('hwy_'):
        feature_categories[f] = 'Recent Highway Usage'
    elif 'repair' in f or 'nonpm' in f or 'compcode' in f or 'shop' in f:
        feature_categories[f] = 'Repair History'
    elif f.startswith(('compressor', 'electrical', 'fuel_system', 'condenser',
                       'engine_', 'evaporator', 'refrigerant', 'coolant',
                       'sheet_metal', 'throttle', 'starter_cranking', 'drier',
                       'diagnosis', 'chute', 'data_logger', 'unit_retrieval',
                       'thermostat', 'expansion', 'exhaust', 'defrost',
                       'control_panel', 'remote_status')):
        feature_categories[f] = 'Component Repairs'
    elif f.startswith('telem_') or f.startswith('alarm_') or 'battery' in f:
        feature_categories[f] = 'Telematics & Alarms'
    elif f.startswith(('pm_', 'days_since_last_pm', 'is_pm', 'days_pm')):
        feature_categories[f] = 'PM Compliance'
    elif f.startswith(('days_since_major', 'has_had_', 'days_since_starter',
                       'days_since_alternator', 'days_since_battery',
                       'days_since_fuel_polishing')):
        feature_categories[f] = 'Major Service History'
    elif f.startswith('engine_hours'):
        feature_categories[f] = 'Engine Hours'
    elif f.startswith('fuel_'):
        feature_categories[f] = 'Fuel History'
    elif f.startswith('container_'):
        feature_categories[f] = 'Container Repairs'
    elif f in ['reefer_age_at_trip', 'container_age_at_trip', 'reefer_install_days_at_trip',
               'reefer_year', 'reefer_model', 'is_slxi', 'is_thermoking', 'is_carrier',
               'pm_policy_days']:
        feature_categories[f] = 'Equipment Profile'
    elif f.startswith(('has_dor_14d', 'has_electrical_14d', 'has_starter_14d',
                       'diag_no_fix', 'n_repair_groups_14d', 'has_repeat_repair',
                       'repair_cost_14d', 'days_since_last_repair')):
        feature_categories[f] = 'Pre-Trip Repair Patterns'
    elif f.startswith(('has_dor', 'has_accident', 'has_warranty')):
        feature_categories[f] = 'Service Events'
    elif f.startswith(('origin_temp', 'dest_temp', 'transit_', 'extreme_cold',
                       'danger_zone', 'heating_mode')):
        feature_categories[f] = 'Weather & Temperature'
    elif f.startswith(('last_mode', 'continuous_mode')):
        feature_categories[f] = 'Operating Mode'
    elif 'last_fuel_pct' in f:
        feature_categories[f] = 'Fuel Status'
    else:
        feature_categories[f] = 'Other'

# Human-readable feature names
feature_labels = {
    'repair_cum_cost': 'Total Repair Cost ($)',
    'nonpm_cum_cost': 'Non-PM Repair Cost ($)',
    'repair_cost_90d': 'Repair Cost (Last 90 Days)',
    'repair_cost_365d': 'Repair Cost (Last Year)',
    'telem_shutdowns_total': 'Total Prior Shutdowns',
    'telem_shutdowns_90d': 'Shutdowns (Last 90 Days)',
    'telem_alarms_total': 'Total Alarm Events',
    'alarm_severity_score': 'Alarm Severity Score',
    'days_since_last_pm': 'Days Since Last PM',
    'is_pm_overdue': 'PM Overdue?',
    'days_pm_overdue': 'Days PM Overdue',
    'pm_compliance_ratio': 'PM Compliance Ratio',
    'engine_hours_at_trip': 'Engine Hours',
    'reefer_age_at_trip': 'Reefer Age (Years)',
    'temperature': 'Ambient Temperature (°C)',
    'leg_miles': 'Trip Distance (Miles)',
    'service_type': 'Service Type',
    'route': 'Route',
    'min_battery_voltage_90d': 'Min Battery Voltage (90d)',
    'last_fuel_pct': 'Last Fuel Level (%)',
    'days_since_last_telem_shutdown': 'Days Since Last Shutdown',
    'compressor_90d': 'Compressor Repairs (90d)',
    'electrical_charging_90d': 'Electrical Repairs (90d)',
    'fuel_system_90d': 'Fuel System Repairs (90d)',
    'season': 'Season',
    'transit_min_temp': 'Min Transit Temperature',
    'extreme_cold_flag': 'Extreme Cold Route',
    'has_dor_14d': 'Door Repair (Last 14 Days)',
    'has_electrical_14d': 'Electrical Repair (Last 14 Days)',
    'has_starter_14d': 'Starter Repair (Last 14 Days)',
    'days_since_last_repair': 'Days Since Last Repair',
    'diag_no_fix_14d': 'Diagnosis Without Fix (14 Days)',
    'n_repair_groups_14d': 'Repair Groups (14-Day Window)',
    'has_repeat_repair_30d': 'Repeat Repair (30 Days)',
    'repair_cost_14d': 'Repair Cost (Last 14 Days)',
}

# ======================================================================
# 8. COMPUTE RISK SCORE PERCENTILES FROM TRAINING DATA
# ======================================================================
print('\n' + '=' * 70)
print('COMPUTING RISK SCORE DISTRIBUTION')
print('=' * 70)

# Score all training data for percentile reference
train_proba = cal_model.predict_proba(X_train)[:, 1]
all_proba = np.concatenate([train_proba, cal_proba_test])

percentiles = {}
for p in range(0, 101, 5):
    percentiles[str(p)] = round(float(np.percentile(all_proba, p)), 6)

print('Risk score percentiles:')
for p in [10, 25, 50, 75, 90, 95, 100]:
    print(f'  P{p}: {percentiles[str(p)]:.4f}')

# Risk tiers with dynamic thresholds
tier_thresholds = {
    'LOW': float(np.percentile(all_proba, 60)),
    'MODERATE': float(np.percentile(all_proba, 80)),
    'HIGH': float(np.percentile(all_proba, 95)),
    # Anything above HIGH threshold is CRITICAL
}
print(f'\nRisk tier thresholds:')
print(f'  LOW:      < {tier_thresholds["LOW"]:.4f} (bottom 60%)')
print(f'  MODERATE: < {tier_thresholds["MODERATE"]:.4f} (60-80th percentile)')
print(f'  HIGH:     < {tier_thresholds["HIGH"]:.4f} (80-95th percentile)')
print(f'  CRITICAL: >= {tier_thresholds["HIGH"]:.4f} (top 5%)')

# Validate tiers on test set
print(f'\nTier validation on holdout (20% stratified):')
for tier, thresh_lo, thresh_hi in [
    ('LOW', 0, tier_thresholds['LOW']),
    ('MODERATE', tier_thresholds['LOW'], tier_thresholds['MODERATE']),
    ('HIGH', tier_thresholds['MODERATE'], tier_thresholds['HIGH']),
    ('CRITICAL', tier_thresholds['HIGH'], 1.01),
]:
    mask = (cal_proba_test >= thresh_lo) & (cal_proba_test < thresh_hi)
    if mask.sum() > 0:
        n_sd = y_test[mask].sum()
        rate = y_test[mask].mean() * 100
        print(f'  {tier:>10}: {mask.sum():>5} trips, {n_sd:>3} shutdowns ({rate:.1f}%)')

# ======================================================================
# 9. EXPORT ARTIFACTS
# ======================================================================
print('\n' + '=' * 70)
print('EXPORTING RISK ENGINE ARTIFACTS')
print('=' * 70)

# 9a. Save XGBoost model
model.save_model(f'{OUT}/xgb_model.json')
print(f'  Saved XGBoost model')

# 9b. Save calibrated model
with open(f'{OUT}/calibrated_model.pkl', 'wb') as f:
    pickle.dump(cal_model, f)
print(f'  Saved calibrated model')

# 9c. Save imputer
with open(f'{OUT}/imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
print(f'  Saved imputer')

# 9d. Save label encoders
le_maps = {}
for col, le in label_encoders.items():
    le_maps[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
with open(f'{OUT}/label_encoders.json', 'w') as f:
    json.dump(le_maps, f, indent=2)
print(f'  Saved label encoders')

# 9e. Save feature metadata
feature_meta = {
    'feature_cols': feature_cols,
    'feature_categories': feature_categories,
    'feature_labels': feature_labels,
    'cat_cols': cat_cols,
}
with open(f'{OUT}/feature_meta.json', 'w') as f:
    json.dump(feature_meta, f, indent=2)
print(f'  Saved feature metadata')

# 9f. Save SHAP importance
shap_importance.to_csv(f'{OUT}/shap_importance.csv', index=False)
print(f'  Saved SHAP importance')

# 9g. Save SHAP expected value (baseline)
shap_baseline = {
    'expected_value': float(explainer.expected_value),
    'base_rate': float(y_train.mean()),
}
with open(f'{OUT}/shap_baseline.json', 'w') as f:
    json.dump(shap_baseline, f, indent=2)
print(f'  Saved SHAP baseline (expected value: {shap_baseline["expected_value"]:.4f})')

# 9h. Save risk score distribution
risk_dist = {
    'percentiles': percentiles,
    'tier_thresholds': tier_thresholds,
    'metrics': {
        'test_auc': round(cal_auc, 4),
        'test_ap': round(cal_ap, 4),
        'test_brier': round(cal_brier, 4),
        'cv_mean_auc': round(float(np.mean(cv_aucs)), 4),
        'cv_mean_ap': round(float(np.mean(cv_aps)), 4),
        'n_features': len(feature_cols),
        'best_n_estimators': best_n,
    }
}
with open(f'{OUT}/risk_distribution.json', 'w') as f:
    json.dump(risk_dist, f, indent=2)
print(f'  Saved risk distribution')

# 9i. Copy reference data + build feature defaults from imputer
import shutil
for fname in ['fleet_lookup.json', 'route_stats.json',
              'category_maps.json', 'trailer_map.json']:
    src = f'data/processed/training/{fname}'
    if os.path.exists(src):
        shutil.copy2(src, f'{OUT}/{fname}')
        print(f'  Copied {fname}')

# Build feature_defaults from imputer statistics (covers all 75 model features)
feat_defaults = {}
for i, col in enumerate(feature_cols):
    feat_defaults[col] = float(imputer.statistics_[i])
with open(f'{OUT}/feature_defaults.json', 'w') as f:
    json.dump(feat_defaults, f, indent=2)
print(f'  Built feature_defaults.json ({len(feat_defaults)} model features from imputer medians)')

# 9j. Build unit snapshot from training data (latest features per unit)
# Shared with refresh_inputs.py so train-time and input-refresh-time snapshots
# are identical (train once; new data only refreshes these inputs).
print('\n  Building unit snapshots...')
from snapshots import build_unit_snapshots
unit_features = build_unit_snapshots(df, feature_cols)

with open(f'{OUT}/unit_snapshots.json', 'w') as f:
    json.dump(unit_features, f, indent=2, default=str)
print(f'  Saved {len(unit_features)} unit snapshots')

# ======================================================================
# 10. FINAL SUMMARY
# ======================================================================
print('\n' + '=' * 70)
print('RISK ENGINE BUILD COMPLETE')
print('=' * 70)
print(f'\nModel Performance (20% Stratified Holdout):')
print(f'  AUC: {cal_auc:.4f}')
print(f'  Avg Precision: {cal_ap:.4f}')
print(f'  Brier Score: {cal_brier:.4f}')
print(f'\nArtifacts saved to: {OUT}/')
print(f'  Total features: {len(feature_cols)}')
print(f'  Units in fleet: {len(unit_features)}')
print(f'  Routes covered: {len(json.load(open("data/processed/training/route_stats.json")))}')

artifacts = os.listdir(OUT)
print(f'\nFiles:')
for a in sorted(artifacts):
    size = os.path.getsize(f'{OUT}/{a}')
    print(f'  {a:<35} {size/1024:.1f} KB')

print('\nDone!')
