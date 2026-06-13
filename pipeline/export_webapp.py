"""
Export Risk Engine Artifacts → Webapp JSON Files
=================================================
Converts the model and reference data in models/risk_engine/ into the
JSON files that webapp/api/predict.py loads at runtime.

Run after build_risk_engine.py to update the webapp deployment.

Output: data/ with all required JSON files.
"""
import json
import os
import pickle
import shutil
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

ENGINE = 'models/risk_engine'
WEBAPP = 'data'
os.makedirs(WEBAPP, exist_ok=True)

print('=' * 70)
print('EXPORTING RISK ENGINE → WEBAPP')
print('=' * 70)

# ======================================================================
# 1. MODEL DUMP — Convert native XGBoost JSON to webapp tree format
# ======================================================================
print('\n1. Building model_dump.json ...')

with open(f'{ENGINE}/xgb_model.json') as f:
    raw = json.load(f)

learner = raw['learner']
gb_model = learner['gradient_booster']['model']
params = learner['learner_model_param']

# base_score may be stored as "[3.3789536E-1]" — strip brackets
_bs = str(params.get('base_score', '0.5')).strip('[]')
base_score = float(_bs)
n_features = int(params.get('num_feature', 0))
native_trees = gb_model['trees']
n_trees = len(native_trees)

# Extract the 6 arrays the webapp tree walker needs per tree
trees = []
for t in native_trees:
    trees.append({
        'split_indices': t['split_indices'],
        'split_conditions': t['split_conditions'],
        'left_children': t['left_children'],
        'right_children': t['right_children'],
        'base_weights': t['base_weights'],
        'default_left': t['default_left'],
    })

model_dump = {
    'format': 'xgboost',
    'base_score': base_score,
    'n_features': n_features,
    'n_trees': n_trees,
    'trees': trees,
}

with open(f'{WEBAPP}/model_dump.json', 'w') as f:
    json.dump(model_dump, f)
print(f'   {n_trees} trees, {n_features} features')

# ======================================================================
# 2. CALIBRATION TABLE — Isotonic regression lookup (1001 entries)
# ======================================================================
print('2. Building calibration_table.json ...')

with open(f'{ENGINE}/calibrated_model.pkl', 'rb') as f:
    cal_model = pickle.load(f)

# The CalibratedClassifierCV wraps calibrators; extract the isotonic one.
# Structure: cal_model.calibrated_classifiers_[0].calibrators[0] (for binary)
calibrators = cal_model.calibrated_classifiers_
# Average across CV folds to get a single calibration mapping
grid = np.linspace(0, 1, 1001)
cal_sums = np.zeros(1001)
for cc in calibrators:
    iso = cc.calibrators[0]  # isotonic calibrator for class 1
    cal_sums += iso.predict(grid)
cal_table = (cal_sums / len(calibrators)).tolist()

with open(f'{WEBAPP}/calibration_table.json', 'w') as f:
    json.dump(cal_table, f)
print(f'   {len(cal_table)} entries (0.000 to 1.000)')

# ======================================================================
# 3. FEATURE NAMES — Ordered list matching model input columns
# ======================================================================
print('3. Building feature_names.json ...')

with open(f'{ENGINE}/feature_meta.json') as f:
    meta = json.load(f)
feature_cols = meta['feature_cols']

with open(f'{WEBAPP}/feature_names.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)
print(f'   {len(feature_cols)} features')

# ======================================================================
# 4. FEATURE DEFAULTS — Median values for imputation
# ======================================================================
print('4. Building feature_defaults.json ...')

with open(f'{ENGINE}/imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

# Map feature names to their median imputation values
defaults = {}
for i, col in enumerate(feature_cols):
    val = float(imputer.statistics_[i])
    defaults[col] = val

with open(f'{WEBAPP}/feature_defaults.json', 'w') as f:
    json.dump(defaults, f, indent=2)
print(f'   {len(defaults)} defaults')

# ======================================================================
# 5. CATEGORY MAPS — Label encoder mappings for categorical features
# ======================================================================
print('5. Building category_maps.json ...')

with open(f'{ENGINE}/label_encoders.json') as f:
    le_maps = json.load(f)

# Convert to {col: {label: int}} — already in this format from build_risk_engine
with open(f'{WEBAPP}/category_maps.json', 'w') as f:
    json.dump(le_maps, f, indent=2)
print(f'   {len(le_maps)} categorical columns: {list(le_maps.keys())}')

# ======================================================================
# 6. DIRECT COPIES — Files that transfer as-is
# ======================================================================
print('6. Copying reference files ...')

copy_files = [
    'unit_snapshots.json',
    'fleet_lookup.json',
    'route_stats.json',
    'risk_distribution.json',
]

# Also copy from data/processed/training/ if present (these may not be in risk_engine/)
training_files = ['trailer_map.json']

for fname in copy_files:
    src = f'{ENGINE}/{fname}'
    if os.path.exists(src):
        shutil.copy2(src, f'{WEBAPP}/{fname}')
        size_kb = os.path.getsize(src) / 1024
        print(f'   {fname} ({size_kb:.1f} KB)')
    else:
        print(f'   WARNING: {src} not found — skipping')

for fname in training_files:
    src = f'data/processed/training/{fname}'
    if os.path.exists(src):
        shutil.copy2(src, f'{WEBAPP}/{fname}')
        size_kb = os.path.getsize(src) / 1024
        print(f'   {fname} ({size_kb:.1f} KB) [from data/processed/training/]')
    else:
        print(f'   WARNING: {src} not found — skipping')

# ======================================================================
# 7. VERIFY
# ======================================================================
print('\n' + '=' * 70)
print('VERIFICATION')
print('=' * 70)

required = [
    'model_dump.json', 'calibration_table.json', 'feature_names.json',
    'feature_defaults.json', 'category_maps.json', 'unit_snapshots.json',
    'fleet_lookup.json', 'route_stats.json', 'trailer_map.json',
    'risk_distribution.json',
]

all_ok = True
for fname in required:
    path = f'{WEBAPP}/{fname}'
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f'  OK  {fname:<30} {size_kb:>8.1f} KB')
    else:
        print(f'  MISSING  {fname}')
        all_ok = False

# Check repair_patterns.json separately (generated by repair_pattern_analysis.py)
rp = f'{WEBAPP}/repair_patterns.json'
if os.path.exists(rp):
    size_kb = os.path.getsize(rp) / 1024
    print(f'  OK  {"repair_patterns.json":<30} {size_kb:>8.1f} KB')
else:
    print(f'  NOTE  repair_patterns.json not yet generated (run repair_pattern_analysis.py)')

print(f'\n{"ALL WEBAPP FILES READY" if all_ok else "SOME FILES MISSING — check above"}')
print('=' * 70)
