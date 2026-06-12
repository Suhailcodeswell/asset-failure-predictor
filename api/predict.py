"""Vercel serverless function: Asset Failure Predictor dashboard.
Serves fleet-wide risk scores, PM analysis, unit lookup, and trip prediction.
Uses pure-Python XGBoost tree walker + isotonic calibration — no ML runtime dependency."""
import json
import os
import math
from datetime import datetime
from http.server import BaseHTTPRequestHandler

AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "AFPredict2026!")

def check_auth(headers):
    token = headers.get('X-Auth-Token', '')
    return token == AUTH_TOKEN

# ---- Load data files ----
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

with open(os.path.join(DATA_DIR, 'category_maps.json')) as f:
    CATEGORY_MAPS = json.load(f)
# Reverse maps: encoded int -> original string label, for decoding snapshots
CATEGORY_REVERSE = {col: {int(v): k for k, v in m.items()} for col, m in CATEGORY_MAPS.items()}
with open(os.path.join(DATA_DIR, 'feature_defaults.json')) as f:
    FEATURE_DEFAULTS = json.load(f)
with open(os.path.join(DATA_DIR, 'feature_names.json')) as f:
    FEATURE_NAMES = json.load(f)
with open(os.path.join(DATA_DIR, 'fleet_lookup.json')) as f:
    FLEET_LOOKUP = json.load(f)
with open(os.path.join(DATA_DIR, 'unit_snapshots.json')) as f:
    UNIT_SNAPSHOTS = json.load(f)
with open(os.path.join(DATA_DIR, 'route_stats.json')) as f:
    ROUTE_STATS = json.load(f)
with open(os.path.join(DATA_DIR, 'trailer_map.json')) as f:
    TRAILER_MAP = json.load(f)
with open(os.path.join(DATA_DIR, 'model_dump.json')) as f:
    MODEL = json.load(f)
with open(os.path.join(DATA_DIR, 'calibration_table.json')) as f:
    CALIBRATION_TABLE = json.load(f)
with open(os.path.join(DATA_DIR, 'repair_patterns.json')) as f:
    REPAIR_PATTERNS = json.load(f)

# ---- Load CPH (Cost Per Hour) model data ----
_cph_pm_path = os.path.join(DATA_DIR, 'cph_pm_model.json')
_cph_genrep_path = os.path.join(DATA_DIR, 'cph_genrep_model.json')
_cph_fleet_path = os.path.join(DATA_DIR, 'cph_fleet.json')
_cph_summary_path = os.path.join(DATA_DIR, 'cph_summary.json')

CPH_AVAILABLE = os.path.exists(_cph_pm_path)

if CPH_AVAILABLE:
    with open(_cph_pm_path) as f:
        CPH_PM_MODEL = json.load(f)
    with open(_cph_genrep_path) as f:
        CPH_GENREP_MODEL = json.load(f)
    with open(_cph_fleet_path) as f:
        CPH_FLEET = json.load(f)
    with open(_cph_summary_path) as f:
        CPH_SUMMARY = json.load(f)
else:
    CPH_PM_MODEL = CPH_GENREP_MODEL = CPH_FLEET = CPH_SUMMARY = {}

# ---- Load CPM (Cost Per Mile) v2 data (pre-computed from CPM1 models) ----
_cpm_fleet_path = os.path.join(DATA_DIR, 'cpm_fleet.json')
_cpm_summary_path = os.path.join(DATA_DIR, 'cpm_summary.json')

CPM_AVAILABLE = os.path.exists(_cpm_fleet_path)

if CPM_AVAILABLE:
    with open(_cpm_fleet_path) as f:
        CPM_FLEET = json.load(f)
    with open(_cpm_summary_path) as f:
        CPM_SUMMARY = json.load(f)
else:
    CPM_FLEET = []
    CPM_SUMMARY = {}

# ---- Load CPM v2 prediction models (GENREP + PM RF trees) ----
_cpm_genrep_model_path = os.path.join(DATA_DIR, 'cpm_genrep_model.json')
_cpm_pm_model_path = os.path.join(DATA_DIR, 'cpm_pm_model.json')

CPM_MODELS_AVAILABLE = os.path.exists(_cpm_genrep_model_path)

if CPM_MODELS_AVAILABLE:
    with open(_cpm_genrep_model_path) as f:
        CPM_GENREP_MODEL = json.load(f)
    with open(_cpm_pm_model_path) as f:
        CPM_PM_MODEL = json.load(f)
else:
    CPM_GENREP_MODEL = CPM_PM_MODEL = {}

# Load tier thresholds from risk engine (kept in sync with model training)
_risk_dist_path = os.path.join(DATA_DIR, '..', '..', 'data', 'risk_engine', 'risk_distribution.json')
_risk_dist_path = os.path.normpath(_risk_dist_path)
if os.path.exists(_risk_dist_path):
    with open(_risk_dist_path) as f:
        _risk_dist = json.load(f)
    TIER_THRESHOLDS = _risk_dist['tier_thresholds']
else:
    # Fallback: also check webapp/data for a deployed copy
    _rd_webapp = os.path.join(DATA_DIR, 'risk_distribution.json')
    if os.path.exists(_rd_webapp):
        with open(_rd_webapp) as f:
            _risk_dist = json.load(f)
        TIER_THRESHOLDS = _risk_dist['tier_thresholds']
    else:
        # Last resort: use values from risk_distribution.json as of 2026-04-09 (Q4 telem fix)
        TIER_THRESHOLDS = {'LOW': 0.020195303857326508, 'MODERATE': 0.04679778590798378, 'HIGH': 0.12891596555709839}


# ---- Pure-Python XGBoost tree predictor + calibration ----

def _sigmoid(x):
    if x > 500: return 1.0
    if x < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def _predict_tree(tree, features):
    node = 0
    while True:
        left = tree['left_children'][node]
        if left == -1:  # leaf node
            return tree['base_weights'][node]
        feat_idx = tree['split_indices'][node]
        threshold = tree['split_conditions'][node]
        val = features[feat_idx] if feat_idx < len(features) else 0
        if val is None or (isinstance(val, float) and math.isnan(val)):
            node = left if tree['default_left'][node] else tree['right_children'][node]
        elif val < threshold:
            node = left
        else:
            node = tree['right_children'][node]

def _calibrate(raw_prob):
    """Isotonic calibration via lookup table (1001 entries, 0.000 to 1.000)."""
    idx = max(0, min(1000, int(raw_prob * 1000)))
    return CALIBRATION_TABLE[idx]

def model_predict(features):
    tree_sum = sum(_predict_tree(t, features) for t in MODEL['trees'])
    # base_score is stored as a probability; convert to log-odds to match XGBoost internals
    bs = MODEL['base_score']
    base_logodds = math.log(bs / (1 - bs)) if 0 < bs < 1 else 0
    raw_prob = _sigmoid(tree_sum + base_logodds)
    return _calibrate(raw_prob)


# ---- Pure-Python RandomForest tree predictor (CPH models) ----

def _rf_predict_tree(tree, features):
    """Walk a single RF decision tree (exported from sklearn)."""
    node = 0
    while True:
        left = tree['left_children'][node]
        if left == -1:  # leaf
            return tree['values'][node]
        feat_idx = tree['split_features'][node]
        threshold = tree['thresholds'][node]
        val = features[feat_idx] if feat_idx < len(features) else 0
        if val is None or (isinstance(val, float) and math.isnan(val)):
            node = left  # default left for missing
        elif val <= threshold:
            node = left
        else:
            node = tree['right_children'][node]


def _rf_predict(model_data, features):
    """Average predictions across all RF trees."""
    preds = [_rf_predict_tree(t, features) for t in model_data['trees']]
    return sum(preds) / len(preds)


def _encode_cph_features(params, model_data):
    """Build numeric feature vector for a CPH model (handles OneHotEncoding)."""
    num_feats = model_data['numeric_features']
    cat_feats = model_data['categorical_features']
    cat_categories = model_data['cat_categories']

    vector = []
    # Numeric features first (same order as training)
    for nf in num_feats:
        val = params.get(nf, 0)
        try:
            vector.append(float(val) if val is not None else 0.0)
        except (ValueError, TypeError):
            vector.append(0.0)

    # OneHotEncoded categorical features
    for cf in cat_feats:
        cats = cat_categories[cf]
        val = str(params.get(cf, '')).strip()
        for cat in cats:
            vector.append(1.0 if val == cat else 0.0)

    return vector


def _rf_classify_tree(tree, features):
    """Walk a single RF classifier tree, return class probability vector."""
    node = 0
    while True:
        left = tree['left_children'][node]
        if left == -1:
            return tree['values'][node]  # list of class probabilities
        feat_idx = tree['split_features'][node]
        threshold = tree['thresholds'][node]
        val = features[feat_idx] if feat_idx < len(features) else 0
        if val is None or (isinstance(val, float) and math.isnan(val)):
            node = left
        elif val <= threshold:
            node = left
        else:
            node = tree['right_children'][node]


def _rf_classify(model_data, features):
    """Majority vote across RF classifier trees. Returns predicted class."""
    n_classes = model_data.get('n_classes', 2)
    votes = [0.0] * n_classes
    for tree in model_data['trees']:
        probs = _rf_classify_tree(tree, features)
        for c in range(min(n_classes, len(probs))):
            votes[c] += probs[c]
    return 1 if votes[1] > votes[0] else 0


def cph_predict(params):
    """Predict PM and GENREP cost/hr using new two-stage model (scenario planner)."""
    if not CPH_AVAILABLE:
        return {'error': 'CPH models not loaded'}

    # PM prediction (pipeline RF with OHE)
    pm_vector = _encode_cph_features(params, CPH_PM_MODEL)
    pm_pred = max(0, _rf_predict(CPH_PM_MODEL, pm_vector))

    # GENREP two-stage prediction
    genrep = CPH_GENREP_MODEL
    class_feats = genrep.get('class_features', [])
    reg_feats = genrep.get('reg_features', [])

    cls_vec = [float(params.get(f, 0) or 0) for f in class_feats]
    is_pit = _rf_classify(genrep['classifier'], cls_vec)

    reg_vec = [float(params.get(f, 0) or 0) for f in reg_feats]
    if is_pit == 1:
        genrep_pred = max(0, _rf_predict(genrep['reg_pit'], reg_vec))
    else:
        genrep_pred = max(0, _rf_predict(genrep['reg_normal'], reg_vec))

    return {
        'pm_cost_per_hour': round(pm_pred, 6),
        'genrep_cost_per_hour': round(genrep_pred, 6),
        'total_cost_per_hour': round(pm_pred + genrep_pred, 6),
        'is_money_pit': is_pit,
    }


# ---- CPM v2 What-If prediction (GENREP + PM RF with scaler) ----

def _cpm_encode_and_predict(params, model_data):
    """Encode features (impute + scale + OHE) and walk RF trees."""
    vec = []
    # Numeric: impute NaN with median, then StandardScaler
    for i, nf in enumerate(model_data['numeric_features']):
        val = params.get(nf)
        if val is None:
            val = model_data['imputer_medians'][i]
        else:
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = model_data['imputer_medians'][i]
        scaled = (val - model_data['scaler_mean'][i]) / model_data['scaler_scale'][i]
        vec.append(scaled)
    # Categorical: OHE
    for cf in model_data['categorical_features']:
        cats = model_data['cat_categories'][cf]
        val = str(params.get(cf, '')).strip()
        for cat in cats:
            vec.append(1.0 if val == cat else 0.0)
    # Walk RF trees and average
    return _rf_predict(model_data, vec)


def cpm_v2_predict(params):
    """Predict GENREP CPM and PM CPM using exported RF models."""
    if not CPM_MODELS_AVAILABLE:
        return {'error': 'CPM prediction models not loaded'}
    genrep_cpm = max(0, _cpm_encode_and_predict(params, CPM_GENREP_MODEL))
    pm_cpm = max(0, _cpm_encode_and_predict(params, CPM_PM_MODEL))
    total_cpm = genrep_cpm + pm_cpm
    total_miles = float(params.get('total_miles', 0) or 0)
    return {
        'pred_genrep_cpm': round(genrep_cpm, 8),
        'pred_pm_cpm': round(pm_cpm, 8),
        'pred_total_cpm': round(total_cpm, 8),
        'estimated_total_cost': round(total_cpm * total_miles, 2) if total_miles > 0 else 0,
        'total_miles': total_miles,
    }


# ---- Risk tier logic ----

def get_risk_tier(score):
    # Thresholds loaded from risk_distribution.json (percentile-based tiers from model training)
    if score < TIER_THRESHOLDS['LOW']:
        return {'tier': 'LOW', 'color': '#10B981', 'bg': '#ECFDF5'}
    elif score < TIER_THRESHOLDS['MODERATE']:
        return {'tier': 'MODERATE', 'color': '#F59E0B', 'bg': '#FFFBEB'}
    elif score < TIER_THRESHOLDS['HIGH']:
        return {'tier': 'HIGH', 'color': '#EF4444', 'bg': '#FEF2F2'}
    else:
        return {'tier': 'CRITICAL', 'color': '#991B1B', 'bg': '#FEF2F2'}


def encode_categorical(col, value):
    if col in CATEGORY_MAPS:
        return CATEGORY_MAPS[col].get(str(value), -1)
    return -1


# ---- Recommendations engine ----

def derive_recommendations(risk_score, features, fleet_info):
    recs = []
    tier = get_risk_tier(risk_score)['tier']

    days_overdue = features.get('days_pm_overdue', 0)
    is_overdue = features.get('is_pm_overdue', 0)
    days_since_pm = features.get('days_since_last_pm', 0)
    pm_compliance = features.get('pm_compliance_ratio', 1.0)
    nonpm_90d = features.get('nonpm_count_90d', 0)
    nonpm_cost_90d = features.get('nonpm_cost_90d', 0)
    repair_count_90d = features.get('repair_count_90d', 0)
    is_slxi = fleet_info.get('is_slxi', 0)
    compressor_cum = features.get('compressor_cum', 0)
    evaporator_cum = features.get('evaporator_cum', 0)
    diagnosis_cum = features.get('diagnosis_cum', 0)

    # PM overdue
    if is_overdue or days_overdue > 0:
        urgency = 'CRITICAL' if days_overdue > 90 else ('HIGH' if days_overdue > 30 else 'MEDIUM')
        recs.append({
            'action': 'Schedule PM immediately',
            'urgency': urgency,
            'detail': f'{int(days_overdue)} days overdue' if days_overdue > 0 else 'PM overdue',
            'icon': 'wrench',
        })
    elif days_since_pm > 60:
        recs.append({
            'action': 'PM approaching due date',
            'urgency': 'LOW',
            'detail': f'{int(days_since_pm)} days since last PM',
            'icon': 'clock',
        })

    # Low PM compliance
    if pm_compliance < 0.3 and features.get('pm_total_intervals', 0) >= 2:
        recs.append({
            'action': 'Review PM compliance — consistently missed',
            'urgency': 'HIGH',
            'detail': f'{round(pm_compliance * 100)}% compliance rate',
            'icon': 'alert',
        })

    # Frequent recent repairs
    if nonpm_90d >= 3:
        recs.append({
            'action': 'Investigate repeat repairs',
            'urgency': 'HIGH',
            'detail': f'{int(nonpm_90d)} non-PM repairs in last 90 days',
            'icon': 'search',
        })
    elif repair_count_90d >= 4:
        recs.append({
            'action': 'High repair frequency — review root cause',
            'urgency': 'MEDIUM',
            'detail': f'{int(repair_count_90d)} total repairs in 90 days',
            'icon': 'search',
        })

    # SLXi campaign
    if is_slxi and tier in ('HIGH', 'CRITICAL'):
        recs.append({
            'action': 'Check SLXi campaign status',
            'urgency': 'MEDIUM',
            'detail': 'SLXi model — known defect campaign ($5-6K per event)',
            'icon': 'flag',
        })

    # Critical component history
    if compressor_cum >= 2 or evaporator_cum >= 2:
        component = 'compressor' if compressor_cum >= evaporator_cum else 'evaporator'
        count = max(compressor_cum, evaporator_cum)
        recs.append({
            'action': f'Recurring {component} issues — consider overhaul',
            'urgency': 'HIGH' if count >= 3 else 'MEDIUM',
            'detail': f'{int(count)} cumulative {component} repairs',
            'icon': 'alert',
        })

    # High cost
    if nonpm_cost_90d > 3000:
        recs.append({
            'action': 'High recent repair spend — cost review',
            'urgency': 'MEDIUM',
            'detail': f'${int(nonpm_cost_90d):,} in last 90 days',
            'icon': 'dollar',
        })

    # If no specific recs but high risk, general recommendation
    if not recs and tier in ('HIGH', 'CRITICAL'):
        recs.append({
            'action': 'Flag for detailed inspection',
            'urgency': 'HIGH',
            'detail': 'Multiple risk factors combined',
            'icon': 'flag',
        })

    if not recs and tier == 'MODERATE':
        recs.append({
            'action': 'Monitor closely',
            'urgency': 'LOW',
            'detail': 'Moderate risk — watch for deterioration',
            'icon': 'eye',
        })

    return recs[:4]


# ---- Pre-compute fleet risk scores on cold start ----

def _score_unit(unit_id):
    features = dict(FEATURE_DEFAULTS)
    snap = UNIT_SNAPSHOTS.get(unit_id, {})
    # Overlay snapshot values, but skip None (which would overwrite good defaults
    # and become NaN in the model — the Python pipeline uses an imputer for these,
    # but the webapp has no imputer so we keep the median defaults instead).
    for k, v in snap.items():
        if v is not None:
            features[k] = v

    fl = FLEET_LOOKUP.get(unit_id, {})
    features['is_slxi'] = fl.get('is_slxi', 0)
    features['is_thermoking'] = fl.get('is_thermoking', 0)
    features['is_carrier'] = fl.get('is_carrier', 0)
    features['reefer_year'] = fl.get('reefer_year', 0)

    # PM overdue: compute DISPLAY values dynamically from last_pm_date,
    # but keep original snapshot days_since_last_pm for MODEL scoring.
    pm_policy = 120 if fl.get('is_slxi', 0) else 90
    last_pm_str = features.get('last_pm_date', '')
    if last_pm_str and isinstance(last_pm_str, str) and len(last_pm_str) == 10:
        try:
            last_pm = datetime.strptime(last_pm_str, '%Y-%m-%d')
            display_days_since_pm = (datetime.now() - last_pm).days
        except ValueError:
            display_days_since_pm = features.get('days_since_last_pm', 0) or 0
    else:
        display_days_since_pm = features.get('days_since_last_pm', 0) or 0
    display_overdue = max(0, display_days_since_pm - pm_policy)
    # Store display values separately — do NOT overwrite model features
    _pm_display = {
        'days_since_last_pm_current': display_days_since_pm,
        'days_pm_overdue': display_overdue,
        'is_pm_overdue': 1 if display_overdue > 0 else 0,
    }

    # Encode categoricals if not already present
    for cat_col in ['season', 'route', 'origin', 'destination', 'service_type', 'reefer_model']:
        enc_key = cat_col + '_enc'
        if enc_key not in features:
            features[enc_key] = 0

    vector = [float('nan') if features.get(fname) is None else float(features.get(fname, 0))
              for fname in FEATURE_NAMES]
    score = model_predict(vector)
    tier_info = get_risk_tier(score)
    # Use dynamic PM display values for recommendations (not stale snapshot)
    rec_features = dict(features)
    rec_features['days_pm_overdue'] = _pm_display['days_pm_overdue']
    rec_features['is_pm_overdue'] = _pm_display['is_pm_overdue']
    rec_features['days_since_last_pm'] = _pm_display['days_since_last_pm_current']
    recs = derive_recommendations(score, rec_features, fl)

    # Reverse-lookup trailer
    trailer = ''
    for t_id, r_id in TRAILER_MAP.items():
        if r_id == unit_id:
            trailer = t_id
            break

    # Decode last-trip context from snapshot for Fleet Dashboard filters
    snap_route_enc = snap.get('route')
    snap_origin_enc = snap.get('origin')
    snap_dest_enc = snap.get('destination')
    snap_svc_enc = snap.get('service_type')
    snap_route = CATEGORY_REVERSE.get('route', {}).get(int(snap_route_enc), '') if snap_route_enc is not None else ''
    snap_origin = CATEGORY_REVERSE.get('origin', {}).get(int(snap_origin_enc), '') if snap_origin_enc is not None else ''
    snap_dest = CATEGORY_REVERSE.get('destination', {}).get(int(snap_dest_enc), '') if snap_dest_enc is not None else ''
    snap_svc = CATEGORY_REVERSE.get('service_type', {}).get(int(snap_svc_enc), '') if snap_svc_enc is not None else ''

    return {
        'unit': unit_id,
        'trailer': trailer,
        'risk_score': round(float(score), 4),
        'risk_percent': round(float(score) * 100, 2),
        'tier': tier_info['tier'],
        'tier_color': tier_info['color'],
        'tier_bg': tier_info['bg'],
        'reefer_model': fl.get('reefer_model', ''),
        'reefer_year': fl.get('reefer_year', 0),
        'last_route': snap_route,
        'last_origin': snap_origin,
        'last_destination': snap_dest,
        'last_service_type': snap_svc,
        'is_slxi': fl.get('is_slxi', 0),
        'is_thermoking': fl.get('is_thermoking', 0),
        'is_carrier': fl.get('is_carrier', 0),
        'shutdown_count': fl.get('shutdown_count', 0),
        'total_rail_legs': fl.get('total_rail_legs', 0),
        'days_since_last_pm': round(_pm_display['days_since_last_pm_current']),
        'days_pm_overdue': round(_pm_display['days_pm_overdue']),
        'is_pm_overdue': int(_pm_display['is_pm_overdue']),
        'pm_compliance_ratio': round(features.get('pm_compliance_ratio', 0), 2),
        'pm_late_count': int(features.get('pm_late_count', 0)),
        'nonpm_cum_cost': round(features.get('nonpm_cum_cost', 0)),
        'nonpm_cum_count': int(features.get('nonpm_cum_count', 0)),
        'nonpm_count_90d': int(features.get('nonpm_count_90d', 0)),
        'nonpm_cost_90d': round(features.get('nonpm_cost_90d', 0)),
        'repair_count_90d': int(features.get('repair_count_90d', 0)),
        'repair_cum_cost': round(features.get('repair_cum_cost', 0)),
        'compressor_cum': int(features.get('compressor_cum', 0)),
        'evaporator_cum': int(features.get('evaporator_cum', 0)),
        'diagnosis_cum': int(features.get('diagnosis_cum', 0)),
        'electrical_charging_cum': int(features.get('electrical_charging_cum', 0)),
        'fuel_system_cum': int(features.get('fuel_system_cum', 0)),
        'condenser_cum': int(features.get('condenser_cum', 0)),
        'engine_cum': int(features.get('engine_cum', 0)),
        'refrigerant_lines_cum': int(features.get('refrigerant_lines_cum', 0)),
        'starter_cranking_cum': int(features.get('starter_cranking_cum', 0)),
        'defrost_cum': int(features.get('defrost_cum', 0)),
        'unique_compcodes_cum': int(features.get('unique_compcodes_cum', 0)),
        'unique_shops': int(features.get('unique_shops', 0)),
        'recommendations': recs,
    }

# Build fleet risk cache
FLEET_RISK = []
for uid in sorted(FLEET_LOOKUP.keys()):
    if uid in UNIT_SNAPSHOTS:
        FLEET_RISK.append(_score_unit(uid))

# Sort by risk descending
FLEET_RISK.sort(key=lambda x: x['risk_score'], reverse=True)
for i, entry in enumerate(FLEET_RISK):
    entry['rank'] = i + 1

# Pre-compute summary stats
FLEET_SUMMARY = {
    'total_units': len(FLEET_RISK),
    'tier_counts': {},
    'avg_risk': 0,
    'pm_overdue_count': 0,
    'slxi_count': 0,
    'slxi_elevated_plus': 0,
    'avg_risk_by_tier': {},
}

tier_scores = {}
total_risk = 0
for entry in FLEET_RISK:
    tier = entry['tier']
    FLEET_SUMMARY['tier_counts'][tier] = FLEET_SUMMARY['tier_counts'].get(tier, 0) + 1
    if tier not in tier_scores:
        tier_scores[tier] = []
    tier_scores[tier].append(entry['risk_score'])
    total_risk += entry['risk_score']
    if entry['is_pm_overdue']:
        FLEET_SUMMARY['pm_overdue_count'] += 1
    if entry['is_slxi']:
        FLEET_SUMMARY['slxi_count'] += 1
        if tier in ('HIGH', 'CRITICAL'):
            FLEET_SUMMARY['slxi_elevated_plus'] += 1

FLEET_SUMMARY['avg_risk'] = round(total_risk / max(len(FLEET_RISK), 1) * 100, 2)
for tier, scores in tier_scores.items():
    FLEET_SUMMARY['avg_risk_by_tier'][tier] = round(sum(scores) / len(scores) * 100, 2)

# Units at HIGH or CRITICAL risk
FLEET_SUMMARY['elevated_plus'] = sum(
    1 for e in FLEET_RISK if e['tier'] in ('HIGH', 'CRITICAL')
)

# Active units, total shutdowns, and fleet-wide shutdown rate
FLEET_SUMMARY['active_units'] = len(FLEET_RISK)
_total_shutdowns = sum(e['shutdown_count'] for e in FLEET_RISK)
_total_trips = sum(e['total_rail_legs'] for e in FLEET_RISK)
FLEET_SUMMARY['total_shutdowns'] = _total_shutdowns
FLEET_SUMMARY['fleet_sd_rate'] = round(_total_shutdowns / max(_total_trips, 1) * 100, 2)

# PM delay buckets (Ken's favorite feature)
PM_BUCKETS = []
bucket_defs = [
    ('On Time', 0, 0),
    ('1-30 days overdue', 1, 30),
    ('31-60 days overdue', 31, 60),
    ('61-90 days overdue', 61, 90),
    ('91-120 days overdue', 91, 120),
    ('120+ days overdue', 121, 9999),
]
for label, lo, hi in bucket_defs:
    units_in_bucket = [e for e in FLEET_RISK
                       if (lo == 0 and not e['is_pm_overdue']) or
                       (lo > 0 and e['days_pm_overdue'] >= lo and e['days_pm_overdue'] <= hi)]
    if not units_in_bucket:
        shutdown_rate = 0
        avg_risk = 0
    else:
        shutdown_rate = round(sum(e['shutdown_count'] for e in units_in_bucket) /
                              max(sum(e['total_rail_legs'] for e in units_in_bucket), 1) * 100, 2)
        avg_risk = round(sum(e['risk_score'] for e in units_in_bucket) /
                         len(units_in_bucket) * 100, 2)

    PM_BUCKETS.append({
        'label': label,
        'count': len(units_in_bucket),
        'avg_risk': avg_risk,
        'shutdown_rate': shutdown_rate,
    })


# ---- Pre-compute model analysis ----

def _build_model_analysis():
    COMP_KEYS = [
        ('compressor_cum', 'Compressor'), ('evaporator_cum', 'Evaporator'),
        ('diagnosis_cum', 'Diagnosis'), ('electrical_charging_cum', 'Electrical'),
        ('fuel_system_cum', 'Fuel System'), ('starter_cranking_cum', 'Starter/Cranking'),
        ('condenser_cum', 'Condenser'), ('engine_cum', 'Engine'),
    ]

    by_model = {}
    for e in FLEET_RISK:
        m = e['reefer_model'] or 'Unknown'
        if m == 'nan':
            m = 'Unknown'
        by_model.setdefault(m, []).append(e)

    # Get snapshot data for component details
    def snap_val(unit_id, key):
        return UNIT_SNAPSHOTS.get(unit_id, {}).get(key, 0) or 0

    models = []
    for model, units in sorted(by_model.items(), key=lambda x: -len(x[1])):
        if len(units) < 2:
            continue
        n = len(units)
        sd = sum(u['shutdown_count'] for u in units)
        trips = sum(u['total_rail_legs'] for u in units)
        sd_rate = round(sd / max(trips, 1) * 100, 2)
        avg_risk = round(sum(u['risk_score'] for u in units) / n * 100, 2)
        avg_repair = round(sum(u['nonpm_cum_cost'] for u in units) / n)
        avg_total_repair = round(sum(u['repair_cum_cost'] for u in units) / n)
        cost_per_trip = round(sum(u['nonpm_cum_cost'] for u in units) / max(trips, 1), 2)
        pm_overdue = sum(1 for u in units if u['is_pm_overdue'])
        elevated = sum(1 for u in units if u['tier'] in ('HIGH', 'CRITICAL'))
        tiers = {}
        for t in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']:
            tiers[t] = sum(1 for u in units if u['tier'] == t)

        # Component averages from snapshots
        components = {}
        for key, label in COMP_KEYS:
            avg = round(sum(snap_val(u['unit'], key) for u in units) / n, 2)
            components[label] = avg

        # Years present
        years = sorted(set(u['reefer_year'] for u in units if u['reefer_year'] > 0))

        # Year cohorts
        cohorts = []
        by_year = {}
        for u in units:
            y = u['reefer_year']
            if y > 0:
                by_year.setdefault(y, []).append(u)
        for year, yunits in sorted(by_year.items(), reverse=True):
            if len(yunits) < 2:
                continue
            yn = len(yunits)
            ysd = sum(u['shutdown_count'] for u in yunits)
            ytrips = sum(u['total_rail_legs'] for u in yunits)
            ysd_rate = round(ysd / max(ytrips, 1) * 100, 2)
            yavg_risk = round(sum(u['risk_score'] for u in yunits) / yn * 100, 2)
            yavg_repair = round(sum(u['nonpm_cum_cost'] for u in yunits) / yn)
            ycost_per_trip = round(sum(u['nonpm_cum_cost'] for u in yunits) / max(ytrips, 1), 2)
            yelev = sum(1 for u in yunits if u['tier'] in ('HIGH', 'CRITICAL'))
            ycomps = {}
            for key, label in COMP_KEYS:
                ycomps[label] = round(sum(snap_val(u['unit'], key) for u in yunits) / yn, 2)
            cohorts.append({
                'year': year, 'age': datetime.now().year - year, 'count': yn,
                'sd_rate': ysd_rate, 'avg_risk': yavg_risk,
                'avg_repair': yavg_repair, 'cost_per_trip': ycost_per_trip,
                'elevated': yelev,
                'components': ycomps,
            })

        models.append({
            'model': model, 'count': n, 'shutdowns': sd, 'trips': trips,
            'sd_rate': sd_rate, 'avg_risk': avg_risk,
            'avg_repair': avg_repair, 'avg_total_repair': avg_total_repair,
            'cost_per_trip': cost_per_trip,
            'pm_overdue': pm_overdue, 'pm_overdue_pct': round(pm_overdue / n * 100),
            'elevated': elevated, 'elevated_pct': round(elevated / n * 100),
            'tiers': tiers, 'components': components,
            'years': years, 'cohorts': cohorts,
        })

    # Worst cohorts across all models
    worst_cohorts = []
    for m in models:
        for c in m['cohorts']:
            worst_cohorts.append({**c, 'model': m['model']})
    worst_cohorts.sort(key=lambda x: -x['sd_rate'])

    # Recommendations per model
    recs = {}
    for m in models:
        mr = []
        if m['sd_rate'] > 3:
            top_comp = max(m['components'].items(), key=lambda x: x[1])
            mr.append({'action': f"Preemptive {top_comp[0].lower()} inspection/replacement",
                        'urgency': 'CRITICAL', 'detail': f"{m['sd_rate']}% shutdown rate, {top_comp[0]} avg {top_comp[1]}/unit"})
        if m['sd_rate'] > 2:
            mr.append({'action': 'Accelerate PM schedule for this model',
                        'urgency': 'HIGH', 'detail': f"{m['pm_overdue_pct']}% of units PM overdue"})
        if m['elevated_pct'] > 20:
            mr.append({'action': 'Flag elevated-risk units for pre-trip inspection',
                        'urgency': 'HIGH', 'detail': f"{m['elevated']} units ({m['elevated_pct']}%) at elevated+ risk"})
        # Check for age-driven risk in cohorts
        for c in m['cohorts']:
            if c['sd_rate'] > 2.5 and c['age'] >= 10:
                mr.append({'action': f"Evaluate phase-out for {c['year']} cohort (age {c['age']}y)",
                            'urgency': 'MEDIUM', 'detail': f"{c['count']} units, {c['sd_rate']}% shutdown rate"})
            if c['sd_rate'] > 3 and c['age'] < 10:
                mr.append({'action': f"Investigate defect pattern in {c['year']} cohort",
                            'urgency': 'HIGH', 'detail': f"Age {c['age']}y but {c['sd_rate']}% shutdown rate — not age-driven"})
        # Component-specific
        if m['components'].get('Compressor', 0) > 1:
            mr.append({'action': 'Compressor preemptive replacement program',
                        'urgency': 'HIGH', 'detail': f"Avg {m['components']['Compressor']} repairs/unit (fleet-high)"})
        if m['components'].get('Fuel System', 0) > 0.8:
            mr.append({'action': 'Fuel system preventive maintenance',
                        'urgency': 'MEDIUM', 'detail': f"Avg {m['components']['Fuel System']} repairs/unit"})
        if m['components'].get('Evaporator', 0) > 0.7:
            mr.append({'action': 'Evaporator inspection program',
                        'urgency': 'HIGH', 'detail': f"Avg {m['components']['Evaporator']} repairs/unit"})
        if not mr:
            mr.append({'action': 'Maintain current maintenance schedule',
                        'urgency': 'LOW', 'detail': f"Model performing within acceptable range ({m['sd_rate']}% SD rate)"})
        recs[m['model']] = mr[:5]

    return {
        'models': models,
        'worst_cohorts': worst_cohorts[:15],
        'recommendations': recs,
    }

MODEL_ANALYSIS = _build_model_analysis()


# ---- Make × Model × Year × Season cohort grid ----
# Re-score every unit under each of the 4 seasons, then pivot.

def _make_label(fl):
    """Derive Make from is_* flags. Carrier > SLXi > ThermoKing > Other priority."""
    if fl.get('is_carrier'):
        return 'Carrier'
    if fl.get('is_slxi'):
        return 'SLXi'
    if fl.get('is_thermoking'):
        return 'ThermoKing'
    return 'Other'

# Representative month per season (used when re-scoring season-swapped feature vectors)
_SEASON_MONTHS = {0: 10, 1: 4, 2: 7, 3: 1}  # Fall=Oct, Spring=Apr, Summer=Jul, Winter=Jan
_SEASON_NAMES = {0: 'Fall', 1: 'Spring', 2: 'Summer', 3: 'Winter'}
SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Fall']


def _build_unit_season_grid():
    """For each unit, score risk under each of the 4 seasons.

    Keeps unit's other features unchanged (temperature, route, etc.). The seasonal
    effect surfaces through the model's `season` + `month` features. This is
    approximate (temperatures still reflect the snapshot's actual season) but
    captures the model's learned seasonal sensitivity per unit.
    """
    season_enc_idx = FEATURE_NAMES.index('season') if 'season' in FEATURE_NAMES else None
    month_idx = FEATURE_NAMES.index('month') if 'month' in FEATURE_NAMES else None

    rows = []  # list of {unit, make, model, year, season, risk}
    for uid in sorted(FLEET_LOOKUP.keys()):
        if uid not in UNIT_SNAPSHOTS:
            continue
        snap = UNIT_SNAPSHOTS[uid]
        fl = FLEET_LOOKUP[uid]

        # Build base feature vector (same as _score_unit)
        features = dict(FEATURE_DEFAULTS)
        for k, v in snap.items():
            if v is not None:
                features[k] = v
        features['is_slxi'] = fl.get('is_slxi', 0)
        features['is_thermoking'] = fl.get('is_thermoking', 0)
        features['is_carrier'] = fl.get('is_carrier', 0)
        features['reefer_year'] = fl.get('reefer_year', 0)
        for cat_col in ['season', 'route', 'origin', 'destination', 'service_type', 'reefer_model']:
            enc_key = cat_col + '_enc'
            if enc_key not in features:
                features[enc_key] = 0

        base_vec = [float('nan') if features.get(f) is None else float(features.get(f, 0))
                    for f in FEATURE_NAMES]

        make = _make_label(fl)
        model = fl.get('reefer_model', '') or 'Unknown'
        year = int(fl.get('reefer_year', 0) or 0)

        for season_code, season_name in _SEASON_NAMES.items():
            vec = list(base_vec)
            if season_enc_idx is not None:
                vec[season_enc_idx] = float(season_code)
            if month_idx is not None:
                vec[month_idx] = float(_SEASON_MONTHS[season_code])
            score = model_predict(vec)
            rows.append({
                'unit': uid,
                'make': make,
                'model': model,
                'year': year,
                'season': season_name,
                'risk': float(score),
            })
    return rows


_UNIT_SEASON_GRID = _build_unit_season_grid()


def _build_model_grid():
    """Aggregate UNIT_SEASON_GRID into (make, model, year, season) cohort summaries."""
    bucket = {}  # (make, model, year, season) -> {risks: [], unit_count: set}
    for r in _UNIT_SEASON_GRID:
        key = (r['make'], r['model'], r['year'], r['season'])
        b = bucket.setdefault(key, {'risks': [], 'units': set()})
        b['risks'].append(r['risk'])
        b['units'].add(r['unit'])

    cohorts = []
    for (make, model, year, season), b in bucket.items():
        n = len(b['units'])
        if not n:
            continue
        risks = b['risks']
        avg_risk = sum(risks) / len(risks) * 100
        cohorts.append({
            'make': make,
            'model': model,
            'year': year,
            'season': season,
            'units': n,
            'avg_risk': round(avg_risk, 2),
        })

    # Distinct dimension values for client-side filters
    makes = sorted({c['make'] for c in cohorts})
    models = sorted({c['model'] for c in cohorts})
    years = sorted({c['year'] for c in cohorts if c['year']}, reverse=True)
    seasons = SEASON_ORDER

    # Fleet-wide baselines per season for delta indicators
    season_baseline = {}
    for s in seasons:
        season_rows = [r['risk'] for r in _UNIT_SEASON_GRID if r['season'] == s]
        season_baseline[s] = round(sum(season_rows) / len(season_rows) * 100, 2) if season_rows else 0

    return {
        'cohorts': cohorts,
        'makes': makes,
        'models': models,
        'years': years,
        'seasons': seasons,
        'season_baseline': season_baseline,
        'n_units_scored': len({r['unit'] for r in _UNIT_SEASON_GRID}),
    }


MODEL_GRID = _build_model_grid()


# ---- Trip-level prediction (existing) ----

def build_feature_vector(params):
    features = dict(FEATURE_DEFAULTS)
    unit_id = params.get('reefer_unit', '')
    if unit_id and unit_id in UNIT_SNAPSHOTS:
        for k, v in UNIT_SNAPSHOTS[unit_id].items():
            if v is not None:
                features[k] = v

    direct_numeric = [
        'leg_miles', 'total_miles', 'temperature', 'month', 'trip_duration_days',
        'days_since_last_pm', 'pm_late_count', 'pm_compliance_ratio',
        'nonpm_cum_cost', 'nonpm_cum_count', 'repair_count_90d',
        'unique_compcodes_cum', 'unique_shops', 'hwy_gap_days', 'hwy_trip_count',
        'evaporator_cum', 'compressor_cum', 'diagnosis_cum',
    ]
    for key in direct_numeric:
        if key in params and params[key] is not None and str(params[key]).strip() != '':
            try:
                features[key] = float(params[key])
            except (ValueError, TypeError):
                pass

    # Map form field names to model feature names where they differ
    if 'nonpm_cum_cost' in params and str(params.get('nonpm_cum_cost', '')).strip() != '':
        try:
            features['repair_cum_cost'] = float(params['nonpm_cum_cost'])
        except (ValueError, TypeError):
            pass
    if 'nonpm_cum_count' in params and str(params.get('nonpm_cum_count', '')).strip() != '':
        try:
            features['repair_count_365d'] = float(params['nonpm_cum_count'])
        except (ValueError, TypeError):
            pass

    if unit_id and unit_id in FLEET_LOOKUP:
        fl = FLEET_LOOKUP[unit_id]
        features['is_slxi'] = fl['is_slxi']
        features['is_thermoking'] = fl['is_thermoking']
        features['is_carrier'] = fl['is_carrier']
        features['reefer_year'] = fl['reefer_year']
        trip_month = float(params.get('month', 6))
        trip_year = datetime.now().year + (trip_month - 1) / 12
        if fl['reefer_year'] > 0:
            features['reefer_age_at_trip'] = trip_year - fl['reefer_year']
        if fl['container_year'] > 0:
            features['container_age_at_trip'] = trip_year - fl['container_year']

    features['season'] = encode_categorical('season', params.get('season', 'Winter'))
    features['route'] = encode_categorical('route', params.get('route', ''))
    features['origin'] = encode_categorical('origin', params.get('origin', ''))
    features['destination'] = encode_categorical('destination', params.get('destination', ''))
    features['service_type'] = encode_categorical('service_type', params.get('service_type', ''))

    reefer_model = ''
    if unit_id and unit_id in FLEET_LOOKUP:
        reefer_model = FLEET_LOOKUP[unit_id].get('reefer_model', '')
    features['reefer_model'] = encode_categorical('reefer_model', reefer_model)
    pm_policy = 120 if features.get('is_slxi', 0) else 90
    features['pm_policy_days'] = pm_policy
    days_since_pm = features.get('days_since_last_pm', 0) or 0
    overdue_days = max(0, days_since_pm - pm_policy)
    features['days_pm_overdue'] = overdue_days
    features['is_pm_overdue'] = 1 if overdue_days > 0 else 0

    trip_month_int = max(1, min(12, int(float(params.get('month', datetime.now().month)))))
    features['day_of_week'] = datetime(datetime.now().year, trip_month_int, 15).weekday()

    vector = [float('nan') if features.get(fname) is None else float(features.get(fname, 0))
              for fname in FEATURE_NAMES]
    return vector, features


def trip_predict(params):
    vector, features = build_feature_vector(params)
    score = model_predict(vector)
    tier_info = get_risk_tier(score)
    fl = FLEET_LOOKUP.get(params.get('reefer_unit', ''), {})
    recs = derive_recommendations(score, features, fl)

    risk_factors = []
    key_features = {
        'nonpm_cum_cost': ('Cumulative repair cost', '$'),
        'nonpm_cum_count': ('Total non-PM repairs', ''),
        'unique_compcodes_cum': ('Repair variety (unique codes)', ''),
        'unique_shops': ('Shops visited', ''),
        'pm_late_count': ('Late PM count', ''),
        'pm_compliance_ratio': ('PM compliance ratio', '%'),
        'days_since_last_pm': ('Days since last PM', 'd'),
        'diagnosis_cum': ('Diagnosis events', ''),
        'evaporator_cum': ('Evaporator repairs', ''),
        'compressor_cum': ('Compressor repairs', ''),
        'hwy_gap_days': ('Days since last rail', 'd'),
    }
    for feat, (label, unit) in key_features.items():
        val = features.get(feat, 0)
        default = FEATURE_DEFAULTS.get(feat, 0)
        if default > 0 and val > default * 1.5:
            display_val = round(val, 1)
            display_default = round(default, 1)
            if feat == 'pm_compliance_ratio':
                display_val = round(val * 100, 1)
                display_default = round(default * 100, 1)
            risk_factors.append({
                'factor': label,
                'value': display_val,
                'fleet_median': display_default,
                'unit': unit,
                'status': 'above_average'
            })
        elif feat == 'pm_compliance_ratio' and val < 0.3 and default > 0:
            risk_factors.append({
                'factor': label,
                'value': round(val * 100, 1),
                'fleet_median': round(default * 100, 1),
                'unit': '%',
                'status': 'below_average'
            })

    risk_factors.sort(key=lambda x: abs(x['value'] - x['fleet_median']), reverse=True)

    return {
        'risk_score': round(float(score), 4),
        'risk_percent': round(float(score) * 100, 2),
        'tier': tier_info['tier'],
        'tier_color': tier_info['color'],
        'tier_bg': tier_info['bg'],
        'risk_factors': risk_factors[:5],
        'recommendations': recs,
    }


# ---- Pilot validation snapshots ----

SNAPSHOTS_DIR = os.path.join(DATA_DIR, 'snapshots')

def _snapshot_summary(snap):
    """Compact summary used in /api/snapshots listing — keeps payload small."""
    fleet = snap.get('fleet') or []
    n = len(fleet)
    tiers = {'LOW': 0, 'MODERATE': 0, 'HIGH': 0, 'CRITICAL': 0}
    risk_sum = 0.0
    shutdowns = 0
    trips = 0
    for u in fleet:
        t = u.get('tier')
        if t in tiers:
            tiers[t] += 1
        risk_sum += float(u.get('risk_score') or 0)
        shutdowns += int(u.get('shutdown_count') or 0)
        trips += int(u.get('total_rail_legs') or 0)
    return {
        'id': snap.get('id'),
        'label': snap.get('label', ''),
        'captured_at': snap.get('captured_at'),
        'notes': snap.get('notes', ''),
        'n_units': n,
        'avg_risk': round(risk_sum / n * 100, 2) if n else 0,
        'shutdowns': shutdowns,
        'trips': trips,
        'tier_counts': tiers,
    }

def list_pilot_snapshots():
    """List all snapshot summaries available in webapp/data/snapshots/."""
    if not os.path.isdir(SNAPSHOTS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(SNAPSHOTS_DIR)):
        if not name.endswith('.json'):
            continue
        path = os.path.join(SNAPSHOTS_DIR, name)
        try:
            with open(path) as f:
                snap = json.load(f)
        except (OSError, ValueError):
            continue
        if not snap.get('id'):
            snap['id'] = name[:-5]
        out.append(_snapshot_summary(snap))
    out.sort(key=lambda s: s.get('captured_at') or '', reverse=True)
    return out

def read_pilot_snapshot(snap_id):
    """Read a full snapshot by id. Returns None if not found."""
    if not snap_id or '/' in snap_id or '..' in snap_id:
        return None
    path = os.path.join(SNAPSHOTS_DIR, snap_id + '.json')
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ---- HTTP Handler ----

class handler(BaseHTTPRequestHandler):
    def _handle_auth(self):
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        password = qs.get('password', [''])[0]
        if password == AUTH_TOKEN:
            self._json_response(200, {'authenticated': True, 'token': AUTH_TOKEN})
        else:
            self._json_response(401, {'authenticated': False})

    def _json_response(self, status, data):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Auth-Token')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _reject_auth(self):
        self._json_response(401, {'error': 'unauthorized'})

    def do_POST(self):
        if not check_auth(self.headers):
            return self._reject_auth()
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        params = json.loads(body) if body else {}

        path = self.path.split('?')[0]
        if '/cph-predict' in path:
            result = cph_predict(params)
        elif '/cpm-predict' in path:
            result = cpm_v2_predict(params)
        else:
            result = trip_predict(params)
        self._json_response(200, result)

    def do_GET(self):
        path = self.path.split('?')[0]
        if '/auth' in path:
            return self._handle_auth()
        if not check_auth(self.headers):
            return self._reject_auth()

        if path.startswith('/api/'):
            endpoint = path[4:]
        else:
            endpoint = path

        # Fleet risk — the main operational endpoint
        if endpoint in ('/fleet-risk', '/api/fleet-risk'):
            data = FLEET_RISK
        elif endpoint in ('/fleet-summary', '/api/fleet-summary'):
            data = FLEET_SUMMARY
        elif endpoint in ('/pm-analysis', '/api/pm-analysis'):
            data = PM_BUCKETS
        elif endpoint in ('/model-analysis', '/api/model-analysis'):
            data = MODEL_ANALYSIS
        elif endpoint in ('/model-grid', '/api/model-grid'):
            data = MODEL_GRID
        elif endpoint in ('/repair-patterns', '/api/repair-patterns'):
            data = REPAIR_PATTERNS
        elif endpoint in ('/units', '/api/units'):
            data = sorted(FLEET_LOOKUP.keys())
        elif endpoint in ('/routes', '/api/routes'):
            data = ROUTE_STATS
        elif endpoint in ('/fleet', '/api/fleet'):
            data = FLEET_LOOKUP
        elif endpoint in ('/trailers', '/api/trailers'):
            data = TRAILER_MAP
        # ---- CPH v2 endpoints (reefer_predictor_deploy) ----
        elif endpoint in ('/cph-fleet', '/api/cph-fleet'):
            data = CPH_FLEET if CPH_AVAILABLE else {'error': 'CPH data not available'}
        elif endpoint in ('/cph-summary', '/api/cph-summary'):
            data = CPH_SUMMARY if CPH_AVAILABLE else {'error': 'CPH data not available'}
        elif endpoint in ('/cph-models', '/api/cph-models'):
            # Summary has dropdown options for scenario planner
            data = CPH_SUMMARY if CPH_AVAILABLE else {'error': 'CPH data not available'}
        # ---- CPM v2 endpoints (pre-computed from CPM1 models) ----
        elif endpoint in ('/cpm-fleet', '/api/cpm-fleet'):
            data = CPM_FLEET if CPM_AVAILABLE else {'error': 'CPM data not available'}
        elif endpoint in ('/cpm-summary', '/api/cpm-summary'):
            data = CPM_SUMMARY if CPM_AVAILABLE else {'error': 'CPM data not available'}
        elif '/unit/' in path:
            unit_id = path.split('/unit/')[-1]
            # Find pre-computed risk entry
            risk_entry = next((e for e in FLEET_RISK if e['unit'] == unit_id), None)
            data = {
                'info': FLEET_LOOKUP.get(unit_id, {}),
                'snapshot': UNIT_SNAPSHOTS.get(unit_id, {}),
                'risk': risk_entry,
            }
        # ---- Pilot validation snapshots ----
        elif endpoint in ('/snapshots', '/api/snapshots'):
            data = list_pilot_snapshots()
        elif '/snapshot/' in path:
            snap_id = path.split('/snapshot/')[-1].rstrip('/')
            snap = read_pilot_snapshot(snap_id)
            if snap is None:
                self._json_response(404, {'error': 'snapshot not found', 'id': snap_id})
                return
            data = snap
        elif endpoint in ('/', '/api', '/api/', ''):
            data = {'status': 'ok', 'units': len(FLEET_RISK)}
        else:
            self._json_response(404, {'error': 'not found', 'path': path})
            return

        self._json_response(200, data)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Auth-Token')
        self.end_headers()
