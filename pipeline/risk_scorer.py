"""
Reefer Risk Scorer — Deployable Scoring Module

Usage:
    from risk_scorer import RiskScorer
    scorer = RiskScorer('models/risk_engine')
    result = scorer.score(unit_id='R8448', trip={
        'route': 'CN TOR-CN CGY',
        'origin': 'TOR',
        'destination': 'CGY',
        'service_type': 'FRZ',
        'temperature': -15,
        'leg_miles': 2127,
        'season': 'Winter',
        'month': 1,
    })
    print(result)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import json, pickle, os


class RiskScorer:
    def __init__(self, engine_dir):
        self.dir = engine_dir

        # Load model
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(f'{engine_dir}/xgb_model.json')

        with open(f'{engine_dir}/calibrated_model.pkl', 'rb') as f:
            self.cal_model = pickle.load(f)

        with open(f'{engine_dir}/imputer.pkl', 'rb') as f:
            self.imputer = pickle.load(f)

        # Load reference data
        with open(f'{engine_dir}/label_encoders.json') as f:
            self.label_encoders = json.load(f)

        with open(f'{engine_dir}/feature_meta.json') as f:
            meta = json.load(f)
            self.feature_cols = meta['feature_cols']
            self.feature_categories = meta['feature_categories']
            self.feature_labels = meta['feature_labels']
            self.cat_cols = meta['cat_cols']

        with open(f'{engine_dir}/feature_defaults.json') as f:
            self.defaults = json.load(f)

        with open(f'{engine_dir}/fleet_lookup.json') as f:
            self.fleet = json.load(f)

        with open(f'{engine_dir}/unit_snapshots.json') as f:
            self.snapshots = json.load(f)

        with open(f'{engine_dir}/route_stats.json') as f:
            self.route_stats = {r['route']: r for r in json.load(f)}

        with open(f'{engine_dir}/risk_distribution.json') as f:
            self.risk_dist = json.load(f)

        with open(f'{engine_dir}/shap_baseline.json') as f:
            self.shap_baseline = json.load(f)

        # SHAP explainer
        self.explainer = shap.TreeExplainer(self.xgb_model)

    def _build_feature_vector(self, unit_id, trip):
        """Build the full feature vector from unit history + trip details."""
        row = {}
        warnings = []

        # Input validation
        if 'temperature' in trip:
            t = trip['temperature']
            if t < -60 or t > 60:
                warnings.append(f'temperature={t} outside expected range [-60, 60]°C')
        if 'leg_miles' in trip and trip['leg_miles'] <= 0:
            warnings.append(f'leg_miles={trip["leg_miles"]} must be positive')
        if 'month' in trip:
            m = trip['month']
            if m < 1 or m > 12:
                warnings.append(f'month={m} outside valid range [1, 12]')

        # Unit lookup warnings
        if unit_id not in self.snapshots:
            warnings.append(f'unit_id={unit_id} not found in snapshots — using defaults')
        if unit_id not in self.fleet:
            warnings.append(f'unit_id={unit_id} not found in fleet lookup — using defaults')

        # Start with defaults
        for col in self.feature_cols:
            if col in self.defaults:
                row[col] = self.defaults[col]
            else:
                row[col] = 0.0

        # Overlay unit snapshot (historical features)
        if unit_id in self.snapshots:
            snap = self.snapshots[unit_id]
            for k, v in snap.items():
                if k in self.feature_cols and not isinstance(v, str):
                    row[k] = v

        # Overlay fleet lookup (equipment profile)
        if unit_id in self.fleet:
            fl = self.fleet[unit_id]
            if 'is_slxi' in fl: row['is_slxi'] = fl['is_slxi']
            if 'is_thermoking' in fl: row['is_thermoking'] = fl['is_thermoking']
            if 'is_carrier' in fl: row['is_carrier'] = fl['is_carrier']
            if 'reefer_year' in fl: row['reefer_year'] = fl['reefer_year']

        # Overlay trip details (user-provided)
        trip_keys = ['leg_miles', 'total_miles', 'temperature', 'month',
                     'day_of_week', 'trip_duration_days', 'origin_temp',
                     'dest_temp', 'transit_min_temp', 'transit_max_temp',
                     'transit_temp_swing', 'origin_dest_temp_diff',
                     # maintenance current-state features — overridable for what-if
                     # scenarios in the Trip Predictor; default to the unit snapshot
                     # when not supplied by the caller.
                     'repair_count_365d', 'repair_cum_cost', 'repair_cost_365d',
                     'repair_count_30d', 'container_repair_cost_365d',
                     'unique_compcodes_365d', 'days_since_last_repair',
                     'pm_gap_mean', 'pm_cum_overdue_days', 'diagnosis_365d',
                     'n_repair_groups_14d', 'has_dor', 'has_dor_90d']
        for k in trip_keys:
            if k in trip:
                row[k] = trip[k]

        # Auto-compute some trip fields if not provided
        if 'total_miles' not in trip and 'leg_miles' in trip:
            row['total_miles'] = trip['leg_miles']

        # Season mapping
        if 'season' in trip:
            season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
            row['season'] = season_map.get(trip['season'], 2)
        elif 'month' in trip:
            m = trip['month']
            if m in [12, 1, 2]: row['season'] = 0  # Winter
            elif m in [3, 4, 5]: row['season'] = 1  # Spring
            elif m in [6, 7, 8]: row['season'] = 2  # Summer
            else: row['season'] = 3  # Fall

        # Temperature-derived features (aligned with training: build_trip_features_v3.py:490-491)
        if 'temperature' in trip:
            t = trip['temperature']
            row['extreme_cold_flag'] = 1 if t < 0 else 0
            row['danger_zone_flag'] = 1 if 0 <= t <= 20 else 0
            row['heating_mode'] = 1 if t < 0 else 0
            if 'origin_temp' not in trip:
                row['origin_temp'] = t
            if 'dest_temp' not in trip:
                row['dest_temp'] = t

        # Encode categoricals
        for col in self.cat_cols:
            if col in trip:
                val = str(trip[col])
                le = self.label_encoders.get(col, {})
                encoded = le.get(val)
                if encoded is None:
                    encoded = le.get('MISSING', 0)
                    warnings.append(f'{col}={val} not in training data — using fallback encoding')
                row[col] = encoded
            elif col in row and isinstance(row[col], str):
                le = self.label_encoders.get(col, {})
                encoded = le.get(row[col])
                if encoded is None:
                    encoded = le.get('MISSING', 0)
                    warnings.append(f'{col}={row[col]} not in training data — using fallback encoding')
                row[col] = encoded

        # Build array in correct order
        vec = np.array([row.get(c, 0.0) for c in self.feature_cols],
                       dtype=np.float32).reshape(1, -1)

        # Impute any NaN
        vec = self.imputer.transform(vec)

        return vec, row, warnings

    def score(self, unit_id, trip):
        """
        Score a trip and return risk assessment with explanations.

        Args:
            unit_id: Reefer unit ID (e.g., 'R8448')
            trip: dict with trip details:
                - route: e.g., 'CN TOR-CN CGY'
                - origin: e.g., 'TOR'
                - destination: e.g., 'CGY'
                - service_type: 'FRZ', 'FRSH', 'HEAT', 'DRY'
                - temperature: ambient temp in °C
                - leg_miles: trip distance
                - month: 1-12
                - season: 'Winter', 'Spring', 'Summer', 'Fall'
                - trip_duration_days: expected duration

        Returns:
            dict with risk_score, risk_tier, risk_factors, etc.
        """
        vec, row_dict, warnings = self._build_feature_vector(unit_id, trip)

        # Get calibrated probability
        risk_score = float(self.cal_model.predict_proba(vec)[0, 1])

        # Get raw XGBoost probability (for SHAP)
        raw_score = float(self.xgb_model.predict_proba(vec)[0, 1])

        # Determine tier
        thresholds = self.risk_dist['tier_thresholds']
        if risk_score < thresholds['LOW']:
            tier = 'LOW'
        elif risk_score < thresholds['MODERATE']:
            tier = 'MODERATE'
        elif risk_score < thresholds['HIGH']:
            tier = 'HIGH'
        else:
            tier = 'CRITICAL'

        # Percentile rank
        pcts = self.risk_dist['percentiles']
        pct_keys = sorted(pcts.keys(), key=lambda x: int(x))
        percentile = 0
        for pk in pct_keys:
            if risk_score >= pcts[pk]:
                percentile = int(pk)
        percentile = min(percentile, 99)

        # SHAP explanation
        shap_values = self.explainer.shap_values(vec)[0]

        # Top risk factors (positive SHAP = increases risk)
        shap_pairs = list(zip(self.feature_cols, shap_values))
        shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        risk_factors = []
        for feat, shap_val in shap_pairs[:10]:
            direction = 'increases' if shap_val > 0 else 'decreases'
            label = self.feature_labels.get(feat, feat.replace('_', ' ').title())
            category = self.feature_categories.get(feat, 'Other')
            value = row_dict.get(feat, None)

            # For encoded categoricals, try to show original value
            display_value = value
            if feat in self.label_encoders and value is not None:
                inv = {v: k for k, v in self.label_encoders[feat].items()}
                display_value = inv.get(int(value), value)

            risk_factors.append({
                'feature': feat,
                'label': label,
                'category': category,
                'value': display_value,
                'shap_value': round(float(shap_val), 4),
                'direction': direction,
                'impact': 'high' if abs(shap_val) > 0.15 else 'moderate' if abs(shap_val) > 0.05 else 'low',
            })

        # Route baseline
        route_name = trip.get('route', '')
        route_info = self.route_stats.get(route_name, None)

        # Unit profile
        unit_info = {}
        if unit_id in self.fleet:
            unit_info = self.fleet[unit_id]
        if unit_id in self.snapshots:
            snap = self.snapshots[unit_id]
            unit_info['total_trips'] = snap.get('total_trips', 0)
            unit_info['total_shutdowns'] = snap.get('total_shutdowns', 0)
            unit_info['historical_shutdown_rate'] = snap.get('shutdown_rate', 0)

        return {
            'unit_id': unit_id,
            'risk_score': round(risk_score, 4),
            'risk_percentage': f'{risk_score*100:.1f}%',
            'risk_tier': tier,
            'percentile': percentile,
            'percentile_label': f'Higher risk than {percentile}% of trips',
            'risk_factors': risk_factors,
            'route_baseline': {
                'route': route_name,
                'historical_rate': route_info['rate'] if route_info else None,
                'historical_trips': route_info['trips'] if route_info else None,
            },
            'unit_profile': unit_info,
            'trip_input': trip,
            'warnings': warnings,
        }

    def explain_text(self, result):
        """Generate a human-readable risk explanation."""
        lines = []
        lines.append(f"RISK ASSESSMENT — {result['unit_id']}")
        lines.append(f"{'='*50}")
        lines.append(f"Risk Score: {result['risk_percentage']} ({result['risk_tier']})")
        lines.append(f"{result['percentile_label']}")

        if result['route_baseline']['historical_rate']:
            lines.append(f"Route baseline: {result['route_baseline']['historical_rate']}% shutdown rate")

        up = result.get('unit_profile', {})
        if up.get('total_trips'):
            lines.append(f"Unit history: {up['total_shutdowns']}/{up['total_trips']} trips had shutdowns ({up.get('historical_shutdown_rate',0)*100:.1f}%)")

        lines.append(f"\nTOP RISK FACTORS:")
        lines.append(f"{'-'*50}")

        for i, f in enumerate(result['risk_factors'][:7], 1):
            arrow = '+' if f['direction'] == 'increases' else '-'
            val_str = f['value']
            if isinstance(val_str, float):
                val_str = f'{val_str:.1f}'
            lines.append(f"  {i}. {arrow} {f['label']}: {val_str}")
            lines.append(f"     [{f['category']}] Impact: {f['impact'].upper()} (SHAP: {f['shap_value']:+.3f})")

        return '\n'.join(lines)


# ======================================================================
# DEMO / CLI
# ======================================================================
if __name__ == '__main__':
    import sys

    engine_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'models', 'risk_engine')
    scorer = RiskScorer(engine_dir)

    print('OpsInsight Reefer Risk Scoring Tool')
    print('=' * 60)
    print(f'Fleet: {len(scorer.fleet)} units')
    print(f'Routes: {len(scorer.route_stats)} routes')
    print(f'Features: {len(scorer.feature_cols)}')

    # Demo scenarios
    demos = [
        {
            'name': 'High-risk winter frozen run',
            'unit': 'R8448',
            'trip': {
                'route': 'CN TOR-CN CGY',
                'origin': 'TOR',
                'destination': 'CGY',
                'service_type': 'FRZ',
                'temperature': -25,
                'leg_miles': 2127,
                'month': 1,
                'season': 'Winter',
                'trip_duration_days': 4.5,
            }
        },
        {
            'name': 'Summer short haul',
            'unit': 'R9378',
            'trip': {
                'route': 'CN TOR-CN MTL',
                'origin': 'TOR',
                'destination': 'MTL',
                'service_type': 'FRSH',
                'temperature': 22,
                'leg_miles': 541,
                'month': 7,
                'season': 'Summer',
                'trip_duration_days': 1.5,
            }
        },
        {
            'name': 'Moderate risk fall long-haul',
            'unit': 'R7839',
            'trip': {
                'route': 'CN TOR-CN VCR',
                'origin': 'TOR',
                'destination': 'VCR',
                'service_type': 'FRZ',
                'temperature': -5,
                'leg_miles': 3400,
                'month': 11,
                'season': 'Fall',
                'trip_duration_days': 6.0,
            }
        },
    ]

    for demo in demos:
        print(f'\n\n{"="*60}')
        print(f'SCENARIO: {demo["name"]}')
        print(f'{"="*60}')

        result = scorer.score(demo['unit'], demo['trip'])
        print(scorer.explain_text(result))
