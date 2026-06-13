"""Score the entire fleet with SHAP explanations and compute aggregate analytics."""
from datetime import datetime

import numpy as np
import pandas as pd

from .data_loader import DashboardData

TIER_ORDER = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']


class FleetScorer:
    """Batch-scores every unit and derives fleet-wide analytics."""

    def __init__(self, data: DashboardData):
        self.data = data

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------
    def score_fleet(self) -> list[dict]:
        """Score all units that have both a fleet entry and a snapshot.

        Returns a list of ScoredUnit dicts sorted by risk descending.
        """
        data = self.data
        unit_ids = sorted(uid for uid in data.fleet_lookup if uid in data.unit_snapshots)
        n = len(unit_ids)

        # Build feature matrix (n x n_features)
        matrix = np.zeros((n, data.n_features), dtype=np.float32)
        row_dicts = []

        for i, uid in enumerate(unit_ids):
            row = dict(data.feature_defaults)
            snap = data.unit_snapshots[uid]
            # Overlay ALL snapshot fields (including component _cum fields not in model)
            for k, v in snap.items():
                if not isinstance(v, str):
                    row[k] = v
            fl = data.fleet_lookup[uid]
            for fk in ('is_slxi', 'is_thermoking', 'is_carrier', 'reefer_year'):
                if fk in fl:
                    row[fk] = fl[fk]

            # PM overdue: compute DISPLAY values dynamically from last_pm_date,
            # but keep original snapshot days_since_last_pm for MODEL scoring.
            pm_policy = 120 if fl.get('is_slxi', 0) else 90
            last_pm_str = row.get('last_pm_date', '')
            if last_pm_str and isinstance(last_pm_str, str) and len(last_pm_str) == 10:
                try:
                    last_pm = datetime.strptime(last_pm_str, '%Y-%m-%d')
                    display_days_since_pm = (datetime.now() - last_pm).days
                except ValueError:
                    display_days_since_pm = row.get('days_since_last_pm', 0) or 0
            else:
                display_days_since_pm = row.get('days_since_last_pm', 0) or 0
            display_overdue = max(0, display_days_since_pm - pm_policy)
            # Store display values — NOT in row (which feeds the model)
            row['_pm_display'] = {
                'days_since_last_pm': display_days_since_pm,
                'days_pm_overdue': display_overdue,
                'is_pm_overdue': 1 if display_overdue > 0 else 0,
            }

            row_dicts.append(row)
            # Use np.nan for None/missing so the imputer can fill properly
            # (0 would mean "just happened" for days_since_* features)
            matrix[i] = [np.nan if row.get(c) is None else float(row.get(c, 0))
                         for c in data.feature_cols]

        # Impute
        matrix = data.imputer.transform(matrix)

        # Calibrated probabilities
        risk_scores = data.cal_model.predict_proba(matrix)[:, 1].astype(float)

        # Batch SHAP
        shap_matrix = data.explainer.shap_values(matrix)  # (n, n_features)

        # Build scored units
        scored = []
        for i, uid in enumerate(unit_ids):
            score = float(risk_scores[i])
            tier = data.get_tier(score)
            fl = data.fleet_lookup[uid]
            row = row_dicts[i]

            # Reverse lookup trailer
            trailer = ''
            for t_id, r_id in data.trailer_map.items():
                if r_id == uid:
                    trailer = t_id
                    break

            # SHAP risk factors (top 10)
            shap_vals = shap_matrix[i]
            risk_factors = self._extract_risk_factors(shap_vals, row)

            # Recommendations (use dynamic PM display values, not stale snapshot)
            rec_row = dict(row)
            pm_d = row.get('_pm_display', {})
            rec_row['days_pm_overdue'] = pm_d.get('days_pm_overdue', 0)
            rec_row['is_pm_overdue'] = pm_d.get('is_pm_overdue', 0)
            rec_row['days_since_last_pm'] = pm_d.get('days_since_last_pm', 0)
            recs = self._derive_recommendations(score, rec_row, fl)

            # Feature values dict (all model features)
            feature_values = {c: float(matrix[i, j]) for j, c in enumerate(data.feature_cols)}

            scored.append({
                'unit': uid,
                'trailer': trailer,
                'risk_score': round(score, 4),
                'risk_percent': round(score * 100, 2),
                'tier': tier,
                'reefer_model': fl.get('reefer_model', ''),
                'reefer_year': fl.get('reefer_year', 0),
                'is_slxi': fl.get('is_slxi', 0),
                'is_thermoking': fl.get('is_thermoking', 0),
                'is_carrier': fl.get('is_carrier', 0),
                'shutdown_count': fl.get('shutdown_count', 0),
                'total_rail_legs': fl.get('total_rail_legs', 0),
                'days_since_last_pm': round(pm_d.get('days_since_last_pm', 0)),
                'days_pm_overdue': round(pm_d.get('days_pm_overdue', 0)),
                'is_pm_overdue': int(pm_d.get('is_pm_overdue', 0)),
                'pm_compliance_ratio': round(row.get('pm_compliance_ratio', 0), 2),
                'nonpm_cum_cost': round(row.get('nonpm_cum_cost', 0) or row.get('repair_cum_cost', 0)),
                'nonpm_cum_count': int(row.get('nonpm_cum_count', 0)),
                'repair_count_90d': int(row.get('repair_count_90d', 0)),
                'repair_cum_cost': round(row.get('repair_cum_cost', 0)),
                'compressor_cum': int(row.get('compressor_cum', 0)),
                'evaporator_cum': int(row.get('evaporator_cum', 0)),
                'diagnosis_cum': int(row.get('diagnosis_cum', 0)),
                'electrical_charging_cum': int(row.get('electrical_charging_cum', 0)),
                'fuel_system_cum': int(row.get('fuel_system_cum', 0)),
                'condenser_cum': int(row.get('condenser_cum', 0)),
                'engine_cum': int(row.get('engine_cum', 0)),
                'refrigerant_lines_cum': int(row.get('refrigerant_lines_cum', 0)),
                'starter_cranking_cum': int(row.get('starter_cranking_cum', 0)),
                'defrost_cum': int(row.get('defrost_cum', 0)),
                'unique_compcodes_cum': int(row.get('unique_compcodes_cum', 0)),
                'unique_shops': int(row.get('unique_shops', 0)),
                'risk_factors': risk_factors,
                'recommendations': recs,
                'feature_values': feature_values,
            })

        # Sort by risk descending, assign ranks
        scored.sort(key=lambda x: x['risk_score'], reverse=True)
        for i, s in enumerate(scored):
            s['rank'] = i + 1

        return scored

    # ------------------------------------------------------------------
    # SHAP factor extraction
    # ------------------------------------------------------------------
    def _extract_risk_factors(self, shap_vals, row_dict):
        data = self.data
        pairs = list(zip(data.feature_cols, shap_vals))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        factors = []
        for feat, sv in pairs[:10]:
            direction = 'increases' if sv > 0 else 'decreases'
            label = data.feature_labels.get(feat, feat.replace('_', ' ').title())
            category = data.feature_categories.get(feat, 'Other')
            value = row_dict.get(feat)

            # Reverse categorical encoding for display
            display_value = value
            if feat in data.label_encoders and value is not None:
                inv = data.reverse_encoder(feat)
                display_value = inv.get(int(value), value)

            factors.append({
                'feature': feat,
                'label': label,
                'category': category,
                'value': display_value,
                'shap_value': round(float(sv), 4),
                'direction': direction,
                'impact': 'high' if abs(sv) > 0.15 else 'moderate' if abs(sv) > 0.05 else 'low',
            })
        return factors

    # ------------------------------------------------------------------
    # Recommendations (mirrors webapp predict.py logic)
    # ------------------------------------------------------------------
    def _derive_recommendations(self, risk_score, features, fleet_info):
        recs = []
        tier = self.data.get_tier(risk_score)
        days_overdue = features.get('days_pm_overdue', 0) or 0
        is_overdue = features.get('is_pm_overdue', 0)
        days_since_pm = features.get('days_since_last_pm', 0) or 0
        pm_compliance = features.get('pm_compliance_ratio', 1.0) or 1.0
        repair_count_90d = features.get('repair_count_90d', 0) or 0
        is_slxi = fleet_info.get('is_slxi', 0)
        compressor = features.get('compressor_cum', 0) or 0
        evaporator = features.get('evaporator_cum', 0) or 0

        if is_overdue or days_overdue > 0:
            urgency = 'CRITICAL' if days_overdue > 90 else 'HIGH' if days_overdue > 30 else 'MEDIUM'
            recs.append({'action': 'Schedule PM immediately', 'urgency': urgency,
                         'detail': f'{int(days_overdue)} days overdue'})
        elif days_since_pm > 60:
            recs.append({'action': 'PM approaching due date', 'urgency': 'LOW',
                         'detail': f'{int(days_since_pm)} days since last PM'})

        if pm_compliance < 0.3:
            recs.append({'action': 'Review PM compliance', 'urgency': 'HIGH',
                         'detail': f'{round(pm_compliance * 100)}% compliance rate'})

        if repair_count_90d >= 3:
            recs.append({'action': 'Investigate repeat repairs', 'urgency': 'HIGH',
                         'detail': f'{int(repair_count_90d)} repairs in 90 days'})

        if is_slxi and tier in ('HIGH', 'CRITICAL'):
            recs.append({'action': 'Check SLXi campaign status', 'urgency': 'MEDIUM',
                         'detail': 'SLXi model — known defect campaign'})

        if compressor >= 2 or evaporator >= 2:
            comp = 'compressor' if compressor >= evaporator else 'evaporator'
            count = max(compressor, evaporator)
            recs.append({'action': f'Recurring {comp} issues — consider overhaul',
                         'urgency': 'HIGH' if count >= 3 else 'MEDIUM',
                         'detail': f'{int(count)} cumulative repairs'})

        if not recs and tier in ('HIGH', 'CRITICAL'):
            recs.append({'action': 'Flag for detailed inspection', 'urgency': 'HIGH',
                         'detail': 'Multiple risk factors combined'})
        if not recs and tier == 'MODERATE':
            recs.append({'action': 'Monitor closely', 'urgency': 'LOW',
                         'detail': 'Moderate risk — watch for deterioration'})

        return recs[:4]

    # ------------------------------------------------------------------
    # Fleet summary
    # ------------------------------------------------------------------
    def compute_fleet_summary(self, scored: list[dict]) -> dict:
        tier_counts = {t: 0 for t in TIER_ORDER}
        total_risk = 0
        pm_overdue = 0
        slxi_count = 0
        slxi_elevated = 0

        for s in scored:
            tier_counts[s['tier']] = tier_counts.get(s['tier'], 0) + 1
            total_risk += s['risk_score']
            if s['is_pm_overdue']:
                pm_overdue += 1
            if s['is_slxi']:
                slxi_count += 1
                if s['tier'] in ('HIGH', 'CRITICAL'):
                    slxi_elevated += 1

        n = len(scored)
        total_sd = sum(s['shutdown_count'] for s in scored)
        total_trips = sum(s['total_rail_legs'] for s in scored)

        return {
            'total_units': n,
            'active_units': n,
            'tier_counts': tier_counts,
            'avg_risk': round(total_risk / max(n, 1) * 100, 2),
            'elevated_plus': tier_counts.get('HIGH', 0) + tier_counts.get('CRITICAL', 0),
            'pm_overdue_count': pm_overdue,
            'slxi_count': slxi_count,
            'slxi_elevated_plus': slxi_elevated,
            'total_shutdowns': total_sd,
            'fleet_sd_rate': round(total_sd / max(total_trips, 1) * 100, 2),
            'total_trips': total_trips,
        }

    # ------------------------------------------------------------------
    # PM delay buckets
    # ------------------------------------------------------------------
    def compute_pm_buckets(self, scored: list[dict]) -> list[dict]:
        bucket_defs = [
            ('On Time', 0, 0),
            ('1-30 days overdue', 1, 30),
            ('31-60 days overdue', 31, 60),
            ('61-90 days overdue', 61, 90),
            ('91-120 days overdue', 91, 120),
            ('120+ days overdue', 121, 9999),
        ]
        buckets = []
        for label, lo, hi in bucket_defs:
            units = [s for s in scored
                     if (lo == 0 and not s['is_pm_overdue'])
                     or (lo > 0 and lo <= s['days_pm_overdue'] <= hi)]
            sd = sum(s['shutdown_count'] for s in units)
            trips = sum(s['total_rail_legs'] for s in units)
            buckets.append({
                'label': label,
                'count': len(units),
                'avg_risk': round(sum(s['risk_score'] for s in units) / max(len(units), 1) * 100, 2),
                'shutdown_rate': round(sd / max(trips, 1) * 100, 2),
            })
        return buckets

    # ------------------------------------------------------------------
    # Model analysis
    # ------------------------------------------------------------------
    def compute_model_analysis(self, scored: list[dict]) -> dict:
        COMP_KEYS = [
            ('compressor_cum', 'Compressor'), ('evaporator_cum', 'Evaporator'),
            ('diagnosis_cum', 'Diagnosis'), ('electrical_charging_cum', 'Electrical'),
            ('fuel_system_cum', 'Fuel System'), ('starter_cranking_cum', 'Starter/Cranking'),
            ('condenser_cum', 'Condenser'), ('engine_cum', 'Engine'),
        ]

        by_model = {}
        for s in scored:
            m = s['reefer_model'] or 'Unknown'
            if m == 'nan':
                m = 'Unknown'
            by_model.setdefault(m, []).append(s)

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
            cost_per_trip = round(sum(u['nonpm_cum_cost'] for u in units) / max(trips, 1), 2)
            pm_overdue = sum(1 for u in units if u['is_pm_overdue'])
            elevated = sum(1 for u in units if u['tier'] in ('HIGH', 'CRITICAL'))
            tiers = {t: sum(1 for u in units if u['tier'] == t) for t in TIER_ORDER}

            components = {}
            for key, label in COMP_KEYS:
                components[label] = round(sum(u.get(key, 0) for u in units) / n, 2)

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
                cohorts.append({
                    'year': year,
                    'age': datetime.now().year - year,
                    'count': yn,
                    'sd_rate': round(ysd / max(ytrips, 1) * 100, 2),
                    'avg_risk': round(sum(u['risk_score'] for u in yunits) / yn * 100, 2),
                    'avg_repair': round(sum(u['nonpm_cum_cost'] for u in yunits) / yn),
                    'cost_per_trip': round(sum(u['nonpm_cum_cost'] for u in yunits) / max(ytrips, 1), 2),
                    'elevated': sum(1 for u in yunits if u['tier'] in ('HIGH', 'CRITICAL')),
                })

            models.append({
                'model': model, 'count': n, 'shutdowns': sd, 'trips': trips,
                'sd_rate': sd_rate, 'avg_risk': avg_risk,
                'avg_repair': avg_repair, 'cost_per_trip': cost_per_trip,
                'pm_overdue': pm_overdue, 'pm_overdue_pct': round(pm_overdue / n * 100),
                'elevated': elevated, 'elevated_pct': round(elevated / n * 100),
                'tiers': tiers, 'components': components,
                'years': years, 'cohorts': cohorts,
            })

        # Worst cohorts across all models
        worst = []
        for m in models:
            for c in m['cohorts']:
                worst.append({**c, 'model': m['model']})
        worst.sort(key=lambda x: -x['sd_rate'])

        return {'models': models, 'worst_cohorts': worst[:15]}

    # ------------------------------------------------------------------
    # SLXi vs Standard component comparison
    # ------------------------------------------------------------------
    def compute_component_comparison(self, scored: list[dict]) -> list[dict]:
        COMPS = [
            ('compressor_cum', 'Compressor'), ('evaporator_cum', 'Evaporator'),
            ('diagnosis_cum', 'Diagnosis'), ('electrical_charging_cum', 'Electrical Charging'),
            ('fuel_system_cum', 'Fuel System'), ('condenser_cum', 'Condenser'),
            ('engine_cum', 'Engine'), ('refrigerant_lines_cum', 'Refrigerant Lines'),
            ('starter_cranking_cum', 'Starter Cranking'), ('defrost_cum', 'Defrost'),
        ]
        slxi = [s for s in scored if s['is_slxi']]
        std = [s for s in scored if not s['is_slxi']]

        result = []
        for key, label in COMPS:
            slxi_avg = sum(u.get(key, 0) for u in slxi) / max(len(slxi), 1)
            std_avg = sum(u.get(key, 0) for u in std) / max(len(std), 1)
            result.append({
                'component': label,
                'slxi_avg': round(slxi_avg, 3),
                'standard_avg': round(std_avg, 3),
                'delta': round(slxi_avg - std_avg, 3),
            })
        return result
