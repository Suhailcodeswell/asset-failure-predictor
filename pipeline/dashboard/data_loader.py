"""Load all risk engine artifacts into a single data container."""
import json
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
import shap


class DashboardData:
    """Loads and holds all risk-engine artifacts needed by the dashboard pipeline."""

    def __init__(self, engine_dir: str):
        self.engine_dir = engine_dir
        self._load_model(engine_dir)
        self._load_reference(engine_dir)
        self._load_shap(engine_dir)
        self._load_repair_patterns()

    # ------------------------------------------------------------------
    # Model & calibration
    # ------------------------------------------------------------------
    def _load_model(self, d):
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(f'{d}/xgb_model.json')

        with open(f'{d}/calibrated_model.pkl', 'rb') as f:
            self.cal_model = pickle.load(f)

        with open(f'{d}/imputer.pkl', 'rb') as f:
            self.imputer = pickle.load(f)

    # ------------------------------------------------------------------
    # Reference data
    # ------------------------------------------------------------------
    def _load_reference(self, d):
        with open(f'{d}/feature_meta.json') as f:
            meta = json.load(f)
        self.feature_cols = meta['feature_cols']
        self.feature_categories = meta['feature_categories']
        self.feature_labels = meta['feature_labels']
        self.cat_cols = meta['cat_cols']

        with open(f'{d}/label_encoders.json') as f:
            self.label_encoders = json.load(f)

        with open(f'{d}/feature_defaults.json') as f:
            self.feature_defaults = json.load(f)

        with open(f'{d}/fleet_lookup.json') as f:
            self.fleet_lookup = json.load(f)

        with open(f'{d}/unit_snapshots.json') as f:
            self.unit_snapshots = json.load(f)

        # Merge enriched webapp snapshots (has component _cum fields added later)
        webapp_snap_path = os.path.join(os.path.dirname(d), '..', 'webapp', 'data', 'unit_snapshots.json')
        webapp_snap_path = os.path.normpath(webapp_snap_path)
        if os.path.exists(webapp_snap_path):
            with open(webapp_snap_path) as f:
                webapp_snaps = json.load(f)
            for uid in self.unit_snapshots:
                if uid in webapp_snaps:
                    for k, v in webapp_snaps[uid].items():
                        if k not in self.unit_snapshots[uid]:
                            self.unit_snapshots[uid][k] = v

        with open(f'{d}/route_stats.json') as f:
            self.route_stats = json.load(f)

        trailer_path = f'{d}/trailer_map.json'
        if not os.path.exists(trailer_path):
            trailer_path = os.path.join(os.path.dirname(d), 'training', 'trailer_map.json')
        if os.path.exists(trailer_path):
            with open(trailer_path) as f:
                self.trailer_map = json.load(f)
        else:
            self.trailer_map = {}

        with open(f'{d}/risk_distribution.json') as f:
            self.risk_distribution = json.load(f)

    # ------------------------------------------------------------------
    # SHAP explainer
    # ------------------------------------------------------------------
    def _load_shap(self, d):
        self.explainer = shap.TreeExplainer(self.xgb_model)

        shap_bl_path = f'{d}/shap_baseline.json'
        if os.path.exists(shap_bl_path):
            with open(shap_bl_path) as f:
                self.shap_baseline = json.load(f)
        else:
            self.shap_baseline = {}

        shap_imp_path = f'{d}/shap_importance.csv'
        if os.path.exists(shap_imp_path):
            self.shap_importance = pd.read_csv(shap_imp_path)
        else:
            self.shap_importance = None

    # ------------------------------------------------------------------
    # Repair patterns (optional — may not exist yet)
    # ------------------------------------------------------------------
    def _load_repair_patterns(self):
        rp_path = os.path.join(os.path.dirname(self.engine_dir),
                               '..', 'webapp', 'data', 'repair_patterns.json')
        rp_path = os.path.normpath(rp_path)
        if os.path.exists(rp_path):
            with open(rp_path) as f:
                self.repair_patterns = json.load(f)
        else:
            self.repair_patterns = None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def tier_thresholds(self):
        return self.risk_distribution['tier_thresholds']

    @property
    def n_features(self):
        return len(self.feature_cols)

    @property
    def n_units(self):
        return len(self.fleet_lookup)

    def reverse_encoder(self, col):
        """Return {int_code: original_label} for a categorical column."""
        le = self.label_encoders.get(col, {})
        return {v: k for k, v in le.items()}

    def get_tier(self, score):
        t = self.tier_thresholds
        if score < t['LOW']:
            return 'LOW'
        elif score < t['MODERATE']:
            return 'MODERATE'
        elif score < t['HIGH']:
            return 'HIGH'
        else:
            return 'CRITICAL'

    def summary(self):
        print(f'DashboardData loaded:')
        print(f'  Features: {self.n_features}')
        print(f'  Fleet units: {self.n_units}')
        print(f'  Snapshots: {len(self.unit_snapshots)}')
        print(f'  Routes: {len(self.route_stats)}')
        print(f'  Trailer map: {len(self.trailer_map)} entries')
        print(f'  Repair patterns: {"loaded" if self.repair_patterns else "not found"}')
