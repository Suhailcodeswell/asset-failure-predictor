"""Export scored fleet data to webapp JSON files."""
import json
import math
import os
import pickle
import shutil

import numpy as np

from .data_loader import DashboardData


class WebappExporter:
    """Produces all JSON files needed by the Vercel webapp."""

    def __init__(self, data: DashboardData, scored_fleet: list[dict],
                 fleet_summary: dict, pm_buckets: list[dict],
                 model_analysis: dict):
        self.data = data
        self.scored = scored_fleet
        self.summary = fleet_summary
        self.pm_buckets = pm_buckets
        self.model_analysis = model_analysis

    def export(self, webapp_dir: str) -> list[str]:
        """Write all webapp JSON files. Returns list of paths written."""
        os.makedirs(webapp_dir, exist_ok=True)
        written = []

        written.append(self._export_model_dump(webapp_dir))
        written.append(self._export_calibration(webapp_dir))
        written.append(self._export_feature_names(webapp_dir))
        written.append(self._export_feature_defaults(webapp_dir))
        written.append(self._export_category_maps(webapp_dir))
        written += self._copy_reference_files(webapp_dir)

        # Repair patterns: copy if exists
        if self.data.repair_patterns:
            p = os.path.join(webapp_dir, 'repair_patterns.json')
            with open(p, 'w') as f:
                json.dump(self.data.repair_patterns, f)
            written.append(p)

        print(f'  Webapp: {len(written)} files written to {webapp_dir}/')
        return written

    def _export_model_dump(self, d):
        """Convert XGBoost model to webapp tree-walker format."""
        src = os.path.join(self.data.engine_dir, 'xgb_model.json')
        with open(src) as f:
            raw = json.load(f)

        learner = raw['learner']
        gb = learner['gradient_booster']['model']
        params = learner['learner_model_param']

        bs = str(params.get('base_score', '0.5')).strip('[]')
        n_features = int(params.get('num_feature', 0))

        trees = []
        for t in gb['trees']:
            trees.append({
                'split_indices': t['split_indices'],
                'split_conditions': t['split_conditions'],
                'left_children': t['left_children'],
                'right_children': t['right_children'],
                'base_weights': t['base_weights'],
                'default_left': t['default_left'],
            })

        out = {'format': 'xgboost', 'base_score': float(bs),
               'n_features': n_features, 'n_trees': len(trees), 'trees': trees}

        path = os.path.join(d, 'model_dump.json')
        with open(path, 'w') as f:
            json.dump(out, f)
        return path

    def _export_calibration(self, d):
        """Build isotonic calibration lookup table."""
        cal = self.data.cal_model
        grid = np.linspace(0, 1, 1001)
        cal_sums = np.zeros(1001)
        for cc in cal.calibrated_classifiers_:
            iso = cc.calibrators[0]
            cal_sums += iso.predict(grid)
        table = (cal_sums / len(cal.calibrated_classifiers_)).tolist()

        path = os.path.join(d, 'calibration_table.json')
        with open(path, 'w') as f:
            json.dump(table, f)
        return path

    def _export_feature_names(self, d):
        path = os.path.join(d, 'feature_names.json')
        with open(path, 'w') as f:
            json.dump(self.data.feature_cols, f, indent=2)
        return path

    def _export_feature_defaults(self, d):
        path = os.path.join(d, 'feature_defaults.json')
        with open(path, 'w') as f:
            json.dump(self.data.feature_defaults, f, indent=2)
        return path

    def _export_category_maps(self, d):
        path = os.path.join(d, 'category_maps.json')
        with open(path, 'w') as f:
            json.dump(self.data.label_encoders, f, indent=2)
        return path

    def _copy_reference_files(self, d):
        """Copy reference JSONs from risk_engine and training dirs."""
        written = []
        eng = self.data.engine_dir

        # Direct copies from risk_engine
        for fname in ('unit_snapshots.json', 'fleet_lookup.json', 'route_stats.json', 'risk_distribution.json'):
            src = os.path.join(eng, fname)
            if os.path.exists(src):
                # For unit_snapshots, use the enriched webapp version if it exists
                if fname == 'unit_snapshots.json':
                    dest = os.path.join(d, fname)
                    # Write the merged snapshots from data_loader (has component fields)
                    with open(dest, 'w') as f:
                        json.dump(self.data.unit_snapshots, f, indent=2)
                else:
                    shutil.copy2(src, os.path.join(d, fname))
                written.append(os.path.join(d, fname))

        # Trailer map from training dir
        for fname in ('trailer_map.json',):
            src = os.path.join(os.path.dirname(eng), 'training', fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(d, fname))
                written.append(os.path.join(d, fname))

        return written
