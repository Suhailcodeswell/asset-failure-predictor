# ML Pipeline

End-to-end workflow for the Asset Failure Predictor — ported from a production
transport-fleet analytics project and adapted for public portfolio use with
synthetic/anonymized data.

## Architecture

```
data/raw/              Synthetic source files (repairs, telematics, mileage, meters)
        ↓
pipeline/build_trip_features_v3.py    →  data/processed/training/trip_features_v3.csv
        ↓
pipeline/build_risk_engine.py         →  models/risk_engine/*  (XGBoost + calibration)
        ↓
pipeline/repair_pattern_analysis.py   →  data/repair_patterns.json
        ↓
pipeline/export_webapp.py             →  data/*.json  (Vercel runtime artifacts)
```

## Quick start

```bash
# Web app runtime (Vercel / local dev)
pip install -r requirements.txt

# Full ML pipeline
pip install -r requirements-ml.txt

# Validate demo inputs exist
python pipeline/refresh.py --check

# Full retrain on bundled demo data (~10–15 min)
python unified_pipeline.py
# or
python pipeline/refresh.py
```

## Pipeline scripts

| Script | Purpose |
|--------|---------|
| `refresh.py` | Master orchestrator — validate → features → train → export |
| `build_trip_features_v3.py` | 199-column trip-level feature engineering (no leakage) |
| `build_risk_engine.py` | Calibrated XGBoost + SHAP + temporal train/test split |
| `repair_pattern_analysis.py` | Pre-trip repair code patterns linked to shutdowns |
| `export_webapp.py` | Convert model artifacts to pure-Python JSON tree walkers |
| `risk_scorer.py` | Deployable Python scoring module with SHAP explanations |
| `refresh_inputs.py` | Update unit snapshots from new monthly data (no retrain) |
| `refresh_current_state.py` | Monthly input refresh from CSV templates |
| `build_cph_webapp_data.py` | Cost-per-hour model export |
| `build_cpm_webapp_data.py` | Cost-per-mile model export |

## Data conventions

- **Reefer units:** `AST-####` (synthetic portfolio IDs)
- **Hubs:** `Hub-East`, `Hub-West`, etc. (anonymized route endpoints)
- **Shops:** `SHP-01` … `SHP-10`
- **Vendors:** `Vendor-A` (ThermoKing-class), `Vendor-B` (Carrier-class)

See `data/templates/README.md` for monthly upload column formats and
`data/README.md` for the demo dataset layout.

## Model outputs

After training, `models/risk_engine/` contains:

- `xgb_model.json` — native XGBoost model
- `calibrated_model.pkl` — isotonic calibration wrapper
- `imputer.pkl`, `label_encoders.json`, `feature_meta.json`
- `unit_snapshots.json`, `fleet_lookup.json`, `route_stats.json`

The web app loads lightweight JSON exports from `data/` (no sklearn/xgboost at runtime).

## NDA note

This repository contains **no confidential client data**. Real identifiers from
the internal project (company names, unit IDs, hub codes) are replaced with
synthetic labels. Statistical structure is preserved for demonstration purposes.
