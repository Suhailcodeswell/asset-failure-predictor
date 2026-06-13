# Demo data (public portfolio)

All identifiers in this repository are **synthetic or anonymized**. No confidential
client data is included.

## Layout

| Path | Purpose |
|------|---------|
| `data/raw/` | Synthetic source files matching the pipeline input schema |
| `data/processed/training/` | Model-ready tables (fleet registry, trip features sample) |
| `data/processed/reference/` | Alarm severity lookup (generic industry codes) |
| `data/demo/sample/` | Example monthly upload CSVs |
| `data/templates/` | Column templates for new data drops |
| `data/*.json` | Exported model artifacts consumed by the Vercel web app |

## Retraining

```bash
pip install -r requirements-ml.txt
python pipeline/refresh.py --check    # validate inputs
python unified_pipeline.py            # full retrain (demo data)
```

The bundled `trip_features_v3.csv` is a **5,000-row sanitized sample** so reviewers
can inspect feature engineering output without downloading multi-GB client files.
