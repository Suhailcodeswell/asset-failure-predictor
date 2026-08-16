# Asset Failure Predictor

Portfolio demo of a transport operations platform for failure risk scoring, cost-per-hour (CPH), and cost-per-mile (CPM). Includes the ML pipeline, synthetic demo data, and a deployable web app.

This public repo uses synthetic and anonymized data only. It does not contain confidential client information.

| Module | Prediction | Model |
| --- | --- | --- |
| Failure risk | Trip-level shutdown probability | Calibrated XGBoost + SHAP |
| Cost per hour (CPH) | PM, GENREP, and fuel cost per engine hour | Two-stage Random Forest |
| Cost per mile (CPM) | Maintenance and repair cost per mile | Random Forest |

**Live demo:** [asset-failure-predictor.vercel.app](https://asset-failure-predictor.vercel.app)

## What's included

| Area | Path |
| --- | --- |
| Web app | `public/index.html`, `api/predict.py` |
| ML pipeline | `pipeline/` |
| Demo data | `data/raw/`, `data/processed/` |
| Model artifacts | `data/*.json` |
| Pipeline docs | `docs/PIPELINE.md` |

## Retrain on demo data

```bash
pip install -r requirements-ml.txt
python pipeline/refresh.py --check
python unified_pipeline.py
```

See [docs/PIPELINE.md](docs/PIPELINE.md) for the full workflow.

## Local web app

```bash
pip install -r requirements.txt
npm i -g vercel
vercel dev
```

Or:

```bash
python scripts/dev_server.py
```

## Stack

HTML, Tailwind CSS, JavaScript, Python (Vercel serverless), pandas, scikit-learn, XGBoost, SHAP

Inference uses pure-Python tree walkers so the server does not need sklearn or XGBoost at runtime.

## Author

Suhail Ahmed
