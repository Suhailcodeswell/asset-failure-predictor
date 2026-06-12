# Asset Failure Predictor

**Transport operations tool for asset failure risk scoring, cost-per-hour (CPH), and cost-per-mile (CPM).**

A portfolio demonstration of an AI-built software product for operational transport fleets. Three calibrated ML models ship in one deployable web application:

| Module | What it predicts | Model |
|--------|------------------|-------|
| **Failure Risk** | Trip-level shutdown / failure probability | Calibrated XGBoost + SHAP explanations |
| **Cost Per Hour (CPH)** | PM, GENREP, and fuel cost per engine hour | Two-stage Random Forest |
| **Cost Per Mile (CPM)** | Maintenance and repair cost per mile | Random Forest (GENREP + PM) |

> **Note:** This repository uses **synthetic, anonymized demo data**. It is a public portfolio version and does not contain confidential client information.

## Live demo

Deploy to [Vercel](https://vercel.com/new/import?s=https://github.com/Suhailcodeswell/asset-failure-predictor):

1. Import the GitHub repo (`asset-failure-predictor`)
2. Leave framework preset as **Other**
3. Deploy (no build command needed)
4. Optional: set `AUTH_TOKEN` in Vercel env vars to change the login code

**Demo access code:** `AFPredict2026!`

## Features

- Trip risk predictor with calibrated probability and SHAP drivers
- Fleet dashboard with risk tiers, PM compliance, and Excel export
- Model comparison and repair pattern analysis
- CPH fleet lookup, scenario planner, and money-pit classification
- CPM unit lookup, what-if scenarios, and fleet ranking
- Pilot validation workflow

## Tech stack

- **Frontend:** HTML, Tailwind CSS, vanilla JavaScript
- **Backend:** Python serverless API (Vercel)
- **ML inference:** Pure-Python tree walkers (no sklearn/xgboost runtime on server)

## Local development

```bash
# Install Vercel CLI
npm i -g vercel

# From repo root
vercel dev
```

Open `http://localhost:3000` and sign in with the demo access code.

## Deployment

```bash
vercel --prod
```

Set `AUTH_TOKEN` in Vercel environment variables to rotate the access code.

## Project structure

```
├── api/predict.py      # Serverless scoring API
├── data/               # Exported model artifacts + demo fleet JSON
├── public/index.html   # Single-page application
├── scripts/            # Data sanitization utilities
└── vercel.json         # Routing config
```

## Author

Built with AI-assisted development as a portfolio piece demonstrating end-to-end ML product delivery for operations analytics.
