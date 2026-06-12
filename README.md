# OpsInsight Analytics

**Asset failure risk scoring, cost-per-hour (CPH), and cost-per-mile (CPM) intelligence for operations teams.**

OpsInsight is a portfolio demonstration of an AI-assisted analytics product built for operational asset management. It packages three calibrated ML models into a single deployable web application:

| Module | What it predicts | Model |
|--------|------------------|-------|
| **Failure Risk** | Trip-level shutdown / failure probability | Calibrated XGBoost + SHAP explanations |
| **Cost Per Hour (CPH)** | PM, GENREP, and fuel cost per engine hour | Two-stage Random Forest |
| **Cost Per Mile (CPM)** | Maintenance and repair cost per mile | Random Forest (GENREP + PM) |

> **Note:** This repository uses **synthetic, anonymized demo data**. It is a public portfolio version and does not contain confidential client information.

## Live demo

Deploy to [Vercel](https://vercel.com) with one click, or run locally (see below).

**Demo access code:** `OpsInsight2026!`

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
