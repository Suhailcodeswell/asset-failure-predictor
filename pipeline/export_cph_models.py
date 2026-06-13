"""
Train PM and GENREP cost-per-hour RandomForest models from the CPH advanced dataset,
then export tree structures + encoders + fleet predictions as JSON for the webapp.

The webapp uses pure-Python tree walkers (no sklearn), so we export each tree's
internal structure (split_features, thresholds, children, leaf_values).

Usage:
    python scripts/export_cph_models.py
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# ---------- paths ----------

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(BASE, "cph", "advanced_modeling", "Advanced_Modeling_Dataset.csv")
OUT_DIR = os.path.join(BASE, "webapp", "data")

# ---------- model configs ----------

PM_CONFIG = {
    "name": "pm",
    "target": "PM_COST_PER_HOUR",
    "features": [
        "COMPCODE", "TARGET_DAYS", "TOTAL_MAINTENANCE_COUNT",
        "ESTIMATED_2YR_HOURS", "NEGLECT_INTENSITY",
        "SHOPID", "MAKE", "MODEL", "MODELYEAR",
    ],
    "categorical": ["COMPCODE", "SHOPID", "MAKE", "MODEL"],
}

GENREP_CONFIG = {
    "name": "genrep",
    "target": "GENREP_COST_PER_HOUR",
    "features": [
        "COMPCODE", "SHOPID", "MAKE", "MODEL", "MODELYEAR",
        "TARGET_DAYS", "SUM_POS_DELAY", "SUM_NEG_DELAY",
        "TOTAL_MAINTENANCE_COUNT", "TOTAL_DELAY_EVENTS",
        "ESTIMATED_2YR_HOURS", "OVERDUE_PERCENTAGE",
        "HAS_TELEMATICS_ALARM", "TOTAL_ALARMS", "CRITICAL_SHUTDOWNS",
        "TRUE_ENGINE_HOURS", "MIN_BATTERY_VOLTAGE", "AVG_BATTERY_VOLTAGE",
        "AVG_DOWNTIME_HOURS", "AVG_DAYS_TO_FAILURE", "MISSING_RESTART_FLAG",
        "NEGLECT_INTENSITY",
    ],
    "categorical": ["COMPCODE", "SHOPID", "MAKE", "MODEL"],
}


def _export_tree(sk_tree):
    """Convert a single sklearn DecisionTreeRegressor to a JSON-serialisable dict."""
    t = sk_tree.tree_
    return {
        "left_children": t.children_left.tolist(),
        "right_children": t.children_right.tolist(),
        "split_features": t.feature.tolist(),
        "thresholds": [round(float(x), 8) for x in t.threshold],
        "values": [round(float(x[0][0]), 8) for x in t.value],
    }


def _export_rf(pipeline, numeric_features, categorical_features):
    """Export an entire RF pipeline to a JSON-serialisable dict."""
    model = pipeline.named_steps["model"]
    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_categories = {
        col: list(cats)
        for col, cats in zip(categorical_features, cat_encoder.categories_)
    }

    trees = [_export_tree(est) for est in model.estimators_]

    # Feature name order after preprocessing: numeric first, then OHE categoricals
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = list(numeric_features) + cat_feature_names

    # Aggregated feature importances
    raw_imp = model.feature_importances_
    imp_df = pd.DataFrame({"feature": all_feature_names, "importance": raw_imp})
    agg = []
    for nf in numeric_features:
        agg.append({"feature": nf, "importance": round(float(imp_df.loc[imp_df.feature == nf, "importance"].values[0]), 6)})
    for cf in categorical_features:
        val = float(imp_df.loc[imp_df.feature.str.startswith(cf + "_"), "importance"].sum())
        agg.append({"feature": cf, "importance": round(val, 6)})
    agg.sort(key=lambda x: -x["importance"])

    return {
        "trees": trees,
        "n_trees": len(trees),
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
        "cat_categories": cat_categories,
        "all_feature_names": all_feature_names,
        "feature_importances": agg,
    }


def train_and_export(df_all, config):
    """Train a single model and return (pipeline, export_dict, metrics)."""
    features = config["features"]
    target = config["target"]
    cat_feats = config["categorical"]
    num_feats = [f for f in features if f not in cat_feats]

    df = df_all.dropna(subset=features + [target]).copy()
    df = df[df["ESTIMATED_2YR_HOURS"] >= 100]

    for c in cat_feats:
        df[c] = df[c].astype(str).str.strip()

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ])
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {"r2": round(r2, 4), "mae": round(mae, 4), "n_train": len(X_train), "n_test": len(X_test)}
    export = _export_rf(pipeline, num_feats, cat_feats)
    export["metrics"] = metrics

    return pipeline, export, metrics


def precompute_fleet(df_all, pm_pipe, genrep_pipe, pm_cfg, genrep_cfg):
    """Run both models on all units, return per-unit aggregated predictions."""
    results = {}

    for config, pipeline, target_col in [
        (pm_cfg, pm_pipe, "PM_COST_PER_HOUR"),
        (genrep_cfg, genrep_pipe, "GENREP_COST_PER_HOUR"),
    ]:
        features = config["features"]
        cat_feats = config["categorical"]

        df = df_all.dropna(subset=features + [target_col]).copy()
        df = df[df["ESTIMATED_2YR_HOURS"] >= 100]
        for c in cat_feats:
            df[c] = df[c].astype(str).str.strip()

        preds = pipeline.predict(df[features])
        df["_pred"] = preds

        for _, row in df.iterrows():
            uid = str(row["UNITNUMBER"]).strip()
            if uid not in results:
                results[uid] = {
                    "unit_id": uid,
                    "make": str(row.get("MAKE", "")).strip(),
                    "model": str(row.get("MODEL", "")).strip(),
                    "model_year": int(row.get("MODELYEAR", 0)) if pd.notna(row.get("MODELYEAR")) else 0,
                    "estimated_2yr_hours": round(float(row.get("ESTIMATED_2YR_HOURS", 0)), 1),
                    "pm_rows": [],
                    "genrep_rows": [],
                }

            entry = {
                "compcode": str(row.get("COMPCODE", "")).strip(),
                "shopid": str(row.get("SHOPID", "")).strip(),
                "target_days": int(row.get("TARGET_DAYS", 0)) if pd.notna(row.get("TARGET_DAYS")) else 0,
                "actual": round(float(row[target_col]), 4),
                "predicted": round(float(row["_pred"]), 4),
            }

            if config["name"] == "pm":
                entry["maintenance_count"] = int(row.get("TOTAL_MAINTENANCE_COUNT", 0)) if pd.notna(row.get("TOTAL_MAINTENANCE_COUNT")) else 0
                entry["neglect_intensity"] = round(float(row.get("NEGLECT_INTENSITY", 0) or 0), 4)
                results[uid]["pm_rows"].append(entry)
            else:
                entry["total_alarms"] = int(row.get("TOTAL_ALARMS", 0)) if pd.notna(row.get("TOTAL_ALARMS")) else 0
                entry["critical_shutdowns"] = int(row.get("CRITICAL_SHUTDOWNS", 0)) if pd.notna(row.get("CRITICAL_SHUTDOWNS")) else 0
                entry["overdue_pct"] = round(float(row.get("OVERDUE_PERCENTAGE", 0) or 0), 4)
                results[uid]["genrep_rows"].append(entry)

    # Aggregate per-unit summaries
    for uid, data in results.items():
        if data["pm_rows"]:
            pm_preds = [r["predicted"] for r in data["pm_rows"]]
            pm_actuals = [r["actual"] for r in data["pm_rows"]]
            data["pm_predicted_avg"] = round(np.mean(pm_preds), 4)
            data["pm_actual_avg"] = round(np.mean(pm_actuals), 4)
            data["pm_count"] = len(data["pm_rows"])
        else:
            data["pm_predicted_avg"] = None
            data["pm_actual_avg"] = None
            data["pm_count"] = 0

        if data["genrep_rows"]:
            gr_preds = [r["predicted"] for r in data["genrep_rows"]]
            gr_actuals = [r["actual"] for r in data["genrep_rows"]]
            data["genrep_predicted_avg"] = round(np.mean(gr_preds), 4)
            data["genrep_actual_avg"] = round(np.mean(gr_actuals), 4)
            data["genrep_count"] = len(data["genrep_rows"])
        else:
            data["genrep_predicted_avg"] = None
            data["genrep_actual_avg"] = None
            data["genrep_count"] = 0

        # Total cost per hour
        pm_val = data["pm_predicted_avg"] or 0
        gr_val = data["genrep_predicted_avg"] or 0
        data["total_cph_predicted"] = round(pm_val + gr_val, 4)

    return results


def build_summary(fleet_preds, pm_metrics, genrep_metrics):
    """Fleet-wide CPH summary stats."""
    units = list(fleet_preds.values())
    pm_preds = [u["pm_predicted_avg"] for u in units if u["pm_predicted_avg"] is not None]
    gr_preds = [u["genrep_predicted_avg"] for u in units if u["genrep_predicted_avg"] is not None]
    total_preds = [u["total_cph_predicted"] for u in units if u["total_cph_predicted"] > 0]

    # By model
    model_stats = {}
    for u in units:
        m = u["model"]
        if m not in model_stats:
            model_stats[m] = {"pm": [], "genrep": [], "total": [], "count": 0}
        model_stats[m]["count"] += 1
        if u["pm_predicted_avg"] is not None:
            model_stats[m]["pm"].append(u["pm_predicted_avg"])
        if u["genrep_predicted_avg"] is not None:
            model_stats[m]["genrep"].append(u["genrep_predicted_avg"])
        if u["total_cph_predicted"] > 0:
            model_stats[m]["total"].append(u["total_cph_predicted"])

    by_model = []
    for m, s in sorted(model_stats.items(), key=lambda x: -x[1]["count"]):
        by_model.append({
            "model": m,
            "unit_count": s["count"],
            "avg_pm_cph": round(np.mean(s["pm"]), 4) if s["pm"] else None,
            "avg_genrep_cph": round(np.mean(s["genrep"]), 4) if s["genrep"] else None,
            "avg_total_cph": round(np.mean(s["total"]), 4) if s["total"] else None,
        })

    # By shop
    shop_stats = {}
    for u in units:
        for row in u.get("pm_rows", []):
            s = row["shopid"]
            if s not in shop_stats:
                shop_stats[s] = {"pm": [], "genrep": [], "count": 0}
            shop_stats[s]["count"] += 1
            shop_stats[s]["pm"].append(row["predicted"])
        for row in u.get("genrep_rows", []):
            s = row["shopid"]
            if s not in shop_stats:
                shop_stats[s] = {"pm": [], "genrep": [], "count": 0}
            shop_stats[s]["count"] += 1
            shop_stats[s]["genrep"].append(row["predicted"])

    by_shop = []
    for s, d in sorted(shop_stats.items(), key=lambda x: -x[1]["count"]):
        by_shop.append({
            "shop": s,
            "record_count": d["count"],
            "avg_pm_cph": round(np.mean(d["pm"]), 4) if d["pm"] else None,
            "avg_genrep_cph": round(np.mean(d["genrep"]), 4) if d["genrep"] else None,
        })

    return {
        "total_units": len(units),
        "fleet_avg_pm_cph": round(np.mean(pm_preds), 4) if pm_preds else None,
        "fleet_avg_genrep_cph": round(np.mean(gr_preds), 4) if gr_preds else None,
        "fleet_avg_total_cph": round(np.mean(total_preds), 4) if total_preds else None,
        "fleet_median_total_cph": round(np.median(total_preds), 4) if total_preds else None,
        "pm_model_metrics": pm_metrics,
        "genrep_model_metrics": genrep_metrics,
        "by_model": by_model,
        "by_shop": by_shop,
        # Distribution buckets for histogram
        "cph_distribution": _build_distribution(total_preds),
    }


def _build_distribution(values, n_bins=10):
    if not values:
        return []
    arr = np.array(values)
    edges = np.linspace(arr.min(), arr.max(), n_bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    return [
        {"min": round(float(edges[i]), 4), "max": round(float(edges[i + 1]), 4), "count": int(counts[i])}
        for i in range(n_bins)
    ]


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_CSV)
    print(f"  {df.shape[0]} rows, {df['UNITNUMBER'].nunique()} units")

    print("\nTraining PM cost-per-hour model (Final/Optimized)...")
    pm_pipe, pm_export, pm_metrics = train_and_export(df, PM_CONFIG)
    print(f"  R2={pm_metrics['r2']}, MAE=${pm_metrics['mae']}")

    print("\nTraining GENREP cost-per-hour model (Advanced w/ telematics)...")
    genrep_pipe, genrep_export, genrep_metrics = train_and_export(df, GENREP_CONFIG)
    print(f"  R2={genrep_metrics['r2']}, MAE=${genrep_metrics['mae']}")

    print("\nPre-computing fleet predictions...")
    fleet_preds = precompute_fleet(df, pm_pipe, genrep_pipe, PM_CONFIG, GENREP_CONFIG)
    print(f"  {len(fleet_preds)} units scored")

    print("\nBuilding summary...")
    summary = build_summary(fleet_preds, pm_metrics, genrep_metrics)

    # Strip detail rows from fleet export (keep summaries only — detail rows bloat the file)
    fleet_slim = {}
    for uid, data in fleet_preds.items():
        slim = {k: v for k, v in data.items() if k not in ("pm_rows", "genrep_rows")}
        # Keep top-level info per row but compact
        slim["pm_compcodes"] = [r["compcode"] for r in data.get("pm_rows", [])]
        slim["genrep_compcodes"] = [r["compcode"] for r in data.get("genrep_rows", [])]
        # Keep best/worst rows for detail view
        if data["pm_rows"]:
            slim["pm_highest"] = max(data["pm_rows"], key=lambda r: r["predicted"])
            slim["pm_lowest"] = min(data["pm_rows"], key=lambda r: r["predicted"])
        if data["genrep_rows"]:
            slim["genrep_highest"] = max(data["genrep_rows"], key=lambda r: r["predicted"])
            slim["genrep_lowest"] = min(data["genrep_rows"], key=lambda r: r["predicted"])
        fleet_slim[uid] = slim

    # Write outputs
    os.makedirs(OUT_DIR, exist_ok=True)

    out_files = {
        "cph_pm_model.json": pm_export,
        "cph_genrep_model.json": genrep_export,
        "cph_fleet.json": fleet_slim,
        "cph_summary.json": summary,
    }

    for fname, obj in out_files.items():
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w") as f:
            json.dump(obj, f, separators=(",", ":"))
        size_kb = os.path.getsize(path) / 1024
        print(f"  Wrote {fname} ({size_kb:.0f} KB)")

    print("\nDone. All CPH artifacts exported to data/")


if __name__ == "__main__":
    main()
