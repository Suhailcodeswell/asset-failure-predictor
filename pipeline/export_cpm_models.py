"""
Train trailer CPM (Cost Per Mile) predictive models and export webapp artifacts.

Uses the time-split datasets (2024 history → 2025 targets):
- trailer_repair_model_dataset.csv  → predicts future repair cost
- trailer_maintenance_model_dataset.csv → predicts future PM cost

Also merges actual CPM data from trailer_cpm_base.csv for display.

Exports:
- data/cpm_fleet.json     — Per-trailer predictions + actual costs/CPM
- data/cpm_summary.json   — Fleet-wide summary, by-division, seasonal
- data/cpm_repair_model.json  — RF tree structures for live prediction
- data/cpm_pm_model.json      — RF tree structures for live prediction

Usage:
    python scripts/export_cpm_models.py
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPM_DATA = os.path.join(BASE, "cpm", "cleaned data")
OUT_DIR = os.path.join(BASE, "webapp", "data")

# ---------- tree export helpers (same as CPH export) ----------

def _export_tree(sk_tree):
    t = sk_tree.tree_
    return {
        "left_children": t.children_left.tolist(),
        "right_children": t.children_right.tolist(),
        "split_features": t.feature.tolist(),
        "thresholds": [round(float(x), 8) for x in t.threshold],
        "values": [round(float(x[0][0]), 8) for x in t.value],
    }

def _export_rf(pipeline, numeric_features, categorical_features):
    model = pipeline.named_steps["model"]
    cat_encoder = pipeline.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
    cat_categories = {
        col: [str(c) for c in cats]
        for col, cats in zip(categorical_features, cat_encoder.categories_)
    }
    trees = [_export_tree(est) for est in model.estimators_]
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = list(numeric_features) + cat_feature_names
    raw_imp = model.feature_importances_
    imp_df = pd.DataFrame({"feature": all_feature_names, "importance": raw_imp})
    agg = []
    for nf in numeric_features:
        rows = imp_df.loc[imp_df.feature == nf, "importance"]
        agg.append({"feature": nf, "importance": round(float(rows.values[0]), 6) if len(rows) else 0})
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

# ---------- model training ----------

DROP_COLS = {"trailer_id", "trailer_id_str", "past_first_repair_date", "past_last_repair_date"}


def _safe_float(val, default=0.0):
    """Convert to float, returning default for NaN/None."""
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except (TypeError, ValueError):
        return default

def _split_features(df, target_col):
    """Determine numeric and categorical features."""
    numeric, categorical = [], []
    for c in df.columns:
        if c == target_col or c in DROP_COLS:
            continue
        if df[c].notna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical

def train_and_export(df, target_col, model_name):
    """Train RF model, return (pipeline, export_dict, metrics, predictions_df)."""
    numeric_feats, cat_feats = _split_features(df, target_col)

    X = df.drop(columns=[target_col] + [c for c in DROP_COLS if c in df.columns]).copy()
    X = X.replace({pd.NA: np.nan})
    for c in cat_feats:
        if c in X.columns:
            X[c] = X[c].astype("object")

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_feats),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
            ]), cat_feats),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    pipeline = Pipeline([
        ("pre", pre),
        ("model", RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred_test = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    # Predict all
    all_preds = pipeline.predict(X)
    pred_df = df[["trailer_id"]].copy() if "trailer_id" in df.columns else pd.DataFrame()
    pred_df[f"pred_{model_name}"] = all_preds

    metrics = {"r2": round(r2, 4), "mae": round(mae, 2), "n_train": len(X_train), "n_test": len(X_test)}
    export = _export_rf(pipeline, numeric_feats, cat_feats)
    export["metrics"] = metrics

    return pipeline, export, metrics, pred_df


def build_fleet_data(repair_preds, pm_preds, base_df, repair_df, pm_df):
    """Merge predictions with actual CPM data."""
    # Start with base CPM data
    fleet = {}

    # Add base CPM info
    for _, row in base_df.iterrows():
        tid = str(row.get("trailer_id", "")).strip()
        if not tid:
            continue

        fleet[tid] = {
            "trailer_id": tid,
            "division": str(row.get("division", "")).strip(),
            "department": str(row.get("DEPTCODE", "")).strip(),
            "total_miles": round(_safe_float(row.get("total_miles"))),
            "highway_miles": round(_safe_float(row.get("highway_miles"))),
            "rail_miles": round(_safe_float(row.get("rail_miles"))),
            # Actual costs
            "actual_repair_cost": round(_safe_float(row.get("trailer_repair_cost_total")), 2),
            "actual_pm_cost": round(_safe_float(row.get("trailer_pm_cost_total")), 2),
            # Actual CPM rates
            "cpm_repair": round(_safe_float(row.get("cpm_repair")), 4),
            "cpm_pm": round(_safe_float(row.get("cpm_pm")), 4),
            "cpm_total": round(_safe_float(row.get("cpm_maint_total")), 4),
            # Seasonal miles
            "miles_winter": round(_safe_float(row.get("total_miles_winter"))),
            "miles_spring": round(_safe_float(row.get("total_miles_spring"))),
            "miles_summer": round(_safe_float(row.get("total_miles_summer"))),
            "miles_fall": round(_safe_float(row.get("total_miles_fall"))),
            # Seasonal repair CPM
            "cpm_repair_winter": round(_safe_float(row.get("cpm_repair_winter")), 4),
            "cpm_repair_spring": round(_safe_float(row.get("cpm_repair_spring")), 4),
            "cpm_repair_summer": round(_safe_float(row.get("cpm_repair_summer")), 4),
            "cpm_repair_fall": round(_safe_float(row.get("cpm_repair_fall")), 4),
        }

    def _tid(val):
        """Normalize trailer_id to string (handles float like 693059.0 → '693059')."""
        s = str(val).strip()
        if s.endswith('.0'):
            s = s[:-2]
        return s

    # Merge predicted costs from time-split models
    for _, row in repair_preds.iterrows():
        tid = _tid(row.get("trailer_id", ""))
        if tid in fleet:
            fleet[tid]["pred_repair_cost"] = round(float(row.get("pred_repair", 0)), 2)

    for _, row in pm_preds.iterrows():
        tid = _tid(row.get("trailer_id", ""))
        if tid in fleet:
            fleet[tid]["pred_pm_cost"] = round(float(row.get("pred_pm", 0)), 2)

    # Add actual target costs from time-split datasets
    for _, row in repair_df.iterrows():
        tid = _tid(row.get("trailer_id", ""))
        if tid in fleet:
            fleet[tid]["actual_2025_repair_cost"] = round(_safe_float(row.get("future_trailer_repair_cost_total")), 2)
            fleet[tid]["past_total_cost"] = round(_safe_float(row.get("past_total_cost_all")), 2)
            fleet[tid]["past_repair_events"] = int(_safe_float(row.get("past_total_repair_events")))

    for _, row in pm_df.iterrows():
        tid = _tid(row.get("trailer_id", ""))
        if tid in fleet:
            fleet[tid]["actual_2025_pm_cost"] = round(_safe_float(row.get("future_trailer_pm_cost_total")), 2)

    # Compute predicted CPM where we have both prediction and miles
    for tid, data in fleet.items():
        miles = data.get("total_miles", 0)
        pred_repair = data.get("pred_repair_cost", 0)
        pred_pm = data.get("pred_pm_cost", 0)
        data["pred_total_cost"] = round(pred_repair + pred_pm, 2)
        if miles > 0:
            data["pred_cpm_repair"] = round(pred_repair / miles, 4)
            data["pred_cpm_pm"] = round(pred_pm / miles, 4)
            data["pred_cpm_total"] = round((pred_repair + pred_pm) / miles, 4)
        else:
            data["pred_cpm_repair"] = 0
            data["pred_cpm_pm"] = 0
            data["pred_cpm_total"] = 0

    return fleet


def build_summary(fleet, repair_metrics, pm_metrics):
    """Fleet-wide CPM summary."""
    units = list(fleet.values())
    with_miles = [u for u in units if u.get("total_miles", 0) > 0]
    with_cpm = [u for u in units if u.get("cpm_total", 0) > 0]

    # Total fleet stats
    total_miles = sum(u["total_miles"] for u in units)
    total_repair = sum(u.get("actual_repair_cost", 0) for u in units)
    total_pm = sum(u.get("actual_pm_cost", 0) for u in units)
    total_pred_repair = sum(u.get("pred_repair_cost", 0) for u in units)
    total_pred_pm = sum(u.get("pred_pm_cost", 0) for u in units)

    # By division
    div_stats = {}
    for u in units:
        d = u.get("division", "UNK") or "UNK"
        if d not in div_stats:
            div_stats[d] = {"count": 0, "miles": 0, "repair_cost": 0, "pm_cost": 0, "pred_repair": 0, "pred_pm": 0}
        div_stats[d]["count"] += 1
        div_stats[d]["miles"] += u.get("total_miles", 0)
        div_stats[d]["repair_cost"] += u.get("actual_repair_cost", 0)
        div_stats[d]["pm_cost"] += u.get("actual_pm_cost", 0)
        div_stats[d]["pred_repair"] += u.get("pred_repair_cost", 0)
        div_stats[d]["pred_pm"] += u.get("pred_pm_cost", 0)

    by_division = []
    for d, s in sorted(div_stats.items(), key=lambda x: -x[1]["count"]):
        cpm = round((s["repair_cost"] + s["pm_cost"]) / max(s["miles"], 1), 4)
        by_division.append({
            "division": d,
            "trailer_count": s["count"],
            "total_miles": round(s["miles"]),
            "avg_repair_cost": round(s["repair_cost"] / max(s["count"], 1), 2),
            "avg_pm_cost": round(s["pm_cost"] / max(s["count"], 1), 2),
            "cpm_total": cpm,
            "avg_pred_repair": round(s["pred_repair"] / max(s["count"], 1), 2),
            "avg_pred_pm": round(s["pred_pm"] / max(s["count"], 1), 2),
        })

    # Seasonal patterns (from trailers with seasonal data)
    seasonal = {}
    for season in ["winter", "spring", "summer", "fall"]:
        miles_key = f"miles_{season}"
        cpm_key = f"cpm_repair_{season}"
        s_miles = sum(u.get(miles_key, 0) for u in units)
        cpm_vals = [u[cpm_key] for u in units if u.get(cpm_key, 0) > 0]
        seasonal[season] = {
            "total_miles": round(s_miles),
            "avg_cpm_repair": round(np.mean(cpm_vals), 4) if cpm_vals else 0,
            "trailers_with_data": len(cpm_vals),
        }

    # Cost distribution
    pred_totals = [u.get("pred_total_cost", 0) for u in units if u.get("pred_total_cost", 0) > 0]
    dist = _build_distribution(pred_totals)

    return {
        "total_trailers": len(units),
        "trailers_with_miles": len(with_miles),
        "trailers_with_cpm": len(with_cpm),
        "total_fleet_miles": round(total_miles),
        "total_repair_cost": round(total_repair, 2),
        "total_pm_cost": round(total_pm, 2),
        "avg_repair_cost": round(total_repair / max(len(units), 1), 2),
        "avg_pm_cost": round(total_pm / max(len(units), 1), 2),
        "fleet_cpm_repair": round(total_repair / max(total_miles, 1), 4),
        "fleet_cpm_pm": round(total_pm / max(total_miles, 1), 4),
        "fleet_cpm_total": round((total_repair + total_pm) / max(total_miles, 1), 4),
        "avg_pred_repair_cost": round(total_pred_repair / max(len(units), 1), 2),
        "avg_pred_pm_cost": round(total_pred_pm / max(len(units), 1), 2),
        "repair_model_metrics": repair_metrics,
        "pm_model_metrics": pm_metrics,
        "by_division": by_division,
        "seasonal": seasonal,
        "pred_cost_distribution": dist,
    }


def _build_distribution(values, n_bins=10):
    if not values:
        return []
    arr = np.array(values)
    # Cap outliers at 99th percentile for better visualization
    cap = np.percentile(arr, 99)
    arr_capped = np.clip(arr, 0, cap)
    edges = np.linspace(0, cap, n_bins + 1)
    counts, _ = np.histogram(arr_capped, bins=edges)
    return [
        {"min": round(float(edges[i]), 2), "max": round(float(edges[i + 1]), 2), "count": int(counts[i])}
        for i in range(n_bins)
    ]


def main():
    print("Loading datasets...")
    repair_df = pd.read_csv(os.path.join(CPM_DATA, "trailer_repair_model_dataset.csv"))
    pm_df = pd.read_csv(os.path.join(CPM_DATA, "trailer_maintenance_model_dataset.csv"))
    base_df = pd.read_csv(os.path.join(CPM_DATA, "trailer_cpm_base.csv"))
    print(f"  Repair dataset: {repair_df.shape}")
    print(f"  PM dataset: {pm_df.shape}")
    print(f"  Base CPM: {base_df.shape}")

    print("\nTraining repair cost model (time-split: 2024→2025)...")
    repair_pipe, repair_export, repair_metrics, repair_preds = train_and_export(
        repair_df, "future_trailer_repair_cost_total", "repair"
    )
    print(f"  R2={repair_metrics['r2']}, MAE=${repair_metrics['mae']}")

    print("\nTraining PM cost model (time-split: 2024→2025)...")
    pm_pipe, pm_export, pm_metrics, pm_preds = train_and_export(
        pm_df, "future_trailer_pm_cost_total", "pm"
    )
    print(f"  R2={pm_metrics['r2']}, MAE=${pm_metrics['mae']}")

    print("\nBuilding fleet data...")
    fleet = build_fleet_data(repair_preds, pm_preds, base_df, repair_df, pm_df)
    print(f"  {len(fleet)} trailers")

    print("\nBuilding summary...")
    summary = build_summary(fleet, repair_metrics, pm_metrics)

    # Write outputs
    os.makedirs(OUT_DIR, exist_ok=True)
    out_files = {
        "cpm_repair_model.json": repair_export,
        "cpm_pm_model.json": pm_export,
        "cpm_fleet.json": fleet,
        "cpm_summary.json": summary,
    }

    for fname, obj in out_files.items():
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w") as f:
            json.dump(obj, f, separators=(",", ":"))
        size_kb = os.path.getsize(path) / 1024
        print(f"  Wrote {fname} ({size_kb:.0f} KB)")

    print("\nDone. All CPM artifacts exported to data/")


if __name__ == "__main__":
    main()
