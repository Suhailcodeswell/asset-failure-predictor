#!/usr/bin/env python3
"""Convert CPM1 model artifacts into webapp-ready JSON data files.

Reads CSVs/JSONs from CPM1/Modeling/ and writes:
  - data/cpm_fleet.json   (per-unit predictions + SHAP + metadata)
  - data/cpm_summary.json (metrics, global SHAP, feature schema)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "CPM1" / "Modeling"
COMBINED = MODEL_DIR / "cpm_v1_combined"
GEN_DIR = MODEL_DIR / "genrep_v1"
PM_DIR = MODEL_DIR / "pm_v1"
WEBAPP_DATA = ROOT / "data"
ASSET_XLSX = ROOT / "data" / "fleet" / "Asset Detail.xlsx"

REF_YEAR = 2026


def clean_for_json(obj):
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(obj, 8)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else round(v, 8)
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    return obj


def build_trailer_meta():
    """One row per container_id from Asset Detail — matches modeling norm_id."""
    ad = pd.read_excel(ASSET_XLSX, sheet_name=0, engine="openpyxl")
    ad.columns = [str(c).strip() for c in ad.columns]
    ad = ad[ad["Associated Trailer"].notna()].copy()
    tail = (
        ad["Associated Trailer"].astype(str).str.strip()
        .str.replace(r"\D+", "", regex=True)
        .str[-6:].str.zfill(6)
    )
    ad["container_id"] = pd.to_numeric(tail, errors="coerce")
    ad = ad.dropna(subset=["container_id"])
    ad["container_id"] = ad["container_id"].astype(int)

    def first_nonempty(s):
        for v in s:
            if pd.isna(v):
                continue
            t = str(v).strip()
            if t and t.lower() != "nan":
                return v
        return np.nan

    return ad.groupby("container_id", as_index=False).agg(
        trailer_year=("Trailer Year", lambda s: pd.to_numeric(s, errors="coerce").max()),
        trailer_type=("Trailer Type", first_nonempty),
        reefer_make=("Make", "first"),
        reefer_model=("Model", "first"),
    )


def clean_feature_name(name):
    """Strip pipeline prefixes like num__ and cat__consignee_ for display."""
    if name.startswith("num__"):
        return name[5:]
    if name.startswith("cat__"):
        return name[5:]
    return name


def main():
    print("Loading CPM1 artifacts...")

    # 1. Unit lookup predictions
    lookup = pd.read_csv(COMBINED / "unit_lookup_predictions.csv")
    print(f"  Predictions: {len(lookup)} units")

    # 2. Per-unit SHAP drivers
    top_shap = pd.read_csv(COMBINED / "unit_lookup_top_shap_drivers.csv")

    # 3. Global SHAP importance
    global_gen = pd.read_csv(COMBINED / "global_shap_importance_genrep.csv")
    global_pm = pd.read_csv(COMBINED / "global_shap_importance_pm.csv")

    # 4. Metrics
    combined_metrics = json.loads((COMBINED / "combined_metrics_summary.json").read_text())
    gen_metrics = json.loads((GEN_DIR / "metrics_summary.json").read_text())
    pm_metrics = json.loads((PM_DIR / "metrics_summary.json").read_text())

    # 5. Feature schema
    schema = json.loads((COMBINED / "website_feature_schema.json").read_text())

    # 6. PM snapshot for shop context
    pm_snap = pd.read_csv(PM_DIR / "model_dataset_snapshot.csv")
    shop_ctx = pm_snap[["container_id", "dominant_shop", "n_shops"]].copy()

    # 7. Trailer metadata
    print("  Building trailer metadata from Asset Detail...")
    trailer_meta = build_trailer_meta()

    # --- Merge everything into fleet records ---
    fleet = lookup.merge(shop_ctx, on="container_id", how="left")
    fleet = fleet.merge(trailer_meta, on="container_id", how="left")

    # Compute trailer age
    fleet["trailer_age"] = fleet["trailer_year"].apply(
        lambda y: max(0, REF_YEAR - int(y)) if pd.notna(y) and y > 1900 else None
    )

    # Build SHAP driver arrays per unit
    shap_map = {}
    for _, row in top_shap.iterrows():
        cid = int(row["container_id"])
        gen_drivers = []
        pm_drivers = []
        for i in range(1, 11):
            gf = row.get(f"genrep_top{i}_feature", "")
            gs = row.get(f"genrep_top{i}_shap", 0.0)
            if pd.notna(gf) and str(gf).strip():
                gen_drivers.append({"feature": clean_feature_name(str(gf)), "shap": float(gs) if pd.notna(gs) else 0.0})
            pf = row.get(f"pm_top{i}_feature", "")
            ps = row.get(f"pm_top{i}_shap", 0.0)
            if pd.notna(pf) and str(pf).strip():
                pm_drivers.append({"feature": clean_feature_name(str(pf)), "shap": float(ps) if pd.notna(ps) else 0.0})
        shap_map[cid] = {"genrep": gen_drivers, "pm": pm_drivers}

    # Build fleet records list
    fleet_records = []
    for _, row in fleet.iterrows():
        cid = int(row["container_id"])
        drivers = shap_map.get(cid, {"genrep": [], "pm": []})
        rec = {
            "container_id": cid,
            "pred_genrep_cpm": row["pred_genrep_cpm"],
            "pred_pm_cpm": row["pred_pm_cpm"],
            "pred_total_cpm": row["pred_total_cpm"],
            "actual_genrep_cpm": row["actual_genrep_cpm"],
            "actual_pm_cpm": row["actual_pm_cpm"],
            "actual_total_cpm": row["actual_total_cpm"],
            "total_miles": row["total_miles"],
            "rail_miles": row["rail_miles"],
            "highway_miles": row["highway_miles"],
            "pred_genrep_cost": row["pred_genrep_cost"],
            "pred_pm_cost": row["pred_pm_cost"],
            "pred_total_cost": row["pred_total_cost"],
            "actual_total_cost": row["actual_total_cost"],
            "trailer_year": row.get("trailer_year"),
            "trailer_age": row.get("trailer_age"),
            "trailer_type": row.get("trailer_type"),
            "reefer_make": row.get("reefer_make"),
            "reefer_model": row.get("reefer_model"),
            "dominant_shop": row.get("dominant_shop"),
            "n_shops": row.get("n_shops"),
            "genrep_shap": drivers["genrep"][:5],
            "pm_shap": drivers["pm"][:5],
        }
        fleet_records.append(rec)

    fleet_records = clean_for_json(fleet_records)

    # --- Build summary ---
    # Compute fleet-level stats
    avg_genrep_cpm = float(lookup["actual_genrep_cpm"].mean())
    avg_pm_cpm = float(lookup["actual_pm_cpm"].mean())
    avg_total_cpm = float(lookup["actual_total_cpm"].mean())
    avg_total_miles = float(lookup["total_miles"].mean())
    q20 = float(lookup["pred_total_cpm"].quantile(0.20))
    q80 = float(lookup["pred_total_cpm"].quantile(0.80))
    median_pred = float(lookup["pred_total_cpm"].median())

    # Clean global SHAP — filter to non-zero and clean names
    gen_shap_list = []
    for _, r in global_gen.iterrows():
        if r["mean_abs_shap"] > 0:
            gen_shap_list.append({
                "feature": clean_feature_name(r["feature"]),
                "mean_abs_shap": round(float(r["mean_abs_shap"]), 8),
            })

    pm_shap_list = []
    for _, r in global_pm.iterrows():
        if r["mean_abs_shap"] > 0:
            pm_shap_list.append({
                "feature": clean_feature_name(r["feature"]),
                "mean_abs_shap": round(float(r["mean_abs_shap"]), 8),
            })

    # Feature glossary
    glossary = {
        "genrep_lines_per_1k_mi": "GENREP line-items per 1,000 miles",
        "wash_events": "Historical count of wash-type GENREP events",
        "wash_gap_median_days": "Typical days between wash events (median)",
        "mile_x_reactive_pm": "Interaction: miles x reactive PM share",
        "genrep_orders_per_1k_mi": "GENREP work order frequency per 1,000 miles",
        "consignee": "Customer/consignee profile",
        "service_fresh_share": "Share of service profile in fresh loads",
        "pm_orders_per_1k_mi": "PM work order frequency per 1,000 miles",
        "reactive_pm_share": "Share of PM done during mixed repair context",
        "total_miles": "Total miles (CPM denominator)",
        "rail_share": "Fraction of miles on rail",
        "pm_lines_per_1k_mi": "PM line-items per 1,000 miles",
        "pm_line_count": "Total PM line-items for the unit",
        "pm_order_count": "Total PM work orders for the unit",
        "pm_repeat_events": "Repeat PM-cycle observations",
        "pm_delay_mean_days": "Average PM timing deviation (days)",
        "pm_delay_pos_rate": "Rate of late PM cycles",
        "pm_delay_neg_rate": "Rate of early PM cycles",
    }

    summary = {
        "cohort_rows": combined_metrics["cohort_rows"],
        "model_combo": combined_metrics["model_combo"],
        "total_train_r2": combined_metrics["total_train_r2"],
        "total_train_mae_cpm": combined_metrics["total_train_mae_cpm"],
        "total_train_rmse_cpm": combined_metrics["total_train_rmse_cpm"],
        "total_train_mae_dollars": combined_metrics["total_train_mae_dollars"],
        "total_train_rmse_dollars": combined_metrics["total_train_rmse_dollars"],
        "avg_genrep_cpm": avg_genrep_cpm,
        "avg_pm_cpm": avg_pm_cpm,
        "avg_total_cpm": avg_total_cpm,
        "avg_total_miles": avg_total_miles,
        "median_pred_total_cpm": median_pred,
        "q20_pred_total_cpm": q20,
        "q80_pred_total_cpm": q80,
        "genrep_features_used": combined_metrics["genrep_features_used"],
        "pm_features_used": combined_metrics["pm_features_used"],
        "genrep_metrics": gen_metrics,
        "pm_metrics": pm_metrics,
        "global_shap": {
            "genrep": gen_shap_list,
            "pm": pm_shap_list,
        },
        "feature_schema": schema,
        "feature_glossary": glossary,
    }
    summary = clean_for_json(summary)

    # --- Write output files ---
    print("Writing webapp JSON files...")

    with open(WEBAPP_DATA / "cpm_fleet.json", "w") as f:
        json.dump(fleet_records, f)
    print(f"  cpm_fleet.json: {len(fleet_records)} units, {(WEBAPP_DATA / 'cpm_fleet.json').stat().st_size / 1024:.0f} KB")

    with open(WEBAPP_DATA / "cpm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  cpm_summary.json: {(WEBAPP_DATA / 'cpm_summary.json').stat().st_size / 1024:.0f} KB")

    print("Done.")


if __name__ == "__main__":
    main()
