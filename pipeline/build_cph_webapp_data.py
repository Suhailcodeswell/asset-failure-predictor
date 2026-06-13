#!/usr/bin/env python3
"""Convert new CPH reefer_predictor_deploy models + data into webapp JSON.

Reads:
  cph/reefer_predictor_deploy/data/*.csv
  cph/reefer_predictor_deploy/models/*.pkl
  cph/reefer_predictor_deploy/data/fuel_lph_deployment.joblib

Writes:
  data/cph_fleet.json       (per-unit predictions + profiles)
  data/cph_summary.json     (fleet metrics + model info)
  data/cph_pm_model.json    (PM RF trees for scenario planner)
  data/cph_genrep_model.json (GENREP two-stage trees for scenario planner)
"""

import json, pickle, math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPH_DIR = ROOT / "cph" / "reefer_predictor_deploy"
WEBAPP_DATA = ROOT / "data"

PM_CODES = {
    "000-011": "Reefer 360 PM Check",
    "000-012": "Major Reefer Service",
    "000-013": "Minor Reefer Service",
    "000-014": "Pre-Trip Inspection",
    "000-038": "Alternator Service",
    "000-047": "Air Filter Service",
    "000-070": "Battery Replacement",
    "000-071": "Fuel Polishing",
}


def clean(obj):
    """Replace NaN/Inf with None for JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 8)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else round(v, 8)
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---- Export sklearn RF trees to JSON ----

def export_rf_trees(rf, is_classifier=False):
    """Extract trees from a RandomForestRegressor or RandomForestClassifier."""
    trees = []
    for est in rf.estimators_:
        tree = est.tree_
        if is_classifier:
            # Classifier: normalize to class probabilities
            vals = []
            for v in tree.value:
                probs = v.flatten()
                total = probs.sum()
                vals.append((probs / total).tolist() if total > 0 else probs.tolist())
        else:
            # Regressor: single float per node
            vals = [float(v.flatten()[0]) for v in tree.value]

        trees.append({
            "left_children": tree.children_left.tolist(),
            "right_children": tree.children_right.tolist(),
            "split_features": tree.feature.tolist(),
            "thresholds": tree.threshold.tolist(),
            "values": vals,
        })
    return trees


def export_pipeline_rf(pipeline):
    """Export a Pipeline(ColumnTransformer + RandomForestRegressor) to JSON."""
    steps = list(pipeline.named_steps.items())
    preprocessor = steps[0][1]
    rf = steps[1][1]

    numeric_features = []
    categorical_features = []
    cat_categories = {}
    has_scaler = False
    scaler_mean = []
    scaler_scale = []

    for trans_name, transformer, columns in preprocessor.transformers_:
        col_list = list(columns)
        if trans_name in ('num', 'numeric'):
            numeric_features = col_list
            if hasattr(transformer, 'named_steps'):
                for sn, ss in transformer.named_steps.items():
                    if hasattr(ss, 'mean_'):
                        has_scaler = True
                        scaler_mean = ss.mean_.tolist()
                        scaler_scale = ss.scale_.tolist()
            elif hasattr(transformer, 'mean_'):
                has_scaler = True
                scaler_mean = transformer.mean_.tolist()
                scaler_scale = transformer.scale_.tolist()
        elif trans_name in ('cat', 'categorical'):
            categorical_features = col_list
            ohe = None
            if hasattr(transformer, 'named_steps'):
                for sn, ss in transformer.named_steps.items():
                    if hasattr(ss, 'categories_'):
                        ohe = ss
            elif hasattr(transformer, 'categories_'):
                ohe = transformer
            if ohe:
                for i, cf in enumerate(categorical_features):
                    cat_categories[cf] = [str(c) for c in ohe.categories_[i]]

    result = {
        "trees": export_rf_trees(rf, is_classifier=False),
        "n_trees": len(rf.estimators_),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "cat_categories": cat_categories,
    }
    if has_scaler:
        result["scaler_mean"] = scaler_mean
        result["scaler_scale"] = scaler_scale
    return result


def export_genrep_two_stage(genrep_dict):
    """Export the two-stage GENREP model (classifier + 2 regressors) to JSON."""
    classifier = genrep_dict["classifier"]
    reg_normal = genrep_dict["reg_normal"]
    reg_pit = genrep_dict["reg_pit"]

    return {
        "class_features": genrep_dict["class_features"],
        "reg_features": genrep_dict["reg_features"],
        "classifier": {
            "trees": export_rf_trees(classifier, is_classifier=True),
            "n_trees": len(classifier.estimators_),
            "n_classes": int(classifier.n_classes_),
        },
        "reg_normal": {
            "trees": export_rf_trees(reg_normal, is_classifier=False),
            "n_trees": len(reg_normal.estimators_),
        },
        "reg_pit": {
            "trees": export_rf_trees(reg_pit, is_classifier=False),
            "n_trees": len(reg_pit.estimators_),
        },
    }


# ---- Prediction functions (verify against sklearn) ----

def rf_predict_vec(trees, vec):
    """Walk RF trees and average predictions."""
    total = 0.0
    for tree in trees:
        node = 0
        while True:
            left = tree["left_children"][node]
            if left == -1:
                total += tree["values"][node]
                break
            feat_idx = tree["split_features"][node]
            threshold = tree["thresholds"][node]
            val = vec[feat_idx] if feat_idx < len(vec) else 0
            if val is None or (isinstance(val, float) and math.isnan(val)):
                node = left
            elif val <= threshold:
                node = left
            else:
                node = tree["right_children"][node]
    return total / len(trees)


def classify_vec(trees, vec, n_classes=2):
    """Walk RF classifier trees, return majority class."""
    class_votes = [0.0] * n_classes
    for tree in trees:
        node = 0
        while True:
            left = tree["left_children"][node]
            if left == -1:
                probs = tree["values"][node]
                for c in range(min(n_classes, len(probs))):
                    class_votes[c] += probs[c]
                break
            feat_idx = tree["split_features"][node]
            threshold = tree["thresholds"][node]
            val = vec[feat_idx] if feat_idx < len(vec) else 0
            if val is None or (isinstance(val, float) and math.isnan(val)):
                node = left
            elif val <= threshold:
                node = left
            else:
                node = tree["right_children"][node]
    return 1 if class_votes[1] > class_votes[0] else 0


def encode_pipeline_vec(params, model_json):
    """Build feature vector for a pipeline-exported model."""
    vec = []
    for nf in model_json["numeric_features"]:
        val = params.get(nf, 0)
        try:
            val = float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            val = 0.0
        if model_json.get("scaler_mean"):
            idx = model_json["numeric_features"].index(nf)
            val = (val - model_json["scaler_mean"][idx]) / model_json["scaler_scale"][idx]
        vec.append(val)
    for cf in model_json["categorical_features"]:
        cats = model_json["cat_categories"].get(cf, [])
        val = str(params.get(cf, "")).strip()
        for cat in cats:
            vec.append(1.0 if val == cat else 0.0)
    return vec


def main():
    print("Loading new CPH data and models...")

    # Load datasets
    next_level = pd.read_csv(CPH_DIR / "data" / "Next_Level_Dataset.csv")
    advanced = pd.read_csv(CPH_DIR / "data" / "Advanced_Modeling_Dataset.csv")
    fuel_df = pd.read_csv(CPH_DIR / "data" / "test_fuel_data.csv")
    print(f"  Next_Level: {len(next_level)} rows, Advanced: {len(advanced)} rows, Fuel: {len(fuel_df)} rows")

    # Load models (saved with joblib, not raw pickle)
    pm_pipeline = joblib.load(CPH_DIR / "models" / "pm_model.pkl")
    genrep_dict = joblib.load(CPH_DIR / "models" / "genrep_model.pkl")
    # Fuel model has sklearn version issues; use measured LPH from test_fuel_data.csv directly
    # fuel_bundle = joblib.load(CPH_DIR / "data" / "fuel_lph_deployment.joblib")

    # ---- Export PM model to JSON ----
    print("Exporting PM model...")
    pm_json = export_pipeline_rf(pm_pipeline)
    print(f"  PM trees: {pm_json['n_trees']}, numeric: {pm_json['numeric_features']}, cat: {pm_json['categorical_features']}")

    # ---- Export GENREP model to JSON ----
    print("Exporting GENREP two-stage model...")
    genrep_json = export_genrep_two_stage(genrep_dict)
    print(f"  Classifier trees: {genrep_json['classifier']['n_trees']}")
    print(f"  Reg normal trees: {genrep_json['reg_normal']['n_trees']}")
    print(f"  Reg pit trees: {genrep_json['reg_pit']['n_trees']}")

    # ---- Pre-compute predictions for all units ----
    print("Pre-computing fleet predictions...")

    # PM features
    pm_features = ["COMPCODE", "TARGET_DAYS", "TOTAL_MAINTENANCE_COUNT",
                    "ESTIMATED_2YR_HOURS", "NEGLECT_INTENSITY", "SHOPID",
                    "MAKE", "MODEL", "MODELYEAR"]

    # GENREP features
    class_features = genrep_dict["class_features"]
    reg_features = genrep_dict["reg_features"]

    # Get unique units from Next_Level dataset
    units = sorted(next_level["UNITNUMBER"].unique())
    print(f"  Units: {len(units)}")

    # Build fuel lookup from measured LPH in test_fuel_data.csv
    fuel_lookup = {}
    if "UNITNUMBER" in fuel_df.columns and "fuel_liters_per_engine_hour_delta" in fuel_df.columns:
        for _, frow in fuel_df.iterrows():
            uid = frow.get("UNITNUMBER")
            lph = frow.get("fuel_liters_per_engine_hour_delta")
            if pd.notna(uid) and pd.notna(lph) and float(lph) > 0:
                fuel_lookup[str(uid)] = round(float(lph), 6)
        print(f"  Fuel LPH lookup: {len(fuel_lookup)} units (from measured data)")

    # Build per-unit records
    fleet_records = []
    for uid in units:
        uid_str = str(uid)
        rows = next_level[next_level["UNITNUMBER"] == uid]
        if rows.empty:
            continue

        first = rows.iloc[0]

        # Profile
        make = str(first.get("MAKE", "")) if pd.notna(first.get("MAKE")) else ""
        model = str(first.get("MODEL", "")) if pd.notna(first.get("MODEL")) else ""
        modelyear = int(first["MODELYEAR"]) if pd.notna(first.get("MODELYEAR")) else None
        est_hours = float(first["ESTIMATED_2YR_HOURS"]) if pd.notna(first.get("ESTIMATED_2YR_HOURS")) else 0

        # Telematics
        has_alarm = int(first.get("HAS_TELEMATICS_ALARM", 0)) if pd.notna(first.get("HAS_TELEMATICS_ALARM")) else 0
        total_alarms = first.get("TOTAL_ALARMS")
        min_batt = first.get("MIN_BATTERY_VOLTAGE")
        crit_shutdowns = first.get("CRITICAL_SHUTDOWNS")
        missing_restart = first.get("MISSING_RESTART_FLAG")
        days_to_failure = first.get("AVG_DAYS_TO_FAILURE")

        # PM predictions — one per COMPCODE track, summed
        pm_breakdown = []
        # Use Advanced dataset for PM rows (it has PM_COST_PER_HOUR actuals too)
        pm_rows = advanced[advanced["UNITNUMBER"] == uid]
        if pm_rows.empty:
            pm_rows = rows  # fallback to next_level
        total_pm_cph = 0.0
        for _, pr in pm_rows.iterrows():
            compcode = str(pr.get("COMPCODE", ""))
            if not compcode or compcode == "nan":
                continue
            try:
                x = pd.DataFrame([{f: pr.get(f) for f in pm_features}])
                pred_pm = max(0, float(pm_pipeline.predict(x)[0]))
            except Exception:
                pred_pm = 0.0
            pm_breakdown.append({
                "compcode": compcode,
                "name": PM_CODES.get(compcode, compcode),
                "pred_cph": round(pred_pm, 6),
                "actual_cph": round(float(pr["PM_COST_PER_HOUR"]), 6) if pd.notna(pr.get("PM_COST_PER_HOUR")) else None,
            })
            total_pm_cph += pred_pm

        # GENREP prediction — two-stage, averaged across rows
        genrep_preds = []
        for _, gr in rows.iterrows():
            try:
                cls_vec = [float(gr.get(f, 0)) if pd.notna(gr.get(f)) else 0 for f in class_features]
                is_pit = genrep_dict["classifier"].predict([cls_vec])[0]
                reg_vec = [float(gr.get(f, 0)) if pd.notna(gr.get(f)) else 0 for f in reg_features]
                if is_pit == 1:
                    pred = float(genrep_dict["reg_pit"].predict([reg_vec])[0])
                else:
                    pred = float(genrep_dict["reg_normal"].predict([reg_vec])[0])
                genrep_preds.append(max(0, pred))
            except Exception:
                pass
        avg_genrep_cph = sum(genrep_preds) / len(genrep_preds) if genrep_preds else 0.0

        # Actual GENREP from data
        actual_genrep = first.get("GENREP_COST_PER_HOUR")
        actual_genrep = round(float(actual_genrep), 6) if pd.notna(actual_genrep) else None

        # Fuel
        fuel_lph = fuel_lookup.get(uid_str)

        # Grand total
        grand_total = total_pm_cph + avg_genrep_cph
        if fuel_lph:
            fuel_cost_default = fuel_lph * 1.45  # default $1.45/L
            grand_total += fuel_cost_default

        rec = {
            "unit": uid_str,
            "make": make,
            "model": model,
            "modelyear": modelyear,
            "est_2yr_hours": round(est_hours, 1),
            "has_telematics_alarm": has_alarm,
            "total_alarms": int(total_alarms) if pd.notna(total_alarms) else None,
            "min_battery_voltage": round(float(min_batt), 2) if pd.notna(min_batt) else None,
            "critical_shutdowns": int(crit_shutdowns) if pd.notna(crit_shutdowns) else None,
            "missing_restart_flag": int(missing_restart) if pd.notna(missing_restart) else None,
            "avg_days_to_failure": round(float(days_to_failure), 1) if pd.notna(days_to_failure) else None,
            "pm_breakdown": pm_breakdown,
            "pred_pm_cph": round(total_pm_cph, 6),
            "pred_genrep_cph": round(avg_genrep_cph, 6),
            "actual_genrep_cph": actual_genrep,
            "fuel_lph": fuel_lph,
            "pred_total_cph": round(grand_total, 6),
            "is_money_pit": int(first.get("IS_MONEY_PIT", 0)) if pd.notna(first.get("IS_MONEY_PIT")) else 0,
            "is_lemon": int(first.get("IS_LEMON", 0)) if pd.notna(first.get("IS_LEMON")) else 0,
            "neglect_intensity": round(float(first.get("NEGLECT_INTENSITY", 0)), 4) if pd.notna(first.get("NEGLECT_INTENSITY")) else 0,
            "double_bill_ratio": round(float(first.get("DOUBLE_BILL_RATIO", 0)), 4) if pd.notna(first.get("DOUBLE_BILL_RATIO")) else 0,
        }
        fleet_records.append(rec)

    fleet_records = clean(fleet_records)
    print(f"  Fleet records built: {len(fleet_records)}")

    # ---- Build summary ----
    valid = [r for r in fleet_records if r.get("est_2yr_hours") and r["est_2yr_hours"] >= 100]
    avg_pm = sum(r["pred_pm_cph"] for r in valid) / len(valid) if valid else 0
    avg_genrep = sum(r["pred_genrep_cph"] for r in valid) / len(valid) if valid else 0
    avg_total = sum(r["pred_total_cph"] for r in valid) / len(valid) if valid else 0
    fuel_units = [r for r in fleet_records if r.get("fuel_lph") is not None]

    # Get unique values for scenario planner dropdowns
    makes = sorted(next_level["MAKE"].dropna().unique().tolist())
    models_list = sorted(next_level["MODEL"].dropna().unique().tolist())
    shops = sorted(advanced["SHOPID"].dropna().unique().tolist()) if "SHOPID" in advanced.columns else []
    compcodes = sorted(advanced["COMPCODE"].dropna().unique().tolist()) if "COMPCODE" in advanced.columns else []

    summary = {
        "total_units": len(fleet_records),
        "units_with_hours": len(valid),
        "units_with_fuel": len(fuel_units),
        "avg_pm_cph": round(avg_pm, 6),
        "avg_genrep_cph": round(avg_genrep, 6),
        "avg_total_cph": round(avg_total, 6),
        "pm_codes": PM_CODES,
        "makes": makes,
        "models": models_list,
        "shops": shops,
        "compcodes": compcodes,
        "genrep_class_features": genrep_dict["class_features"],
        "genrep_reg_features": genrep_dict["reg_features"],
    }
    summary = clean(summary)

    # ---- Validate predictions ----
    print("\nValidation — spot-checking 3 units...")
    for rec in fleet_records[:3]:
        print(f"  {rec['unit']}: PM=${rec['pred_pm_cph']:.4f}/hr, GENREP=${rec['pred_genrep_cph']:.4f}/hr, "
              f"fuel={'%.4f L/hr' % rec['fuel_lph'] if rec.get('fuel_lph') else 'N/A'}, total=${rec['pred_total_cph']:.4f}/hr")

    # ---- Write files ----
    print("\nWriting webapp JSON files...")

    with open(WEBAPP_DATA / "cph_fleet.json", "w") as f:
        json.dump(fleet_records, f)
    print(f"  cph_fleet.json: {len(fleet_records)} units, {(WEBAPP_DATA / 'cph_fleet.json').stat().st_size / 1024:.0f} KB")

    with open(WEBAPP_DATA / "cph_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  cph_summary.json: {(WEBAPP_DATA / 'cph_summary.json').stat().st_size / 1024:.0f} KB")

    pm_json = clean(pm_json)
    with open(WEBAPP_DATA / "cph_pm_model.json", "w") as f:
        json.dump(pm_json, f)
    print(f"  cph_pm_model.json: {(WEBAPP_DATA / 'cph_pm_model.json').stat().st_size / 1024:.0f} KB")

    genrep_json = clean(genrep_json)
    with open(WEBAPP_DATA / "cph_genrep_model.json", "w") as f:
        json.dump(genrep_json, f)
    print(f"  cph_genrep_model.json: {(WEBAPP_DATA / 'cph_genrep_model.json').stat().st_size / 1024:.0f} KB")

    print("Done.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
