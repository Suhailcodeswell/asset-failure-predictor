"""Refresh per-unit model INPUTS from new reefer data — without retraining.

Architecture: the failure risk model is TRAINED ONCE (build_risk_engine.py, on
2025 data). New monthly reefer data does NOT retrain it; it only updates each
unit's input snapshot — the current feature vector the frozen model scores to
predict the unit's NEXT trip.

Flow:
    1. (optional) rebuild trip features from data/raw (which now includes any
       newly-ingested months) via build_trip_features_v3.py
    2. recompute each unit's latest-trip snapshot (shared with training)
    3. overwrite models/risk_engine/unit_snapshots.json
    4. the model artifacts (xgb_model, calibrated_model, feature_meta, …) are
       left untouched

Usage:
    python pipeline/refresh_inputs.py                 # use existing features csv
    python pipeline/refresh_inputs.py --rebuild-features   # rebuild from data/raw first
    python pipeline/refresh_inputs.py --score-sample 5     # show fresh predictions
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Repo root = nearest ancestor containing paths.py (depth-independent).
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))  # so `import snapshots` / `risk_scorer` work

from snapshots import build_unit_snapshots  # noqa: E402

FEATURES_CSV = REPO_ROOT / "data" / "processed" / "training" / "trip_features_v3.csv"
ENGINE_DIR = REPO_ROOT / "models" / "risk_engine"


def rebuild_features() -> None:
    print("Rebuilding trip features from data/raw …")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "build_trip_features_v3.py")],
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise SystemExit(f"feature rebuild failed (exit {proc.returncode})")


def _prepare_encoded_df(features_csv: Path, engine_dir: Path):
    """Load trip features and reproduce build_risk_engine's preprocessing so the
    snapshots match the FROZEN model: merge highway features, then encode
    categoricals with the SAVED label_encoders (not re-fit — model is frozen)."""
    meta = json.loads((engine_dir / "feature_meta.json").read_text())
    feature_cols = meta["feature_cols"]
    cat_cols = meta.get("cat_cols", [])
    label_encoders = json.loads((engine_dir / "label_encoders.json").read_text())

    df = pd.read_csv(features_csv, low_memory=False)

    # Merge highway features (mirror build_risk_engine.py §1)
    final_csv = features_csv.parent / "trip_features_final.csv"
    if final_csv.exists():
        try:
            final = pd.read_csv(final_csv, low_memory=False)
            hwy = [c for c in final.columns if c.startswith("hwy_")] + ["pm_cum_overdue_days"]
            hwy = [c for c in hwy if c in final.columns]
            if hwy and {"manifest", "leg"}.issubset(df.columns) and {"manifest", "leg"}.issubset(final.columns):
                df["_key"] = df["manifest"].astype(str) + "_" + df["leg"].astype(str)
                final["_key"] = final["manifest"].astype(str) + "_" + final["leg"].astype(str)
                df = df.merge(final[["_key"] + hwy], on="_key", how="left").drop(columns=["_key"])
        except Exception as e:  # pragma: no cover
            print(f"  highway merge skipped: {e}")

    # Encode categoricals with the SAVED maps (string -> int), unseen -> MISSING.
    for col in cat_cols:
        if col in df.columns:
            le = label_encoders.get(col, {})
            miss = le.get("MISSING", 0)
            df[col] = df[col].fillna("MISSING").astype(str).map(le).fillna(miss).astype(int)

    return df, feature_cols


def refresh(features_csv: Path = FEATURES_CSV, engine_dir: Path = ENGINE_DIR) -> dict:
    df, feature_cols = _prepare_encoded_df(features_csv, engine_dir)

    snap_path = engine_dir / "unit_snapshots.json"
    prev_units = set()
    if snap_path.exists():
        # back up before overwrite (rollback safety; model is untouched)
        backup = engine_dir / "unit_snapshots.prev.json"
        backup.write_text(snap_path.read_text())
        prev_units = set(json.loads(snap_path.read_text()).keys())

    snapshots = build_unit_snapshots(df, feature_cols)
    snap_path.write_text(json.dumps(snapshots, indent=2, default=str))

    new_units = set(snapshots) - prev_units
    latest = max((s.get("last_trip_date", "") for s in snapshots.values()), default="")
    report = {
        "units": len(snapshots),
        "new_units": sorted(new_units),
        "latest_trip_date": latest,
        "features_csv": str(features_csv.relative_to(REPO_ROOT)),
        "refreshed_at": datetime.now().isoformat(timespec="seconds"),
        "model_retrained": False,
    }
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Refresh frozen-model inputs from new data (no retrain).")
    p.add_argument("--rebuild-features", action="store_true",
                   help="Rebuild trip features from data/raw before refreshing snapshots.")
    p.add_argument("--features-csv", type=Path, default=FEATURES_CSV)
    p.add_argument("--engine-dir", type=Path, default=ENGINE_DIR)
    p.add_argument("--score-sample", type=int, default=0,
                   help="After refresh, print fresh next-trip predictions for N highest-risk units.")
    args = p.parse_args(argv)

    if args.rebuild_features:
        rebuild_features()

    report = refresh(args.features_csv, args.engine_dir)
    print("\nINPUT REFRESH COMPLETE (model NOT retrained)")
    print(f"  units refreshed : {report['units']}")
    print(f"  new units       : {len(report['new_units'])} {report['new_units'][:8]}")
    print(f"  latest trip date: {report['latest_trip_date']}")
    print(f"  snapshots -> {args.engine_dir / 'unit_snapshots.json'}")

    if args.score_sample > 0:
        from risk_scorer import RiskScorer
        scorer = RiskScorer(str(args.engine_dir))
        # Score every unit's "next trip" on a neutral template and rank by risk.
        print(f"\n  Fresh next-trip risk (top {args.score_sample}) using refreshed inputs:")
        results = []
        snaps = json.loads((args.engine_dir / "unit_snapshots.json").read_text())
        for uid in snaps:
            try:
                r = scorer.score(uid, {})
                results.append((uid, r.get("risk_score", 0.0), r.get("risk_tier", "")))
            except Exception:
                continue
        results.sort(key=lambda x: x[1], reverse=True)
        for uid, score, tier in results[: args.score_sample]:
            print(f"    {uid:>10}  risk={score:6.2%}  {tier}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
