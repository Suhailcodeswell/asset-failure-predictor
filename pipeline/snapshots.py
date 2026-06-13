"""Per-unit input snapshots for the (frozen) failure risk model.

A "snapshot" is one reefer unit's CURRENT input feature vector — the values the
trained model scores to predict that unit's NEXT trip. It is the most recent
trip's engineered features plus a few derived fields (PM-overdue, trip/shutdown
counts).

This module is the single source of truth for snapshot construction so that
TRAIN-time (build_risk_engine.py) and INPUT-REFRESH-time (refresh_inputs.py)
produce byte-identical snapshots. The model is trained once; new reefer data
only changes these inputs, not the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_unit_snapshots(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Build {reefer_unit: snapshot_dict} from a trip-features DataFrame.

    Args:
        df: trip-level features; must contain 'reefer_unit', 'trip_start',
            'shutdown' and every column in feature_cols.
        feature_cols: the model's deployed feature list (from feature_meta.json).

    Returns:
        dict mapping unit id -> snapshot dict (model features + PM-overdue +
        metadata), with NaN preserved as None so the imputer handles it.
    """
    df_sorted = df.sort_values("trip_start")
    unit_features: dict = {}

    for unit_id, group in df_sorted.groupby("reefer_unit"):
        last_row = group.iloc[-1]
        snapshot: dict = {}

        # All model features — latest trip's values as the unit's current input.
        for col in feature_cols:
            val = last_row[col] if col in last_row.index else np.nan
            if pd.notna(val):
                if isinstance(val, (np.floating, float)):
                    snapshot[col] = float(val)
                elif isinstance(val, (np.integer, int)):
                    snapshot[col] = int(val)
                else:
                    snapshot[col] = str(val)
            else:
                snapshot[col] = None  # preserve NaN as null for the imputer

        # PM compliance derived from days_since_last_pm vs policy.
        is_slxi = last_row.get("is_slxi", 0) if "is_slxi" in last_row.index else 0
        policy = 90 if is_slxi else 180
        dspm = last_row.get("days_since_last_pm", np.nan) if "days_since_last_pm" in last_row.index else np.nan
        if pd.notna(dspm):
            snapshot["is_pm_overdue"] = 1 if dspm > policy else 0
            snapshot["days_pm_overdue"] = max(0, dspm - policy)
        else:
            snapshot["is_pm_overdue"] = None
            snapshot["days_pm_overdue"] = None

        snapshot["last_trip_date"] = str(last_row["trip_start"])
        snapshot["total_trips"] = int(len(group))
        snapshot["total_shutdowns"] = int(group["shutdown"].sum())
        snapshot["shutdown_rate"] = round(float(group["shutdown"].mean()), 4)
        unit_features[unit_id] = snapshot

    return unit_features
