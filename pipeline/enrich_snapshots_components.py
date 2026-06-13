"""
Enrich data/unit_snapshots.json with component repair fields
from the training data (trip_features_v3.csv).

For each reefer unit, takes the LAST trip and extracts component cumulative,
90-day, and 365-day repair count fields, then merges them into the existing
unit_snapshots.json without overwriting any existing fields.
"""

import json
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "data/processed/training/trip_features_v3.csv")
SNAPSHOTS_PATH = os.path.join(PROJECT_ROOT, "data/unit_snapshots.json")

# Component fields to extract (_cum, _90d, _365d variants)
COMPONENT_BASES = [
    "compressor",
    "evaporator",
    "diagnosis",
    "electrical_charging",
    "fuel_system",
    "condenser",
    "refrigerant_lines",
    "starter_cranking",
    "defrost",
    "engine",
]

SUFFIXES = ["_cum", "_90d", "_365d"]

# Additional non-component fields missing from snapshots but present in CSV
EXTRA_FIELDS = [
    "nonpm_cum_cost",
    "nonpm_cum_count",
    "nonpm_count_90d",
    "nonpm_cost_90d",
    "pm_compliance_ratio",
    "pm_late_count",
    "pm_total_intervals",
]

# Build full list of target columns
TARGET_COLS = []
for base in COMPONENT_BASES:
    for suffix in SUFFIXES:
        TARGET_COLS.append(f"{base}{suffix}")
TARGET_COLS.extend(EXTRA_FIELDS)

print(f"Target component fields ({len(TARGET_COLS)}):")
for col in TARGET_COLS:
    print(f"  {col}")
print()

# Step 1: Read CSV
print("Reading trip_features_v3.csv...")
df = pd.read_csv(CSV_PATH)
print(f"  Total rows: {len(df)}")
print(f"  Unique reefer units: {df['reefer_unit'].nunique()}")

# Verify all target columns exist in CSV
missing_cols = [c for c in TARGET_COLS if c not in df.columns]
if missing_cols:
    print(f"  WARNING: Missing columns in CSV: {missing_cols}")
    TARGET_COLS = [c for c in TARGET_COLS if c in df.columns]

# Step 2: For each unit, get the LAST trip (by trip_start then index)
print("\nExtracting last-trip component values per unit...")
df["trip_start"] = pd.to_datetime(df["trip_start"])
df = df.sort_values(["reefer_unit", "trip_start"])

# Take the last row per unit (which is the latest trip after sorting)
last_trips = df.groupby("reefer_unit").last().reset_index()
print(f"  Units with last-trip data: {len(last_trips)}")

# Step 3: Load existing snapshots
print(f"\nLoading {SNAPSHOTS_PATH}...")
with open(SNAPSHOTS_PATH, "r") as f:
    snapshots = json.load(f)
print(f"  Units in snapshots: {len(snapshots)}")

# Step 4: Merge component fields + compute last_pm_date
enriched_count = 0
fields_added_total = 0
skipped_existing = 0

for _, row in last_trips.iterrows():
    unit_id = row["reefer_unit"]
    if unit_id not in snapshots:
        continue

    enriched_count += 1
    for col in TARGET_COLS:
        # Only ADD missing fields, don't overwrite existing ones
        if col not in snapshots[unit_id]:
            val = row[col]
            # Handle NaN -> 0
            if pd.isna(val):
                val = 0
            else:
                # Convert numpy types to native Python types
                val = int(val) if float(val) == int(float(val)) else float(val)
            snapshots[unit_id][col] = val
            fields_added_total += 1
        else:
            skipped_existing += 1

    # Compute last_pm_date = trip_start - days_since_last_pm (ISO string for webapp)
    dslpm = row.get("days_since_last_pm")
    trip_date = row["trip_start"]
    if not pd.isna(dslpm) and not pd.isna(trip_date) and dslpm > 0:
        last_pm = trip_date - pd.Timedelta(days=dslpm)
        snapshots[unit_id]["last_pm_date"] = last_pm.strftime("%Y-%m-%d")
        fields_added_total += 1
    # Also store last_trip_date for reference
    if not pd.isna(trip_date):
        snapshots[unit_id]["last_trip_date"] = trip_date.strftime("%Y-%m-%d")

# Step 5: Save updated snapshots
print(f"\nSaving enriched snapshots...")
with open(SNAPSHOTS_PATH, "w") as f:
    json.dump(snapshots, f, indent=2)
print(f"  Saved to {SNAPSHOTS_PATH}")

# Stats
units_in_csv = set(last_trips["reefer_unit"])
units_in_snapshots = set(snapshots.keys())
matched = units_in_csv & units_in_snapshots
unmatched_snapshot = units_in_snapshots - units_in_csv

print(f"\n{'='*60}")
print(f"ENRICHMENT SUMMARY")
print(f"{'='*60}")
print(f"  Units in CSV:              {len(units_in_csv)}")
print(f"  Units in snapshots:        {len(units_in_snapshots)}")
print(f"  Units matched & enriched:  {enriched_count}")
print(f"  Units in snapshots w/o CSV: {len(unmatched_snapshot)}")
print(f"  Component fields per unit: {len(TARGET_COLS)}")
print(f"  Total fields added:        {fields_added_total}")
print(f"  Existing fields skipped:   {skipped_existing}")
print(f"{'='*60}")

# Sample output
print(f"\nSample enriched unit (first matched):")
sample_unit = list(matched)[0]
print(f"  Unit: {sample_unit}")
for col in TARGET_COLS:
    if col in snapshots[sample_unit]:
        print(f"    {col}: {snapshots[sample_unit][col]}")
