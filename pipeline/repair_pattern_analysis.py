"""
Pre-Trip Repair Pattern Analysis
================================
Joins reefer repair records to trip data to identify which repair codes,
sequences, combinations, and description keywords in the days before a
trip are predictive of a shutdown on that trip.

Outputs: data/repair_patterns.json
"""
import pandas as pd
import numpy as np
import json
import os
import re
from collections import Counter, defaultdict
from itertools import combinations

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

print("=" * 80)
print("REPAIR PATTERN ANALYSIS — Pre-Trip Failure Indicators")
print("=" * 80)

# ======================================================================
# 1. LOAD DATA
# ======================================================================

print("\nLoading trip data...")
trips = pd.read_csv(
    "data/processed/training/trip_features_v3.csv",
    usecols=["reefer_unit", "trip_start", "trip_end", "shutdown", "route",
             "leg_miles", "service_type", "season"],
)
trips["trip_start"] = pd.to_datetime(trips["trip_start"])
trips["trip_end"] = pd.to_datetime(trips["trip_end"])
print(f"  {len(trips)} trips, {trips['shutdown'].sum()} shutdowns ({trips['shutdown'].mean()*100:.1f}%)")

print("Loading reefer repairs...")
repairs = pd.read_excel("data/raw/repairs/Updated Reefer Unit Repairs 2024-2025.xlsx")
repairs["COMPCODE"] = repairs["COMPCODE"].astype(str).str.strip()
repairs["REPREASON"] = repairs["REPREASON"].astype(str).str.strip()
repairs["UNITNUMBER"] = repairs["UNITNUMBER"].astype(str).str.strip()
repairs["DESCRIP"] = repairs["DESCRIP"].astype(str).str.strip()
repairs["SHOPID"] = repairs["SHOPID"].astype(str).str.strip()
repairs["LINETYPE"] = repairs["LINETYPE"].astype(str).str.strip()
repairs["OPENED"] = pd.to_datetime(repairs["OPENED"])
repairs["SumOfLINETOTAL"] = pd.to_numeric(repairs["SumOfLINETOTAL"], errors="coerce").fillna(0)
print(f"  {len(repairs)} repair records")

# ======================================================================
# 2. BUILD COMP CODE LABELS
# ======================================================================

# Map comp code prefixes to human-readable component groups
COMP_GROUP = {
    "000": "PM / Scheduled Service",
    "001": "Compressor",
    "002": "Condenser",
    "003": "Evaporator",
    "014": "Refrigerant Lines",
    "016": "Drier",
    "031": "Starter / Cranking",
    "032": "Refrigeration Capacity",
    "034": "Thermostat",
    "035": "Electrical / Charging",
    "041": "Coolant",
    "042": "Engine",
    "043": "Exhaust / EGR",
    "044": "Fuel System",
    "045": "Throttle",
    "050": "Sheet Metal / Structure",
    "051": "Expansion Valve",
    "052": "Control Panel",
    "053": "Data Logger / Remote",
    "063": "Defrost",
    "064": "Chute / Bulkhead",
    "071": "Belts / Clutch",
    "077": "Alternator",
    "078": "Battery",
    "082": "Diagnosis / General Repair",
    "085": "Unit Retrieval",
    "151": "Container Reefer Specific",
    "DOR": "Door / In-Out",
    "DOT": "DOT Inspection",
    "DRY": "Dry Run",
    "TNX": "Fleet Internal",
    "UNR": "Unit Recovery / Tow",
    "OUT": "Out of Service",
}


def get_comp_group(code):
    if pd.isna(code) or code == "nan":
        return "Unknown"
    prefix = code.split("-")[0] if "-" in code else code[:3]
    return COMP_GROUP.get(prefix, f"Other ({prefix})")


def is_diag_only(code):
    """Return True if code is a diagnosis-only code."""
    return str(code).endswith("-DIA") or str(code) == "082"


repairs["comp_group"] = repairs["COMPCODE"].apply(get_comp_group)

# ======================================================================
# 3. AGGREGATE REPAIRS INTO WORK ORDERS (by unit + date)
# ======================================================================
# Multiple repair lines can belong to the same shop visit.
# Group by unit + date to get one "work order" per visit.

print("\nAggregating repair records into work orders...")
wo = (
    repairs.groupby(["UNITNUMBER", repairs["OPENED"].dt.date])
    .agg(
        comp_codes=("COMPCODE", lambda x: list(sorted(set(x)))),
        comp_groups=("comp_group", lambda x: list(sorted(set(x)))),
        reasons=("REPREASON", lambda x: list(sorted(set(x)))),
        descriptions=("DESCRIP", lambda x: " | ".join(sorted(set(x.dropna())))),
        total_cost=("SumOfLINETOTAL", "sum"),
        line_count=("COMPCODE", "count"),
        shops=("SHOPID", lambda x: list(sorted(set(x)))),
        has_diag=("COMPCODE", lambda x: any(is_diag_only(c) for c in x)),
        has_component_fix=("COMPCODE", lambda x: any(not is_diag_only(c) and not c.startswith("000") for c in x)),
    )
    .reset_index()
)
wo.columns = ["unit", "date", "comp_codes", "comp_groups", "reasons",
              "descriptions", "total_cost", "line_count", "shops",
              "has_diag", "has_component_fix"]
wo["date"] = pd.to_datetime(wo["date"])
print(f"  {len(wo)} unique work orders")

# ======================================================================
# 4. JOIN REPAIRS TO TRIPS BY TIME WINDOW
# ======================================================================

WINDOWS = [7, 14, 30]  # days before trip
BASELINE_RATE = trips["shutdown"].mean()

print(f"\nJoining repairs to trips (windows: {WINDOWS} days)...")
print(f"  Baseline shutdown rate: {BASELINE_RATE*100:.2f}%")

# Build a dict of work orders per unit for fast lookup
wo_by_unit = defaultdict(list)
for _, row in wo.iterrows():
    wo_by_unit[row["unit"]].append(row)

# For each trip, find pre-trip work orders
trip_repair_links = []  # (trip_idx, window_days, work_order_row)

for idx, trip in trips.iterrows():
    unit = trip["reefer_unit"]
    trip_date = trip["trip_start"]
    unit_wos = wo_by_unit.get(unit, [])

    for w_row in unit_wos:
        days_before = (trip_date - w_row["date"]).days
        if 0 < days_before <= 30:  # max window
            trip_repair_links.append({
                "trip_idx": idx,
                "shutdown": trip["shutdown"],
                "unit": unit,
                "trip_start": trip_date,
                "wo_date": w_row["date"],
                "days_before": days_before,
                "comp_codes": w_row["comp_codes"],
                "comp_groups": w_row["comp_groups"],
                "reasons": w_row["reasons"],
                "descriptions": w_row["descriptions"],
                "total_cost": w_row["total_cost"],
                "has_diag": w_row["has_diag"],
                "has_component_fix": w_row["has_component_fix"],
            })

links_df = pd.DataFrame(trip_repair_links)
print(f"  {len(links_df)} trip-repair links found")

# ======================================================================
# 5. ANALYSIS: Failure rate by individual comp code (per window)
# ======================================================================

print("\nComputing failure rates by component code...")

results_by_code = {}

for window in WINDOWS:
    window_links = links_df[links_df["days_before"] <= window]

    # Find which trips had each comp code in the window
    code_trip_map = defaultdict(set)  # code -> set of trip_idx
    for _, link in window_links.iterrows():
        for code in link["comp_codes"]:
            code_trip_map[code].add(link["trip_idx"])

    code_stats = []
    for code, trip_idxs in code_trip_map.items():
        trip_subset = trips.loc[list(trip_idxs)]
        n_trips = len(trip_subset)
        n_shutdowns = int(trip_subset["shutdown"].sum())
        if n_trips < 5:
            continue
        failure_rate = n_shutdowns / n_trips
        lift = failure_rate / BASELINE_RATE if BASELINE_RATE > 0 else 0
        group = get_comp_group(code)

        code_stats.append({
            "code": code,
            "group": group,
            "window": window,
            "trips_with_code": n_trips,
            "shutdowns": n_shutdowns,
            "failure_rate": round(failure_rate * 100, 2),
            "lift": round(lift, 2),
            "baseline_rate": round(BASELINE_RATE * 100, 2),
        })

    code_stats.sort(key=lambda x: -x["failure_rate"])
    results_by_code[window] = code_stats

# ======================================================================
# 6. ANALYSIS: Failure rate by component GROUP (cleaner view)
# ======================================================================

print("Computing failure rates by component group...")

results_by_group = {}

for window in WINDOWS:
    window_links = links_df[links_df["days_before"] <= window]

    group_trip_map = defaultdict(set)
    for _, link in window_links.iterrows():
        for group in link["comp_groups"]:
            group_trip_map[group].add(link["trip_idx"])

    group_stats = []
    for group, trip_idxs in group_trip_map.items():
        trip_subset = trips.loc[list(trip_idxs)]
        n_trips = len(trip_subset)
        n_shutdowns = int(trip_subset["shutdown"].sum())
        if n_trips < 5:
            continue
        failure_rate = n_shutdowns / n_trips
        lift = failure_rate / BASELINE_RATE if BASELINE_RATE > 0 else 0

        group_stats.append({
            "group": group,
            "window": window,
            "trips_with_group": n_trips,
            "shutdowns": n_shutdowns,
            "failure_rate": round(failure_rate * 100, 2),
            "lift": round(lift, 2),
        })

    group_stats.sort(key=lambda x: -x["failure_rate"])
    results_by_group[window] = group_stats

# ======================================================================
# 7. ANALYSIS: Repair REASON before trip
# ======================================================================

print("Computing failure rates by repair reason...")

results_by_reason = {}

for window in WINDOWS:
    window_links = links_df[links_df["days_before"] <= window]

    reason_trip_map = defaultdict(set)
    for _, link in window_links.iterrows():
        for reason in link["reasons"]:
            reason_trip_map[reason].add(link["trip_idx"])

    reason_stats = []
    for reason, trip_idxs in reason_trip_map.items():
        trip_subset = trips.loc[list(trip_idxs)]
        n_trips = len(trip_subset)
        n_shutdowns = int(trip_subset["shutdown"].sum())
        if n_trips < 3:
            continue
        failure_rate = n_shutdowns / n_trips
        lift = failure_rate / BASELINE_RATE if BASELINE_RATE > 0 else 0

        reason_stats.append({
            "reason": reason,
            "window": window,
            "trips": n_trips,
            "shutdowns": n_shutdowns,
            "failure_rate": round(failure_rate * 100, 2),
            "lift": round(lift, 2),
        })

    reason_stats.sort(key=lambda x: -x["failure_rate"])
    results_by_reason[window] = reason_stats

# ======================================================================
# 8. ANALYSIS: 2-code COMBINATIONS (most dangerous pairs)
# ======================================================================

print("Computing dangerous 2-code combinations (14d window)...")

window_links_14 = links_df[links_df["days_before"] <= 14]

# For each trip, get all comp groups that appeared
trip_groups_14 = defaultdict(set)
for _, link in window_links_14.iterrows():
    for group in link["comp_groups"]:
        trip_groups_14[link["trip_idx"]].add(group)

# Find all 2-group combos and their failure rates
combo_trips = defaultdict(set)
for trip_idx, groups in trip_groups_14.items():
    if len(groups) < 2:
        continue
    for combo in combinations(sorted(groups), 2):
        combo_trips[combo].add(trip_idx)

combo_stats = []
for combo, trip_idxs in combo_trips.items():
    trip_subset = trips.loc[list(trip_idxs)]
    n_trips = len(trip_subset)
    n_shutdowns = int(trip_subset["shutdown"].sum())
    if n_trips < 5:
        continue
    failure_rate = n_shutdowns / n_trips
    lift = failure_rate / BASELINE_RATE if BASELINE_RATE > 0 else 0

    combo_stats.append({
        "combo": list(combo),
        "trips": n_trips,
        "shutdowns": n_shutdowns,
        "failure_rate": round(failure_rate * 100, 2),
        "lift": round(lift, 2),
    })

combo_stats.sort(key=lambda x: -x["failure_rate"])
top_combos = combo_stats[:30]

# ======================================================================
# 9. ANALYSIS: SEQUENCE (last 2 work orders before trip)
# ======================================================================

print("Computing repair sequences (last 2 work orders before trip)...")

# For each trip, get the last 2 work orders in order
sequence_stats = defaultdict(lambda: {"total": 0, "shutdowns": 0})

for trip_idx, trip in trips.iterrows():
    unit = trip["reefer_unit"]
    trip_date = trip["trip_start"]
    unit_wos = wo_by_unit.get(unit, [])

    # Get work orders in 30 days before trip, sorted by date
    recent_wos = [
        w for w in unit_wos
        if 0 < (trip_date - w["date"]).days <= 30
    ]
    recent_wos.sort(key=lambda x: x["date"])

    if len(recent_wos) >= 2:
        # Last 2 work orders -> sequence of primary groups
        seq = []
        for w in recent_wos[-2:]:
            # Primary group = most specific non-PM group, or PM
            groups = [g for g in w["comp_groups"] if g != "PM / Scheduled Service"]
            primary = groups[0] if groups else "PM / Scheduled Service"
            seq.append(primary)
        seq_key = tuple(seq)
        sequence_stats[seq_key]["total"] += 1
        if trip["shutdown"]:
            sequence_stats[seq_key]["shutdowns"] += 1

sequence_results = []
for seq_key, stats in sequence_stats.items():
    if stats["total"] < 5:
        continue
    failure_rate = stats["shutdowns"] / stats["total"]
    lift = failure_rate / BASELINE_RATE if BASELINE_RATE > 0 else 0
    sequence_results.append({
        "sequence": list(seq_key),
        "trips": stats["total"],
        "shutdowns": stats["shutdowns"],
        "failure_rate": round(failure_rate * 100, 2),
        "lift": round(lift, 2),
    })

sequence_results.sort(key=lambda x: -x["failure_rate"])
top_sequences = sequence_results[:30]

# ======================================================================
# 10. ANALYSIS: Diagnosis-without-resolution pattern
# ======================================================================

print("Analyzing diagnosis-without-resolution patterns...")

# Trips where last work order had diagnosis but no component fix
diag_no_fix_trips = set()
diag_with_fix_trips = set()
no_diag_trips = set()

for trip_idx, trip in trips.iterrows():
    unit = trip["reefer_unit"]
    trip_date = trip["trip_start"]
    unit_wos = wo_by_unit.get(unit, [])

    recent_wos = [
        w for w in unit_wos
        if 0 < (trip_date - w["date"]).days <= 14
    ]
    if not recent_wos:
        no_diag_trips.add(trip_idx)
        continue

    recent_wos.sort(key=lambda x: x["date"])
    last_wo = recent_wos[-1]

    if last_wo["has_diag"] and not last_wo["has_component_fix"]:
        diag_no_fix_trips.add(trip_idx)
    elif last_wo["has_diag"] and last_wo["has_component_fix"]:
        diag_with_fix_trips.add(trip_idx)

diag_resolution = []
for label, idx_set in [
    ("Diagnosis only (no fix)", diag_no_fix_trips),
    ("Diagnosis + component fix", diag_with_fix_trips),
    ("No recent repair", no_diag_trips),
]:
    subset = trips.loc[list(idx_set)]
    n = len(subset)
    sd = int(subset["shutdown"].sum())
    rate = sd / n if n > 0 else 0
    diag_resolution.append({
        "pattern": label,
        "trips": n,
        "shutdowns": sd,
        "failure_rate": round(rate * 100, 2),
        "lift": round(rate / BASELINE_RATE, 2) if BASELINE_RATE > 0 else 0,
    })

# ======================================================================
# 11. ANALYSIS: Repeat repairs (same group 2+ times in 30 days)
# ======================================================================

print("Analyzing repeat repair patterns...")

repeat_trip_map = defaultdict(lambda: {"repeat": set(), "single": set()})

for trip_idx, trip in trips.iterrows():
    unit = trip["reefer_unit"]
    trip_date = trip["trip_start"]
    unit_wos = wo_by_unit.get(unit, [])

    recent_wos = [
        w for w in unit_wos
        if 0 < (trip_date - w["date"]).days <= 30
    ]
    if len(recent_wos) < 2:
        continue

    # Count how many times each group appears across work orders
    group_counts = Counter()
    for w in recent_wos:
        for g in set(w["comp_groups"]):
            group_counts[g] += 1

    for group, count in group_counts.items():
        if group == "PM / Scheduled Service":
            continue
        if count >= 2:
            repeat_trip_map[group]["repeat"].add(trip_idx)
        else:
            repeat_trip_map[group]["single"].add(trip_idx)

repeat_stats = []
for group, sets in repeat_trip_map.items():
    for label, idx_set in [("Repeat (2+)", sets["repeat"]), ("Single", sets["single"])]:
        if len(idx_set) < 5:
            continue
        subset = trips.loc[list(idx_set)]
        n = len(subset)
        sd = int(subset["shutdown"].sum())
        rate = sd / n if n > 0 else 0
        repeat_stats.append({
            "group": group,
            "pattern": label,
            "trips": n,
            "shutdowns": sd,
            "failure_rate": round(rate * 100, 2),
            "lift": round(rate / BASELINE_RATE, 2) if BASELINE_RATE > 0 else 0,
        })

repeat_stats.sort(key=lambda x: (-x["failure_rate"]))

# ======================================================================
# 12. ANALYSIS: Turnaround time (days between last repair and trip)
# ======================================================================

print("Analyzing repair-to-trip turnaround times...")

turnaround_buckets = [
    ("0-2 days", 0, 2),
    ("3-5 days", 3, 5),
    ("6-10 days", 6, 10),
    ("11-20 days", 11, 20),
    ("21-30 days", 21, 30),
]

turnaround_stats = []
for label, lo, hi in turnaround_buckets:
    # Find trips whose LAST repair was in this bucket
    bucket_trips = set()
    for trip_idx, trip in trips.iterrows():
        unit = trip["reefer_unit"]
        trip_date = trip["trip_start"]
        unit_wos = wo_by_unit.get(unit, [])
        recent_wos = [
            w for w in unit_wos
            if 0 < (trip_date - w["date"]).days <= 30
        ]
        if not recent_wos:
            continue
        recent_wos.sort(key=lambda x: x["date"])
        last_days = (trip_date - recent_wos[-1]["date"]).days
        if lo <= last_days <= hi:
            bucket_trips.add(trip_idx)

    if not bucket_trips:
        continue
    subset = trips.loc[list(bucket_trips)]
    n = len(subset)
    sd = int(subset["shutdown"].sum())
    rate = sd / n if n > 0 else 0
    turnaround_stats.append({
        "bucket": label,
        "trips": n,
        "shutdowns": sd,
        "failure_rate": round(rate * 100, 2),
        "lift": round(rate / BASELINE_RATE, 2) if BASELINE_RATE > 0 else 0,
    })

# ======================================================================
# 13. ANALYSIS: Description keyword mining
# ======================================================================

print("Mining repair description keywords...")

KEYWORDS = [
    "leak", "intermittent", "replaced", "adjusted", "diagnos",
    "no start", "no crank", "overhe", "low fuel", "belt",
    "compressor", "alternator", "starter", "battery", "sensor",
    "wiring", "harness", "contamina", "clogged", "frozen",
    "cracked", "worn", "corroded", "campaign", "recall",
    "pm check", "swap", "retrieval", "shutdown", "alarm",
    "not reporting", "fuel polish", "coolant",
]

keyword_stats = []
for window in [14]:
    window_links = links_df[links_df["days_before"] <= window]

    for keyword in KEYWORDS:
        # Find trips where keyword appeared in any pre-trip repair description
        kw_trips = set()
        for _, link in window_links.iterrows():
            if keyword.lower() in str(link["descriptions"]).lower():
                kw_trips.add(link["trip_idx"])

        if len(kw_trips) < 5:
            continue

        subset = trips.loc[list(kw_trips)]
        n = len(subset)
        sd = int(subset["shutdown"].sum())
        rate = sd / n if n > 0 else 0
        lift = rate / BASELINE_RATE if BASELINE_RATE > 0 else 0

        keyword_stats.append({
            "keyword": keyword,
            "trips": n,
            "shutdowns": sd,
            "failure_rate": round(rate * 100, 2),
            "lift": round(lift, 2),
        })

keyword_stats.sort(key=lambda x: -x["failure_rate"])

# ======================================================================
# 14. SUMMARY STATS
# ======================================================================

# Trips with ANY repair in 14-day window vs trips without
trips_with_repair = set(links_df[links_df["days_before"] <= 14]["trip_idx"])
trips_without_repair = set(trips.index) - trips_with_repair

with_subset = trips.loc[list(trips_with_repair)]
without_subset = trips.loc[list(trips_without_repair)]

summary = {
    "total_trips": len(trips),
    "total_shutdowns": int(trips["shutdown"].sum()),
    "baseline_rate": round(BASELINE_RATE * 100, 2),
    "trips_with_repair_14d": len(with_subset),
    "shutdowns_with_repair_14d": int(with_subset["shutdown"].sum()),
    "rate_with_repair_14d": round(with_subset["shutdown"].mean() * 100, 2) if len(with_subset) > 0 else 0,
    "trips_without_repair_14d": len(without_subset),
    "shutdowns_without_repair_14d": int(without_subset["shutdown"].sum()),
    "rate_without_repair_14d": round(without_subset["shutdown"].mean() * 100, 2) if len(without_subset) > 0 else 0,
    "total_work_orders": len(wo),
    "unique_comp_codes": int(repairs["COMPCODE"].nunique()),
    "unique_comp_groups": len(COMP_GROUP),
    "repair_date_range": f"{repairs['OPENED'].min().strftime('%Y-%m-%d')} to {repairs['OPENED'].max().strftime('%Y-%m-%d')}",
    "trip_date_range": f"{trips['trip_start'].min().strftime('%Y-%m-%d')} to {trips['trip_start'].max().strftime('%Y-%m-%d')}",
}

# ======================================================================
# 15. EXPORT
# ======================================================================

output = {
    "summary": summary,
    "by_code": {str(k): v for k, v in results_by_code.items()},
    "by_group": {str(k): v for k, v in results_by_group.items()},
    "by_reason": {str(k): v for k, v in results_by_reason.items()},
    "combos": top_combos,
    "sequences": top_sequences,
    "diag_resolution": diag_resolution,
    "repeat_repairs": repeat_stats,
    "turnaround": turnaround_stats,
    "keywords": keyword_stats,
    "comp_group_labels": COMP_GROUP,
}

out_path = "data/repair_patterns.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n{'=' * 80}")
print(f"EXPORT COMPLETE: {out_path}")
print(f"  Summary: {summary['total_trips']} trips, {summary['total_shutdowns']} shutdowns")
print(f"  Rate with repair in 14d window: {summary['rate_with_repair_14d']}%")
print(f"  Rate without: {summary['rate_without_repair_14d']}%")
print(f"  Component codes analyzed: {len(results_by_code.get(14, []))}")
print(f"  Dangerous combos found: {len(top_combos)}")
print(f"  Sequences analyzed: {len(top_sequences)}")
print(f"  Keywords analyzed: {len(keyword_stats)}")
print(f"{'=' * 80}")
