"""Refresh the FROZEN failure model's per-unit CURRENT-STATE inputs AS OF a date.

This is a sibling of `refresh_inputs.py`. Where `refresh_inputs.py` rebuilds the
*latest-trip* snapshot from the engineered trip-features CSV, this tool recomputes
each reefer's maintenance / telematics CURRENT STATE directly from the combined
2024-2025 + Term-2 (Jan-Apr 2026) raw data, anchored to an `--as-of` cutoff date
instead of the unit's last trip date. It then OVERWRITES ONLY the numeric
current-state feature keys in `models/risk_engine/unit_snapshots.json`.

The model is NOT retrained: xgb_model.json / calibrated_model.pkl / feature_meta /
imputer / encoders are left byte-identical. We only refresh the inputs the frozen
model scores.

Snapshot contract (see risk_scorer.py:103-107): a snapshot value is overlaid only
if it is NOT a str. Categoricals are stored label-encoded as ints. This tool only
ever writes NUMERIC current-state features, so the categorical / trip-context keys
are untouched.

Recomputed features (23) — exact formulas mirrored from build_trip_features_v3.py,
anchored to `as_of` instead of `trip_date`:

  Repairs (build_trip_features_v3.py:786-948, 854-878):
    days_since_last_repair, repair_count_30d, repair_count_365d, repair_cum_cost,
    repair_cost_365d, n_repair_groups_14d, diagnosis_365d, unique_compcodes_365d,
    has_dor, has_dor_90d
  Container repairs (build_trip_features_v3.py:1066-1075):
    container_repair_cost_365d
  PM (build_trip_features_v3.py:541-545, 754-762):
    pm_gap_mean
  Engine hours (build_trip_features_v3.py:650-674, 1050-1054):
    engine_hours_at_trip, engine_hours_per_day
  Telematics (build_trip_features_v3.py:954-1048) — recency/counts/severity, from
  the Term-2 Alarm-History-Summary adapter:
    days_since_last_telem_shutdown, days_since_last_alarm, days_since_last_low_fuel,
    days_since_last_telem_event, telem_shutdowns_90d, alarm_red_total,
    alarm_yellow_90d, telem_low_fuel_90d

LEFT AT LAST-KNOWN (flagged): pm_cum_overdue_days — this column is NOT computed by
any current build script; it is a legacy column baked into trip_features_final.csv,
so its exact definition cannot be matched confidently. Per the no-guess rule it is
left at its existing snapshot value.

Usage:
    python pipeline/refresh_current_state.py
    python pipeline/refresh_current_state.py --as-of 2026-04-30
    python pipeline/refresh_current_state.py --engine-dir models/risk_engine
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root = nearest ancestor containing paths.py (depth-independent).
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from paths import raw, processed, model  # noqa: E402

ENGINE_DIR = REPO_ROOT / "models" / "risk_engine"

# Monthly client drop zone (unambiguous per-type subfolders). The client fills the
# templates in data/templates/ and drops each new month's file here; this tool reads
# the canonical baseline + the one-time Term-2 export + everything dropped here.
MONTHLY = raw("monthly_updates")


def _monthly_files(subdir: str, ext: str) -> list[Path]:
    d = MONTHLY / subdir
    return sorted(d.glob(f"*.{ext}")) if d.is_dir() else []

# --- The 23 numeric current-state features this tool recomputes -------------
RECOMPUTE_FEATURES = [
    "days_since_last_telem_shutdown", "days_since_last_alarm", "days_since_last_low_fuel",
    "days_since_last_telem_event", "telem_shutdowns_90d", "alarm_red_total",
    "alarm_yellow_90d", "telem_low_fuel_90d", "days_since_last_repair", "repair_count_30d",
    "repair_count_365d", "repair_cum_cost", "repair_cost_365d", "n_repair_groups_14d",
    "diagnosis_365d", "unique_compcodes_365d", "container_repair_cost_365d", "has_dor",
    "has_dor_90d", "pm_cum_overdue_days", "pm_gap_mean", "engine_hours_at_trip",
    "engine_hours_per_day",
]
# Features in RECOMPUTE_FEATURES we CANNOT match confidently -> leave last-known.
LEAVE_AT_LAST_KNOWN = {"pm_cum_overdue_days"}

# diagnosis category compcodes (build_trip_features_v3.py:600).
DIAGNOSIS_COMPCODES = {"082-DIA", "DOR-DIA"}


# ======================================================================
# DATA LOADING — combine canonical 2024-2025 + Term-2 Jan-Apr 2026
# ======================================================================
def _read_table(p: Path):
    """Read a repairs/meter table from .csv (template format) or .xlsx."""
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.name.startswith("Updated Reefer Unit Repairs"):
        return pd.read_excel(p, sheet_name="Sheet1")
    return pd.read_excel(p)


def _input_files(canonical: Path, monthly_subdir: str) -> list[Path]:
    """Baseline canonical file + everything the client has dropped under
    data/raw/monthly_updates/<subdir>/ (CSV templates or XLSX).

    Note: new data flows in ONLY through the monthly drop zone, so uploading one
    month at a time updates the inputs incrementally (and cumulatively, since
    earlier months stay in the drop folder)."""
    files = [canonical] + _monthly_files(monthly_subdir, "xlsx") + _monthly_files(monthly_subdir, "csv")
    return [p for p in files if p.exists()]


def _extract_num(val) -> str | None:
    """Numeric portion of a container/trailer id (mirrors build_trip_features_v3
    extract_container_num)."""
    if pd.isna(val):
        return None
    nums = re.sub(r"[^0-9]", "", str(val).strip())
    return nums if nums else None


def load_reefer_repairs() -> dict:
    """Per-reefer-unit sorted repair-order list (mirrors build_trip_features_v3
    repairs loader §34-44 + repair_orders aggregation §571-584). Combines the
    canonical 2024-2025 workbook with the Term-2 export, dedupes on ORDERNUM."""
    files = _input_files(raw("repairs", "Updated Reefer Unit Repairs 2024-2025.xlsx"), "reefer_repairs")
    rep = pd.concat([_read_table(p) for p in files], ignore_index=True)
    for c in ("COMPCODE", "REPREASON", "UNITNUMBER", "DESCRIP", "SHOPID"):
        if c in rep.columns:
            rep[c] = rep[c].astype(str).str.strip()
    rep["OPENED"] = pd.to_datetime(rep["OPENED"], errors="coerce")
    rep["SumOfLINETOTAL"] = pd.to_numeric(rep["SumOfLINETOTAL"], errors="coerce").fillna(0)
    rep = rep.dropna(subset=["OPENED"])
    # A repair order can appear in BOTH workbooks (overlap month) -> dedupe by line.
    rep = rep.drop_duplicates(subset=["UNITNUMBER", "ORDERNUM", "COMPCODE",
                                      "OPENED", "SumOfLINETOTAL"])

    orders = rep.groupby(["UNITNUMBER", "ORDERNUM"]).agg(
        date=("OPENED", "min"),
        total_cost=("SumOfLINETOTAL", "sum"),
        repreason=("REPREASON", "first"),
        compcodes=("COMPCODE", lambda x: list(set(x))),
        shopid=("SHOPID", lambda x: x.iloc[0].strip() if len(x) > 0 else ""),
    ).reset_index()
    orders["date"] = pd.to_datetime(orders["date"])
    by_unit = {}
    for unit, grp in orders.groupby("UNITNUMBER"):
        by_unit[unit] = grp.sort_values("date").to_dict("records")
    return by_unit


def load_pm_dates() -> dict:
    """Per-reefer-unit sorted PM-360 (PM / 000-011) date list (mirrors
    build_trip_features_v3.py:541-545). Combined + deduped per (unit, day)."""
    files = _input_files(raw("repairs", "Updated Reefer Unit Repairs 2024-2025.xlsx"), "reefer_repairs")
    rep = pd.concat([_read_table(p) for p in files], ignore_index=True)
    rep["COMPCODE"] = rep["COMPCODE"].astype(str).str.strip()
    rep["REPREASON"] = rep["REPREASON"].astype(str).str.strip()
    rep["UNITNUMBER"] = rep["UNITNUMBER"].astype(str).str.strip()
    rep["OPENED"] = pd.to_datetime(rep["OPENED"], errors="coerce")
    pm = rep[(rep["REPREASON"] == "PM") & (rep["COMPCODE"] == "000-011")].dropna(subset=["OPENED"]).copy()
    pm["date_only"] = pm["OPENED"].dt.date
    pm = pm.drop_duplicates(subset=["UNITNUMBER", "date_only"])
    return pm.groupby("UNITNUMBER")["OPENED"].apply(lambda x: sorted(x)).to_dict()


def load_container_repairs() -> dict:
    """Per-container-number sorted repair-order list (mirrors
    build_trip_features_v3.py:263-283). Combined + deduped."""
    files = _input_files(raw("repairs", "Container Repair Detail.xlsx"), "container_repairs")
    crep = pd.concat([_read_table(p) for p in files], ignore_index=True)
    crep["UNITNUMBER"] = crep["UNITNUMBER"].astype(str).str.strip()
    crep["OPENED"] = pd.to_datetime(crep["OPENED"], errors="coerce")
    crep["REPREASON"] = crep["REPREASON"].astype(str).str.strip()
    crep["COMPCODE"] = crep["COMPCODE"].astype(str).str.strip()
    crep["SumOfLINETOTAL"] = pd.to_numeric(crep["SumOfLINETOTAL"], errors="coerce").fillna(0)
    crep = crep.dropna(subset=["OPENED"])
    crep = crep.drop_duplicates(subset=["UNITNUMBER", "ORDERNUM", "COMPCODE",
                                        "OPENED", "SumOfLINETOTAL"])
    orders = crep.groupby(["UNITNUMBER", "ORDERNUM"]).agg(
        date=("OPENED", "min"),
        total_cost=("SumOfLINETOTAL", "sum"),
        repreason=("REPREASON", "first"),
    ).reset_index()
    orders["date"] = pd.to_datetime(orders["date"])
    # Key by the *numeric* container number so it matches the reefer->container map.
    by_num = {}
    for cnum, grp in orders.groupby("UNITNUMBER"):
        key = _extract_num(cnum) or str(cnum).strip()
        by_num[key] = grp.sort_values("date").to_dict("records")
    return by_num


def load_meter() -> dict:
    """Per-reefer-unit sorted [(date, reading)] hour-meter list (mirrors
    build_trip_features_v3.py:241-259). Combined + deduped."""
    files = _input_files(raw("hours meter reading", "Meter Reading History.xlsx"), "meter_readings")
    meter = pd.concat([_read_table(p) for p in files], ignore_index=True)
    meter["UNITNUMBER"] = meter["UNITNUMBER"].astype(str).str.strip()
    meter["READDAY"] = pd.to_datetime(meter["READDAY"], errors="coerce")
    meter["READING"] = pd.to_numeric(meter["READING"], errors="coerce")
    meter = meter[(meter["METERTYPE"] == "HOUR METER") &
                  (meter["READDAY"] < "2030-01-01") & (meter["READING"] > 0)].copy()
    meter = meter.sort_values(["UNITNUMBER", "READDAY"])
    by_unit = {}
    for unit, grp in meter.groupby("UNITNUMBER"):
        rd = grp[["READDAY", "READING"]].drop_duplicates().sort_values("READDAY")
        by_unit[unit] = list(zip(rd["READDAY"], rd["READING"]))
    return by_unit


def load_term2_telematics(trailer_to_reefer: dict) -> dict:
    """Adapter for the Term-2 Alarm-History-Summary CSV.

    The file is NOT the per-event quarterly format; it is an aggregated summary:
    one row per (Vehicle, Alarm Type) with a count (`#`) and First/Last Logged
    datetimes. Severity: R = Shutdown(High), Y = Check(Medium), G = Log(Low).

    We adapt it to the telematics fields build_trip_features_v3.py derives:
      - shutdown  : Severity == 'R' OR alarm type mentions Shutdown / Pretrip Abort
      - alarm     : Severity in {R, Y}  (matches alarm_red/yellow severity scoring)
      - low_fuel  : alarm type mentions 'fuel'
    Recency uses Last Logged; 90d windows use rows whose Last Logged falls in the
    window; counts are weighted by `#`. `alarm_red_total` = total R count (all R
    rows, weighted), mirroring alarm_red_total = sum of red severity scores.

    Returns {reefer_unit: list of event dicts} with keys: last, first, type
    ('SHUTDOWN'|'ALARM'|'LOW_FUEL'|'OTHER'), severity, count.
    """
    # one-time Term-2 export + any alarm CSVs the client drops monthly
    files = _monthly_files("telematics", "csv")
    by_unit: dict = defaultdict(list)
    frames = []
    for p in files:
        if p.exists():
            try:
                frames.append(pd.read_csv(p, skiprows=4))
            except Exception:
                continue
    if not frames:
        return by_unit
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Vehicle Name"].notna() & (df["Vehicle Name"].astype(str).str.strip() != "")]
    df["Last Logged"] = pd.to_datetime(df["Last Logged"], errors="coerce")
    df["First Logged"] = pd.to_datetime(df["First Logged"], errors="coerce")
    df["#"] = pd.to_numeric(df["#"], errors="coerce").fillna(0).astype(int)

    for _, r in df.iterrows():
        cnum = _extract_num(r["Vehicle Name"])
        reefer = trailer_to_reefer.get(cnum) if cnum else None
        if not reefer:
            continue
        last = r["Last Logged"]
        if pd.isna(last):
            continue
        sev = str(r["Severity"]).strip().upper()
        atype = str(r["Alarm Type"]).lower()
        cnt = int(r["#"])

        is_shutdown = sev == "R" or "shutdown" in atype or "pretrip or self-check abort" in atype
        is_low_fuel = "fuel" in atype
        by_unit[reefer].append({
            "last": last, "first": r["First Logged"], "severity": sev,
            "count": cnt, "is_shutdown": is_shutdown, "is_low_fuel": is_low_fuel,
        })
    return dict(by_unit)


# ======================================================================
# PER-UNIT CURRENT-STATE FEATURE COMPUTATION (anchored to as_of)
# ======================================================================
def _interpolate_engine_hours(readings, as_of):
    """Mirror build_trip_features_v3.py:650-674 with as_of in place of trip_date."""
    if not readings:
        return np.nan, np.nan
    before = [(d, rdg) for d, rdg in readings if d < as_of]
    if not before:
        return np.nan, np.nan
    last_date, last_reading = before[-1]
    hours_at_trip = last_reading
    if len(before) >= 2:
        first_date, first_reading = before[0]
        days_span = (last_date - first_date).total_seconds() / 86400
        rate = (last_reading - first_reading) / days_span if days_span > 0 else np.nan
    else:
        rate = np.nan
    return hours_at_trip, rate


def compute_repair_features(reps, as_of) -> dict:
    """Repair + PM + container features, formulas from build_trip_features_v3.py
    (cited inline), anchored to as_of."""
    out: dict = {}
    cutoff_14 = as_of - pd.Timedelta(days=14)
    cutoff_30 = as_of - pd.Timedelta(days=30)
    cutoff_90 = as_of - pd.Timedelta(days=90)
    cutoff_365 = as_of - pd.Timedelta(days=365)

    reps_before = [r for r in reps if r["date"] < as_of]  # :793

    # repair_cum_cost (:800), windowed cost/count (:805-810)
    out["repair_cum_cost"] = float(sum(r["total_cost"] for r in reps_before))
    out["repair_count_30d"] = int(sum(1 for r in reps_before if r["date"] >= cutoff_30))
    out["repair_count_365d"] = int(sum(1 for r in reps_before if r["date"] >= cutoff_365))
    out["repair_cost_365d"] = float(sum(r["total_cost"] for r in reps_before if r["date"] >= cutoff_365))

    # has_dor / has_dor_90d / unique_compcodes_365d / diagnosis_365d (:854-885)
    has_dor = 0
    has_dor_90 = 0
    unique_codes_365: set = set()
    diagnosis_365 = 0
    for r in reps_before:
        for cc in r["compcodes"]:
            if r["date"] >= cutoff_365:
                unique_codes_365.add(cc)
                if cc in DIAGNOSIS_COMPCODES:  # diagnosis category (:600)
                    diagnosis_365 += 1
            if cc == "DOR-DIA":
                has_dor = 1
                if r["date"] >= cutoff_90:
                    has_dor_90 = 1
    out["has_dor"] = int(has_dor)
    out["has_dor_90d"] = int(has_dor_90)
    out["unique_compcodes_365d"] = int(len(unique_codes_365))
    out["diagnosis_365d"] = int(diagnosis_365)

    # days_since_last_repair (:908-912)
    if reps_before:
        out["days_since_last_repair"] = float((as_of - reps_before[-1]["date"]).days)
    else:
        out["days_since_last_repair"] = None

    # n_repair_groups_14d (:927-934)
    reps_14d = [r for r in reps_before if r["date"] >= cutoff_14]
    groups_14d: set = set()
    for r in reps_14d:
        for cc in r["compcodes"]:
            prefix = cc.split("-")[0] if "-" in cc else cc[:3]
            groups_14d.add(prefix)
    out["n_repair_groups_14d"] = int(len(groups_14d))
    return out


def compute_pm_features(pm_dates, as_of) -> dict:
    """pm_gap_mean from PM-360 history (build_trip_features_v3.py:754-762)."""
    pms_before = [p for p in pm_dates if p < as_of]
    if len(pms_before) >= 2:
        gaps = [(pms_before[i + 1] - pms_before[i]).days for i in range(len(pms_before) - 1)]
        return {"pm_gap_mean": float(np.mean(gaps))}
    return {"pm_gap_mean": None}


def compute_container_features(creps, as_of) -> dict:
    """container_repair_cost_365d (build_trip_features_v3.py:1066-1075)."""
    cutoff_365 = as_of - pd.Timedelta(days=365)
    creps_before = [r for r in creps if r["date"] < as_of]
    return {"container_repair_cost_365d":
            float(sum(r["total_cost"] for r in creps_before if r["date"] >= cutoff_365))}


def compute_engine_features(readings, as_of) -> dict:
    """engine_hours_at_trip / engine_hours_per_day (build_trip_features_v3.py:1050-1054)."""
    eng_hours, eng_rate = _interpolate_engine_hours(readings, as_of)
    return {
        "engine_hours_at_trip": (None if pd.isna(eng_hours) else float(eng_hours)),
        "engine_hours_per_day": (None if pd.isna(eng_rate) else float(eng_rate)),
    }


def compute_telem_features(events, as_of) -> dict:
    """Telematics recency / counts / severity from the Term-2 alarm-summary adapter,
    mirroring build_trip_features_v3.py:954-1048 (days_since_*, *_90d, alarm_red_total,
    alarm_yellow_90d) with as_of as the anchor.

    Recency uses each row's Last Logged; 90d windows count rows whose Last Logged is
    in [as_of-90, as_of); counts are weighted by the row's event count `#`.
    Returns {} when the unit has no Term-2 telematics (so its last-known values stay).
    """
    out: dict = {}
    cutoff_90 = as_of - pd.Timedelta(days=90)
    ev_before = [e for e in events if e["last"] < as_of]
    if not ev_before:
        return out  # no 2026 telem for this unit -> keep last-known snapshot values

    shutdowns = [e for e in ev_before if e["is_shutdown"]]
    low_fuel = [e for e in ev_before if e["is_low_fuel"]]
    # alarm = Severity R or Y (matches alarm severity scoring red+yellow, :1031-1039)
    alarms = [e for e in ev_before if e["severity"] in ("R", "Y")]

    def _days_since(evs):
        if not evs:
            return None
        last = max(e["last"] for e in evs)
        return float((as_of - last).total_seconds() / 86400)

    # days_since_* (:981-999) — telem_event = any alarm row
    out["days_since_last_telem_shutdown"] = _days_since(shutdowns)
    out["days_since_last_alarm"] = _days_since(alarms)
    out["days_since_last_low_fuel"] = _days_since(low_fuel)
    out["days_since_last_telem_event"] = _days_since(ev_before)

    # 90d windowed counts (:973-977), weighted by event count `#`
    def _win_count(evs):
        return int(sum(e["count"] for e in evs if e["last"] >= cutoff_90))

    out["telem_shutdowns_90d"] = _win_count(shutdowns)
    out["telem_low_fuel_90d"] = _win_count(low_fuel)

    # alarm_red_total = total R count (all-time in export), :1034/1041
    out["alarm_red_total"] = int(sum(e["count"] for e in ev_before if e["severity"] == "R"))
    # alarm_yellow_90d = Y count in 90d window, :1039/1045
    out["alarm_yellow_90d"] = int(sum(e["count"] for e in ev_before
                                      if e["severity"] == "Y" and e["last"] >= cutoff_90))
    return out


# ======================================================================
# DRIVER
# ======================================================================
def _coerce(v):
    """Snapshot JSON stores ints as ints, floats as floats, NaN as null."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if pd.isna(v) else float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def refresh(as_of: pd.Timestamp, engine_dir: Path = ENGINE_DIR) -> dict:
    snap_path = engine_dir / "unit_snapshots.json"
    snapshots = json.loads(snap_path.read_text())

    # back up before overwrite (rollback safety; model untouched)
    (engine_dir / "unit_snapshots.prev.json").write_text(snap_path.read_text())

    # reefer<->container maps (numeric container -> reefer R####)
    trailer_to_reefer = json.loads((engine_dir / "trailer_map.json").read_text())
    reefer_to_container = {v: k for k, v in trailer_to_reefer.items()}

    print("Loading combined 2024-2025 + Term-2 data ...")
    repairs_by_unit = load_reefer_repairs()
    pm_by_unit = load_pm_dates()
    container_repairs_by_num = load_container_repairs()
    meter_by_unit = load_meter()
    telem_by_unit = load_term2_telematics(trailer_to_reefer)
    print(f"  reefer-repair units={len(repairs_by_unit)}  pm units={len(pm_by_unit)}  "
          f"container-repair containers={len(container_repairs_by_num)}  "
          f"meter units={len(meter_by_unit)}  term2-telem units={len(telem_by_unit)}")

    changed_units = 0
    sample_before_after = {}
    telem_units_updated = 0

    for unit, snap in snapshots.items():
        before = {k: snap.get(k) for k in RECOMPUTE_FEATURES}

        new_vals: dict = {}
        new_vals.update(compute_repair_features(repairs_by_unit.get(unit, []), as_of))
        new_vals.update(compute_pm_features(pm_by_unit.get(unit, []), as_of))
        cnum = reefer_to_container.get(unit)
        new_vals.update(compute_container_features(
            container_repairs_by_num.get(cnum, []) if cnum else [], as_of))
        new_vals.update(compute_engine_features(meter_by_unit.get(unit, []), as_of))
        telem_new = compute_telem_features(telem_by_unit.get(unit, []), as_of)
        new_vals.update(telem_new)
        if telem_new:
            telem_units_updated += 1

        unit_changed = False
        for feat in RECOMPUTE_FEATURES:
            if feat in LEAVE_AT_LAST_KNOWN:
                continue  # flagged: cannot match definition -> keep existing value
            if feat not in new_vals:
                continue  # not computed for this unit (e.g. no 2026 telem) -> keep
            nv = _coerce(new_vals[feat])
            if snap.get(feat) != nv:
                unit_changed = True
            snap[feat] = nv

        if unit_changed:
            changed_units += 1
        if telem_new and len(sample_before_after) < 5:
            sample_before_after[unit] = {
                "days_since_last_telem_shutdown": (
                    before.get("days_since_last_telem_shutdown"),
                    snap.get("days_since_last_telem_shutdown")),
                "days_since_last_alarm": (
                    before.get("days_since_last_alarm"), snap.get("days_since_last_alarm")),
                "days_since_last_repair": (
                    before.get("days_since_last_repair"), snap.get("days_since_last_repair")),
            }

        snap["as_of_date"] = str(as_of.date())
        snap["inputs_refreshed_at"] = datetime.now().isoformat(timespec="seconds")

    snap_path.write_text(json.dumps(snapshots, indent=2, default=str))

    return {
        "as_of": str(as_of.date()),
        "units_total": len(snapshots),
        "units_changed": changed_units,
        "units_with_2026_telem": telem_units_updated,
        "left_at_last_known": sorted(LEAVE_AT_LAST_KNOWN),
        "sample_before_after": sample_before_after,
    }


def _discover_max_date() -> pd.Timestamp:
    """Default as_of = latest date across the dropped monthly data (the new data).
    Falls back to the 2025 baseline cutoff when nothing has been dropped yet."""
    candidates = []
    for p in _monthly_files("reefer_repairs", "csv") + _monthly_files("reefer_repairs", "xlsx"):
        try:
            candidates.append(pd.to_datetime(_read_table(p)["OPENED"], errors="coerce").max())
        except Exception:
            pass
    for p in _monthly_files("meter_readings", "csv") + _monthly_files("meter_readings", "xlsx"):
        try:
            rd = pd.to_datetime(_read_table(p)["READDAY"], errors="coerce")
            candidates.append(rd[rd < "2030-01-01"].max())  # ignore stray far-future dates
        except Exception:
            pass
    for p in _monthly_files("telematics", "csv"):
        try:
            candidates.append(pd.to_datetime(pd.read_csv(p, skiprows=4)["Last Logged"], errors="coerce").max())
        except Exception:
            pass
    candidates = [c for c in candidates if pd.notna(c)]
    return max(candidates) if candidates else pd.Timestamp("2025-12-31")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Refresh frozen-model CURRENT-STATE inputs as-of a cutoff (no retrain).")
    p.add_argument("--as-of", type=str, default=None,
                   help="Cutoff date YYYY-MM-DD (default: latest date across dropped monthly data; "
                        "2025-12-31 if none dropped).")
    p.add_argument("--engine-dir", type=Path, default=ENGINE_DIR)
    args = p.parse_args(argv)

    as_of = pd.Timestamp(args.as_of) if args.as_of else _discover_max_date().normalize()
    print("=" * 70)
    print("CURRENT-STATE INPUT REFRESH (model NOT retrained)")
    print(f"  as-of cutoff : {as_of.date()}")
    print(f"  engine dir   : {args.engine_dir}")
    print("=" * 70)

    report = refresh(as_of, args.engine_dir)

    print(f"\n  units total          : {report['units_total']}")
    print(f"  units changed        : {report['units_changed']}")
    print(f"  units w/ 2026 telem  : {report['units_with_2026_telem']}")
    print(f"  left at last-known   : {report['left_at_last_known']} "
          f"(definition not matchable from build_trip_features_v3.py)")
    print(f"\n  Sample before/after (units with 2026 alarms):")
    for unit, ba in report["sample_before_after"].items():
        print(f"    {unit}:")
        for feat, (b, a) in ba.items():
            bs = f"{b:.2f}" if isinstance(b, float) else str(b)
            as_ = f"{a:.2f}" if isinstance(a, float) else str(a)
            print(f"      {feat:34s} {bs:>12s} -> {as_:>12s}")
    print(f"\n  snapshots -> {args.engine_dir / 'unit_snapshots.json'}")
    print(f"  backup     -> {args.engine_dir / 'unit_snapshots.prev.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
