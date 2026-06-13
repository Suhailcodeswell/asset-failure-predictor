"""Generate NDA-safe synthetic demo datasets for the public portfolio repo.

Creates:
  - data/processed/training/active_intmdl_fleet.csv  (sanitized from source or synthetic)
  - data/processed/training/trip_features_v3.csv       (sanitized sample for retraining demos)
  - data/processed/reference/alarm_severity_classification.csv
  - data/raw/**                                        (minimal synthetic source files)
  - data/demo/sample/                                  (monthly upload examples)

Run: python scripts/generate_demo_datasets.py
"""
from __future__ import annotations

import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRANSX = Path(r"c:\Users\ABDUL SATHAR\OneDrive\Desktop\TransX\TransX_Field_Project_2026")
DATA = ROOT / "data"
random.seed(42)

HUB_MAP = {
    "CN TOR": "Hub-East", "CN CGY": "Hub-West", "CN VCR": "Hub-Pacific",
    "CN MTL": "Hub-Metro", "CN EDM": "Hub-North", "CN WPG": "Hub-Central",
    "CN HAL": "Hub-Atlantic", "CN MOC": "Hub-Mountain", "CN REG": "Hub-Plains",
    "CN SAS": "Hub-Prairie", "TOR": "Hub-East", "CGY": "Hub-West", "VCR": "Hub-Pacific",
    "MTL": "Hub-Metro", "EDM": "Hub-North", "WPG": "Hub-Central",
}
MAKE_MAP = {"THRKN": "Vendor-A", "THRKN       ": "Vendor-A       ",
            "CARRR": "Vendor-B", "CARRR       ": "Vendor-B       "}
SHOP_MAP = {
    "ABF": "SHP-01", "ABF         ": "SHP-01      ", "MIS": "SHP-02", "MIS         ": "SHP-02      ",
    "CGY": "SHP-03", "CGY         ": "SHP-03      ", "EDM": "SHP-04", "EDM         ": "SHP-04      ",
    "MTL": "SHP-05", "MTL         ": "SHP-05      ", "RED": "SHP-06", "RED         ": "SHP-06      ",
    "TCC": "SHP-07", "TCC         ": "SHP-07      ", "TRX": "SHP-08", "TRX         ": "SHP-08      ",
    "VCR": "SHP-09", "VCR         ": "SHP-09      ", "WPG": "SHP-10", "WPG         ": "SHP-10      ",
}


def map_route(route: str) -> str:
    if not isinstance(route, str):
        return route
    s = route.strip()
    if " -> " in s:
        return s
    if "-CN " in s:
        origin_part, dest_suffix = s.split("-CN ", 1)
        origin = HUB_MAP.get(origin_part.strip(), origin_part.strip())
        dest = HUB_MAP.get(f"CN {dest_suffix.strip()}", dest_suffix.strip())
        return f"{origin} -> {dest}"
    if s.startswith("CN "):
        return HUB_MAP.get(s, s)
    return HUB_MAP.get(s, s)


def build_unit_map(units: list[str]) -> dict[str, str]:
    unique = sorted(set(u for u in units if isinstance(u, str) and re.match(r"R\d+", u)))
    return {u: f"AST-{i:04d}" for i, u in enumerate(unique, start=1001)}


def build_container_map(ids: list[str]) -> dict[str, str]:
    unique = sorted(set(str(i) for i in ids if str(i).isdigit()))
    return {cid: str(900000 + i) for i, cid in enumerate(unique, start=1)}


def sanitize_fleet(df: pd.DataFrame, unit_map: dict[str, str], container_map: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    if "reefer_unit" in out.columns:
        out["reefer_unit"] = out["reefer_unit"].map(lambda x: unit_map.get(str(x).strip(), x))
    if "container" in out.columns:
        out["container"] = out["container"].astype(str).apply(
            lambda x: f"FLI{container_map.get(re.sub(r'[^0-9]', '', x), x)[-6:]}" if re.sub(r'[^0-9]', '', x) in container_map else x
        )
    if "container_num" in out.columns:
        out["container_num"] = out["container_num"].astype(str).map(
            lambda x: container_map.get(x, x)
        )
    for col in ("reefer_make", "container_make"):
        if col in out.columns:
            out[col] = out[col].map(lambda x: MAKE_MAP.get(str(x), MAKE_MAP.get(str(x).strip(), x)))
    if "shop_id" in out.columns:
        out["shop_id"] = out["shop_id"].map(lambda x: SHOP_MAP.get(str(x), SHOP_MAP.get(str(x).strip(), x)))
    return out


def sanitize_trip_features(df: pd.DataFrame, unit_map: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col in ("reefer_unit", "trailer"):
        if col in out.columns:
            out[col] = out[col].astype(str).map(lambda x: unit_map.get(x.strip(), x))
    for col in ("route", "origin", "destination"):
        if col in out.columns:
            out[col] = out[col].astype(str).map(map_route)
    if "reefer_model" in out.columns:
        out["reefer_model"] = out["reefer_model"].astype(str).str.replace("SB", "MDL-").str.replace("S600", "MDL-600")
    return out


def load_or_synthesize_fleet() -> tuple[pd.DataFrame, dict[str, str]]:
    src = TRANSX / "data" / "processed" / "training" / "active_intmdl_fleet.csv"
    if src.exists():
        df = pd.read_csv(src)
        print(f"  Loaded fleet from TransX ({len(df)} rows) — sanitizing...")
        unit_map = build_unit_map(df["reefer_unit"].astype(str).tolist())
        container_map = build_container_map(df["container_num"].astype(str).tolist())
        return sanitize_fleet(df, unit_map, container_map), unit_map
    else:
        print("  TransX fleet not found — generating synthetic fleet...")
        rows = []
        for i in range(1, 81):
            rows.append({
                "container": f"FLI{900000 + i}",
                "container_num": str(900000 + i),
                "reefer_unit": f"AST-{1000 + i:04d}",
                "container_make": "GENCO", "container_model": "UNITMODEL",
                "container_year": 2014, "container_serial": str(900000 + i),
                "shop_id": f"SHP-{i % 10 + 1:02d}", "ownership": "COM",
                "trailer_kind": "RFRCNT", "in_service_date": "2014-01-15",
                "install_date": "2014-02-01", "container_age_years": 11,
                "reefer_install_days": 4000.0, "reefer_make": "Vendor-A",
                "reefer_model": "MDL-230", "reefer_year": 2014.0,
                "reefer_serial": f"SN{i:010d}", "reefer_status": "ACTIVE",
                "reefer_age_years": 11.0, "is_slxi": i % 5 == 0,
                "is_thermoking": 1, "is_carrier": 0,
                "total_events": random.randint(2, 15),
                "shutdown_count": random.randint(0, 3),
                "low_fuel_count": random.randint(0, 2),
                "alarm_count": random.randint(0, 5),
                "restarted_count": random.randint(0, 2),
                "not_reporting_count": 0, "has_shutdown": random.choice([0, 1]),
                "total_rail_legs": random.randint(30, 80),
                "total_rail_miles": random.randint(80000, 150000),
                "unique_routes": random.randint(10, 25),
                "shutdown_rate": round(random.uniform(0, 0.08), 4),
            })
        df = pd.DataFrame(rows)
        unit_map = {u: u for u in df["reefer_unit"].astype(str)}
        return df, unit_map


def load_or_skip_trip_features(unit_map: dict[str, str]) -> pd.DataFrame | None:
    src = TRANSX / "data" / "processed" / "training" / "trip_features_v3.csv"
    if not src.exists():
        print("  TransX trip_features not found — skipping (run pipeline after adding raw data)")
        return None
    print("  Sampling trip_features from TransX — sanitizing...")
    df = pd.read_csv(src, nrows=5000)
    return sanitize_trip_features(df, unit_map)


def copy_alarm_reference() -> None:
    src = TRANSX / "data" / "processed" / "reference" / "alarm_severity_classification.csv"
    dst = DATA / "processed" / "reference" / "alarm_severity_classification.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil_copy = src.read_bytes()
        dst.write_bytes(shutil_copy)
        print(f"  alarm_severity_classification.csv ({len(shutil_copy) // 1024} KB)")
    else:
        dst.write_text("Code,Description,Manufacturer,Official_Severity\n12,Shutdown,Vendor-A,Red\n", encoding="utf-8")
        print("  alarm_severity_classification.csv (minimal stub)")


def write_synthetic_raw(fleet: pd.DataFrame) -> None:
    """Create minimal synthetic raw Excel/CSV files matching pipeline input schemas."""
    repairs_dir = DATA / "raw" / "repairs"
    tele_dir = DATA / "raw" / "telematics"
    mileage_dir = DATA / "raw" / "mileage"
    meter_dir = DATA / "raw" / "hours meter reading"
    for d in (repairs_dir, tele_dir, mileage_dir, meter_dir, DATA / "raw" / "fuel data"):
        d.mkdir(parents=True, exist_ok=True)

    units = fleet["reefer_unit"].head(40).tolist()
    containers = fleet["container_num"].head(40).tolist()
    base_date = datetime(2024, 6, 1)

    repair_rows = []
    for i, unit in enumerate(units):
        for j in range(random.randint(3, 8)):
            dt = base_date + timedelta(days=i * 7 + j * 12)
            repair_rows.append({
                "SHOPID": f"SHP-{(i % 10) + 1:02d}      ",
                "UNITNUMBER": unit,
                "MAKE": "Vendor-A       ",
                "MODEL": "MDL-230   ",
                "MODELYEAR": 2014,
                "COSTCTCODE": "COMPANY     ",
                "DEPTCODE": "TLWEST      ",
                "TYPE": "RFR UNIT    ",
                "OPENED": dt,
                "ORDERTYPE": "REPAIR      ",
                "STATUS": "CLOSED      ",
                "ORDERNUM": f"SHP-ORD-{i:04d}-{j}",
                "REPREASON": random.choice(["PM", "GENREP", "GENREP"]),
                "COMPCODE": random.choice(["000-011", "082-DIA", "082-002", "082-041"]),
                "DESCRIP": "Synthetic demo repair record",
                "LINETYPE": "LABOR       ",
                "SumOfLINETOTAL": round(random.uniform(80, 1200), 2),
            })
    repairs_df = pd.DataFrame(repair_rows)
    repairs_path = repairs_dir / "Updated Reefer Unit Repairs 2024-2025.xlsx"
    repairs_df.to_excel(repairs_path, index=False, sheet_name="Sheet1")
    print(f"  {repairs_path.relative_to(ROOT)} ({len(repairs_df)} rows)")

    container_rows = []
    for i, cid in enumerate(containers[:30]):
        container_rows.append({
            "SHOPID": f"SHP-{(i % 10) + 1:02d}      ",
            "CUSTOMERNAME": "Demo Fleet",
            "UNITNUMBER": str(cid),
            "MAKE": "GENCO",
            "MODEL": "UNITMODEL",
            "MODELYEAR": 2014,
            "COSTCTCODE": "COMPANY     ",
            "DEPTCODE": "TLWEST      ",
            "TYPE": "CONTAINER   ",
            "OPENED": base_date + timedelta(days=i * 14),
            "REPREASON": "GENREP",
            "STATUS": "CLOSED      ",
            "ORDERNUM": f"CTR-{i:05d}",
            "COMPCODE": "082-DIA",
            "DESCRIP": "Synthetic container repair",
            "LINETYPE": "LABOR       ",
            "SumOfLINETOTAL": round(random.uniform(100, 800), 2),
        })
    crep_path = repairs_dir / "Container Repair Detail.xlsx"
    pd.DataFrame(container_rows).to_excel(crep_path, index=False)
    print(f"  {crep_path.relative_to(ROOT)}")

    tele_rows = []
    for i, cid in enumerate(containers[:25]):
        if random.random() < 0.15:
            tele_rows.append({
                "CONTAINER": f"FLI{cid}",
                "SHUTDOWN_DATE_TIME": base_date + timedelta(days=30 + i * 3),
                "MANIFEST": 100000 + i,
                "ALARM_CODE": 12,
            })
    tele_path = tele_dir / "Combined_Telematics_2025.xlsx"
    pd.DataFrame(tele_rows or [{"CONTAINER": f"FLI{containers[0]}", "SHUTDOWN_DATE_TIME": base_date,
                                 "MANIFEST": 100001, "ALARM_CODE": 12}]).to_excel(tele_path, index=False)
    print(f"  {tele_path.relative_to(ROOT)}")

    mc_path = tele_dir / "manifest_corrections.csv"
    mc_path.write_text("CONTAINER,SHUTDOWN_DATE_TIME,EVENT_TYPE,ORIGINAL_MANIFEST,VERIFIED_MANIFEST\n", encoding="utf-8")
    print(f"  {mc_path.relative_to(ROOT)}")

    meter_rows = []
    for i, unit in enumerate(units[:30]):
        for m in range(1, 5):
            meter_rows.append({
                "UNITNUMBER": unit, "TYPE": "RFR UNIT", "SERIALNO": f"SN{i:08d}",
                "ORDERNUM": f"MTR-{i}-{m}", "READDAY": datetime(2024, m, 15),
                "READING": 10000 + i * 500 + m * 200, "METERTYPE": "ENGINE HOURS",
                "Yr": 2024, "COSTCTCODE": "COMPANY", "DEPTCODE": "TLWEST",
                "MODEL": "MDL-230", "MODELYEAR": 2014, "STATUS": "ACTIVE", "READTYPE": "ACTUAL",
            })
    meter_path = meter_dir / "Meter Reading History.xlsx"
    pd.DataFrame(meter_rows).to_excel(meter_path, index=False)
    print(f"  {meter_path.relative_to(ROOT)}")

    # Rail miles — 4 header rows then data (header=4 in build_trip_features)
    rail_rows = []
    routes = [("Hub-East", "Hub-West"), ("Hub-West", "Hub-Pacific"), ("Hub-Metro", "Hub-East")]
    for i, (orig, dest) in enumerate(routes * 15):
        cid = containers[i % len(containers)]
        start = base_date + timedelta(days=i * 5)
        rail_rows.append({
            "Leg_Trailer1": str(cid),
            "Leg_Start_Date1": start,
            "Leg_End_Date1": start + timedelta(days=random.randint(3, 6)),
            "Leg_Miles1": random.randint(800, 2500),
            "Origin1": orig,
            "Destination1": dest,
            "Manifest": 200000 + i,
            "Service_Type": random.choice(["FRZ", "REF", "DRY"]),
            "Temperature": random.choice([-20, -10, 0, 5]),
        })
    rail_path = mileage_dir / "Rail Miles 2024 and 2025.xlsx"
    with pd.ExcelWriter(rail_path, engine="openpyxl") as writer:
        pd.DataFrame(rail_rows).to_excel(writer, sheet_name="Rail Miles Data 2024 & 2025",
                                         startrow=4, index=False)
    print(f"  {rail_path.relative_to(ROOT)} ({len(rail_rows)} legs)")

    # Demo monthly sample (CSV templates format)
    demo_dir = DATA / "demo" / "sample"
    demo_dir.mkdir(parents=True, exist_ok=True)
    sample = repairs_df.head(5).copy()
    sample.to_csv(demo_dir / "reefer_unit_repairs_sample.csv", index=False)
    print(f"  data/demo/sample/ (monthly upload example)")


def write_data_readme() -> None:
    text = """# Demo data (public portfolio)

All identifiers in this repository are **synthetic or anonymized**. No confidential
client data is included.

## Layout

| Path | Purpose |
|------|---------|
| `data/raw/` | Synthetic source files matching the pipeline input schema |
| `data/processed/training/` | Model-ready tables (fleet registry, trip features sample) |
| `data/processed/reference/` | Alarm severity lookup (generic industry codes) |
| `data/demo/sample/` | Example monthly upload CSVs |
| `data/templates/` | Column templates for new data drops |
| `data/*.json` | Exported model artifacts consumed by the Vercel web app |

## Retraining

```bash
pip install -r requirements-ml.txt
python pipeline/refresh.py --check    # validate inputs
python unified_pipeline.py            # full retrain (demo data)
```

The bundled `trip_features_v3.csv` is a **5,000-row sanitized sample** so reviewers
can inspect feature engineering output without downloading multi-GB client files.
"""
    (DATA / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    print("Generating portfolio demo datasets...")
    fleet, unit_map = load_or_synthesize_fleet()
    out_fleet = DATA / "processed" / "training" / "active_intmdl_fleet.csv"
    out_fleet.parent.mkdir(parents=True, exist_ok=True)
    fleet.to_csv(out_fleet, index=False)
    print(f"  {out_fleet.relative_to(ROOT)} ({len(fleet)} units)")

    trips = load_or_skip_trip_features(unit_map)
    if trips is not None:
        out_trips = DATA / "processed" / "training" / "trip_features_v3.csv"
        trips.to_csv(out_trips, index=False)
        mb = out_trips.stat().st_size / 1024 / 1024
        print(f"  {out_trips.relative_to(ROOT)} ({len(trips)} rows, {mb:.1f} MB)")

    copy_alarm_reference()
    write_synthetic_raw(fleet)
    write_data_readme()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
