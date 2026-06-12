"""Anonymize fleet data for public portfolio deployment.

Replaces client-specific identifiers with synthetic operational labels while
preserving model structure and statistically plausible values.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

random.seed(42)

HUB_MAP = {
    "CN TOR": "Hub-East",
    "CN CGY": "Hub-West",
    "CN VCR": "Hub-Pacific",
    "CN MTL": "Hub-Metro",
    "CN EDM": "Hub-North",
    "CN WPG": "Hub-Central",
    "CN HAL": "Hub-Atlantic",
    "CN MOC": "Hub-Mountain",
    "CN REG": "Hub-Plains",
    "CN SAS": "Hub-Prairie",
}

MAKE_MAP = {
    "THRKN": "Vendor-A",
    "THRKN       ": "Vendor-A       ",
    "CARRR": "Vendor-B",
    "CARRR       ": "Vendor-B       ",
}

SHOP_MAP = {
    "ABF": "SHP-01",
    "ABF         ": "SHP-01      ",
    "MIS": "SHP-02",
    "MIS         ": "SHP-02      ",
    "CGY": "SHP-03",
    "CGY         ": "SHP-03      ",
    "EDM": "SHP-04",
    "EDM         ": "SHP-04      ",
    "MTL": "SHP-05",
    "MTL         ": "SHP-05      ",
    "RED": "SHP-06",
    "RED         ": "SHP-06      ",
    "TCC": "SHP-07",
    "TCC         ": "SHP-07      ",
    "TRX": "SHP-08",
    "TRX         ": "SHP-08      ",
    "VCR": "SHP-09",
    "VCR         ": "SHP-09      ",
    "WPG": "SHP-10",
    "WPG         ": "SHP-10      ",
}

TEXT_REPLACEMENTS = [
    ("TransX Internal", "Fleet Internal"),
    ("TRANSX", "FLEETOPS"),
    ("TransX", "OpsInsight"),
    ("transx", "opsinsight"),
    ("TNX", "FLI"),
]


def map_route(route: str) -> str:
    if " -> " in route:
        return route
    if "-CN " in route:
        origin_part, dest_suffix = route.split("-CN ", 1)
        origin = HUB_MAP.get(origin_part.strip(), origin_part.strip())
        dest_key = f"CN {dest_suffix.strip()}"
        dest = HUB_MAP.get(dest_key, dest_key)
        return f"{origin} -> {dest}"
    if route.startswith("CN "):
        return HUB_MAP.get(route.strip(), route.strip())
    return route


def replace_text(value: str) -> str:
    out = value
    for old, new in TEXT_REPLACEMENTS:
        out = out.replace(old, new)
    if out.startswith("CN "):
        out = map_route(out)
    return out


def jitter_number(value: float, pct: float = 0.04) -> float:
    factor = 1.0 + random.uniform(-pct, pct)
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, int(round(value * factor)))
    if isinstance(value, float):
        return round(value * factor, 6)
    return value


def build_unit_map(keys: list[str]) -> dict[str, str]:
    units = sorted(set(k for k in keys if isinstance(k, str) and k.startswith("R")))
    return {u: f"AST-{i:04d}" for i, u in enumerate(units, start=1001)}


def build_container_map(keys: list[str]) -> dict[str, str]:
    ids = sorted(set(str(k) for k in keys if str(k).isdigit()))
    return {cid: 900000 + i for i, cid in enumerate(ids, start=1)}


def remap_dict_keys(obj: dict, mapping: dict) -> dict:
    return {mapping.get(str(k), k): v for k, v in obj.items()}


def sanitize_json_obj(obj, unit_map: dict, container_map: dict):
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            nk = k
            if isinstance(k, str):
                if k in unit_map:
                    nk = unit_map[k]
                elif k.isdigit() and k in container_map:
                    nk = str(container_map[k])
                elif k.startswith("R") and k in unit_map:
                    nk = unit_map[k]
                else:
                    nk = replace_text(k)
            new[nk] = sanitize_json_obj(v, unit_map, container_map)
        return new
    if isinstance(obj, list):
        return [sanitize_json_obj(v, unit_map, container_map) for v in obj]
    if isinstance(obj, str):
        s = replace_text(obj)
        if s in unit_map:
            return unit_map[s]
        if s in MAKE_MAP:
            return MAKE_MAP[s]
        if s.strip() in {k.strip(): v for k, v in SHOP_MAP.items()}:
            return SHOP_MAP.get(s, SHOP_MAP.get(s.strip() + " " * (len(s) - len(s.strip())), s))
        for shop_old, shop_new in SHOP_MAP.items():
            if s == shop_old:
                return shop_new
        if s.startswith("CN ") and "-" in s:
            return map_route(s)
        if re.fullmatch(r"R\d+", s) and s in unit_map:
            return unit_map[s]
        return s
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return jitter_number(obj)
    return obj


def load_json(name: str):
    with open(DATA / name, encoding="utf-8") as f:
        return json.load(f)


def save_json(name: str, obj):
    with open(DATA / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":") if name.endswith("_fleet.json") else None)
        if name.endswith("_fleet.json"):
            pass
        else:
            f.write("\n")


def main():
    fleet_lookup = load_json("fleet_lookup.json")
    trailer_map = load_json("trailer_map.json")
    unit_snapshots = load_json("unit_snapshots.json")

    unit_keys = (
        list(fleet_lookup.keys())
        + list(unit_snapshots.keys())
        + list(trailer_map.values())
        + [row.get("unit", "") for row in load_json("cph_fleet.json")]
    )
    unit_map = build_unit_map([str(k) for k in unit_keys if k and str(k) != "nan"])

    container_keys = list(trailer_map.keys()) + [
        str(row.get("container_id", "")) for row in load_json("cpm_fleet.json") if isinstance(row, dict)
    ]
    container_map = build_container_map(container_keys)

    # category_maps: remap route/origin/destination labels only
    category_maps = load_json("category_maps.json")
    for section in ("route", "origin", "destination"):
        if section in category_maps:
            remapped = {}
            for label, code in category_maps[section].items():
                remapped[map_route(label) if section == "route" else HUB_MAP.get(label, label)] = code
            category_maps[section] = remapped
    save_json("category_maps.json", category_maps)

    # route_stats
    route_stats = load_json("route_stats.json")
    for row in route_stats:
        row["route"] = map_route(row["route"])
        row["trips"] = jitter_number(row["trips"])
        row["shutdowns"] = jitter_number(row["shutdowns"])
        row["rate"] = round(row["shutdowns"] / max(row["trips"], 1) * 100, 2)
    save_json("route_stats.json", route_stats)

    # Keyed JSON files
    for name in ("fleet_lookup.json", "unit_snapshots.json"):
        obj = load_json(name)
        sanitized = sanitize_json_obj(obj, unit_map, container_map)
        save_json(name, sanitized)

    # trailer_map
    new_trailer = {}
    for cid, unit in trailer_map.items():
        if unit is None or (isinstance(unit, float) and str(unit) == "nan"):
            continue
        new_cid = str(container_map.get(str(cid), cid))
        new_unit = unit_map.get(str(unit), str(unit))
        new_trailer[new_cid] = new_unit
    save_json("trailer_map.json", new_trailer)

    # Fleet arrays
    for name in ("cph_fleet.json", "cpm_fleet.json"):
        rows = load_json(name)
        for row in rows:
            if "unit" in row and str(row["unit"]) in unit_map:
                row["unit"] = unit_map[str(row["unit"])]
            if "make" in row:
                row["make"] = MAKE_MAP.get(row["make"], MAKE_MAP.get(row["make"].strip(), row["make"]))
            if "container_id" in row:
                old = str(int(row["container_id"]))
                row["container_id"] = container_map.get(old, row["container_id"])
            if "reefer_make" in row and row["reefer_make"] in MAKE_MAP:
                row["reefer_make"] = MAKE_MAP[row["reefer_make"]]
            if "dominant_shop" in row and row["dominant_shop"] in SHOP_MAP:
                row["dominant_shop"] = SHOP_MAP[row["dominant_shop"]]
            for key in list(row.keys()):
                if isinstance(row[key], (int, float)) and key not in ("container_id", "trailer_year", "trailer_age", "n_shops", "modelyear"):
                    row[key] = jitter_number(row[key])
        save_json(name, rows)

    # Summaries
    cph_summary = load_json("cph_summary.json")
    cph_summary["makes"] = [MAKE_MAP.get(m, m) for m in cph_summary.get("makes", [])]
    cph_summary["shops"] = [SHOP_MAP.get(s, s) for s in cph_summary.get("shops", [])]
    for k in ("avg_pm_cph", "avg_genrep_cph", "avg_total_cph"):
        if k in cph_summary:
            cph_summary[k] = jitter_number(cph_summary[k])
    save_json("cph_summary.json", cph_summary)

    cpm_summary = load_json("cpm_summary.json")
    save_json("cpm_summary.json", sanitize_json_obj(cpm_summary, unit_map, container_map))

    repair_patterns = load_json("repair_patterns.json")
    save_json("repair_patterns.json", sanitize_json_obj(repair_patterns, unit_map, container_map))

    print(f"Sanitized data in {DATA}")
    print(f"  Units remapped: {len(unit_map)}")
    print(f"  Containers remapped: {len(container_map)}")


if __name__ == "__main__":
    main()
