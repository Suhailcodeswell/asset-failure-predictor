"""One-time port of TransX ML pipeline into the public portfolio repo.

Copies pipeline scripts, adapts paths/branding, and generates synthetic demo data.
Run from repo root: python scripts/setup_portfolio_from_transx.py
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSX = Path(r"c:\Users\ABDUL SATHAR\OneDrive\Desktop\TransX\TransX_Field_Project_2026")
SRC_SCRIPTS = TRANSX / "failure" / "scripts"
DST_PIPELINE = ROOT / "pipeline"

# Scripts to include in the public portfolio (core ML + export + refresh).
PIPELINE_FILES = [
    "refresh.py",
    "refresh_inputs.py",
    "refresh_current_state.py",
    "build_trip_features_v3.py",
    "build_risk_engine.py",
    "export_webapp.py",
    "repair_pattern_analysis.py",
    "risk_scorer.py",
    "snapshots.py",
    "build_dashboard.py",
    "build_cph_webapp_data.py",
    "build_cpm_webapp_data.py",
    "export_cph_models.py",
    "export_cpm_models.py",
    "experiment_feature_sets.py",
    "experiment_shap_table.py",
    "enrich_snapshots_components.py",
    "repair_sequence_v3.py",
]

REPLACEMENTS = [
    (r"os\.path\.dirname\(os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)\)",
     "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))"),
    ("apps/webapp/data", "data"),
    ('ROOT / "webapp" / "data"', 'ROOT / "data"'),
    ('WEBAPP_DATA = ROOT / "webapp" / "data"', 'WEBAPP_DATA = ROOT / "data"'),
    ("TransX Model Refresh Pipeline", "Asset Failure Predictor — Model Refresh Pipeline"),
    ("TRANSX MODEL REFRESH PIPELINE", "ASSET FAILURE PREDICTOR — MODEL REFRESH"),
    ("TransX Reefer Risk Scoring Engine", "Reefer Risk Scoring Engine"),
    ("TransX Reefer Risk Scorer", "Reefer Risk Scorer"),
    ("TransX Internal", "Fleet Internal"),
    ("TRANSX", "FLEETOPS"),
    ("TransX", "OpsInsight"),
    ("transx", "opsinsight"),
    ("'scripts/", "'pipeline/"),
    ('"scripts/', '"pipeline/'),
    ("failure/scripts/", "pipeline/"),
    ("failure/output/TransX_Risk_Dashboard.xlsx", "output/Risk_Dashboard.xlsx"),
    ("from scripts.risk_scorer import RiskScorer", "from pipeline.risk_scorer import RiskScorer"),
    ("python scripts/refresh.py", "python pipeline/refresh.py"),
]


def adapt_content(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new) if not old.startswith("os\\.") else re.sub(old, new, text)
    return text


def copy_pipeline_scripts() -> None:
    DST_PIPELINE.mkdir(parents=True, exist_ok=True)
    for name in PIPELINE_FILES:
        src = SRC_SCRIPTS / name
        if not src.exists():
            print(f"  SKIP missing: {name}")
            continue
        dst = DST_PIPELINE / name
        content = adapt_content(src.read_text(encoding="utf-8"))
        dst.write_text(content, encoding="utf-8")
        print(f"  pipeline/{name}")

    dash_src = SRC_SCRIPTS / "dashboard"
    if dash_src.is_dir():
        dash_dst = DST_PIPELINE / "dashboard"
        if dash_dst.exists():
            shutil.rmtree(dash_dst)
        shutil.copytree(dash_src, dash_dst)
        for py in dash_dst.rglob("*.py"):
            py.write_text(adapt_content(py.read_text(encoding="utf-8")), encoding="utf-8")
        print("  pipeline/dashboard/")


def copy_support_files() -> None:
    # paths.py — portfolio version (no cph/cpm ref paths)
    paths_content = '''"""Canonical filesystem layout for the portfolio repo."""
from __future__ import annotations
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
DATA: Path = ROOT / "data"
RAW: Path = DATA / "raw"
PROCESSED: Path = DATA / "processed"
MODELS: Path = ROOT / "models"
OUTPUT: Path = ROOT / "output"


def raw(*parts: str) -> Path:
    return RAW.joinpath(*parts)


def processed(*parts: str) -> Path:
    return PROCESSED.joinpath(*parts)


def model(*parts: str) -> Path:
    return MODELS.joinpath(*parts)
'''
    (ROOT / "paths.py").write_text(paths_content, encoding="utf-8")
    print("  paths.py")

    unified = '''"""Unified pipeline — retrain failure-risk models from canonical data."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import paths

ROOT = paths.ROOT


def _run(script: str, label: str) -> None:
    print(f"\\n{'=' * 64}\\n[{label}]\\n{'=' * 64}", flush=True)
    proc = subprocess.run([sys.executable, script], cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"[{label}] FAILED (exit {proc.returncode})")


def run_failure() -> None:
    _run("pipeline/build_trip_features_v3.py", "FAILURE 1/4 — feature engineering")
    _run("pipeline/build_risk_engine.py", "FAILURE 2/4 — model training")
    _run("pipeline/repair_pattern_analysis.py", "FAILURE 3/4 — repair patterns")
    _run("pipeline/export_webapp.py", "FAILURE 4/4 — webapp export")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Retrain failure-risk models from data/raw.")
    p.add_argument("--only", choices=["failure"], help="Workstream to run (default: failure)")
    args = p.parse_args(argv)
    run_failure()
    print(f"\\nDone — artifacts under {ROOT / 'models'}/ and {ROOT / 'data'}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    (ROOT / "unified_pipeline.py").write_text(unified, encoding="utf-8")
    print("  unified_pipeline.py")

    # Data templates
    tpl_src = TRANSX / "data" / "templates"
    tpl_dst = ROOT / "data" / "templates"
    if tpl_src.is_dir():
        tpl_dst.mkdir(parents=True, exist_ok=True)
        for f in tpl_src.iterdir():
            if f.is_file():
                content = adapt_content(f.read_text(encoding="utf-8"))
                (tpl_dst / f.name).write_text(content, encoding="utf-8")
        print("  data/templates/")

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / ".gitkeep").touch()
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / ".gitkeep").touch()
    (ROOT / "data" / "processed" / "training").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed" / "reference").mkdir(parents=True, exist_ok=True)


def main() -> int:
    if not TRANSX.exists():
        print(f"ERROR: TransX source not found at {TRANSX}", file=sys.stderr)
        return 1
    print("Copying pipeline scripts...")
    copy_pipeline_scripts()
    print("Copying support files...")
    copy_support_files()
    print("Done. Next: python scripts/generate_demo_datasets.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
