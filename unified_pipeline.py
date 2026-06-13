"""Unified pipeline — retrain failure-risk models from canonical data."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import paths

ROOT = paths.ROOT


def _run(script: str, label: str) -> None:
    print(f"\n{'=' * 64}\n[{label}]\n{'=' * 64}", flush=True)
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
    print(f"\nDone — artifacts under {ROOT / 'models'}/ and {ROOT / 'data'}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
