#!/usr/bin/env python3
"""
Asset Failure Predictor — Model Refresh Pipeline
==============================
Single entry point to rebuild the risk model from updated data.

Usage:
    python pipeline/refresh.py              # Full pipeline
    python pipeline/refresh.py --check      # Validate data files only (no training)
    python pipeline/refresh.py --skip-features  # Skip feature engineering (reuse existing trip_features_v3.csv)

Steps:
    1. Validate all required data files exist
    2. Feature engineering  (build_trip_features_v3.py → trip_features_v3.csv)
    3. Model training       (build_risk_engine.py → models/risk_engine/*)
    4. Repair patterns      (repair_pattern_analysis.py → data/repair_patterns.json)
    5. Dashboard build      (build_dashboard.py → Excel + data/*)
    6. Validation & summary
"""
import os
import sys
import subprocess
import time
import json
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

# ======================================================================
# CONFIGURATION — Required data files
# ======================================================================

REQUIRED_FILES = {
    # Repairs
    'data/raw/repairs/Updated Reefer Unit Repairs 2024-2025.xlsx': 'Reefer repair orders',
    'data/raw/repairs/Container Repair Detail.xlsx': 'Container repair orders',
    # Fleet
    'data/processed/training/active_intmdl_fleet.csv': 'Active fleet registry',
    # Telematics
    'data/raw/telematics/Combined_Telematics_2025.xlsx': 'Telematics shutdown events',
    'data/raw/telematics/manifest_corrections.csv': 'Manifest corrections',
    # Mileage
    'data/raw/mileage/Rail Miles 2024 and 2025.xlsx': 'Rail miles / trip manifests',
    'data/raw/hours meter reading/Meter Reading History.xlsx': 'Engine hour meter readings',
    # Reference
    'data/processed/reference/alarm_severity_classification.csv': 'Alarm severity codes',
}

# Optional files — pipeline continues if missing
OPTIONAL_FILES = {
    'data/raw/telematics/quarterly/Daily Telematics Update -YTD 2025 Q1.xlsx': 'Q1 telematics',
    'data/raw/telematics/quarterly/Daily Telematics Update -YTD 2025 Q2.xlsx': 'Q2 telematics',
    'data/raw/telematics/quarterly/Daily Telematics Update -YTD 2025 Q3.xlsx': 'Q3 telematics',
    'data/raw/telematics/quarterly/Daily Telematics Update -YTD 2025 Q4.xlsx': 'Q4 telematics',
    'data/raw/fuel data/MMA 2024 Fuel Data.xlsx': '2024 fuel data',
    'data/raw/fuel data/MMA 2025 Fuel Data.xlsx': '2025 fuel data',
    'data/processed/training/weather_cache.csv': 'Weather cache (will re-fetch if missing)',
}

# Pipeline scripts in execution order
SCRIPTS = [
    ('pipeline/build_trip_features_v3.py', 'Feature Engineering'),
    ('pipeline/build_risk_engine.py', 'Model Training'),
    ('pipeline/repair_pattern_analysis.py', 'Repair Pattern Analysis'),
    ('pipeline/build_dashboard.py', 'Dashboard Build (Excel + Webapp)'),
]

# Expected outputs after full pipeline
EXPECTED_OUTPUTS = {
    # From feature engineering
    'data/processed/training/trip_features_v3.csv': 'Training features',
    # From model training
    'models/risk_engine/xgb_model.json': 'XGBoost model',
    'models/risk_engine/calibrated_model.pkl': 'Calibrated model',
    'models/risk_engine/imputer.pkl': 'Feature imputer',
    'models/risk_engine/label_encoders.json': 'Label encoders',
    'models/risk_engine/feature_meta.json': 'Feature metadata',
    'models/risk_engine/unit_snapshots.json': 'Unit snapshots',
    'models/risk_engine/risk_distribution.json': 'Risk distribution',
    # From webapp export
    'data/model_dump.json': 'Webapp model',
    'data/calibration_table.json': 'Calibration table',
    'data/feature_names.json': 'Feature names',
    'data/feature_defaults.json': 'Feature defaults',
    'data/unit_snapshots.json': 'Webapp unit snapshots',
    # From repair analysis
    'data/repair_patterns.json': 'Repair patterns',
    # From dashboard build
    'failure/output/OpsInsight_Risk_Dashboard.xlsx': 'Excel dashboard',
}


def print_header(text):
    print(f'\n{"=" * 70}')
    print(f'  {text}')
    print(f'{"=" * 70}')


def validate_data():
    """Check that all required data files exist."""
    print_header('STEP 1: VALIDATING DATA FILES')

    missing = []
    for path, desc in REQUIRED_FILES.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f'  OK    {desc:<45} ({size_mb:.1f} MB)')
        else:
            print(f'  MISS  {desc:<45} <- {path}')
            missing.append((path, desc))

    print()
    for path, desc in OPTIONAL_FILES.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f'  OK    {desc:<45} ({size_mb:.1f} MB)')
        else:
            print(f'  SKIP  {desc:<45} (optional)')

    if missing:
        print(f'\n  ERROR: {len(missing)} required file(s) missing:')
        for path, desc in missing:
            print(f'    - {path}')
        print('\n  Place the required files and re-run.')
        return False

    print(f'\n  All {len(REQUIRED_FILES)} required files present.')
    return True


def run_script(script_path, description):
    """Run a Python script as a subprocess, streaming output."""
    print_header(f'{description.upper()}  ({script_path})')
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE,
        capture_output=False,  # stream to terminal
    )

    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f'\n  FAILED after {elapsed:.0f}s (exit code {result.returncode})')
        return False

    print(f'\n  Completed in {elapsed:.0f}s')
    return True


def validate_outputs():
    """Check that all expected output files exist and are recent."""
    print_header('VALIDATION: CHECKING OUTPUTS')

    now = time.time()
    all_ok = True
    stale = []

    for path, desc in EXPECTED_OUTPUTS.items():
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            age_min = (now - os.path.getmtime(path)) / 60
            status = 'OK' if age_min < 30 else 'STALE'
            if status == 'STALE':
                stale.append(path)
            print(f'  {status:>5}  {desc:<35} {size_kb:>8.1f} KB  ({age_min:.0f} min ago)')
        else:
            print(f'  MISS   {desc:<35} <- {path}')
            all_ok = False

    if stale:
        print(f'\n  WARNING: {len(stale)} file(s) older than 30 min — may not have been rebuilt:')
        for p in stale:
            print(f'    - {p}')

    return all_ok


def run_quick_test():
    """Smoke test: load the scorer and predict one trip."""
    print_header('SMOKE TEST: SCORING A SAMPLE TRIP')
    try:
        # Import fresh to pick up new artifacts
        sys.path.insert(0, BASE)
        from pipeline.risk_scorer import RiskScorer

        scorer = RiskScorer('models/risk_engine')
        # Use first available unit from fleet lookup after training
        sample_unit = 'AST-1001'
        result = scorer.score(sample_unit, {
            'route': 'Hub-East -> Hub-West',
            'service_type': 'FRZ',
            'temperature': -20,
            'leg_miles': 2127,
            'month': 1,
            'season': 'Winter',
        })

        print(f'  Unit:       {sample_unit}')
        print(f'  Route:      Hub-East -> Hub-West (Winter, -20°C)')
        print(f'  Risk Score: {result["risk_score"]:.4f}')
        print(f'  Risk Tier:  {result["risk_tier"]}')
        print(f'  Top Factor: {result["risk_factors"][0]["feature"]} ({result["risk_factors"][0]["direction"]})')
        print(f'  Warnings:   {len(result.get("warnings", []))}')
        print(f'\n  Smoke test PASSED')
        return True
    except Exception as e:
        print(f'  Smoke test FAILED: {e}')
        return False


def print_summary(results, total_time):
    """Print final summary."""
    print_header('PIPELINE SUMMARY')
    print(f'  Date:     {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'  Duration: {total_time:.0f}s ({total_time/60:.1f} min)')
    print()

    for step, (ok, elapsed) in results.items():
        status = 'PASS' if ok else 'FAIL'
        print(f'  {status}  {step:<40} ({elapsed:.0f}s)')

    failed = [s for s, (ok, _) in results.items() if not ok]
    if failed:
        print(f'\n  {len(failed)} step(s) FAILED. Fix errors above and re-run.')
    else:
        print(f'\n  ALL STEPS PASSED — model is ready for deployment.')
        print(f'  Next: deploy webapp/ to Vercel (vercel --prod)')


def main():
    args = sys.argv[1:]
    skip_features = '--skip-features' in args
    check_only = '--check' in args

    print_header('ASSET FAILURE PREDICTOR — MODEL REFRESH')
    print(f'  Working directory: {BASE}')
    print(f'  Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    if skip_features:
        print(f'  Mode: --skip-features (reusing existing trip_features_v3.csv)')
    if check_only:
        print(f'  Mode: --check (validation only, no training)')

    t_total = time.time()
    results = {}

    # Step 1: Validate data
    t0 = time.time()
    data_ok = validate_data()
    results['Data validation'] = (data_ok, time.time() - t0)

    if not data_ok:
        print('\nAborting: required data files missing.')
        sys.exit(1)

    if check_only:
        print('\n--check mode: data validation passed. Exiting.')
        sys.exit(0)

    # Delete stale weather cache so it gets rebuilt with current years
    weather_cache = 'data/processed/training/weather_cache.csv'
    if os.path.exists(weather_cache) and not skip_features:
        age_days = (time.time() - os.path.getmtime(weather_cache)) / 86400
        if age_days > 180:
            print(f'\n  Deleting stale weather cache ({age_days:.0f} days old)...')
            os.remove(weather_cache)

    # Step 2-5: Run pipeline scripts
    scripts_to_run = SCRIPTS.copy()
    if skip_features:
        scripts_to_run = scripts_to_run[1:]  # skip feature engineering
        # Verify features file exists
        if not os.path.exists('data/processed/training/trip_features_v3.csv'):
            print('\nERROR: --skip-features but data/processed/training/trip_features_v3.csv not found.')
            sys.exit(1)

    for script_path, description in scripts_to_run:
        t0 = time.time()
        ok = run_script(script_path, description)
        results[description] = (ok, time.time() - t0)
        if not ok:
            print(f'\nPipeline STOPPED at: {description}')
            print(f'Fix the error and re-run. You can use --skip-features to skip')
            print(f'feature engineering if trip_features_v3.csv is already up to date.')
            print_summary(results, time.time() - t_total)
            sys.exit(1)

    # Step 6: Validate outputs
    t0 = time.time()
    outputs_ok = validate_outputs()
    results['Output validation'] = (outputs_ok, time.time() - t0)

    # Step 7: Smoke test
    t0 = time.time()
    test_ok = run_quick_test()
    results['Smoke test'] = (test_ok, time.time() - t0)

    # Summary
    print_summary(results, time.time() - t_total)

    if all(ok for ok, _ in results.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
