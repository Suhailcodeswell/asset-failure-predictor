"""
OpsInsight Unified Risk Dashboard Builder
======================================
Scores the entire fleet with SHAP explanations and produces:
  1. Excel workbook  (output/OpsInsight_Risk_Dashboard.xlsx)
  2. Webapp JSON data (data/)

Usage:
    python scripts/build_dashboard.py                # Both outputs
    python scripts/build_dashboard.py --excel-only    # Excel only
    python scripts/build_dashboard.py --webapp-only   # Webapp only
"""
import argparse
import os
import sys
import time

# Ensure scripts/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboard.data_loader import DashboardData
from dashboard.fleet_scorer import FleetScorer
from dashboard.excel_builder import ExcelDashboardBuilder
from dashboard.webapp_exporter import WebappExporter


def main():
    parser = argparse.ArgumentParser(description='OpsInsight Risk Dashboard Builder')
    parser.add_argument('--excel-only', action='store_true', help='Only build the Excel workbook')
    parser.add_argument('--webapp-only', action='store_true', help='Only export webapp JSON files')
    parser.add_argument('--engine-dir', default='models/risk_engine', help='Risk engine artifacts directory')
    parser.add_argument('--output', default='failure/output/OpsInsight_Risk_Dashboard.xlsx', help='Excel output path')
    parser.add_argument('--webapp-dir', default='data', help='Webapp data directory')
    args = parser.parse_args()

    print('=' * 70)
    print('FLEETOPS UNIFIED RISK DASHBOARD BUILDER')
    print('=' * 70)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    t0 = time.time()
    print('\n1. Loading risk engine artifacts...')
    data = DashboardData(args.engine_dir)
    data.summary()
    print(f'   Done in {time.time() - t0:.1f}s')

    # ------------------------------------------------------------------
    # Step 2: Score fleet
    # ------------------------------------------------------------------
    t1 = time.time()
    print('\n2. Scoring fleet with SHAP explanations...')
    scorer = FleetScorer(data)
    scored_fleet = scorer.score_fleet()
    print(f'   Scored {len(scored_fleet)} units')
    print(f'   Top risk: {scored_fleet[0]["unit"]} at {scored_fleet[0]["risk_percent"]}% ({scored_fleet[0]["tier"]})')
    print(f'   Done in {time.time() - t1:.1f}s')

    # ------------------------------------------------------------------
    # Step 3: Compute analytics
    # ------------------------------------------------------------------
    t2 = time.time()
    print('\n3. Computing fleet analytics...')
    fleet_summary = scorer.compute_fleet_summary(scored_fleet)
    pm_buckets = scorer.compute_pm_buckets(scored_fleet)
    model_analysis = scorer.compute_model_analysis(scored_fleet)
    component_comparison = scorer.compute_component_comparison(scored_fleet)
    print(f'   Fleet: {fleet_summary["total_units"]} units, SD rate {fleet_summary["fleet_sd_rate"]}%')
    print(f'   Models analyzed: {len(model_analysis["models"])}')
    print(f'   Done in {time.time() - t2:.1f}s')

    # ------------------------------------------------------------------
    # Step 4: Build outputs
    # ------------------------------------------------------------------
    if not args.webapp_only:
        t3 = time.time()
        print('\n4a. Building Excel workbook...')
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        builder = ExcelDashboardBuilder(
            data, scored_fleet, fleet_summary,
            pm_buckets, model_analysis, component_comparison
        )
        builder.build(args.output)
        print(f'   Done in {time.time() - t3:.1f}s')

    if not args.excel_only:
        t4 = time.time()
        print('\n4b. Exporting webapp JSON files...')
        exporter = WebappExporter(
            data, scored_fleet, fleet_summary,
            pm_buckets, model_analysis
        )
        exporter.export(args.webapp_dir)
        print(f'   Done in {time.time() - t4:.1f}s')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = time.time() - t0
    print('\n' + '=' * 70)
    print(f'DASHBOARD BUILD COMPLETE ({total:.1f}s)')
    print('=' * 70)
    if not args.webapp_only:
        print(f'  Excel:  {args.output}')
    if not args.excel_only:
        print(f'  Webapp: {args.webapp_dir}/')


if __name__ == '__main__':
    main()
