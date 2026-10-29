#!/usr/bin/env python3
"""
main.py - Master comparison runner for min-jerk-flatness-planning.

Executes all comparison scripts and generates publication-quality PDFs, PNGs, and JSON metrics:
  1. Open-loop vs Planned Trajectories (plot_openloop_trajectories.py)
  2. RealtimeController Velocities Comparison (9 subplots) (plot_realtime_velocities.py)
  3. RealtimeController Control Inputs Comparison (6 subplots) (plot_realtime_actuation.py)
  4. Real-time Response / Solve Times Comparison (plot_response_times.py)
  5. Tracking & Actuation Error Metrics JSON + PDF (compute_error_metrics.py)
  6. Academic Benchmark Comparison (compare_cases.py)
"""

import sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Import modules
from plot_openloop_trajectories import plot_openloop_comparison
from plot_realtime_velocities import plot_velocities_comparison
from plot_realtime_actuation import plot_actuation_comparison
from plot_response_times import plot_response_times_comparison
from compute_error_metrics import main as run_compute_error_metrics
from compare_cases import main as run_compare_cases

def run_all():
    print("=" * 70)
    print("  MIN-JERK FLATNESS PLANNING - COMPREHENSIVE COMPARISON SUITE")
    print("=" * 70)

    print("\n[1/6] Generating Open-loop vs Planned Trajectory comparison...")
    plot_openloop_comparison()

    print("\n[2/6] Generating RealtimeController Velocities (9-subplot grid)...")
    plot_velocities_comparison()

    print("\n[3/6] Generating RealtimeController Control Inputs (6-subplot grid)...")
    plot_actuation_comparison()

    print("\n[4/6] Generating Real-time Response Times & 30 Hz Headroom comparison...")
    plot_response_times_comparison()

    print("\n[5/6] Computing Tracking & Actuation Error Metrics (JSON & PDF)...")
    run_compute_error_metrics()

    print("\n[6/6] Generating Benchmark Timing Comparison (compare_cases)...")
    run_compare_cases()

    print("\n" + "=" * 70)
    print("  ALL COMPARISON ARTIFACTS SUCCESSFULLY GENERATED IN:")
    print(f"  {SCRIPT_DIR}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Run comparison scripts for min-jerk-flatness-planning.")
    parser.add_argument('--all', action='store_true', default=True, help="Run all comparison scripts (default)")
    parser.add_argument('--trajectories', action='store_true', help="Run open-loop trajectories comparison only")
    parser.add_argument('--velocities', action='store_true', help="Run velocities (9 subplots) comparison only")
    parser.add_argument('--actuation', action='store_true', help="Run control inputs (6 subplots) comparison only")
    parser.add_argument('--timing', action='store_true', help="Run response times comparison only")
    parser.add_argument('--metrics', action='store_true', help="Run error metrics calculation (JSON + PDF) only")
    parser.add_argument('--benchmark', action='store_true', help="Run compare_cases benchmark only")

    args = parser.parse_args()

    if any([args.trajectories, args.velocities, args.actuation, args.timing, args.metrics, args.benchmark]):
        if args.trajectories:
            plot_openloop_comparison()
        if args.velocities:
            plot_velocities_comparison()
        if args.actuation:
            plot_actuation_comparison()
        if args.timing:
            plot_response_times_comparison()
        if args.metrics:
            run_compute_error_metrics()
        if args.benchmark:
            run_compare_cases()
    else:
        run_all()

if __name__ == '__main__':
    main()
