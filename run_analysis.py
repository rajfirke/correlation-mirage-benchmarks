#!/usr/bin/env python3
"""
Pipeline orchestrator for the copula benchmark analysis.

Runs the full analysis pipeline in order. Each script is independent and
can also be run standalone. Pre-computed results are provided in results/;
re-running regenerates them from the data in data/.

Usage:
    python run_analysis.py              # run all (core + validation)
    python run_analysis.py --core       # core pipeline only (Tables 1-3, Figs 1-4)
    python run_analysis.py --validate   # validation scripts only (Appendices)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

CORE_SCRIPTS = [
    "analysis/01_download_and_clean.py",
    "analysis/02_marginals_pit.py",
    "analysis/03_bivariate_copulas.py",
    "analysis/04_vine_copula.py",
    "analysis/05_benchmark_selection.py",
    "analysis/06_figures.py",
    "analysis/07_nonparametric_tail.py",
    "analysis/07_statistical_robustness.py",
    "analysis/08_deduplication.py",
    "analysis/08_revised_selection.py",
    "analysis/09_distance_corr_comparison.py",
    "analysis/10_sensitivity_analysis.py",
    "analysis/11_ollm_full_pipeline.py",
    "analysis/12_final_figures.py",
]

VALIDATION_SCRIPTS = [
    "scripts/13_wild_copula_pipeline.py",
    "scripts/14_downstream_ranking.py",
    "scripts/15_expanded_copula_candidates.py",
    "scripts/16_berkson_calibration.py",
    "scripts/17_threshold_sensitivity.py",
    "scripts/18_deduplicated_inference.py",
    "scripts/19_rebuttal_su96.py",
]


def run_script(script_path: str) -> bool:
    full_path = PROJECT_ROOT / script_path
    if not full_path.exists():
        print(f"  ✗ {script_path} — not found, skipping")
        return False

    print(f"\n{'='*60}")
    print(f"  Running: {script_path}")
    print(f"{'='*60}")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=str(PROJECT_ROOT),
    )

    elapsed = time.time() - t0
    status = "✓" if result.returncode == 0 else "✗"
    print(f"  {status} {script_path} ({elapsed:.1f}s)")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run copula benchmark analysis pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--core", action="store_true", help="Core pipeline only")
    group.add_argument("--validate", action="store_true", help="Validation scripts only")
    args = parser.parse_args()

    if args.core:
        scripts = CORE_SCRIPTS
    elif args.validate:
        scripts = VALIDATION_SCRIPTS
    else:
        scripts = CORE_SCRIPTS + VALIDATION_SCRIPTS

    print(f"Copula Benchmark Analysis Pipeline")
    print(f"Running {len(scripts)} scripts\n")

    t0 = time.time()
    results = [(s, run_script(s)) for s in scripts]
    total = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total:.1f}s")
    passed = sum(1 for _, ok in results if ok)
    print(f"  {passed}/{len(results)} scripts succeeded")
    for script, ok in results:
        if not ok:
            print(f"  ✗ FAILED: {script}")
    print(f"{'='*60}")

    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
