#!/usr/bin/env python3
"""
run_parallel_experiments.py
===========================
GCP Multi-core parallel execution script for Courthoud freeze/unfreeze experiments.

Since the four competing platforms (agents) are tightly coupled through the 
shared taxi marketplace (competitive customer choice model, joint taxi transitions, 
and step-by-step sequential learning updates), they cannot be split across cores.

However, independent simulation runs (different deviating platforms and different 
random seeds) can be run in parallel. This script leverages Python's 
ProcessPoolExecutor to run multiple independent experiments concurrently, 
fully utilizing multi-core GCP virtual machines.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def run_single_experiment(script_name: str, deviator: int, seed: int, out_dir: str, extra_args: list[str]) -> tuple[int, int, float, bool]:
    """Runs a single simulation run as a subprocess to avoid state corruption/locks."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / script_name),
        "--deviator", str(deviator),
        "--seed", str(seed),
        "--out-dir", str(Path(out_dir) / f"deviator_{deviator}_seed_{seed}"),
    ] + extra_args

    print(f"[START] Running: deviator={deviator}, seed={seed} (Command: {' '.join(cmd)})")
    start_time = time.time()
    try:
        # Run subprocess and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = time.time() - start_time
        print(f"[SUCCESS] Finished: deviator={deviator}, seed={seed} in {duration:.1f}s")
        return deviator, seed, duration, True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"[FAILURE] Failed: deviator={deviator}, seed={seed} in {duration:.1f}s\nError:\n{e.stderr}")
        return deviator, seed, duration, False


def main():
    parser = argparse.ArgumentParser(description="Run Courthoud Freeze/Unfreeze Defection experiments in parallel across CPU cores")
    parser.add_argument("--script", type=str, default="courthoud_freeze_unfreeze_monopoly_reset.py", 
                        choices=["courthoud_freeze_unfreeze_monopoly.py", "courthoud_freeze_unfreeze_monopoly_reset.py"],
                        help="Which script to run")
    parser.add_argument("--deviators", type=int, nargs="+", default=[0, 1, 2, 3], help="List of deviators to simulate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="List of random seeds to simulate")
    parser.add_argument("--cores", type=int, default=None, help="Number of parallel CPU cores to use (default: all available)")
    parser.add_argument("--out-dir", type=str, default="experiments/parallel_runs", help="Output directory")
    
    # Forward common args to the simulation scripts
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--baseline-eval-episodes", type=int, default=70)
    parser.add_argument("--freeze-episodes", type=int, default=70)
    parser.add_argument("--post-unfreeze-episodes", type=int, default=70)
    parser.add_argument("--steps", type=int, default=120)
    
    args, unknown = parser.parse_known_args()

    # Build extra arguments to forward
    extra_args = [
        "--episodes", str(args.episodes),
        "--baseline-eval-episodes", str(args.baseline_eval_episodes),
        "--freeze-episodes", str(args.freeze_episodes),
        "--post-unfreeze-episodes", str(args.post_unfreeze_episodes),
        "--steps", str(args.steps),
    ] + unknown

    # Generate task list (cartesian product of deviators and seeds)
    tasks = []
    for seed in args.seeds:
        for deviator in args.deviators:
            tasks.append((args.script, deviator, seed, args.out_dir, extra_args))

    print(f"=== Starting {len(tasks)} parallel runs on {args.cores or 'all'} cores ===")
    total_start = time.time()
    
    success_count = 0
    failures = []

    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        futures = {executor.submit(run_single_experiment, *task): task for task in tasks}
        
        for future in as_completed(futures):
            deviator, seed, duration, success = future.result()
            if success:
                success_count += 1
            else:
                failures.append((deviator, seed))

    total_duration = time.time() - total_start
    print(f"\n=== Execution Summary ===")
    print(f"Total time elapsed: {total_duration:.1f} seconds")
    print(f"Successful runs: {success_count}/{len(tasks)}")
    if failures:
        print(f"Failed runs: {failures}")
    else:
        print("All parallel runs completed successfully!")


if __name__ == "__main__":
    main()
