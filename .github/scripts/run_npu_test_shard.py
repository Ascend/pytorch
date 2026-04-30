#!/usr/bin/env python3
"""
Run NPU test cases from a shard JSON file.

Each test case runs in an independent subprocess for crash isolation.
Results are collected and saved to JSON files.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def run_single_case(
    case_id: str,
    test_dir: Path,
    timeout: int,
    verbose: bool = False
) -> Dict:
    """Run a single test case in a subprocess."""
    result = {
        "case_id": case_id,
        "status": "unknown",
        "duration": 0,
        "output": "",
        "error": "",
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Parse case_id to get test file path
    parts = case_id.split("::")
    test_file = parts[0]

    # Add test file parent directory to PYTHONPATH for sibling imports
    test_file_path = test_dir / test_file
    parent_dir = str(test_file_path.parent)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{parent_dir}:{existing_pythonpath}" if existing_pythonpath else parent_dir

    cmd = ["pytest", "-v", "--timeout=300", "-x", case_id]

    if verbose:
        print(f"Running: {case_id}")

    start_time = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(test_dir),
            env=env
        )
        duration = time.time() - start_time
        result["duration"] = round(duration, 2)
        result["output"] = proc.stdout
        result["error"] = proc.stderr

        if proc.returncode == 0:
            result["status"] = "passed"
        elif proc.returncode == 1:
            result["status"] = "failed"
        elif proc.returncode == 2:
            result["status"] = "error"
        elif proc.returncode == 3:
            result["status"] = "skipped"
        elif proc.returncode == 4:
            result["status"] = "xfail"
        elif proc.returncode == 5:
            result["status"] = "xpass"
        else:
            result["status"] = f"unknown_exit_{proc.returncode}"

        if verbose:
            print(f"  [{result['status']}] {duration:.2f}s")

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["duration"] = timeout
        result["error"] = f"Test timed out after {timeout} seconds"
        if verbose:
            print(f"  [timeout] {timeout}s")

    except Exception as e:
        result["status"] = "crashed"
        result["error"] = str(e)
        if verbose:
            print(f"  [crashed] {e}")

    return result


def run_shard(
    cases_json: str,
    test_dir: str,
    report_dir: str,
    timeout: int,
    max_workers: int,
    verbose: bool = False
) -> Dict:
    """Run all cases from a shard JSON file."""
    cases_file = Path(cases_json)
    if not cases_file.exists():
        raise FileNotFoundError(f"Cases JSON file not found: {cases_json}")

    with open(cases_file) as f:
        shard_data = json.load(f)

    cases = shard_data.get("cases", [])
    shard_index = shard_data.get("shard_index", 1)
    total_shards = shard_data.get("total_shards", 1)

    print(f"Loaded shard {shard_index}/{total_shards} with {len(cases)} cases")

    test_dir_path = Path(test_dir)
    if not test_dir_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    report_dir_path = Path(report_dir)
    report_dir_path.mkdir(parents=True, exist_ok=True)

    results = []
    stats = {
        "passed": 0,
        "failed": 0,
        "error": 0,
        "skipped": 0,
        "timeout": 0,
        "crashed": 0,
        "xfail": 0,
        "xpass": 0,
        "unknown": 0,
    }

    print(f"Running tests with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_case, case, test_dir_path, timeout, verbose): case
            for case in cases
        }

        for future in as_completed(futures):
            case = futures[future]
            result = future.result()
            results.append(result)

            status = result["status"]
            if status in stats:
                stats[status] += 1
            else:
                stats["unknown"] += 1

    # Save results
    results_file = report_dir_path / f"shard_{shard_index}_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "shard_index": shard_index,
            "total_shards": total_shards,
            "total_cases": len(cases),
            "stats": stats,
            "results": results,
        }, f, indent=2)

    print(f"\nShard {shard_index} Results saved to {results_file}")
    print(f"Statistics: {stats}")

    return {"stats": stats, "results_file": str(results_file)}


def main():
    parser = argparse.ArgumentParser(description="Run NPU test shard")
    parser.add_argument("--cases-json", required=True, help="JSON file with test cases")
    parser.add_argument("--test-dir", required=True, help="PyTorch test directory")
    parser.add_argument("--report-dir", required=True, help="Directory to save results")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per test case (seconds, default 300)")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    run_shard(
        cases_json=args.cases_json,
        test_dir=args.test_dir,
        report_dir=args.report_dir,
        timeout=args.timeout,
        max_workers=args.max_workers,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()