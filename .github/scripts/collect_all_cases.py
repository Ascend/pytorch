#!/usr/bin/env python3
"""
Collect all test cases and split into shards.

This script runs in prepare job (once) to:
1. Discover test files by type (distributed/regular)
2. Collect all test cases via pytest --collect-only
3. Split cases evenly into N shards
4. Output shard JSON files for each type
5. Save collection error logs for failed files

Usage:
    python collect_all_cases.py \
        --test-dir /path/to/pytorch/test \
        --case-paths-config /path/to/case_paths_ci.yml \
        --distributed-shards 2 \
        --regular-shards 5 \
        --output-dir /path/to/output \
        --error-log-dir /path/to/error_logs \
        --parallel 16
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

# Import discover_test_files module
import discover_test_files


def get_test_file_parent_dir(test_file: str, test_dir: Path) -> Path:
    """
    Get the parent directory of a test file.

    This directory should be added to PYTHONPATH to enable
    imports of sibling modules (e.g., model_registry.py).

    Args:
        test_file: Test file path (e.g., "test/distributed/pipelining/test_backward.py")
        test_dir: Path to PyTorch test directory

    Returns:
        Path to the test file's parent directory
    """
    if test_file.startswith("test/"):
        test_file_rel = test_file[5:]
    else:
        test_file_rel = test_file

    test_file_path = Path(test_file_rel)
    return test_dir / test_file_path.parent


def collect_cases_for_file(test_file: str, test_dir: Path) -> Tuple[str, List[str], bool, str]:
    """
    Collect test cases from a single file.

    Adds test file's parent directory to PYTHONPATH to enable
    imports of sibling modules (e.g., 'from model_registry import MLPModule').

    Returns:
        Tuple of (test_file, nodeids, success, error_message)
        - test_file: Original test file path
        - nodeids: List of collected test case nodeids
        - success: True if collection succeeded without errors
        - error_message: Error details if collection failed, empty string otherwise
    """
    if test_file.startswith("test/"):
        test_file_rel = test_file[5:]
    else:
        test_file_rel = test_file

    # Extract display name (remove test/ prefix and .py suffix)
    display_name = test_file_rel
    if display_name.endswith(".py"):
        display_name = display_name[:-3]

    # Get test file's parent directory for PYTHONPATH
    test_file_dir = get_test_file_parent_dir(test_file, test_dir)

    # Build environment with test file directory in PYTHONPATH
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(test_file_dir) + (":" + existing_pythonpath if existing_pythonpath else "")

    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "--quiet",
        test_file_rel,
    ]

    try:
        result = subprocess.run(
            command,
            cwd=str(test_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )

        nodeids = []
        for line in result.stdout.splitlines():
            if "::" in line and not line.strip().startswith("<"):
                nodeids.append(line.strip())

        # Check for collection errors based on pytest exit codes:
        #   0: all passed (success)
        #   2: pytest error (includes collection errors like ImportError)
        #   3: all skipped (success)
        #   4: command line error (error)
        #   5: no tests collected (NOT an error - file has no test cases)
        # Key insight: returncode 5 is normal, not an error
        if result.returncode in (0, 3, 5):
            # Normal: passed, skipped, or no tests in file
            return (test_file, nodeids, True, "")
        else:
            # returncode 2, 4: real collection error
            # Combine stdout (has ERRORS section) and stderr for error message
            error_msg = result.stdout.strip()
            if result.stderr.strip():
                error_msg += "\n--- stderr ---\n" + result.stderr.strip()
            return (test_file, nodeids, False, error_msg)

    except subprocess.TimeoutExpired:
        error_msg = f"TIMEOUT: Collection took >120s for {display_name}"
        return (test_file, [], False, error_msg)
    except Exception as e:
        error_msg = f"ERROR: {e}"
        return (test_file, [], False, error_msg)


def collect_all_cases(
    test_files: List[str],
    test_dir: Path,
    error_log_dir: Path,
    parallel: int = 16,
) -> List[Dict]:
    """
    Collect all cases from all files.

    Args:
        test_files: List of test file paths
        test_dir: Path to PyTorch test directory
        error_log_dir: Directory to save error logs for failed collections
        parallel: Number of parallel workers

    Returns:
        List of dicts with nodeid and file for each collected case
    """
    all_cases = []
    failed_files = []  # Track files with collection errors for logging

    print(f"Collecting cases from {len(test_files)} files with {parallel} workers...")
    print("=" * 60)

    # Create error log directory
    error_log_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(collect_cases_for_file, f, test_dir): f
            for f in test_files
        }

        completed = 0
        successful_count = 0
        failed_count = 0
        total_cases = 0

        for future in as_completed(futures):
            test_file, nodeids, success, error_msg = future.result()
            completed += 1

            # Extract display name for logging
            if test_file.startswith("test/"):
                display_name = test_file[5:]
            else:
                display_name = test_file
            if display_name.endswith(".py"):
                display_name = display_name[:-3]

            if success:
                successful_count += 1
                # Print concise log for successful files
                print(f"  {display_name}: {len(nodeids)} cases")
                for nodeid in nodeids:
                    all_cases.append({
                        "nodeid": nodeid,
                        "file": test_file,
                    })
            else:
                failed_count += 1
                # Print concise log for failed files
                print(f"  [FAILED] {display_name}: {len(nodeids)} cases")
                # Save error details to log file
                failed_files.append({
                    "file": display_name,
                    "error": error_msg,
                    "cases": len(nodeids),
                    "test_file": test_file,
                })
                # Still add any cases that were collected despite errors
                for nodeid in nodeids:
                    all_cases.append({
                        "nodeid": nodeid,
                        "file": test_file,
                    })

            # Update total cases count for progress display
            total_cases += len(nodeids)

            # Print progress summary every 100 files
            if completed % 100 == 0:
                print(f"  [Progress: {completed}/{len(test_files)} files, {successful_count} ok, {failed_count} failed, {total_cases} cases]")

    print("=" * 60)

    # Save error logs to files
    if failed_files:
        save_error_logs(failed_files, error_log_dir)

    # Final summary
    print(f"Collection complete: {len(all_cases)} cases from {successful_count}/{len(test_files)} files")
    if failed_count > 0:
        print(f"  WARNING: {failed_count} files had collection errors (logs saved to {error_log_dir})")

    return all_cases


def save_error_logs(failed_files: List[Dict], error_log_dir: Path) -> None:
    """
    Save collection error logs to individual files and create a summary.

    Args:
        failed_files: List of dicts with file, error, cases info
        error_log_dir: Directory to save error logs
    """
    print(f"Saving error logs for {len(failed_files)} failed files...")

    # Save individual error log files
    for failed in failed_files:
        # Create safe filename from display name (replace / with _)
        safe_name = failed['file'].replace('/', '_')
        log_file = error_log_dir / f"{safe_name}.log"

        # Write error log
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"File: {failed['file']}\n")
            f.write(f"Cases collected: {failed['cases']}\n")
            f.write(f"Test file path: {failed['test_file']}\n")
            f.write("=" * 80 + "\n")
            f.write("Collection Error:\n")
            f.write("=" * 80 + "\n")
            f.write(failed['error'])
            f.write("\n")

    # Save summary JSON
    summary_file = error_log_dir / "collection_errors_summary.json"
    summary_data = {
        "total_failed": len(failed_files),
        "failed_files": [
            {
                "file": f['file'],
                "cases": f['cases'],
                "test_file": f['test_file'],
                "log_file": f"{f['file'].replace('/', '_')}.log",
            }
            for f in failed_files
        ],
    }
    summary_file.write_text(json.dumps(summary_data, indent=2), encoding='utf-8')

    print(f"  Error logs saved to {error_log_dir}")
    print(f"  Summary: {summary_file}")


def split_cases_into_shards(cases: List[Dict], num_shards: int) -> List[List[Dict]]:
    """Split cases evenly into shards."""
    total = len(cases)
    base_size = total // num_shards
    remainder = total % num_shards

    shards = []
    start = 0
    for i in range(num_shards):
        size = base_size + (1 if i < remainder else 0)
        shards.append(cases[start:start + size])
        start += size

    return shards


def save_shards(
    cases: List[Dict],
    num_shards: int,
    test_type: str,
    output_dir: Path,
) -> Dict:
    """Save shard JSONs and return summary."""
    shards = split_cases_into_shards(cases, num_shards)

    print(f"\nSaving {test_type} shards...")
    for i, shard_cases in enumerate(shards, 1):
        shard_file = output_dir / f"{test_type}_cases_shard_{i}.json"
        shard_data = {
            "shard": i,
            "num_shards": num_shards,
            "test_type": test_type,
            "total_cases": len(shard_cases),
            "cases": shard_cases,
        }
        shard_file.write_text(json.dumps(shard_data, indent=2), encoding="utf-8")
        print(f"  Shard {i}: {len(shard_cases)} cases -> {shard_file}")

    return {
        "test_type": test_type,
        "num_shards": num_shards,
        "total_cases": len(cases),
        "shard_sizes": [len(s) for s in shards],
    }


def main():
    args = parse_args()

    test_dir = Path(args.test_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Error log directory for failed collections
    error_log_dir = Path(args.error_log_dir).resolve() if args.error_log_dir else output_dir / "collection_errors"
    error_log_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    # ========================================
    # Step 1: Collect distributed test cases
    # ========================================
    print("=" * 80)
    print("Collecting distributed test cases")
    print("=" * 80)

    dist_files, dist_meta = discover_test_files.discover_test_files(
        test_dir=test_dir,
        test_type="distributed",
        case_paths_config=args.case_paths_config,
    )
    print(f"Found {len(dist_files)} distributed test files")

    dist_cases = collect_all_cases(dist_files, test_dir, error_log_dir / "distributed", args.parallel)
    print(f"Total distributed cases: {len(dist_cases)}")

    dist_summary = save_shards(dist_cases, args.distributed_shards, "distributed", output_dir)
    summaries.append(dist_summary)

    # ========================================
    # Step 2: Collect regular test cases
    # ========================================
    print("\n" + "=" * 80)
    print("Collecting regular test cases")
    print("=" * 80)

    reg_files, reg_meta = discover_test_files.discover_test_files(
        test_dir=test_dir,
        test_type="regular",
        case_paths_config=args.case_paths_config,
    )
    print(f"Found {len(reg_files)} regular test files")

    reg_cases = collect_all_cases(reg_files, test_dir, error_log_dir / "regular", args.parallel)
    print(f"Total regular cases: {len(reg_cases)}")

    reg_summary = save_shards(reg_cases, args.regular_shards, "regular", output_dir)
    summaries.append(reg_summary)

    # ========================================
    # Step 3: Save overall summary
    # ========================================
    overall_summary = {
        "distributed": {
            "cases_summary": dist_summary,
            "discovery_metadata": dist_meta,
        },
        "regular": {
            "cases_summary": reg_summary,
            "discovery_metadata": reg_meta,
        },
        "total_cases": len(dist_cases) + len(reg_cases),
        "total_files_scanned": dist_meta.get("total_files", 0) + reg_meta.get("total_files", 0),
    }
    summary_file = output_dir / "cases_collection_summary.json"
    summary_file.write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")
    print(f"\nOverall summary saved to {summary_file}")

    print("\n" + "=" * 80)
    print("Collection Complete")
    print("=" * 80)
    print(f"Distributed: {len(dist_cases)} cases -> {args.distributed_shards} shards (serial execution)")
    print(f"Regular: {len(reg_cases)} cases -> {args.regular_shards} shards (64 parallel workers)")
    print(f"Total: {len(dist_cases) + len(reg_cases)} cases")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect and shard test cases")
    parser.add_argument("--test-dir", required=True, help="PyTorch test directory")
    parser.add_argument("--case-paths-config", help="case_paths_ci.yml path")
    parser.add_argument("--distributed-shards", type=int, default=2, help="Distributed test shards")
    parser.add_argument("--regular-shards", type=int, default=5, help="Regular test shards")
    parser.add_argument("--output-dir", required=True, help="Output directory for shard JSONs")
    parser.add_argument("--error-log-dir", help="Output directory for collection error logs (default: output-dir/collection_errors)")
    parser.add_argument("--parallel", type=int, default=16, help="Parallel collection workers")
    return parser.parse_args()


if __name__ == "__main__":
    main()