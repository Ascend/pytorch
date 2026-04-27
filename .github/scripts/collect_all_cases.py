#!/usr/bin/env python3
"""
Collect all test cases and split into shards.

This script runs in prepare job (once) to:
1. Discover test files by type (distributed/regular)
2. Collect all test cases via pytest --collect-only
3. Split cases evenly into N shards
4. Output shard JSON files for each type

Usage:
    python collect_all_cases.py \
        --test-dir /path/to/pytorch/test \
        --case-paths-config /path/to/case_paths_ci.yml \
        --distributed-shards 2 \
        --regular-shards 5 \
        --output-dir /path/to/output \
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

        # Check for collection errors
        # pytest outputs errors to stdout (ERRORS section), warnings to stderr
        if result.returncode != 0:
            # Collection failed - combine stdout (has ERRORS section) and stderr
            error_msg = result.stdout.strip()
            if result.stderr.strip():
                error_msg += "\n--- stderr ---\n" + result.stderr.strip()
            return (test_file, nodeids, False, error_msg)
        elif result.stderr.strip():
            # Only warnings (returncode == 0) - success if cases were collected
            if len(nodeids) > 0:
                return (test_file, nodeids, True, "")
            # No cases collected but has stderr warnings - treat as failed
            return (test_file, nodeids, False, result.stderr.strip())

        # Success: returncode == 0, no stderr warnings
        return (test_file, nodeids, True, "")

    except subprocess.TimeoutExpired:
        error_msg = f"TIMEOUT: Collection took >120s for {display_name}"
        return (test_file, [], False, error_msg)
    except Exception as e:
        error_msg = f"ERROR: {e}"
        return (test_file, [], False, error_msg)


def collect_all_cases(
    test_files: List[str],
    test_dir: Path,
    parallel: int = 16,
) -> List[Dict]:
    """Collect all cases from all files."""
    all_cases = []
    failed_files = []  # Track files with collection errors

    print(f"Collecting cases from {len(test_files)} files with {parallel} workers...")
    print("=" * 60)

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
                failed_files.append({
                    "file": display_name,
                    "error": error_msg,
                    "cases": len(nodeids),
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

    # Output collection errors to stderr (visible in action logs)
    if failed_files:
        print(f"\n!!! Collection errors ({failed_count} files) !!!", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        for failed in failed_files:
            print(f"\n[FAILED] {failed['file']}: {failed['cases']} cases", file=sys.stderr)
            print("-" * 40, file=sys.stderr)
            # Print the full error message
            print(failed['error'], file=sys.stderr)
        print("=" * 80, file=sys.stderr)

    # Final summary
    print(f"Collection complete: {len(all_cases)} cases from {successful_count}/{len(test_files)} files")
    if failed_count > 0:
        print(f"  WARNING: {failed_count} files had collection errors (see stderr for details)")

    return all_cases


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

    dist_cases = collect_all_cases(dist_files, test_dir, args.parallel)
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

    reg_cases = collect_all_cases(reg_files, test_dir, args.parallel)
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
    parser.add_argument("--parallel", type=int, default=16, help="Parallel collection workers")
    return parser.parse_args()


if __name__ == "__main__":
    main()