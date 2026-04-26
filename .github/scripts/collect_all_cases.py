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
        --distributed-shards 4 \
        --regular-shards 6 \
        --output-dir /path/to/output \
        --parallel 16
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

# Import discover_test_files module
import discover_test_files


def collect_cases_for_file(test_file: str, test_dir: Path) -> Tuple[str, List[str]]:
    """Collect test cases from a single file."""
    if test_file.startswith("test/"):
        test_file_rel = test_file[5:]
    else:
        test_file_rel = test_file

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

        return (test_file, nodeids)
    except Exception as e:
        print(f"WARNING: Failed to collect {test_file}: {e}")
        return (test_file, [])


def collect_all_cases(
    test_files: List[str],
    test_dir: Path,
    parallel: int = 16,
) -> List[Dict]:
    """Collect all cases from all files."""
    all_cases = []

    print(f"Collecting cases from {len(test_files)} files with {parallel} workers...")

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(collect_cases_for_file, f, test_dir): f
            for f in test_files
        }

        completed = 0
        for future in as_completed(futures):
            test_file, nodeids = future.result()
            completed += 1

            for nodeid in nodeids:
                all_cases.append({
                    "nodeid": nodeid,
                    "file": test_file,
                })

            if completed % 50 == 0 or completed == len(test_files):
                print(f"  Progress: {completed}/{len(test_files)} files, {len(all_cases)} cases")

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
        "distributed": dist_summary,
        "regular": reg_summary,
        "total_cases": len(dist_cases) + len(reg_cases),
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
    parser.add_argument("--distributed-shards", type=int, default=4, help="Distributed test shards")
    parser.add_argument("--regular-shards", type=int, default=6, help="Regular test shards")
    parser.add_argument("--output-dir", required=True, help="Output directory for shard JSONs")
    parser.add_argument("--parallel", type=int, default=16, help="Parallel collection workers")
    return parser.parse_args()


if __name__ == "__main__":
    main()