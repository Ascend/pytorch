#!/usr/bin/env python3
"""
Collect all test cases from PyTorch test directory and shard them.

This script scans the test directory, collects all test cases using pytest --collect-only,
classifies them as distributed or regular tests, and shards them for parallel execution.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple


DISTRIBUTED_TEST_DIRS = [
    "distributed",
    "distributed/algorithms/nn",
]


def is_distributed_test(test_file: str) -> bool:
    """Check if a test file is in distributed test directories."""
    for dir_prefix in DISTRIBUTED_TEST_DIRS:
        if test_file.startswith(dir_prefix):
            return True
    return False


def collect_cases_from_file(test_dir: Path, test_file: str, parallel: int = 1) -> List[str]:
    """Collect test cases from a single test file using pytest --collect-only."""
    full_path = test_dir / test_file
    if not full_path.exists():
        return []

    try:
        result = subprocess.run(
            ["pytest", "--collect-only", "-q", str(full_path)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(test_dir),
            env={**os.environ, "PYTEST_ADDOPTS": ""}
        )

        cases = []
        for line in result.stdout.splitlines():
            # Parse pytest output format: "test_file.py::TestClass::test_method"
            if "::" in line and not line.startswith("="):
                case_id = line.strip()
                if case_id and not case_id.startswith("<"):
                    cases.append(case_id)

        return cases
    except subprocess.TimeoutExpired:
        print(f"Timeout collecting cases from {test_file}")
        return []
    except Exception as e:
        print(f"Error collecting cases from {test_file}: {e}")
        return []


def discover_test_files(test_dir: Path) -> List[str]:
    """Discover all test_*.py files in the test directory."""
    test_files = []
    for py_file in test_dir.rglob("test_*.py"):
        rel_path = str(py_file.relative_to(test_dir))
        test_files.append(rel_path)
    return sorted(test_files)


def shard_cases(cases: List[str], num_shards: int) -> List[List[str]]:
    """Shard cases evenly across shards."""
    shards = [[] for _ in range(num_shards)]
    for i, case in enumerate(cases):
        shard_idx = i % num_shards
        shards[shard_idx].append(case)
    return shards


def collect_all_cases(
    test_dir: str,
    distributed_shards: int,
    regular_shards: int,
    output_dir: str,
    parallel: int = 1
) -> Dict:
    """Collect all test cases and shard them."""
    test_dir_path = Path(test_dir)
    if not test_dir_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Discovering test files in {test_dir}...")
    test_files = discover_test_files(test_dir_path)
    print(f"Found {len(test_files)} test files")

    distributed_files = [f for f in test_files if is_distributed_test(f)]
    regular_files = [f for f in test_files if not is_distributed_test(f)]

    print(f"Distributed test files: {len(distributed_files)}")
    print(f"Regular test files: {len(regular_files)}")

    # Collect cases in parallel
    print("Collecting distributed cases...")
    distributed_cases = []
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(collect_cases_from_file, test_dir_path, f, parallel): f
            for f in distributed_files
        }
        for future in as_completed(futures):
            file = futures[future]
            cases = future.result()
            distributed_cases.extend(cases)

    print("Collecting regular cases...")
    regular_cases = []
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(collect_cases_from_file, test_dir_path, f, parallel): f
            for f in regular_files
        }
        for future in as_completed(futures):
            file = futures[future]
            cases = future.result()
            regular_cases.extend(cases)

    print(f"Total distributed cases: {len(distributed_cases)}")
    print(f"Total regular cases: {len(regular_cases)}")

    # Shard cases
    distributed_sharded = shard_cases(distributed_cases, distributed_shards)
    regular_sharded = shard_cases(regular_cases, regular_shards)

    # Save shards to JSON files
    for i, shard in enumerate(distributed_sharded, 1):
        shard_file = output_dir_path / f"distributed_cases_shard_{i}.json"
        with open(shard_file, "w") as f:
            json.dump({
                "shard_index": i,
                "total_shards": distributed_shards,
                "cases": shard,
                "count": len(shard)
            }, f, indent=2)
        print(f"Saved distributed shard {i} with {len(shard)} cases to {shard_file}")

    for i, shard in enumerate(regular_sharded, 1):
        shard_file = output_dir_path / f"regular_cases_shard_{i}.json"
        with open(shard_file, "w") as f:
            json.dump({
                "shard_index": i,
                "total_shards": regular_shards,
                "cases": shard,
                "count": len(shard)
            }, f, indent=2)
        print(f"Saved regular shard {i} with {len(shard)} cases to {shard_file}")

    # Save summary
    summary = {
        "total_cases": len(distributed_cases) + len(regular_cases),
        "distributed_cases": len(distributed_cases),
        "regular_cases": len(regular_cases),
        "distributed_shards": distributed_shards,
        "regular_shards": regular_shards,
        "distributed_files": len(distributed_files),
        "regular_files": len(regular_files),
    }

    summary_file = output_dir_path / "cases_collection_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Collect and shard PyTorch test cases")
    parser.add_argument("--test-dir", required=True, help="PyTorch test directory path")
    parser.add_argument("--distributed-shards", type=int, default=2, help="Number of distributed test shards")
    parser.add_argument("--regular-shards", type=int, default=5, help="Number of regular test shards")
    parser.add_argument("--output-dir", required=True, help="Output directory for shard JSON files")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel collectors")

    args = parser.parse_args()

    summary = collect_all_cases(
        test_dir=args.test_dir,
        distributed_shards=args.distributed_shards,
        regular_shards=args.regular_shards,
        output_dir=args.output_dir,
        parallel=args.parallel
    )

    print("\nCollection Summary:")
    print(f"  Total cases: {summary['total_cases']}")
    print(f"  Distributed cases: {summary['distributed_cases']} ({summary['distributed_shards']} shards)")
    print(f"  Regular cases: {summary['regular_cases']} ({summary['regular_shards']} shards)")


if __name__ == "__main__":
    main()