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


def collect_cases_from_file(test_dir: Path, test_file: str, parallel: int = 1, verbose: bool = False) -> Tuple[List[str], str]:
    """Collect test cases from a single test file using pytest --collect-only.

    Returns:
        Tuple of (cases list, error message or empty string)
    """
    full_path = test_dir / test_file
    if not full_path.exists():
        error = f"File not found: {full_path}"
        if verbose:
            print(f"[SKIP] {test_file}: {error}")
        return [], error

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

        # Check for errors
        error_msg = ""
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            if verbose:
                print(f"[ERROR] {test_file}: pytest returned {result.returncode}")
                if result.stderr:
                    # 打印完整 stderr，不截断
                    for line in result.stderr.splitlines():
                        print(f"  {line}")
        elif len(cases) == 0:
            # No cases collected, might be import error or empty file
            if result.stderr:
                error_msg = result.stderr.strip()
                if verbose:
                    print(f"[WARN] {test_file}: 0 cases collected")
                    # 打印完整 stderr
                    for line in result.stderr.splitlines():
                        print(f"  {line}")
            else:
                if verbose:
                    print(f"[WARN] {test_file}: 0 cases collected (possibly empty or all skipped)")
        else:
            if verbose:
                print(f"[OK] {test_file}: {len(cases)} cases collected")

        return cases, error_msg
    except subprocess.TimeoutExpired:
        error = "Timeout after 60s"
        if verbose:
            print(f"[TIMEOUT] {test_file}: {error}")
        return [], error
    except Exception as e:
        error = str(e)
        if verbose:
            print(f"[EXCEPTION] {test_file}: {error}")
        return [], error


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
    parallel: int = 1,
    verbose: bool = False
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

    if verbose:
        print("\n=== Collecting distributed cases ===")

    # Collect cases in parallel
    print("Collecting distributed cases...")
    distributed_cases = []
    distributed_errors = {}
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(collect_cases_from_file, test_dir_path, f, parallel, verbose): f
            for f in distributed_files
        }
        for future in as_completed(futures):
            file = futures[future]
            cases, error = future.result()
            distributed_cases.extend(cases)
            if error:
                distributed_errors[file] = error

    if verbose:
        print("\n=== Collecting regular cases ===")

    print("Collecting regular cases...")
    regular_cases = []
    regular_errors = {}
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(collect_cases_from_file, test_dir_path, f, parallel, verbose): f
            for f in regular_files
        }
        for future in as_completed(futures):
            file = futures[future]
            cases, error = future.result()
            regular_cases.extend(cases)
            if error:
                regular_errors[file] = error

    print(f"Total distributed cases: {len(distributed_cases)}")
    print(f"Total regular cases: {len(regular_cases)}")

    # Print summary of errors if any
    if distributed_errors or regular_errors:
        print("\n=== Collection Errors Summary ===")
        if distributed_errors:
            print(f"Distributed files with errors: {len(distributed_errors)}")
            for file, error in sorted(distributed_errors.items())[:10]:
                # 打印完整错误信息
                print(f"  {file}:")
                for line in error.splitlines()[:5]:  # 只打印前5行避免过长
                    print(f"    {line}")
            if len(distributed_errors) > 10:
                print(f"  ... and {len(distributed_errors) - 10} more files")
        if regular_errors:
            print(f"Regular files with errors: {len(regular_errors)}")
            for file, error in sorted(regular_errors.items())[:10]:
                print(f"  {file}:")
                for line in error.splitlines()[:5]:
                    print(f"    {line}")
            if len(regular_errors) > 10:
                print(f"  ... and {len(regular_errors) - 10} more files")

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
    parser.add_argument("--verbose", action="store_true", help="Print detailed collection progress")

    args = parser.parse_args()

    summary = collect_all_cases(
        test_dir=args.test_dir,
        distributed_shards=args.distributed_shards,
        regular_shards=args.regular_shards,
        output_dir=args.output_dir,
        parallel=args.parallel,
        verbose=args.verbose
    )

    print("\nCollection Summary:")
    print(f"  Total cases: {summary['total_cases']}")
    print(f"  Distributed cases: {summary['distributed_cases']} ({summary['distributed_shards']} shards)")
    print(f"  Regular cases: {summary['regular_cases']} ({summary['regular_shards']} shards)")


if __name__ == "__main__":
    main()