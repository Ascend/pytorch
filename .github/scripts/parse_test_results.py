#!/usr/bin/env python3
"""
Utility functions for test result processing.

This module provides file operations and summary printing for test execution:
    - Create shard info dictionaries
    - Save results to JSON files (stats, info, cases, test plan)
    - Print test summary to stdout
    - Aggregate case results by test file

Usage as module:
    from parse_test_results import (
        create_shard_info,
        get_shard_log_file,
        save_stats_file,
        save_info_file,
        save_cases_file,
        save_test_plan_file,
        print_stats_summary,
        aggregate_all_cases_by_file,
    )
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ==============================================================================
# Stats Processing
# ==============================================================================


def create_shard_info(shard: int, num_shards: int, timestamp: str) -> Dict:
    """Create shard info dictionary template."""
    return {
        "shard": shard,
        "num_shards": num_shards,
        "selection_mode": "pytest_direct",
        "total_files": 0,
        "selected_test_files": 0,
        "shard_files": 0,
        "path_filtered_out_files": 0,
        "excluded_test_files": 0,
        "disabled_count": 0,
        "whitelist_entries": 0,
        "blacklist_entries": 0,
        "junit_generated": False,
        "junit_xml_files": 0,
        "zero_item_test_files": 0,
        "startup_failures": 0,
        "import_failures": 0,
        "test_failures": 0,
        "timestamp": timestamp,
    }


# ==============================================================================
# Utility Functions
# ==============================================================================


def get_shard_type_prefix(shard_type: str) -> str:
    """Convert shard type to short prefix for file naming."""
    if shard_type == "distributed":
        return "dist"
    elif shard_type == "custom":
        return "custom"
    else:
        return "reg"


def get_shard_log_file(report_dir: Path, shard: int, shard_type: str = "regular") -> Path:
    """Get path for shard log file."""
    prefix = get_shard_type_prefix(shard_type)
    return report_dir / f"test_shard_{prefix}-{shard}.log"


def load_disabled_testcases_count(json_file: str) -> int:
    """Count entries in disabled_testcases.json."""
    if not json_file or not os.path.exists(json_file):
        return 0

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, (dict, list)):
        return len(data)
    return 0


# ==============================================================================
# File Save Functions
# ==============================================================================


def save_stats_file(report_dir: str, shard: int, stats: Dict, shard_type: str = "regular") -> str:
    """Save statistics to JSON file."""
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    stats_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats_file


def save_info_file(report_dir: str, shard: int, info: Dict, shard_type: str = "regular") -> str:
    """Save info to JSON file."""
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    info_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_info.json")
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info_file


def save_test_plan_file(report_dir: str, shard: int, planned_tests: List[str], shard_type: str = "regular") -> str:
    """Save planned test files list."""
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    plan_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_planned_test_files.txt")
    with open(plan_file, "w", encoding="utf-8") as f:
        for target in planned_tests:
            f.write(f"{target}\n")
    return plan_file


def save_cases_file(report_dir: str, shard: int, cases_data: Dict, shard_type: str = "regular") -> str:
    """Save case-level results to JSON file."""
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    cases_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_cases.json")
    with open(cases_file, "w", encoding="utf-8") as f:
        json.dump(cases_data, f, indent=2, ensure_ascii=False)
    return cases_file


# ==============================================================================
# Case Aggregation by File
# ==============================================================================


def aggregate_cases_by_file(cases_list: List[Dict]) -> Dict[str, Dict]:
    """
    Aggregate case results by test file.

    This function groups test cases by their source file and computes
    statistics (passed, failed, errors, etc.) per file. It also collects
    detailed failure information for reporting.

    Args:
        cases_list: List of case result dicts with "nodeid", "file", "status" keys

    Returns:
        Dict mapping test file path -> aggregated stats
        Each entry contains:
            - file: test file path
            - total: total cases in file
            - passed, failed, errors, crashed, timeout, skipped: counts
            - failed_cases: list of failed/error/crashed/timeout cases with details
            - duration: total execution time for file
    """
    file_stats = {}

    for case in cases_list:
        test_file = case.get("file", "unknown")
        if not test_file:
            # Try to extract file from nodeid
            nodeid = case.get("nodeid", "")
            if "::" in nodeid:
                test_file = nodeid.split("::")[0]
            else:
                test_file = "unknown"

        status = case.get("status", "error")
        duration = case.get("duration", 0.0)

        if test_file not in file_stats:
            file_stats[test_file] = {
                "file": test_file,
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "timeout": 0,
                "skipped": 0,
                "failed_cases": [],
                "duration": 0.0,
            }

        stats = file_stats[test_file]
        stats["total"] += 1
        stats["duration"] += duration

        if status == "passed":
            stats["passed"] += 1
        elif status == "failed":
            stats["failed"] += 1
            stats["failed_cases"].append({
                "nodeid": case.get("nodeid"),
                "status": "failed",
                "message": case.get("message", ""),
                "duration": duration,
            })
        elif status == "error":
            stats["errors"] += 1
            stats["failed_cases"].append({
                "nodeid": case.get("nodeid"),
                "status": "error",
                "message": case.get("message", ""),
                "duration": duration,
            })
        elif status == "timeout":
            stats["timeout"] += 1
            stats["failed_cases"].append({
                "nodeid": case.get("nodeid"),
                "status": "timeout",
                "message": f"Timeout after {duration}s",
                "duration": duration,
            })
        elif status == "skipped":
            stats["skipped"] += 1

    return file_stats


def aggregate_all_cases_by_file(cases_results: Dict) -> Dict[str, Dict]:
    """
    Aggregate all cases from multiple shards by test file.

    Args:
        cases_results: Dict mapping shard_key -> cases_data (from shard_*_cases.json)

    Returns:
        Dict mapping test file -> aggregated stats across all shards
    """
    all_file_stats = {}

    for shard_key, cases_data in cases_results.items():
        shard_cases = cases_data.get("cases", [])
        file_stats = aggregate_cases_by_file(shard_cases)

        for test_file, stats in file_stats.items():
            if test_file not in all_file_stats:
                all_file_stats[test_file] = {
                    "file": test_file,
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "errors": 0,
                    "timeout": 0,
                    "skipped": 0,
                    "failed_cases": [],
                    "duration": 0.0,
                }

            existing = all_file_stats[test_file]
            existing["total"] += stats["total"]
            existing["passed"] += stats["passed"]
            existing["failed"] += stats["failed"]
            existing["errors"] += stats["errors"]
            existing["timeout"] += stats["timeout"]
            existing["skipped"] += stats["skipped"]
            existing["duration"] += stats["duration"]
            existing["failed_cases"].extend(stats["failed_cases"])

    # Sort failed_cases within each file
    for test_file in all_file_stats:
        all_file_stats[test_file]["failed_cases"].sort(
            key=lambda x: x.get("nodeid", "")
        )

    return all_file_stats


# ==============================================================================
# Summary Printing
# ==============================================================================


def print_stats_summary(shard: int, stats: Dict, shard_type: str = "regular") -> None:
    """Print statistics summary to stdout."""
    prefix = get_shard_type_prefix(shard_type)
    print(f"\n{'=' * 60}")
    print(f"Test Results for Shard {prefix}-{shard}")
    print(f"{'=' * 60}")
    print(f"Total:   {stats['total']}")
    print(f"Passed:  {stats['passed']}")
    print(f"Failed:  {stats['failed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors:  {stats['errors']}")
    print(f"Duration: {stats['duration']:.2f}s")
    if stats.get("missing_files_count"):
        print(f"Missing files: {stats['missing_files_count']}")
    if stats.get("crashed"):
        print(f"Crash signal: {stats.get('crash_signal', 'unknown')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Module only, no CLI functionality
    pass