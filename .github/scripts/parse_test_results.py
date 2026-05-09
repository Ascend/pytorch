#!/usr/bin/env python3
"""
Parse test results from JUnit XML files and pytest logs.

This script provides utilities for:
    - Parsing JUnit XML reports
    - Aggregating test statistics
    - Analyzing pytest log files
    - Generating result reports (JSON, text)

Usage as module:
    from parse_test_results import (
        parse_junit_xml,
        aggregate_junit_stats,
        analyze_pytest_log,
        finalize_stats,
        save_stats_file,
        save_info_file,
        print_stats_summary,
    )

Usage as CLI:
    python parse_test_results.py \
        --report-dir test-reports \
        --shard 1 \
        --shard-type distributed \
        --output-dir parsed-results
"""

import argparse
import json
import os
import re
import signal
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ==============================================================================
# JUnit XML Parsing
# ==============================================================================


def parse_junit_xml(xml_file: str) -> Dict:
    """
    Parse a single JUnit XML file and extract test statistics.

    Args:
        xml_file: Path to JUnit XML file

    Returns:
        Dict with keys: total, passed, failed, skipped, errors, duration
    """
    stats = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "duration": 0.0,
    }

    if not os.path.exists(xml_file):
        return stats

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for testsuite in root.iter("testsuite"):
            stats["total"] += int(testsuite.get("tests", 0))
            stats["failed"] += int(testsuite.get("failures", 0))
            stats["skipped"] += int(testsuite.get("skipped", 0))
            stats["errors"] += int(testsuite.get("errors", 0))
            stats["duration"] += float(testsuite.get("time", 0))
        stats["passed"] = stats["total"] - stats["failed"] - stats["skipped"] - stats["errors"]
    except Exception as exc:
        print(f"Warning: Failed to parse XML report {xml_file}: {exc}")

    return stats


def aggregate_junit_stats(report_roots: List[Path], pattern: str = "*.xml") -> Dict:
    """
    Aggregate statistics from multiple JUnit XML files.

    Args:
        report_roots: List of directories to search for XML files
        pattern: Glob pattern for XML files (default: "*.xml")

    Returns:
        Dict with aggregated stats: total, passed, failed, skipped, errors, duration
    """
    totals = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "duration": 0.0,
    }

    seen_files = set()
    for report_root in report_roots:
        if not report_root.exists():
            continue
        for xml_file in report_root.rglob(pattern):
            try:
                resolved = str(xml_file.resolve())
            except OSError:
                resolved = str(xml_file)
            if resolved in seen_files:
                continue
            seen_files.add(resolved)

            stats = parse_junit_xml(str(xml_file))
            for key in totals:
                totals[key] += stats[key]

    totals["xml_files_count"] = len(seen_files)
    return totals


def parse_shard_xml_files(report_dir: Path, shard: int, shard_type: str = "regular") -> Dict:
    """
    Parse all JUnit XML files for a specific shard.

    Args:
        report_dir: Directory containing test reports
        shard: Shard number
        shard_type: "distributed" or "regular"

    Returns:
        Dict with aggregated stats for the shard
    """
    prefix = get_shard_type_prefix(shard_type)
    xml_pattern = f"shard_{prefix}-{shard}_pytest*.xml"

    xml_files = sorted(report_dir.glob(xml_pattern))
    if not xml_files:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "duration": 0.0,
            "junit_generated": False,
            "junit_xml_files": 0,
        }

    stats = aggregate_junit_stats([report_dir], xml_pattern)
    stats["junit_generated"] = True
    stats["junit_xml_files"] = len(xml_files)
    return stats


# ==============================================================================
# Log Analysis
# ==============================================================================


def analyze_pytest_log(log_file: Path, returncode: int) -> Dict:
    """
    Analyze pytest log file for failure patterns.

    Args:
        log_file: Path to pytest log file
        returncode: pytest process return code

    Returns:
        Dict with: zero_item_test_files, startup_failures, import_failures, test_failures
    """
    metrics = {
        "zero_item_test_files": 0,
        "startup_failures": 0,
        "import_failures": 0,
        "test_failures": 0,
    }

    if not log_file.exists():
        return metrics

    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return metrics

    # Detect "no tests collected" scenarios
    if returncode == 5 or "collected 0 items" in content or "no tests ran" in content:
        metrics["zero_item_test_files"] = 1

    # Count import errors
    metrics["import_failures"] = len(
        re.findall(r"^ImportError while importing test module", content, flags=re.MULTILINE)
    )

    # Count collection errors (excluding import errors)
    collection_errors = len(re.findall(r"^ERROR collecting ", content, flags=re.MULTILINE))
    metrics["startup_failures"] = max(collection_errors - metrics["import_failures"], 0)

    return metrics


# ==============================================================================
# Stats Processing
# ==============================================================================


def create_empty_stats() -> Dict:
    """Create empty statistics dictionary."""
    return {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "duration": 0.0,
        "junit_generated": False,
        "junit_xml_files": 0,
        "zero_item_test_files": 0,
        "startup_failures": 0,
        "import_failures": 0,
        "test_failures": 0,
    }


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


def finalize_stats(base_stats: Dict, returncode: int, duration: float, error_message: str = "") -> Dict:
    """
    Finalize statistics with returncode and duration.

    Args:
        base_stats: Base statistics dict
        returncode: Process return code
        duration: Execution duration in seconds
        error_message: Optional error message

    Returns:
        Finalized stats dict
    """
    stats = dict(base_stats)
    stats["duration"] = max(float(stats.get("duration", 0.0)), duration)

    if returncode != 0:
        stats["returncode"] = returncode

        # Handle signal crashes (negative returncode)
        if returncode < 0:
            signal_num = abs(returncode)
            try:
                signal_name = signal.Signals(signal_num).name
            except ValueError:
                signal_name = f"SIG{signal_num}"
            stats["crashed"] = True
            stats["crash_signal"] = signal_name

        # Mark incomplete if no tests
        if stats.get("total", 0) == 0:
            stats["errors"] = max(stats.get("errors", 0), 1)
            stats["incomplete"] = True

        if error_message:
            stats["error_message"] = error_message
    else:
        stats["returncode"] = 0

    return stats


def get_shard_status(stats: Dict, has_xml: bool) -> str:
    """
    Determine shard status from stats.

    Args:
        stats: Statistics dict
        has_xml: Whether XML files were generated

    Returns:
        Status string: MISSING, CRASHED, TIMEOUT, ERROR, FAILED, NO_TESTS, PASSED
    """
    if not has_xml:
        return "MISSING"

    if stats.get("crashed"):
        return "CRASHED"

    if stats.get("timed_out"):
        return "TIMEOUT"

    if stats.get("incomplete"):
        return "INCOMPLETE"

    if stats.get("errors", 0) > 0:
        return "ERROR"

    if stats.get("failed", 0) > 0:
        return "FAILED"

    if stats.get("total", 0) == 0:
        return "NO_TESTS"

    return "PASSED"


# ==============================================================================
# Utility Functions
# ==============================================================================


def get_shard_type_prefix(shard_type: str) -> str:
    """Convert shard type to short prefix for file naming."""
    return "dist" if shard_type == "distributed" else "reg"


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


def save_excluded_test_files_file(report_dir: str, shard: int, excluded_files: List[str], shard_type: str = "regular") -> str:
    """Save excluded test files list."""
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    excluded_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_excluded_test_files.txt")
    with open(excluded_file, "w", encoding="utf-8") as f:
        for target in excluded_files:
            f.write(f"{target}\n")
    return excluded_file


def save_missing_files_file(report_dir: str, shard: int, missing_files: List[str], shard_type: str = "regular") -> str:
    """Save missing files list (crashed files without XML)."""
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    missing_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_missing_files.txt")
    with open(missing_file, "w", encoding="utf-8") as f:
        for file_path in missing_files:
            f.write(f"{file_path}\n")
    return missing_file


def save_cases_file(report_dir: str, shard: int, cases_data: Dict, shard_type: str = "regular") -> str:
    """Save case-level results to JSON file."""
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    cases_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_cases.json")
    with open(cases_file, "w", encoding="utf-8") as f:
        json.dump(cases_data, f, indent=2, ensure_ascii=False)
    return cases_file


def load_cases_file(report_dir: Path, shard: int, shard_type: str = "regular") -> Dict:
    """Load case-level results from JSON file."""
    prefix = get_shard_type_prefix(shard_type)
    cases_file = report_dir / f"shard_{prefix}-{shard}_cases.json"
    if not cases_file.exists():
        return {}
    try:
        with open(cases_file, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load cases file {cases_file}: {e}")
        return {}


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
                "crashed": 0,
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
        elif status == "crashed":
            stats["crashed"] += 1
            stats["failed_cases"].append({
                "nodeid": case.get("nodeid"),
                "status": "crashed",
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
                    "crashed": 0,
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
            existing["crashed"] += stats["crashed"]
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


def get_files_with_failures(file_stats: Dict[str, Dict]) -> List[Dict]:
    """
    Get list of test files that have failures/errors/crashes/timeout.

    Args:
        file_stats: Dict from aggregate_all_cases_by_file()

    Returns:
        List of file stats dicts sorted by file name, only including files with failures
    """
    failed_files = []
    for test_file, stats in file_stats.items():
        if stats["failed"] > 0 or stats["errors"] > 0 or stats["crashed"] > 0 or stats["timeout"] > 0:
            failed_files.append(stats)

    failed_files.sort(key=lambda x: x["file"])
    return failed_files


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


def create_result_summary(stats: Dict, shard: int, shard_type: str) -> str:
    """Create a formatted result summary string."""
    prefix = get_shard_type_prefix(shard_type)
    status = get_shard_status(stats, stats.get("junit_generated", False))

    lines = [
        f"Shard {prefix}-{shard} Results:",
        f"  Status: {status}",
        f"  Total: {stats.get('total', 0)}",
        f"  Passed: {stats.get('passed', 0)}",
        f"  Failed: {stats.get('failed', 0)}",
        f"  Errors: {stats.get('errors', 0)}",
        f"  Duration: {stats.get('duration', 0.0):.2f}s",
    ]

    if stats.get("missing_files_count"):
        lines.append(f"  Missing: {stats['missing_files_count']}")

    return "\n".join(lines)


# ==============================================================================
# High-Level Parsing Functions
# ==============================================================================


def parse_shard_results(
    report_dir: Path,
    shard: int,
    shard_type: str,
    returncode: int,
    duration: float,
    missing_files: List[str] = None,
) -> Tuple[Dict, Dict]:
    """
    Parse all results for a shard and return (stats, log_metrics).

    This is the main entry point for result parsing.

    Args:
        report_dir: Directory containing test reports
        shard: Shard number
        shard_type: "distributed" or "regular"
        returncode: pytest process return code
        duration: Execution duration
        missing_files: List of files that crashed (no XML generated)

    Returns:
        Tuple of (stats_dict, log_metrics_dict)
    """
    missing_files = missing_files or []

    # Parse JUnit XML files
    stats = parse_shard_xml_files(report_dir, shard, shard_type)

    # Add per-file isolation metadata
    stats["per_file_isolation"] = True
    stats["missing_files_count"] = len(missing_files)

    # Analyze log file
    log_file = get_shard_log_file(report_dir, shard, shard_type)
    log_metrics = analyze_pytest_log(log_file, returncode)

    # Finalize stats
    stats = finalize_stats(stats, returncode, duration)

    # Merge log metrics
    log_metrics["test_failures"] = stats.get("failed", 0) + stats.get("errors", 0)
    log_metrics["missing_files_count"] = len(missing_files)
    stats.update(log_metrics)

    # Handle returncode=5 (no tests collected) as success
    if returncode == 5 and stats.get("total", 0) == 0:
        stats["returncode"] = 0

    return stats, log_metrics


def generate_shard_reports(
    report_dir: str,
    shard: int,
    shard_type: str,
    stats: Dict,
    info: Dict,
    missing_files: List[str] = None,
) -> Dict[str, str]:
    """
    Generate all report files for a shard.

    Args:
        report_dir: Output directory
        shard: Shard number
        shard_type: "distributed" or "regular"
        stats: Statistics dict
        info: Info dict
        missing_files: List of missing/crashed files

    Returns:
        Dict mapping report type to file path
    """
    report_files = {}

    # Save stats
    report_files["stats"] = save_stats_file(report_dir, shard, stats, shard_type)

    # Save info
    report_files["info"] = save_info_file(report_dir, shard, info, shard_type)

    # Save missing files if any
    if missing_files:
        report_files["missing"] = save_missing_files_file(report_dir, shard, missing_files, shard_type)

    return report_files


# ==============================================================================
# CLI Interface
# ==============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Parse test results from JUnit XML files")
    parser.add_argument("--report-dir", type=str, required=True, help="Directory containing test reports")
    parser.add_argument("--shard", type=int, required=True, help="Shard number")
    parser.add_argument(
        "--shard-type",
        type=str,
        choices=["distributed", "regular"],
        default="regular",
        help="Shard type",
    )
    parser.add_argument("--returncode", type=int, default=0, help="pytest return code")
    parser.add_argument("--duration", type=float, default=0.0, help="Execution duration in seconds")
    parser.add_argument("--output-stats", type=str, help="Output file for stats JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser.parse_args()


def main():
    """CLI entry point."""
    args = parse_args()

    report_dir = Path(args.report_dir).resolve()
    if not report_dir.exists():
        print(f"Error: Report directory not found: {report_dir}")
        sys.exit(1)

    # Parse results
    stats, log_metrics = parse_shard_results(
        report_dir=report_dir,
        shard=args.shard,
        shard_type=args.shard_type,
        returncode=args.returncode,
        duration=args.duration,
    )

    # Output
    if args.output_stats:
        output_path = Path(args.output_stats)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Stats saved to: {output_path}")

    if args.verbose:
        print(json.dumps(stats, indent=2))

    print_stats_summary(args.shard, stats, args.shard_type)

    # Exit with appropriate code
    sys.exit(stats.get("returncode", 0))


if __name__ == "__main__":
    main()