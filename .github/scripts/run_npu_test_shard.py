#!/usr/bin/env python3
"""
Run a shard of patched upstream PyTorch tests via per-file isolation pytest execution.

Test types:
- Distributed tests: NPU distributed tests
- Regular tests: All other tests

Each shard applies whitelist/blacklist filtering from case_paths_ci.yml
and item-level deselection from disabled_testcases.json.

All tests are executed using per-file isolation: each test file runs in its own
pytest subprocess, preventing NPU kernel crashes from affecting other test files.
Files are executed concurrently using ThreadPoolExecutor for parallel execution.

Execution Flow (4 Steps):
    Step 1: Test file discovery (scan all test_*.py)
    Step 2: Shard type filtering (distributed/regular)
    Step 3: Whitelist/blacklist filtering
    Step 4: Shard assignment

Each step is implemented as an independent, testable function.
"""

import argparse
import dataclasses
import fnmatch
import json
import os
import re
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Dict, List, Optional, Tuple
import threading


# ==============================================================================
# Data Classes for Test Planning Results
# ==============================================================================

@dataclasses.dataclass
class DiscoveryResult:
    """Result of Step 1: Test file discovery."""
    test_dir: Path
    all_test_files: List[str]  # All discovered test files with 'test/' prefix
    total_count: int


@dataclasses.dataclass
class TypeFilterResult:
    """Result of Step 2: Shard type filtering."""
    shard_type: str  # "distributed" or "regular"
    selected_files: List[str]  # Files matching shard type
    excluded_files: List[str]  # Files not matching shard type
    selected_count: int
    excluded_count: int


@dataclasses.dataclass
class PathRulesFilterResult:
    """Result of Step 3: Whitelist/blacklist filtering."""
    whitelist: List[str]
    blacklist: List[str]
    selected_files: List[str]  # Files after whitelist/blacklist
    excluded_files: List[str]  # Files filtered out by rules
    selected_count: int
    excluded_count: int


@dataclasses.dataclass
class ShardAssignmentResult:
    """Result of Step 4: Shard assignment."""
    shard: int  # Current shard number (1-indexed)
    num_shards: int  # Total shards for this type
    planned_tests: List[str]  # Files assigned to this shard
    planned_count: int


@dataclasses.dataclass
class ShardPlanResult:
    """Complete result of all 4 steps of test planning."""
    discovery: DiscoveryResult
    type_filter: TypeFilterResult
    path_rules_filter: PathRulesFilterResult
    shard_assignment: ShardAssignmentResult

    def get_planned_tests(self) -> List[str]:
        """Get the final list of planned tests for this shard."""
        return self.shard_assignment.planned_tests

    def to_info_dict(self) -> Dict:
        """Convert to info dict for JSON output."""
        return {
            "total_files": self.discovery.total_count,
            "test_discovery_mode": "raw_file_scan",
            "shard_type": self.type_filter.shard_type,
            "type_selected_files": self.type_filter.selected_count,
            "type_excluded_files": self.type_filter.excluded_count,
            "whitelist_entries": len(self.path_rules_filter.whitelist),
            "blacklist_entries": len(self.path_rules_filter.blacklist),
            "whitelist_blacklist_selected": self.path_rules_filter.selected_count,
            "whitelist_blacklist_excluded": self.path_rules_filter.excluded_count,
            "shard": self.shard_assignment.shard,
            "num_shards": self.shard_assignment.num_shards,
            "shard_files": self.shard_assignment.planned_count,
        }


# ==============================================================================
# Step 1: Test File Discovery
# ==============================================================================


def get_shard_type_prefix(shard_type: str) -> str:
    """
    Convert shard type to short prefix for file naming.

    Args:
        shard_type: "distributed" or "regular"

    Returns:
        "dist" for distributed, "reg" for regular
    """
    return "dist" if shard_type == "distributed" else "reg"


def parse_args():
    parser = argparse.ArgumentParser(description="Run PyTorch NPU tests for a shard via direct pytest")
    parser.add_argument("--shard", type=int, required=True, help="Shard number (1-indexed within test-type)")
    parser.add_argument("--num-shards", type=int, required=True, help="Total number of shards for this test-type")
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["distributed", "regular"],
        default="regular",
        help="Test type: 'distributed' for distributed tests, 'regular' for other tests",
    )
    parser.add_argument("--test-dir", type=str, required=True, help="Path to the PyTorch test directory")
    parser.add_argument("--disabled-testcases", type=str, help="Path to disabled_testcases.json")
    parser.add_argument(
        "--case-paths-config",
        type=str,
        help="Path to case_paths_ci.yml for file-level whitelist/blacklist control",
    )
    parser.add_argument("--report-dir", type=str, default="test-reports", help="Directory for test reports")
    parser.add_argument("--timeout", type=int, default=600, help="Per-test timeout passed to pytest")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", type=int, default=2, help="Number of parallel workers for concurrent file execution (default: 2)")
    return parser.parse_args()


def normalize_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def normalize_rule_path(rule: str) -> str:
    normalized = normalize_path(rule)
    if not normalized:
        return ""
    if normalized == "test" or normalized.startswith("test/"):
        return normalized.rstrip("/")
    return f"test/{normalized}".rstrip("/")


def load_disabled_testcases_count(json_file: str) -> int:
    if not json_file or not os.path.exists(json_file):
        return 0

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, (dict, list)):
        return len(data)
    return 0


def parse_simple_yaml_lists(raw_text: str) -> Dict[str, List[str]]:
    parsed = {"whitelist": [], "blacklist": []}
    current_key = None

    for raw_line in raw_text.splitlines():
        without_comment = raw_line.split("#", 1)[0].rstrip()
        if not without_comment.strip():
            continue

        stripped = without_comment.lstrip()
        if not raw_line.startswith((" ", "\t")) and stripped.endswith(":"):
            key = stripped[:-1].strip()
            current_key = key if key in parsed else None
            continue

        if current_key and stripped.startswith("- "):
            value = stripped[2:].strip().strip("\"'")
            if value:
                parsed[current_key].append(value)

    return parsed


def coerce_rule_list(value, key: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Expected '{key}' to be a list, got {type(value).__name__}")

    normalized_values = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Expected every '{key}' entry to be a string, got {type(item).__name__}")
        normalized = normalize_rule_path(item)
        if normalized:
            normalized_values.append(normalized)
    return normalized_values


def load_case_path_rules(config_file: str) -> Tuple[str, List[str], List[str]]:
    if not config_file:
        return "", [], []

    config_path = Path(config_file).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"case_paths_ci config not found: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        payload = parse_simple_yaml_lists(raw_text)
    else:
        payload = yaml.safe_load(raw_text) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML object in {config_path}, got {type(payload).__name__}")

    whitelist = coerce_rule_list(payload.get("whitelist"), "whitelist")
    blacklist = coerce_rule_list(payload.get("blacklist"), "blacklist")
    return str(config_path), whitelist, blacklist




def discover_raw_test_files(test_dir: Path) -> List[str]:
    """Scan all test_*.py files in test directory."""
    files = []
    for test_file in test_dir.rglob("test_*.py"):
        rel_path = test_file.relative_to(test_dir).as_posix()
        files.append(f"test/{rel_path}")
    return sorted(files)


def path_matches_rule(test_path: str, rule: str) -> bool:
    normalized_path = normalize_path(test_path)
    normalized_rule = normalize_rule_path(rule)
    if not normalized_rule:
        return False

    if any(char in normalized_rule for char in "*?[]"):
        return fnmatch.fnmatch(normalized_path, normalized_rule)

    return normalized_path == normalized_rule or normalized_path.startswith(f"{normalized_rule}/")


def apply_case_path_rules(
    test_files: List[str], whitelist: List[str], blacklist: List[str]
) -> Tuple[List[str], List[str]]:
    if whitelist:
        selected = [path for path in test_files if any(path_matches_rule(path, rule) for rule in whitelist)]
    else:
        selected = list(test_files)

    if blacklist:
        selected = [path for path in selected if not any(path_matches_rule(path, rule) for rule in blacklist)]

    selected_set = set(selected)
    excluded = [path for path in test_files if path not in selected_set]
    return selected, excluded


def select_shard_files(test_files: List[str], shard: int, num_shards: int) -> List[str]:
    """
    Select test files for a shard using contiguous range-based selection.

    This approach assigns consecutive files to each shard (sorted alphabetically),
    ensuring that files from the same directory end up in the same or adjacent shards.
    This is different from round-robin (modulo) distribution which spreads adjacent
    files across different shards.

    Args:
        test_files: List of test file paths, already sorted alphabetically
        shard: Shard number (1-indexed, 1 <= shard <= num_shards)
        num_shards: Total number of shards (max 100)

    Returns:
        List of test files assigned to this shard
    """
    if not test_files:
        return []

    shard_index = shard - 1  # Convert to 0-indexed
    total_files = len(test_files)

    # Calculate base size and remainder for even distribution
    base_size = total_files // num_shards
    remainder = total_files % num_shards

    # Shards with index < remainder get one extra file
    # This ensures files are distributed as evenly as possible
    if shard_index < remainder:
        start = shard_index * (base_size + 1)
        end = start + base_size + 1
    else:
        start = remainder * (base_size + 1) + (shard_index - remainder) * base_size
        end = start + base_size

    return test_files[start:end]


def filter_tests_by_type(test_files: List[str], shard_type: str) -> Tuple[List[str], List[str]]:
    """
    Filter test files by shard type.

    Args:
        test_files: List of test file paths (with test/ prefix)
        shard_type: "distributed" or "regular"

    Returns:
        Tuple of (selected_files, excluded_files)
    """
    if shard_type == "distributed":
        # Distributed tests: files starting with test/distributed/
        selected = [f for f in test_files if f.startswith("test/distributed/")]
        excluded = [f for f in test_files if not f.startswith("test/distributed/")]
    else:
        # Regular tests: all non-distributed tests
        # Whitelist/blacklist filtering happens in Step 3
        selected = [f for f in test_files if not f.startswith("test/distributed/")]
        excluded = [f for f in test_files if f.startswith("test/distributed/")]

    return selected, excluded


def strip_test_prefix_and_suffix(test_path: str) -> str:
    """
    Remove 'test/' prefix and '.py' suffix from path.

    Example: 'test/test_autograd.py' -> 'test_autograd'
             'test/distributed/test_c10d.py' -> 'distributed/test_c10d'
    """
    path = test_path
    if path.startswith("test/"):
        path = path[5:]  # Remove 'test/' prefix
    if path.endswith(".py"):
        path = path[:-3]  # Remove '.py' suffix
    return path


# ==============================================================================
# Step 1-4 Independent Testable Functions
# ==============================================================================

def step1_discover_test_files(test_dir: Path) -> DiscoveryResult:
    """
    Step 1: Test File Discovery

    Scan all test_*.py files in the test directory.

    Args:
        test_dir: Path to the PyTorch test directory

    Returns:
        DiscoveryResult containing all discovered test files

    Example:
        >>> result = step1_discover_test_files(Path("/path/to/pytorch/test"))
        >>> print(f"Found {result.total_count} test files")
    """
    all_test_files = discover_raw_test_files(test_dir)
    return DiscoveryResult(
        test_dir=test_dir,
        all_test_files=all_test_files,
        total_count=len(all_test_files),
    )


def step2_filter_by_type(discovery_result: DiscoveryResult, shard_type: str) -> TypeFilterResult:
    """
    Step 2: Shard Type Filtering

    Filter test files by shard type (distributed or regular).

    Args:
        discovery_result: Result from Step 1
        shard_type: "distributed" or "regular"

    Returns:
        TypeFilterResult containing selected and excluded files

    Example:
        >>> discovery = step1_discover_test_files(test_dir)
        >>> type_result = step2_filter_by_type(discovery, "distributed")
        >>> print(f"Distributed files: {type_result.selected_count}")
    """
    selected_files, excluded_files = filter_tests_by_type(
        discovery_result.all_test_files,
        shard_type
    )
    return TypeFilterResult(
        shard_type=shard_type,
        selected_files=selected_files,
        excluded_files=excluded_files,
        selected_count=len(selected_files),
        excluded_count=len(excluded_files),
    )


def step3_apply_path_rules(
    type_filter_result: TypeFilterResult,
    whitelist: List[str],
    blacklist: List[str]
) -> PathRulesFilterResult:
    """
    Step 3: Whitelist/Blacklist Filtering

    Apply whitelist and blacklist rules to the selected files.

    Args:
        type_filter_result: Result from Step 2
        whitelist: List of whitelist rules (paths with 'test/' prefix)
        blacklist: List of blacklist rules (paths with 'test/' prefix)

    Returns:
        PathRulesFilterResult containing filtered files

    Example:
        >>> whitelist = ["test/test_autograd.py", "test/nn"]
        >>> blacklist = ["test/nn/test_convolution.py"]
        >>> rules_result = step3_apply_path_rules(type_result, whitelist, blacklist)
    """
    selected_files, excluded_files = apply_case_path_rules(
        type_filter_result.selected_files,
        whitelist,
        blacklist
    )
    return PathRulesFilterResult(
        whitelist=whitelist,
        blacklist=blacklist,
        selected_files=selected_files,
        excluded_files=excluded_files,
        selected_count=len(selected_files),
        excluded_count=len(excluded_files),
    )


def step4_assign_shard(
    path_rules_result: PathRulesFilterResult,
    shard: int,
    num_shards: int
) -> ShardAssignmentResult:
    """
    Step 4: Shard Assignment

    Assign test files to a specific shard using contiguous range-based selection.

    Args:
        path_rules_result: Result from Step 3
        shard: Shard number (1-indexed, 1 <= shard <= num_shards)
        num_shards: Total number of shards

    Returns:
        ShardAssignmentResult containing files assigned to this shard

    Example:
        >>> shard_result = step4_assign_shard(rules_result, shard=1, num_shards=50)
        >>> print(f"This shard has {shard_result.planned_count} files")
    """
    planned_tests = select_shard_files(
        path_rules_result.selected_files,
        shard,
        num_shards
    )
    return ShardAssignmentResult(
        shard=shard,
        num_shards=num_shards,
        planned_tests=planned_tests,
        planned_count=len(planned_tests),
    )


def plan_shard_tests(
    test_dir: Path,
    shard: int,
    num_shards: int,
    shard_type: str,
    whitelist: List[str],
    blacklist: List[str],
) -> ShardPlanResult:
    """
    Complete Test Planning: Execute all 4 steps to get planned tests for a shard.

    This is the main coordination function that orchestrates all 4 steps:
        Step 1: Test file discovery
        Step 2: Shard type filtering
        Step 3: Whitelist/blacklist filtering
        Step 4: Shard assignment

    Args:
        test_dir: Path to the PyTorch test directory
        shard: Shard number (1-indexed)
        num_shards: Total number of shards
        shard_type: "distributed" or "regular"
        whitelist: List of whitelist rules
        blacklist: List of blacklist rules

    Returns:
        ShardPlanResult containing all step results and final planned tests

    Example:
        >>> result = plan_shard_tests(
        ...     test_dir=Path("/path/to/pytorch/test"),
        ...     shard=1,
        ...     num_shards=50,
        ...     shard_type="regular",
        ...     whitelist=["test/test_autograd.py"],
        ...     blacklist=["test/test_cuda.py"],
        ... )
        >>> planned_tests = result.get_planned_tests()
        >>> print(f"Shard 1 has {len(planned_tests)} tests")
    """
    # Step 1: Discover all test files
    discovery_result = step1_discover_test_files(test_dir)

    # Step 2: Filter by shard type
    type_filter_result = step2_filter_by_type(discovery_result, shard_type)

    # Step 3: Apply whitelist/blacklist rules
    path_rules_result = step3_apply_path_rules(
        type_filter_result,
        whitelist,
        blacklist
    )

    # Step 4: Assign files to shard
    shard_assignment_result = step4_assign_shard(
        path_rules_result,
        shard,
        num_shards
    )

    return ShardPlanResult(
        discovery=discovery_result,
        type_filter=type_filter_result,
        path_rules_filter=path_rules_result,
        shard_assignment=shard_assignment_result,
    )


# ==============================================================================
# Utility Functions for Testing
# ==============================================================================

def create_test_plan_summary(result: ShardPlanResult) -> str:
    """
    Create a human-readable summary of the test planning result.

    Args:
        result: ShardPlanResult from plan_shard_tests()

    Returns:
        Formatted string summary
    """
    lines = [
        "=" * 60,
        "Test Planning Summary",
        "=" * 60,
        f"Step 1 - Discovery: {result.discovery.total_count} files found",
        f"Step 2 - Type Filter: {result.type_filter.selected_count} selected ({result.type_filter.shard_type})",
        f"Step 3 - Path Rules: {result.path_rules_filter.selected_count} after whitelist/blacklist",
        f"         Whitelist: {len(result.path_rules_filter.whitelist)} entries",
        f"         Blacklist: {len(result.path_rules_filter.blacklist)} entries",
        f"         Filtered out: {result.path_rules_filter.excluded_count} files",
        f"Step 4 - Shard Assignment: {result.shard_assignment.planned_count} files for shard {result.shard_assignment.shard}/{result.shard_assignment.num_shards}",
        "=" * 60,
    ]
    return "\n".join(lines)


def parse_junit_xml(xml_file: str) -> Dict:
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


def aggregate_junit_stats(report_roots: List[Path]) -> Dict:
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
        for xml_file in report_root.rglob("*.xml"):
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
    return totals


def create_empty_stats() -> Dict:
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
    return {
        "shard": shard,
        "num_shards": num_shards,
        "selection_mode": "pytest_direct",
        "total_files": 0,
        "selected_test_entries": 0,
        "selected_test_files": 0,
        "shard_files": 0,
        "path_filtered_out_files": 0,
        "excluded_test_files": 0,
        "disabled_count": 0,
        "disabled_count_matched": 0,
        "disabled_count_deselected": 0,
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
    stats = dict(base_stats)
    stats["duration"] = max(float(stats.get("duration", 0.0)), duration)
    if returncode != 0:
        stats["returncode"] = returncode
        if returncode < 0:
            signal_num = abs(returncode)
            try:
                signal_name = signal.Signals(signal_num).name
            except ValueError:
                signal_name = f"SIG{signal_num}"
            stats["crashed"] = True
            stats["crash_signal"] = signal_name
        if stats.get("total", 0) == 0:
            stats["errors"] = max(stats.get("errors", 0), 1)
            stats["incomplete"] = True
        if error_message:
            stats["error_message"] = error_message
    else:
        stats["returncode"] = 0
    return stats


def save_stats_file(report_dir: str, shard: int, stats: Dict, shard_type: str = "regular") -> str:
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    stats_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats_file


def save_info_file(report_dir: str, shard: int, info: Dict, shard_type: str = "regular") -> str:
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    info_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_info.json")
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info_file


def save_test_plan_file(report_dir: str, shard: int, planned_tests: List[str], shard_type: str = "regular") -> str:
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    plan_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_planned_test_files.txt")
    with open(plan_file, "w", encoding="utf-8") as f:
        for target in planned_tests:
            f.write(f"{target}\n")
    return plan_file


def save_excluded_test_files_file(report_dir: str, shard: int, test_targets: List[str], shard_type: str = "regular") -> str:
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    excluded_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_excluded_test_files.txt")
    with open(excluded_file, "w", encoding="utf-8") as f:
        for target in test_targets:
            f.write(f"{target}\n")
    return excluded_file


def save_unhandled_upstream_tests_file(report_dir: str, shard: int, test_targets: List[str], shard_type: str = "regular") -> str:
    os.makedirs(report_dir, exist_ok=True)
    prefix = get_shard_type_prefix(shard_type)
    unhandled_file = os.path.join(report_dir, f"shard_{prefix}-{shard}_unhandled_upstream_tests.txt")
    with open(unhandled_file, "w", encoding="utf-8") as f:
        for target in test_targets:
            f.write(f"{target}\n")
    return unhandled_file


def get_disabled_testcases_report_file(report_dir: str, shard: int, shard_type: str = "regular") -> str:
    prefix = get_shard_type_prefix(shard_type)
    return os.path.join(report_dir, f"shard_{prefix}-{shard}_disabled_testcases.json")


def load_disabled_testcases_report(report_dir: str, shard: int, shard_type: str = "regular") -> Dict:
    report_file = get_disabled_testcases_report_file(report_dir, shard, shard_type)
    if not os.path.exists(report_file):
        return {
            "disabled_count_matched": 0,
            "disabled_count_deselected": 0,
        }

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "disabled_count_matched": data.get("disabled_count_matched", 0),
            "disabled_count_deselected": data.get("disabled_count_deselected", 0),
        }
    except Exception as exc:
        print(f"Warning: Failed to read disabled testcase report: {exc}")
        return {
            "disabled_count_matched": 0,
            "disabled_count_deselected": 0,
        }


def print_stats_summary(shard: int, stats: Dict, shard_type: str = "regular") -> None:
    prefix = get_shard_type_prefix(shard_type)
    print(f"\n{'=' * 60}")
    print(f"Test Results for Shard {prefix}-{shard}")
    print(f"{'=' * 60}")
    print(f"Total:  {stats['total']}")
    print(f"Passed: {stats['passed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print(f"Duration: {stats['duration']:.2f}s")
    print(f"{'=' * 60}")


def load_installed_torch_root() -> str:
    try:
        import torch
    except Exception as exc:
        print(f"Warning: Failed to import installed torch while preparing PYTHONPATH: {exc}")
        return ""

    return str(Path(torch.__file__).resolve().parent.parent)


def build_execution_env(
    test_dir: Path,
    script_dir: Path,
    disabled_testcases_file: str,
    report_dir: str,
    shard: int,
    shard_type: str = "regular",
) -> Dict[str, str]:
    repo_root = test_dir.parent
    pythonpath_parts = [str(script_dir)]

    torch_path = load_installed_torch_root()
    if torch_path:
        pythonpath_parts.append(torch_path)

    pythonpath_parts.extend([str(repo_root), str(test_dir)])

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    updates = {
        "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        "PYTORCH_TEST_NPU": "1",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "1",
        "NO_TD": "1",
        "PYTEST_ADDOPTS": os.environ.get("PYTEST_ADDOPTS", ""),
        "PYTHONUNBUFFERED": "1",
        # Enable CI mode for slow/disabled test import behavior
        "CI": "true",
    }

    if disabled_testcases_file:
        updates["NPU_DISABLED_TESTCASES_JSON"] = os.path.abspath(disabled_testcases_file)
        updates["NPU_DISABLED_TESTCASES_REPORT"] = os.path.abspath(
            get_disabled_testcases_report_file(report_dir, shard, shard_type)
        )

    return updates


def clean_existing_junit_xml(report_dir: Path) -> None:
    if not report_dir.exists():
        return
    for xml_file in report_dir.rglob("*.xml"):
        xml_file.unlink(missing_ok=True)


def remove_existing_file(path: Path) -> None:
    path.unlink(missing_ok=True)


def get_shard_log_file(report_dir: Path, shard: int, shard_type: str = "regular") -> Path:
    prefix = get_shard_type_prefix(shard_type)
    return report_dir / f"test_shard_{prefix}-{shard}.log"


def run_command_with_tee(command: List[str], cwd: Path, env: Dict[str, str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    merged_env.update(env)

    with log_file.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        try:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_handle.write(line)
            return process.wait()
        except BaseException:
            process.kill()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()


def analyze_pytest_log(log_file: Path, returncode: int) -> Dict:
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

    if returncode == 5 or "collected 0 items" in content or "no tests ran" in content:
        metrics["zero_item_test_files"] = 1

    metrics["import_failures"] = len(
        re.findall(r"^ImportError while importing test module", content, flags=re.MULTILINE)
    )
    collection_errors = len(re.findall(r"^ERROR collecting ", content, flags=re.MULTILINE))
    metrics["startup_failures"] = max(collection_errors - metrics["import_failures"], 0)
    return metrics


def build_pytest_command(
    planned_tests: List[str],
    report_dir: Path,
    shard: int,
    timeout: int,
    verbose: bool,
    parallel: int,
    shard_type: str,
    xml_suffix: str = "",
) -> List[str]:
    """
    Build pytest command for direct test execution.

    All shard types (distributed, excluded, regular) use this unified command
    builder. Tests can run with pytest-xdist for file-level parallelism
    (when parallel > 0) or sequentially (when parallel == 0, for serial tests).

    Args:
        planned_tests: List of test file paths (with 'test/' prefix)
        report_dir: Directory for test reports
        shard: Shard number
        timeout: Per-test timeout
        verbose: Verbose output flag
        parallel: Number of parallel workers (0 = sequential, >0 = pytest-xdist)
        shard_type: "distributed", "excluded", or "regular"
        xml_suffix: Optional suffix appended to the XML filename
                    (e.g. "_test_nn" for serial per-file runs)

    Returns:
        Command list for subprocess execution
    """
    prefix = get_shard_type_prefix(shard_type)
    xml_file = report_dir / f"shard_{prefix}-{shard}_pytest{xml_suffix}.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--color=no",
        "-ra",
        "--tb=short",
        "--continue-on-collection-errors",
        f"--junitxml={xml_file}",
        "-p",
        "pytest_disabled_testcases_plugin",
    ]

    if parallel > 0:
        command.extend([
            f"-n={parallel}",       # pytest-xdist for parallel execution
            "--dist=loadfile",      # Distribute by file for locality
        ])

    if timeout > 0:
        command.append(f"--timeout={timeout}")

    if verbose:
        command.append("-vv")
    else:
        command.append("-v")

    # Add test files (strip only 'test/' prefix, keep '.py' suffix for pytest)
    # pytest needs actual file paths with .py extension
    for test in planned_tests:
        if test.startswith("test/"):
            test_stripped = test[5:]  # Remove only 'test/' prefix, keep '.py'
        else:
            test_stripped = test
        command.append(test_stripped)

    return command


def run_tests_via_pytest(
    planned_tests: List[str],
    shard: int,
    test_dir: Path,
    report_dir: Path,
    env_updates: Dict[str, str],
    timeout: int,
    verbose: bool,
    parallel: int,
    shard_type: str,
) -> Tuple[int, Dict, Dict, List[str]]:
    """
    Run tests directly via pytest with per-file isolation.

    Each test file runs in its own pytest subprocess. This prevents NPU kernel
    crashes from affecting other test files. If a file crashes, it won't generate
    a JUnit XML report, and the summary script will show "Missing" for that file.

    Files are executed concurrently using ThreadPoolExecutor with the specified
    parallel worker count for crash isolation + performance.

    Args:
        planned_tests: List of test file paths (with 'test/' prefix)
        shard: Shard number
        test_dir: Path to the test directory (working directory)
        report_dir: Directory for test reports
        env_updates: Environment variable updates
        timeout: Per-test timeout
        verbose: Verbose output flag
        parallel: Number of parallel workers for concurrent file execution
        shard_type: "distributed" or "regular"

    Returns:
        Tuple of (returncode, stats, log_metrics, missing_files)
        missing_files: List of test files that crashed and didn't generate XML report
    """
    start = monotonic()
    prefix = get_shard_type_prefix(shard_type)
    log_file = get_shard_log_file(report_dir, shard, shard_type)

    merged_env = os.environ.copy()
    merged_env.update(env_updates)

    # Track which files generated XML reports
    executed_files = {}  # test_file -> {"xml_expected": Path, "xml_generated": bool, "returncode": int}

    # Per-file isolation: run each file in separate subprocess
    isolation_parallel = parallel if parallel > 0 else 2

    worst_returncode = 0

    with log_file.open("w", encoding="utf-8") as log_handle:
        log_handle.write("=" * 80 + "\n")
        log_handle.write(f"Per-file isolation pytest execution ({shard_type} shard)\n")
        log_handle.write("=" * 80 + "\n")
        log_handle.write(f"Total test files: {len(planned_tests)}\n")
        log_handle.write(f"Parallel workers: {isolation_parallel}\n")
        log_handle.write("Each file runs in its own subprocess for crash isolation\n")
        log_handle.write("=" * 80 + "\n\n")
        log_handle.flush()

        print(f"\n{'=' * 80}")
        print(f"Per-file isolation mode: {len(planned_tests)} files with {isolation_parallel} parallel workers")
        print("Each file runs in its own subprocess for crash isolation")
        print(f"{'=' * 80}\n")

        # Thread lock for synchronized log writing
        log_lock = threading.Lock()

        def run_single_file_isolated(test_file: str, idx: int) -> Tuple[str, int, bool, str]:
            """
            Run a single test file in isolated subprocess.

            Returns: (test_file, returncode, xml_generated, test_name)
            """
            test_name = strip_test_prefix_and_suffix(test_file)
            safe_name = test_name.replace("/", "_")
            expected_xml = report_dir / f"shard_{prefix}-{shard}_pytest_{safe_name}.xml"

            command = build_pytest_command(
                [test_file],
                report_dir,
                shard,
                timeout,
                verbose,
                parallel=0,  # No xdist within the file
                shard_type=shard_type,
                xml_suffix=f"_{safe_name}",
            )

            # Full command string for printing
            full_command_str = " ".join(command)

            # Write per-file log to separate file for isolation
            file_log_path = report_dir / f"shard_{prefix}-{shard}_log_{safe_name}.txt"

            with log_lock:
                log_handle.write(f"\n{'=' * 80}\n")
                log_handle.write(f"[File {idx}/{len(planned_tests)}] {test_name}\n")
                log_handle.write(f"{'=' * 80}\n")
                log_handle.write(f"Full pytest command:\n")
                log_handle.write(f"  {full_command_str}\n")
                log_handle.write(f"Working directory: {test_dir}\n")
                log_handle.write(f"Expected XML report: {expected_xml.name}\n")
                log_handle.write(f"Log file: {file_log_path.name}\n")
                log_handle.write(f"{'=' * 80}\n")
                log_handle.flush()

            # Print command to stdout
            print(f"\n[{idx}/{len(planned_tests)}] {test_name}")
            print(f"  Command: {full_command_str}")

            # Run pytest subprocess, capture output to per-file log
            rc = _run_pytest_subprocess_to_file(command, test_dir, merged_env, file_log_path)

            # Check if XML was generated
            xml_generated = expected_xml.exists()

            # Append per-file log to main log
            if file_log_path.exists():
                with log_lock:
                    try:
                        file_log_content = file_log_path.read_text(encoding="utf-8", errors="replace")
                        log_handle.write(file_log_content)
                        log_handle.write("\n")
                        if xml_generated:
                            log_handle.write(f"  Result: SUCCESS - XML generated (returncode={rc})\n")
                        else:
                            log_handle.write(f"  Result: CRASH/ERROR - NO XML generated (returncode={rc})\n")
                            log_handle.write(f"  This file likely crashed and will be reported as MISSING\n")
                        log_handle.write(f"{'=' * 80}\n\n")
                        log_handle.flush()
                    except Exception:
                        pass

            if not xml_generated:
                print(f"    WARNING: No XML report generated for {test_name} (likely crashed)")

            return (test_file, rc, xml_generated, test_name)

        # Execute files in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=isolation_parallel) as executor:
            futures = {}
            for idx, test_file in enumerate(planned_tests, 1):
                future = executor.submit(run_single_file_isolated, test_file, idx)
                futures[future] = test_file

            # Collect results as they complete
            for future in as_completed(futures):
                test_file, rc, xml_generated, test_name = future.result()
                executed_files[test_file] = {
                    "xml_expected": report_dir / f"shard_{prefix}-{shard}_pytest_{test_name.replace('/', '_')}.xml",
                    "xml_generated": xml_generated,
                    "returncode": rc,
                    "test_name": test_name,
                }
                if rc != 0 and rc != 5:
                    worst_returncode = rc if worst_returncode == 0 else worst_returncode

    # --- Identify missing files (crashed without generating XML) ---
    missing_files = []
    for test_file, info in executed_files.items():
        if not info["xml_generated"]:
            missing_files.append(test_file)

    # Save missing files list for summary script
    if missing_files:
        missing_file_path = report_dir / f"shard_{prefix}-{shard}_missing_files.txt"
        with missing_file_path.open("w", encoding="utf-8") as f:
            for test_file in missing_files:
                f.write(f"{test_file}\n")
        print(f"\nWARNING: {len(missing_files)} files did not generate XML reports (likely crashed)")

    # --- Aggregate stats from all generated XML files ---
    xml_files = sorted(report_dir.glob(f"shard_{prefix}-{shard}_pytest*.xml"))
    stats = aggregate_junit_stats([report_dir])
    stats["junit_generated"] = bool(xml_files)
    stats["junit_xml_files"] = len(xml_files)
    stats["per_file_isolation"] = True
    stats["parallel_workers"] = isolation_parallel
    stats["missing_files_count"] = len(missing_files)

    # Handle returncode=5 (no tests collected) as success when no real failures
    returncode = worst_returncode
    if worst_returncode == 5 and stats.get("total", 0) == 0 and stats.get("failed", 0) == 0 and stats.get("errors", 0) == 0:
        returncode = 0
        print("Tests collected no items after file filtering and testcase deselection.")

    elapsed = monotonic() - start
    stats = finalize_stats(stats, returncode, elapsed)

    log_metrics = analyze_pytest_log(log_file, worst_returncode)
    log_metrics["test_failures"] = stats.get("failed", 0) + stats.get("errors", 0)
    log_metrics["missing_files_count"] = len(missing_files)
    stats.update(log_metrics)

    if returncode != 0:
        print(f"\n{shard_type.capitalize()} shard tests completed with errors (returncode: {returncode})")
    else:
        print(f"\n{shard_type.capitalize()} shard tests completed successfully")

    return returncode, stats, log_metrics, missing_files


def _run_pytest_subprocess(
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    log_handle,
) -> int:
    """
    Run a pytest subprocess, streaming output to both stdout and log_handle.

    Returns the process return code.
    """
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    raw_returncode = 0
    try:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
        raw_returncode = process.wait()
    except BaseException:
        process.kill()
        raw_returncode = 1
    finally:
        if process.stdout is not None:
            process.stdout.close()

    return raw_returncode


def _run_pytest_subprocess_to_file(
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    log_file: Path,
) -> int:
    """
    Run a pytest subprocess, capturing output to a file (for parallel execution).

    This function is used in per-file isolation mode where multiple files run
    concurrently. Each file's output is captured to a separate file, then merged
    into the main log after completion.

    Returns the process return code.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    raw_returncode = 0
    try:
        with log_file.open("w", encoding="utf-8") as f:
            assert process.stdout is not None
            for line in process.stdout:
                f.write(line)
                # Also print to stdout for real-time visibility
                sys.stdout.write(line)
                sys.stdout.flush()
        raw_returncode = process.wait()
    except BaseException:
        process.kill()
        raw_returncode = 1
    finally:
        if process.stdout is not None:
            process.stdout.close()

    return raw_returncode


def main():
    args = parse_args()

    # Validate shard number
    if args.shard < 1 or args.shard > args.num_shards:
        raise ValueError(f"Invalid shard {args.shard}; expected 1 <= shard <= {args.num_shards}")

    shard_type = args.test_type
    timestamp = datetime.now().isoformat()

    # Resolve paths
    test_dir = Path(args.test_dir).resolve()
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    repo_root = test_dir.parent
    script_dir = Path(__file__).resolve().parent
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load whitelist/blacklist rules
    case_paths_file, whitelist, blacklist = load_case_path_rules(args.case_paths_config)

    # ==========================================================================
    # Execute 4-step test planning
    # ==========================================================================
    plan_result = plan_shard_tests(
        test_dir=test_dir,
        shard=args.shard,
        num_shards=args.num_shards,
        shard_type=shard_type,
        whitelist=whitelist,
        blacklist=blacklist,
    )
    planned_tests = plan_result.get_planned_tests()

    # ==========================================================================
    # Create info dict from planning result
    # ==========================================================================
    info = create_shard_info(args.shard, args.num_shards, timestamp)
    info.update(plan_result.to_info_dict())
    info["shard_type"] = shard_type
    info["shard_index"] = args.shard
    info["shard_total"] = args.num_shards
    info["disabled_count"] = load_disabled_testcases_count(args.disabled_testcases)
    info["parallel_procs"] = args.parallel
    info["selected_test_files"] = plan_result.path_rules_filter.selected_count
    info["excluded_test_files"] = plan_result.path_rules_filter.excluded_count
    info["shard_files"] = plan_result.shard_assignment.planned_count
    if case_paths_file:
        info["path_rules_file"] = case_paths_file

    # Save test plan files
    save_test_plan_file(str(report_dir), args.shard, planned_tests, shard_type)
    save_excluded_test_files_file(str(report_dir), args.shard, plan_result.path_rules_filter.excluded_files, shard_type)
    save_unhandled_upstream_tests_file(str(report_dir), args.shard, [], shard_type)

    # Print summary
    print(create_test_plan_summary(plan_result))
    print(f"\nRepository root: {repo_root}")
    print(f"Test directory: {test_dir}")
    print(f"Parallel workers: {args.parallel} (concurrent file execution)")
    print("Execution mode: per-file isolation (each file in separate subprocess)")
    if case_paths_file:
        print(f"Case path rules: {case_paths_file}")
    print(f"Disabled testcase entries: {info['disabled_count']}")
    print(f"\n{'=' * 80}\n")

    for index, target in enumerate(planned_tests, 1):
        # Show test name without 'test/' prefix for clarity
        display_name = strip_test_prefix_and_suffix(target)
        print(f"  [{index:03d}] {display_name}")

    clean_existing_junit_xml(report_dir)
    remove_existing_file(Path(get_disabled_testcases_report_file(str(report_dir), args.shard, shard_type)))
    remove_existing_file(get_shard_log_file(report_dir, args.shard, shard_type))

    env_updates = build_execution_env(test_dir, script_dir, args.disabled_testcases, str(report_dir), args.shard, shard_type)

    missing_files = []
    if planned_tests:
        # Always use per-file isolation: each file runs in its own pytest subprocess
        # This prevents crashes from affecting other test files
        _, stats, log_metrics, missing_files = run_tests_via_pytest(
            planned_tests,
            args.shard,
            test_dir,
            report_dir,
            env_updates,
            args.timeout,
            args.verbose,
            args.parallel,
            shard_type,
        )
        info["per_file_isolation"] = True
        info["effective_parallel"] = args.parallel
        info["missing_files_count"] = len(missing_files)
    else:
        print("No test files assigned to this shard after file-level filtering.")
        stats = finalize_stats(create_empty_stats(), 0, 0.0)
        log_metrics = {
            "zero_item_test_files": 0,
            "startup_failures": 0,
            "import_failures": 0,
            "test_failures": 0,
            "missing_files_count": 0,
        }

    info["junit_generated"] = bool(stats.get("junit_generated", False))
    info["junit_xml_files"] = int(stats.get("junit_xml_files", 0))
    info["zero_item_test_files"] = int(log_metrics.get("zero_item_test_files", 0))
    info["startup_failures"] = int(log_metrics.get("startup_failures", 0))
    info["import_failures"] = int(log_metrics.get("import_failures", 0))
    info["test_failures"] = int(log_metrics.get("test_failures", 0))
    info.update(load_disabled_testcases_report(str(report_dir), args.shard, shard_type))

    save_info_file(str(report_dir), args.shard, info, shard_type)
    save_stats_file(str(report_dir), args.shard, stats, shard_type)
    print_stats_summary(args.shard, stats, shard_type)
    sys.exit(stats.get("returncode", 1))


if __name__ == "__main__":
    main()
