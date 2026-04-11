#!/usr/bin/env python3
"""
Run a shard of patched upstream PyTorch tests directly with pytest.

Selection is controlled in two layers:
1. File-level allow/deny rules from case_paths_ci.yml
2. Item-level deselection from disabled_testcases.json via a pytest plugin
"""

import argparse
import fnmatch
import json
import os
import re
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Run PyTorch NPU tests for a shard")
    parser.add_argument("--shard", type=int, required=True, help="Shard number (1-indexed)")
    parser.add_argument("--num-shards", type=int, required=True, help="Total number of shards")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to the PyTorch test directory")
    parser.add_argument("--disabled-testcases", type=str, help="Path to disabled_testcases.json")
    parser.add_argument(
        "--case-paths-config",
        type=str,
        help="Path to case_paths_ci.yml for file-level whitelist/blacklist control",
    )
    parser.add_argument(
        "--crashed-files-config",
        type=str,
        help="Path to CRASHED.yml for crashed test files blacklist",
    )
    parser.add_argument("--report-dir", type=str, default="test-reports", help="Directory for test reports")
    parser.add_argument("--timeout", type=int, default=600, help="Per-test timeout passed to pytest")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
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


def load_crashed_files_list(config_file: str) -> Tuple[str, List[str]]:
    """
    Load crashed test files blacklist from CRASHED.yml.

    This file contains test files that cause segmentation fault or process crash
    on NPU, which would break CI execution. These files are excluded before
    the whitelist/blacklist rules are applied.

    Args:
        config_file: Path to CRASHED.yml file

    Returns:
        Tuple of (config_file_path, list_of_crashed_files)
    """
    if not config_file:
        return "", []

    config_path = Path(config_file).resolve()
    if not config_path.exists():
        # File doesn't exist, return empty list (not an error)
        return "", []

    raw_text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        # Parse manually - CRASHED.yml has simpler format
        crashed_files = []
        for line in raw_text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("- "):
                value = stripped[2:].strip()
                if value:
                    crashed_files.append(value)
    else:
        payload = yaml.safe_load(raw_text) or {}
        if not isinstance(payload, dict):
            return str(config_path), []
        crashed_files = payload.get("crashed_files", [])
        if not isinstance(crashed_files, list):
            crashed_files = []

    # Normalize the paths
    normalized_files = coerce_rule_list(crashed_files, "crashed_files")
    return str(config_path), normalized_files


def discover_raw_test_files(test_dir: Path) -> List[str]:
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


def save_stats_file(report_dir: str, shard: int, stats: Dict) -> str:
    os.makedirs(report_dir, exist_ok=True)
    stats_file = os.path.join(report_dir, f"shard_{shard}_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats_file


def save_info_file(report_dir: str, shard: int, info: Dict) -> str:
    os.makedirs(report_dir, exist_ok=True)
    info_file = os.path.join(report_dir, f"shard_{shard}_info.json")
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info_file


def save_test_plan_file(report_dir: str, shard: int, planned_tests: List[str]) -> str:
    os.makedirs(report_dir, exist_ok=True)
    plan_file = os.path.join(report_dir, f"shard_{shard}_planned_test_files.txt")
    with open(plan_file, "w", encoding="utf-8") as f:
        for target in planned_tests:
            f.write(f"{target}\n")
    return plan_file


def save_excluded_test_files_file(report_dir: str, shard: int, test_targets: List[str]) -> str:
    os.makedirs(report_dir, exist_ok=True)
    excluded_file = os.path.join(report_dir, f"shard_{shard}_excluded_test_files.txt")
    with open(excluded_file, "w", encoding="utf-8") as f:
        for target in test_targets:
            f.write(f"{target}\n")
    return excluded_file


def save_unhandled_upstream_tests_file(report_dir: str, shard: int, test_targets: List[str]) -> str:
    os.makedirs(report_dir, exist_ok=True)
    unhandled_file = os.path.join(report_dir, f"shard_{shard}_unhandled_upstream_tests.txt")
    with open(unhandled_file, "w", encoding="utf-8") as f:
        for target in test_targets:
            f.write(f"{target}\n")
    return unhandled_file


def get_disabled_testcases_report_file(report_dir: str, shard: int) -> str:
    return os.path.join(report_dir, f"shard_{shard}_disabled_testcases.json")


def load_disabled_testcases_report(report_dir: str, shard: int) -> Dict:
    report_file = get_disabled_testcases_report_file(report_dir, shard)
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


def print_stats_summary(shard: int, stats: Dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"Test Results for Shard {shard}")
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
    }

    if disabled_testcases_file:
        updates["NPU_DISABLED_TESTCASES_JSON"] = os.path.abspath(disabled_testcases_file)
        updates["NPU_DISABLED_TESTCASES_REPORT"] = os.path.abspath(
            get_disabled_testcases_report_file(report_dir, shard)
        )

    return updates


def clean_existing_junit_xml(report_dir: Path) -> None:
    if not report_dir.exists():
        return
    for xml_file in report_dir.rglob("*.xml"):
        xml_file.unlink(missing_ok=True)


def remove_existing_file(path: Path) -> None:
    path.unlink(missing_ok=True)


def get_shard_log_file(report_dir: Path, shard: int) -> Path:
    return report_dir / f"test_shard_{shard}.log"


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
) -> List[str]:
    xml_file = report_dir / f"shard_{shard}_pytest.xml"
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

    if timeout > 0:
        command.append(f"--timeout={timeout}")

    command.append("-vv" if verbose else "-v")
    command.extend(planned_tests)
    return command


def run_pytest_shard(
    planned_tests: List[str],
    shard: int,
    repo_root: Path,
    report_dir: Path,
    env_updates: Dict[str, str],
    timeout: int,
    verbose: bool,
) -> Tuple[int, Dict, Dict]:
    start = monotonic()
    raw_returncode = 0
    error_message = ""
    log_file = get_shard_log_file(report_dir, shard)
    command = build_pytest_command(planned_tests, report_dir, shard, timeout, verbose)

    print("Executing pytest command:")
    print("  " + " ".join(command))

    try:
        raw_returncode = run_command_with_tee(command, repo_root, env_updates, log_file)
    except Exception as exc:
        raw_returncode = 1
        error_message = f"Failed to execute pytest: {exc}"

    duration = monotonic() - start

    xml_files = sorted(report_dir.rglob("*.xml"))
    stats = aggregate_junit_stats([report_dir])
    stats["junit_generated"] = bool(xml_files)
    stats["junit_xml_files"] = len(xml_files)

    returncode = raw_returncode
    if raw_returncode == 5 and stats.get("total", 0) == 0 and stats.get("failed", 0) == 0 and stats.get("errors", 0) == 0:
        returncode = 0
        print("Pytest collected no tests for this shard after file filtering and testcase deselection.")

    log_metrics = analyze_pytest_log(log_file, raw_returncode)
    log_metrics["test_failures"] = int(stats.get("failed", 0))
    stats.update(log_metrics)

    if returncode != 0 and not error_message:
        error_message = f"pytest exited with code {raw_returncode}"

    stats = finalize_stats(stats or create_empty_stats(), returncode, duration, error_message)
    return returncode, stats, log_metrics


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError(f"Invalid num_shards {args.num_shards}; expected a positive integer")
    if args.shard < 1 or args.shard > args.num_shards:
        raise ValueError(f"Invalid shard {args.shard}; expected 1 <= shard <= {args.num_shards}")

    timestamp = datetime.now().isoformat()
    info = create_shard_info(args.shard, args.num_shards, timestamp)
    info["disabled_count"] = load_disabled_testcases_count(args.disabled_testcases)

    test_dir = Path(args.test_dir).resolve()
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    repo_root = test_dir.parent
    script_dir = Path(__file__).resolve().parent
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load crashed files blacklist (applied first to prevent CI crashes)
    crashed_config_file, crashed_files_list = load_crashed_files_list(args.crashed_files_config)
    info["crashed_files_count"] = len(crashed_files_list)
    if crashed_config_file:
        info["crashed_files_config"] = crashed_config_file

    case_paths_file, whitelist, blacklist = load_case_path_rules(args.case_paths_config)
    info["whitelist_entries"] = len(whitelist)
    info["blacklist_entries"] = len(blacklist)
    if case_paths_file:
        info["path_rules_file"] = case_paths_file

    raw_test_files = discover_raw_test_files(test_dir)

    # First exclude crashed files to prevent process crashes
    if crashed_files_list:
        crashed_excluded = [path for path in raw_test_files if any(path_matches_rule(path, rule) for rule in crashed_files_list)]
        raw_test_files = [path for path in raw_test_files if not any(path_matches_rule(path, rule) for rule in crashed_files_list)]
        info["crashed_excluded_files"] = len(crashed_excluded)
    else:
        info["crashed_excluded_files"] = 0

    # Then apply whitelist/blacklist rules
    selected_test_files, excluded_test_files = apply_case_path_rules(raw_test_files, whitelist, blacklist)
    planned_tests = select_shard_files(selected_test_files, args.shard, args.num_shards)

    info["total_files"] = len(discover_raw_test_files(test_dir))  # Original count before crashed exclusion
    info["selected_test_files"] = len(selected_test_files)
    info["path_filtered_out_files"] = len(excluded_test_files)
    info["excluded_test_files"] = len(excluded_test_files)
    info["shard_files"] = len(planned_tests)

    save_test_plan_file(str(report_dir), args.shard, planned_tests)
    save_excluded_test_files_file(str(report_dir), args.shard, excluded_test_files)
    save_unhandled_upstream_tests_file(str(report_dir), args.shard, [])

    print(f"\n{'=' * 60}")
    print("PyTorch NPU Pytest Shard Runner")
    print(f"{'=' * 60}")
    print(f"Shard: {args.shard}/{args.num_shards}")
    print(f"Repository root: {repo_root}")
    print(f"Test directory: {test_dir}")
    if crashed_config_file:
        print(f"Crashed files config: {crashed_config_file}")
        print(f"Crashed files excluded: {info['crashed_excluded_files']}")
    if case_paths_file:
        print(f"Case path rules: {case_paths_file}")
    print(f"Discovered test files: {info['total_files']}")
    print(f"After crashed exclusion: {len(raw_test_files)}")
    print(f"Selected by path rules: {len(selected_test_files)}")
    print(f"Path-filtered out: {len(excluded_test_files)}")
    print(f"Tests in shard: {len(planned_tests)}")
    print(f"Disabled testcase entries: {info['disabled_count']}")
    print(f"{'=' * 60}\n")

    for index, target in enumerate(planned_tests, 1):
        print(f"  [{index:03d}] {target}")

    clean_existing_junit_xml(report_dir)
    remove_existing_file(Path(get_disabled_testcases_report_file(str(report_dir), args.shard)))
    remove_existing_file(get_shard_log_file(report_dir, args.shard))

    env_updates = build_execution_env(test_dir, script_dir, args.disabled_testcases, str(report_dir), args.shard)

    if planned_tests:
        _, stats, log_metrics = run_pytest_shard(
            planned_tests,
            args.shard,
            repo_root,
            report_dir,
            env_updates,
            args.timeout,
            args.verbose,
        )
    else:
        print("No test files assigned to this shard after file-level filtering.")
        stats = finalize_stats(create_empty_stats(), 0, 0.0)
        log_metrics = {
            "zero_item_test_files": 0,
            "startup_failures": 0,
            "import_failures": 0,
            "test_failures": 0,
        }

    info["junit_generated"] = bool(stats.get("junit_generated", False))
    info["junit_xml_files"] = int(stats.get("junit_xml_files", 0))
    info["zero_item_test_files"] = int(log_metrics.get("zero_item_test_files", 0))
    info["startup_failures"] = int(log_metrics.get("startup_failures", 0))
    info["import_failures"] = int(log_metrics.get("import_failures", 0))
    info["test_failures"] = int(log_metrics.get("test_failures", 0))
    info.update(load_disabled_testcases_report(str(report_dir), args.shard))

    save_info_file(str(report_dir), args.shard, info)
    save_stats_file(str(report_dir), args.shard, stats)
    print_stats_summary(args.shard, stats)
    sys.exit(stats.get("returncode", 1))


if __name__ == "__main__":
    main()
