#!/usr/bin/env python3
"""
Run a shard of patched upstream PyTorch tests via per-case isolation pytest execution.

This script focuses on:
    - Test discovery (via discover_test_files.py)
    - Shard assignment (Step 4)
    - Per-case isolation execution (strict serial, each case in own process)

Result parsing is handled by parse_test_results.py.

Test types:
    - distributed: NPU distributed tests (test/distributed/*)
    - regular: All other tests

Each shard executes tests in per-case isolation mode:
    - First collect all test cases via pytest --collect-only
    - Each case runs in its own pytest subprocess (strict serial)
    - NPU kernel crashes won't cascade to other cases
    - Results recorded in cases.json file

Usage:
    python run_npu_test_shard.py \
        --shard 1 \
        --num-shards 50 \
        --test-type distributed \
        --test-dir /path/to/pytorch/test \
        --case-paths-config /path/to/case_paths_ci.yml \
        --disabled-testcases /path/to/disabled_testcases.json \
        --report-dir test-reports \
        --timeout 1200 \
        --verbose
"""

import argparse
import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Dict, List, Optional, Tuple


# ==============================================================================
# Import Result Parser Module
# ==============================================================================


def load_parse_test_results_module(script_dir: Path):
    """Load parse_test_results module dynamically."""
    module_path = script_dir / "parse_test_results.py"
    if not module_path.exists():
        raise FileNotFoundError(f"parse_test_results.py not found at {module_path}")

    spec = importlib.util.spec_from_file_location("parse_test_results", str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclasses.dataclass
class DiscoveryResult:
    """Result from discover_test_files.py."""
    test_files: List[str]
    metadata: Dict
    total_files: int


@dataclasses.dataclass
class ShardAssignmentResult:
    """Result of Step 4: Shard assignment."""
    shard: int
    num_shards: int
    planned_tests: List[str]
    planned_count: int


@dataclasses.dataclass
class ShardPlanResult:
    """Complete result of discovery + shard assignment."""
    discovery: DiscoveryResult
    shard_assignment: ShardAssignmentResult

    def get_planned_tests(self) -> List[str]:
        return self.shard_assignment.planned_tests

    def to_info_dict(self) -> Dict:
        return {
            "total_files": self.discovery.metadata.get("total_files", 0),
            "test_type": self.discovery.metadata.get("test_type", "regular"),
            "type_selected_files": self.discovery.metadata.get("type_selected", 0),
            "type_excluded_files": self.discovery.metadata.get("type_excluded", 0),
            "whitelist_entries": self.discovery.metadata.get("whitelist_entries", 0),
            "blacklist_entries": self.discovery.metadata.get("blacklist_entries", 0),
            "rules_selected": self.discovery.metadata.get("rules_selected", 0),
            "rules_excluded": self.discovery.metadata.get("rules_excluded", 0),
            "shard": self.shard_assignment.shard,
            "num_shards": self.shard_assignment.num_shards,
            "shard_files": self.shard_assignment.planned_count,
        }


# ==============================================================================
# Discovery Integration
# ==============================================================================


def load_discover_module(script_dir: Path):
    """Load discover_test_files module dynamically."""
    module_path = script_dir / "discover_test_files.py"
    if not module_path.exists():
        raise FileNotFoundError(f"discover_test_files.py not found at {module_path}")

    spec = importlib.util.spec_from_file_location("discover_test_files", str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_discovery(
    test_dir: Path,
    test_type: str,
    case_paths_config: Optional[str],
    discover_module,
) -> DiscoveryResult:
    """Run test discovery and return DiscoveryResult."""
    test_files, metadata = discover_module.discover_test_files(
        test_dir=test_dir,
        test_type=test_type,
        case_paths_config=case_paths_config,
    )

    return DiscoveryResult(
        test_files=test_files,
        metadata=metadata,
        total_files=len(test_files),
    )


# ==============================================================================
# Shard Assignment (Step 4)
# ==============================================================================


def select_shard_files(test_files: List[str], shard: int, num_shards: int) -> List[str]:
    """
    Select test files for a shard using contiguous range-based selection.

    Args:
        test_files: List of test file paths, already sorted alphabetically
        shard: Shard number (1-indexed, 1 <= shard <= num_shards)
        num_shards: Total number of shards

    Returns:
        List of test files assigned to this shard
    """
    if not test_files:
        return []

    shard_index = shard - 1
    total_files = len(test_files)

    base_size = total_files // num_shards
    remainder = total_files % num_shards

    if shard_index < remainder:
        start = shard_index * (base_size + 1)
        end = start + base_size + 1
    else:
        start = remainder * (base_size + 1) + (shard_index - remainder) * base_size
        end = start + base_size

    return test_files[start:end]


def assign_shard(discovery_result: DiscoveryResult, shard: int, num_shards: int) -> ShardAssignmentResult:
    """Assign test files to a specific shard."""
    planned_tests = select_shard_files(discovery_result.test_files, shard, num_shards)
    return ShardAssignmentResult(
        shard=shard,
        num_shards=num_shards,
        planned_tests=planned_tests,
        planned_count=len(planned_tests),
    )


# ==============================================================================
# Complete Test Planning
# ==============================================================================


def plan_shard_tests(
    test_dir: Path,
    shard: int,
    num_shards: int,
    test_type: str,
    case_paths_config: Optional[str],
    discover_module,
) -> ShardPlanResult:
    """Complete test planning: discovery + shard assignment."""
    discovery_result = run_discovery(test_dir, test_type, case_paths_config, discover_module)
    shard_assignment_result = assign_shard(discovery_result, shard, num_shards)

    return ShardPlanResult(
        discovery=discovery_result,
        shard_assignment=shard_assignment_result,
    )


def create_test_plan_summary(result: ShardPlanResult) -> str:
    """Create human-readable summary."""
    lines = [
        "=" * 60,
        "Test Planning Summary",
        "=" * 60,
        f"Discovery (Steps 1-3): {result.discovery.metadata.get('total_files', 0)} files scanned",
        f"  Test type: {result.discovery.metadata.get('test_type', 'regular')}",
        f"  Type filter: {result.discovery.metadata.get('type_selected', 0)} selected",
        f"  Rules filter: {result.discovery.metadata.get('rules_selected', 0)} after whitelist/blacklist",
        f"Shard Assignment (Step 4): {result.shard_assignment.planned_count} files for shard {result.shard_assignment.shard}/{result.shard_assignment.num_shards}",
        "=" * 60,
    ]
    return "\n".join(lines)


# ==============================================================================
# Utility Functions
# ==============================================================================


def strip_test_prefix_and_suffix(test_path: str) -> str:
    """Remove 'test/' prefix and '.py' suffix from path."""
    path = test_path
    if path.startswith("test/"):
        path = path[5:]
    if path.endswith(".py"):
        path = path[:-3]
    return path


def load_installed_torch_root() -> str:
    """Get installed torch root directory."""
    try:
        import torch
        return str(Path(torch.__file__).resolve().parent.parent)
    except Exception as exc:
        print(f"Warning: Failed to import torch: {exc}")
        return ""


# ==============================================================================
# Case Collection
# ==============================================================================


def collect_test_cases(test_file: str, test_dir: Path, env: Dict) -> List[str]:
    """
    Collect all test cases from a test file via pytest --collect-only.

    Args:
        test_file: Test file path (e.g., "test/test_autograd.py")
        test_dir: Path to PyTorch test directory
        env: Environment dict for subprocess

    Returns:
        List of case nodeids (e.g., ["test_autograd.py::TestAutograd::test_grad"])
    """
    # Strip test/ prefix if present
    if test_file.startswith("test/"):
        test_file = test_file[5:]

    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "--quiet",
        test_file,
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
            timeout=60,  # Collection timeout
        )

        # Check for collection errors (import failures, etc.)
        if result.stderr:
            stderr_lines = result.stderr.strip().splitlines()
            # Print warning if there are errors
            if stderr_lines and ("error" in result.stderr.lower() or "failed" in result.stderr.lower()):
                print(f"    WARNING: Collection errors for {test_file}:")
                for line in stderr_lines[-10:]:
                    print(f"      {line}")

        # Parse nodeids from output
        nodeids = []
        for line in result.stdout.splitlines():
            # pytest --collect-only outputs nodeids like:
            # <Function test_grad>
            # or with verbose:
            # test_autograd.py::TestAutograd::test_grad
            if "::" in line:
                # Extract nodeid (remove leading spaces and markers)
                nodeid = line.strip()
                # Remove pytest markers like <Function ...>
                if nodeid.startswith("<"):
                    continue
                nodeids.append(nodeid)

        return nodeids

    except subprocess.TimeoutExpired:
        print(f"WARNING: Collection timeout for {test_file}")
        return []
    except Exception as e:
        print(f"WARNING: Collection failed for {test_file}: {e}")
        return []


# ==============================================================================
# Case Execution
# ==============================================================================


def run_single_test_case(
    case_nodeid: str,
    test_dir: Path,
    env: Dict,
    timeout: int,
    verbose: bool,
) -> Dict:
    """
    Run a single test case in isolated subprocess.

    Args:
        case_nodeid: Test case nodeid (e.g., "test_autograd.py::TestAutograd::test_grad")
        test_dir: Path to PyTorch test directory
        env: Environment dict for subprocess
        timeout: Per-case timeout in seconds
        verbose: Verbose output

    Returns:
        Dict with: nodeid, status, duration, returncode, message, command
    """
    start_time = monotonic()

    # Preserve original nodeid for result reporting
    original_nodeid = case_nodeid

    # Strip test/ prefix from nodeid if present (pytest --collect-only outputs with test/ prefix)
    # When cwd is test_dir, the path should be relative to test_dir, not include test/
    if case_nodeid.startswith("test/"):
        case_nodeid = case_nodeid[5:]

    command = [
        sys.executable,
        "-m",
        "pytest",
        "--color=no",
        "-ra",
        "--tb=short",
        case_nodeid,
    ]

    if timeout > 0:
        command.append(f"--timeout={timeout}")

    if verbose:
        command.append("-vv")
    else:
        command.append("-v")

    # Print command to log
    command_str = " ".join(command)
    print(f"    Command: {command_str}")

    try:
        result = subprocess.run(
            command,
            cwd=str(test_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout + 10,  # Extra buffer for timeout handling
        )

        duration = monotonic() - start_time
        returncode = result.returncode

        # Determine status
        if returncode == 0:
            status = "passed"
        elif returncode == 1:
            status = "failed"
        elif returncode == 2:
            status = "error"
        elif returncode == 3:
            status = "skipped"
        elif returncode == 4:
            status = "error"  # usage error
        elif returncode == 5:
            status = "no_tests"
        elif returncode < 0:
            status = "crashed"
        else:
            status = "error"

        # Extract error message from output
        message = ""
        if status in ("failed", "error", "crashed"):
            # Extract last meaningful lines from stderr/stdout
            output = result.stderr + result.stdout
            lines = output.splitlines()
            error_lines = [l for l in lines[-20:] if l.strip()]
            message = "\n".join(error_lines[-5:])[:500]  # Limit message length

        return {
            "nodeid": original_nodeid,
            "status": status,
            "duration": duration,
            "returncode": returncode,
            "message": message,
            "command": command_str,
        }

    except subprocess.TimeoutExpired:
        duration = monotonic() - start_time
        return {
            "nodeid": original_nodeid,
            "status": "timeout",
            "duration": duration,
            "returncode": -1,
            "message": f"Timeout after {timeout}s",
            "command": command_str,
        }
    except Exception as e:
        duration = monotonic() - start_time
        return {
            "nodeid": original_nodeid,
            "status": "error",
            "duration": duration,
            "returncode": 1,
            "message": str(e)[:500],
            "command": command_str,
        }


def run_tests_with_case_isolation(
    planned_tests: List[str],
    shard: int,
    test_dir: Path,
    report_dir: Path,
    env_updates: Dict[str, str],
    timeout: int,
    verbose: bool,
    shard_type: str,
    result_module,
) -> Tuple[int, float, List[Dict]]:
    """
    Execute tests with per-case isolation (strict serial execution).

    Each test case runs in its own pytest subprocess for crash isolation.
    No parallel execution - strict serial processing.

    Returns:
        Tuple of (worst_returncode, duration, cases_list)
    """
    start = monotonic()
    prefix = result_module.get_shard_type_prefix(shard_type)
    log_file = result_module.get_shard_log_file(report_dir, shard, shard_type)

    merged_env = os.environ.copy()
    merged_env.update(env_updates)

    cases_list = []
    worst_returncode = 0

    with log_file.open("w", encoding="utf-8") as log_handle:
        log_handle.write("=" * 80 + "\n")
        log_handle.write(f"Per-case isolation pytest execution ({shard_type} shard)\n")
        log_handle.write("=" * 80 + "\n")
        log_handle.write(f"Total test files: {len(planned_tests)}\n")
        log_handle.write("Execution mode: strict serial, each case in own process\n")
        log_handle.write("=" * 80 + "\n\n")
        log_handle.flush()

        print(f"\n{'=' * 80}")
        print(f"Per-case isolation mode: {len(planned_tests)} files")
        print("Execution mode: strict serial, each case in own process")
        print(f"{'=' * 80}\n")

        total_cases = 0
        case_idx = 0

        for file_idx, test_file in enumerate(planned_tests, 1):
            test_name = strip_test_prefix_and_suffix(test_file)

            log_handle.write(f"\n{'=' * 80}\n")
            log_handle.write(f"[File {file_idx}/{len(planned_tests)}] {test_name}\n")
            log_handle.write(f"{'=' * 80}\n")
            log_handle.flush()

            print(f"\n[File {file_idx}/{len(planned_tests)}] {test_name}")
            print("  Collecting test cases...")

            # Collect cases for this file
            case_nodeids = collect_test_cases(test_file, test_dir, merged_env)

            if not case_nodeids:
                log_handle.write(f"  No cases collected\n")
                print(f"    No cases collected")
                continue

            log_handle.write(f"  Collected {len(case_nodeids)} cases\n")
            log_handle.flush()
            print(f"    Collected {len(case_nodeids)} cases")

            # Execute each case serially
            for nodeid in case_nodeids:
                case_idx += 1
                total_cases += 1

                log_handle.write(f"\n  [{case_idx}] {nodeid}\n")
                log_handle.flush()

                print(f"    [{case_idx}] {nodeid}")

                # Run single case
                case_result = run_single_test_case(
                    nodeid,
                    test_dir,
                    merged_env,
                    timeout,
                    verbose,
                )

                # Add file info
                case_result["file"] = test_file

                # Log result
                status_str = case_result["status"]
                duration_str = f"{case_result['duration']:.2f}s"
                command_str = case_result.get("command", "")
                log_handle.write(f"    Command: {command_str}\n")
                log_handle.write(f"    Status: {status_str}, Duration: {duration_str}\n")
                if case_result["message"]:
                    log_handle.write(f"    Message: {case_result['message'][:200]}\n")
                log_handle.flush()

                print(f"      {status_str} ({duration_str})")

                cases_list.append(case_result)

                # Track worst returncode
                rc = case_result["returncode"]
                if rc != 0 and rc != 3 and rc != 5:  # Ignore skipped/no_tests
                    if worst_returncode == 0:
                        worst_returncode = rc

        # Summary
        elapsed = monotonic() - start

        passed_count = sum(1 for c in cases_list if c["status"] == "passed")
        failed_count = sum(1 for c in cases_list if c["status"] == "failed")
        error_count = sum(1 for c in cases_list if c["status"] in ("error", "crashed", "timeout"))
        skipped_count = sum(1 for c in cases_list if c["status"] == "skipped")

        log_handle.write(f"\n{'=' * 80}\n")
        log_handle.write(f"Summary: {total_cases} cases executed\n")
        log_handle.write(f"  Passed: {passed_count}\n")
        log_handle.write(f"  Failed: {failed_count}\n")
        log_handle.write(f"  Errors: {error_count}\n")
        log_handle.write(f"  Skipped: {skipped_count}\n")
        log_handle.write(f"  Duration: {elapsed:.2f}s\n")
        log_handle.write(f"{'=' * 80}\n")
        log_handle.flush()

        print(f"\n{'=' * 80}")
        print(f"Summary: {total_cases} cases executed")
        print(f"  Passed: {passed_count}, Failed: {failed_count}, Errors: {error_count}")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"{'=' * 80}")

    return worst_returncode, elapsed, cases_list


def build_execution_env(
    test_dir: Path,
    script_dir: Path,
    disabled_testcases_file: str,
    shard: int,
    shard_type: str,
) -> Dict[str, str]:
    """Build environment variables for test execution."""
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
        "PYTHONUNBUFFERED": "1",
        # Note: Do NOT set CI=true here, as some test files have conditional
        # test generation logic like:
        #   if not (IS_CI and torch.cuda.is_available()):
        #       globals().update(generate_tests(...))
        # Setting CI=true would prevent test case generation in those files.
    }

    # Use PyTorch's built-in DISABLED_TESTS_FILE mechanism for skipping test cases
    if disabled_testcases_file:
        # The disabled_testcases.json format is similar to .pytorch-disabled-tests.json
        # Set DISABLED_TESTS_FILE to use PyTorch's built-in skip mechanism
        updates["DISABLED_TESTS_FILE"] = os.path.abspath(disabled_testcases_file)

    return updates


def clean_existing_junit_xml(report_dir: Path) -> None:
    """Clean existing JUnit XML files."""
    if not report_dir.exists():
        return
    for xml_file in report_dir.rglob("*.xml"):
        xml_file.unlink(missing_ok=True)


def remove_existing_file(path: Path) -> None:
    """Remove existing file."""
    path.unlink(missing_ok=True)


# ==============================================================================
# Test Files Input Parser
# ==============================================================================


def parse_test_files_input(test_files_str: str, test_dir: Path) -> List[str]:
    """
    Parse comma-separated test file input and return standardized test file paths.

    Args:
        test_files_str: Comma-separated test file paths (e.g., "test_meta.py,test_nn.py")
        test_dir: Path to PyTorch test directory

    Returns:
        List of standardized test file paths (e.g., ["test/test_meta.py", "test/test_nn.py"])

    Raises:
        FileNotFoundError: If any specified test file does not exist
    """
    files = [f.strip() for f in test_files_str.split(",") if f.strip()]
    result = []

    for f in files:
        # Normalize path format: ensure starts with "test/"
        if not f.startswith("test/"):
            f = "test/" + f

        # Remove leading "test/" prefix if it's duplicated
        if f.startswith("test/test/"):
            f = f[5:]

        # Verify file exists
        full_path = test_dir.parent / f
        if not full_path.exists():
            # Try with .py extension if not provided
            if not f.endswith(".py"):
                f_with_ext = f + ".py"
                full_path_with_ext = test_dir.parent / f_with_ext
                if full_path_with_ext.exists():
                    f = f_with_ext
                    full_path = full_path_with_ext
                else:
                    raise FileNotFoundError(f"Test file not found: {f} or {f_with_ext}")
            else:
                raise FileNotFoundError(f"Test file not found: {f}")

        result.append(f)

    return result


# ==============================================================================
# CLI
# ==============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run PyTorch NPU tests for a shard via per-case isolation"
    )
    parser.add_argument("--test-files", type=str, help="Comma-separated test file paths to run directly (skip shard assignment, e.g., 'test_meta.py,test_nn.py')")
    parser.add_argument("--shard", type=int, help="Shard number (1-indexed, required if --test-files not set)")
    parser.add_argument("--num-shards", type=int, help="Total number of shards (required if --test-files not set)")
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["distributed", "regular"],
        default="regular",
        help="Test type (ignored if --test-files is set)",
    )
    parser.add_argument("--test-dir", type=str, required=True, help="Path to PyTorch test directory")
    parser.add_argument("--disabled-testcases", type=str, help="Path to disabled_testcases.json")
    parser.add_argument("--case-paths-config", type=str, help="Path to case_paths_ci.yml")
    parser.add_argument("--report-dir", type=str, default="test-reports", help="Directory for reports")
    parser.add_argument("--timeout", type=int, default=1200, help="Per-case timeout in seconds (default: 1200 = 20 minutes)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Validate required arguments based on mode
    if not args.test_files:
        if not args.shard or not args.num_shards:
            parser.error("--shard and --num-shards are required when --test-files is not set")

    return args


def main():
    """Main entry point."""
    args = parse_args()

    # Resolve paths
    test_dir = Path(args.test_dir).resolve()
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    repo_root = test_dir.parent
    script_dir = Path(__file__).resolve().parent
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load modules
    discover_module = load_discover_module(script_dir)
    result_module = load_parse_test_results_module(script_dir)

    timestamp = datetime.now().isoformat()

    # ==========================================================================
    # Mode: Direct execution of specified test files
    # ==========================================================================
    if args.test_files:
        print("=" * 80)
        print("Custom Test Files Execution Mode")
        print("=" * 80)

        # Parse test files input
        planned_tests = parse_test_files_input(args.test_files, test_dir)

        # Use fixed shard number for custom mode
        shard = 1
        num_shards = 1
        shard_type = "custom"

        print(f"Test files specified: {len(planned_tests)}")
        print(f"Test directory: {test_dir}")
        print("Execution mode: per-case isolation (strict serial)")
        if args.disabled_testcases:
            disabled_count = result_module.load_disabled_testcases_count(args.disabled_testcases)
            print(f"Disabled testcase entries: {disabled_count}")
        print(f"\n{'=' * 80}\n")

        for index, target in enumerate(planned_tests, 1):
            display_name = strip_test_prefix_and_suffix(target)
            print(f"  [{index:03d}] {display_name}")

        # Create info dict for custom mode
        info = result_module.create_shard_info(shard, num_shards, timestamp)
        info["selection_mode"] = "custom_files"
        info["shard_type"] = shard_type
        info["shard_files"] = len(planned_tests)
        info["total_files"] = len(planned_tests)
        info["selected_test_files"] = len(planned_tests)
        if args.disabled_testcases:
            info["disabled_count"] = result_module.load_disabled_testcases_count(args.disabled_testcases)

        # Save test plan
        result_module.save_test_plan_file(str(report_dir), shard, planned_tests, shard_type)

        # Clean old files
        clean_existing_junit_xml(report_dir)
        remove_existing_file(result_module.get_shard_log_file(report_dir, shard, shard_type))

        # Build execution env
        env_updates = build_execution_env(
            test_dir, script_dir, args.disabled_testcases, shard, shard_type
        )

        # Execute tests
        cases_list = []
        if planned_tests:
            returncode, duration, cases_list = run_tests_with_case_isolation(
                planned_tests,
                shard,
                test_dir,
                report_dir,
                env_updates,
                args.timeout,
                args.verbose,
                shard_type,
                result_module,
            )
            info["per_case_isolation"] = True
            info["returncode"] = returncode
            info["duration"] = duration

        # Build cases.json data
        passed_count = sum(1 for c in cases_list if c["status"] == "passed")
        failed_count = sum(1 for c in cases_list if c["status"] == "failed")
        error_count = sum(1 for c in cases_list if c["status"] in ("error", "crashed", "timeout"))
        skipped_count = sum(1 for c in cases_list if c["status"] == "skipped")

        cases_data = {
            "shard": shard,
            "shard_type": shard_type,
            "execution_mode": "case_isolation",
            "total_cases": len(cases_list),
            "passed": passed_count,
            "failed": failed_count,
            "errors": error_count,
            "skipped": skipped_count,
            "crashed": sum(1 for c in cases_list if c["status"] == "crashed"),
            "timeout": sum(1 for c in cases_list if c["status"] == "timeout"),
            "duration": duration,
            "cases": cases_list,
        }

        # Save cases.json
        result_module.save_cases_file(str(report_dir), shard, cases_data, shard_type)

        # Save info and stats
        result_module.save_info_file(str(report_dir), shard, info, shard_type)

        stats = {
            "total": len(cases_list),
            "passed": passed_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "errors": error_count,
            "duration": duration,
            "returncode": returncode,
            "per_case_isolation": True,
        }

        result_module.save_stats_file(str(report_dir), shard, stats, shard_type)

        # Print summary
        result_module.print_stats_summary(shard, stats, shard_type)

        sys.exit(returncode)

    # ==========================================================================
    # Mode: Shard-based execution (original logic)
    # ==========================================================================

    # Validate shard number
    if args.shard < 1 or args.shard > args.num_shards:
        raise ValueError(f"Invalid shard {args.shard}; expected 1 <= shard <= {args.num_shards}")

    shard_type = args.test_type
    timestamp = datetime.now().isoformat()

    # ==========================================================================
    # Execute test planning
    # ==========================================================================
    plan_result = plan_shard_tests(
        test_dir=test_dir,
        shard=args.shard,
        num_shards=args.num_shards,
        test_type=shard_type,
        case_paths_config=args.case_paths_config,
        discover_module=discover_module,
    )
    planned_tests = plan_result.get_planned_tests()

    # ==========================================================================
    # Create info dict
    # ==========================================================================
    info = result_module.create_shard_info(args.shard, args.num_shards, timestamp)
    info.update(plan_result.to_info_dict())
    info["shard_type"] = shard_type
    info["disabled_count"] = result_module.load_disabled_testcases_count(args.disabled_testcases)
    info["selected_test_files"] = plan_result.discovery.metadata.get("rules_selected", 0)
    info["excluded_test_files"] = plan_result.discovery.metadata.get("rules_excluded", 0)
    info["shard_files"] = plan_result.shard_assignment.planned_count

    if args.case_paths_config:
        info["path_rules_file"] = args.case_paths_config

    # Save test plan
    result_module.save_test_plan_file(str(report_dir), args.shard, planned_tests, shard_type)

    # Save excluded files (not assigned to this shard)
    all_selected = plan_result.discovery.test_files
    excluded_for_shard = [f for f in all_selected if f not in planned_tests]
    result_module.save_excluded_test_files_file(str(report_dir), args.shard, excluded_for_shard, shard_type)

    # Print summary
    print(create_test_plan_summary(plan_result))
    print(f"\nRepository root: {repo_root}")
    print(f"Test directory: {test_dir}")
    print("Execution mode: per-case isolation (strict serial)")
    if args.case_paths_config:
        print(f"Case path rules: {args.case_paths_config}")
    print(f"Disabled testcase entries: {info['disabled_count']}")
    print(f"\n{'=' * 80}\n")

    for index, target in enumerate(planned_tests, 1):
        display_name = strip_test_prefix_and_suffix(target)
        print(f"  [{index:03d}] {display_name}")

    # Clean old files
    clean_existing_junit_xml(report_dir)
    remove_existing_file(result_module.get_shard_log_file(report_dir, args.shard, shard_type))

    # Build execution env
    env_updates = build_execution_env(
        test_dir, script_dir, args.disabled_testcases, args.shard, shard_type
    )

    # ==========================================================================
    # Execute tests
    # ==========================================================================
    cases_list = []
    if planned_tests:
        returncode, duration, cases_list = run_tests_with_case_isolation(
            planned_tests,
            args.shard,
            test_dir,
            report_dir,
            env_updates,
            args.timeout,
            args.verbose,
            shard_type,
            result_module,
        )
        info["per_case_isolation"] = True
    else:
        print("No test files assigned to this shard after file-level filtering.")
        returncode = 0
        duration = 0.0

    # Build cases.json data
    passed_count = sum(1 for c in cases_list if c["status"] == "passed")
    failed_count = sum(1 for c in cases_list if c["status"] == "failed")
    error_count = sum(1 for c in cases_list if c["status"] in ("error", "crashed", "timeout"))
    skipped_count = sum(1 for c in cases_list if c["status"] == "skipped")

    cases_data = {
        "shard": args.shard,
        "shard_type": shard_type,
        "execution_mode": "case_isolation",
        "total_cases": len(cases_list),
        "passed": passed_count,
        "failed": failed_count,
        "errors": error_count,
        "skipped": skipped_count,
        "crashed": sum(1 for c in cases_list if c["status"] == "crashed"),
        "timeout": sum(1 for c in cases_list if c["status"] == "timeout"),
        "duration": duration,
        "cases": cases_list,
    }

    # Save cases.json
    result_module.save_cases_file(str(report_dir), args.shard, cases_data, shard_type)

    # ==========================================================================
    # Generate reports
    # ==========================================================================
    stats = {
        "total": len(cases_list),
        "passed": passed_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "errors": error_count,
        "duration": duration,
        "returncode": returncode,
        "per_case_isolation": True,
    }

    result_module.save_info_file(str(report_dir), args.shard, info, shard_type)
    result_module.save_stats_file(str(report_dir), args.shard, stats, shard_type)
    result_module.print_stats_summary(args.shard, stats, shard_type)

    sys.exit(stats.get("returncode", 1))


if __name__ == "__main__":
    main()