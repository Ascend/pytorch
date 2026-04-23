#!/usr/bin/env python3
"""
Run a shard of patched upstream PyTorch tests via per-file isolation pytest execution.

This script focuses on:
    - Test discovery (via discover_test_files.py)
    - Shard assignment (Step 4)
    - Per-file isolation execution

Result parsing is handled by parse_test_results.py.

Test types:
    - distributed: NPU distributed tests (test/distributed/*)
    - regular: All other tests

Each shard executes tests in per-file isolation mode:
    - Each test file runs in its own pytest subprocess
    - ThreadPoolExecutor for parallel file execution
    - NPU kernel crashes won't cascade to other files
    - Missing XML reports indicate crashed files

Usage:
    python run_npu_test_shard.py \
        --shard 1 \
        --num-shards 50 \
        --test-type distributed \
        --test-dir /path/to/pytorch/test \
        --case-paths-config /path/to/case_paths_ci.yml \
        --disabled-testcases /path/to/disabled_testcases.json \
        --report-dir test-reports \
        --timeout 600 \
        --parallel 2
"""

import argparse
import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        "CI": "true",
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
# Test Execution
# ==============================================================================


def build_pytest_command(
    planned_tests: List[str],
    report_dir: Path,
    shard: int,
    timeout: int,
    verbose: bool,
    shard_type: str,
    result_module,
    xml_suffix: str = "",
) -> List[str]:
    """Build pytest command."""
    prefix = result_module.get_shard_type_prefix(shard_type)
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
    ]

    if timeout > 0:
        command.append(f"--timeout={timeout}")

    if verbose:
        command.append("-vv")
    else:
        command.append("-v")

    for test in planned_tests:
        if test.startswith("test/"):
            test_stripped = test[5:]
        else:
            test_stripped = test
        command.append(test_stripped)

    return command


def _run_pytest_subprocess_to_file(
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    log_file: Path,
) -> int:
    """Run pytest subprocess, capture output to file."""
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
    result_module,
) -> Tuple[int, float, List[str]]:
    """
    Run tests with per-file isolation.

    Each test file runs in its own pytest subprocess for crash isolation.
    ThreadPoolExecutor handles parallel execution.

    Returns:
        Tuple of (worst_returncode, duration, missing_files)
    """
    start = monotonic()
    prefix = result_module.get_shard_type_prefix(shard_type)
    log_file = result_module.get_shard_log_file(report_dir, shard, shard_type)

    merged_env = os.environ.copy()
    merged_env.update(env_updates)

    executed_files = {}
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

        log_lock = threading.Lock()

        def run_single_file_isolated(test_file: str, idx: int) -> Tuple[str, int, bool, str]:
            """Run single test file in isolated subprocess."""
            test_name = strip_test_prefix_and_suffix(test_file)
            safe_name = test_name.replace("/", "_")
            expected_xml = report_dir / f"shard_{prefix}-{shard}_pytest_{safe_name}.xml"

            command = build_pytest_command(
                [test_file],
                report_dir,
                shard,
                timeout,
                verbose,
                shard_type,
                result_module,
                xml_suffix=f"_{safe_name}",
            )

            full_command_str = " ".join(command)
            file_log_path = report_dir / f"shard_{prefix}-{shard}_log_{safe_name}.txt"

            with log_lock:
                log_handle.write(f"\n{'=' * 80}\n")
                log_handle.write(f"[File {idx}/{len(planned_tests)}] {test_name}\n")
                log_handle.write(f"{'=' * 80}\n")
                log_handle.write(f"Command: {full_command_str}\n")
                log_handle.write(f"Expected XML: {expected_xml.name}\n")
                log_handle.flush()

            print(f"\n[{idx}/{len(planned_tests)}] {test_name}")
            print(f"  Command: {full_command_str}")

            rc = _run_pytest_subprocess_to_file(command, test_dir, merged_env, file_log_path)
            xml_generated = expected_xml.exists()

            if file_log_path.exists():
                with log_lock:
                    try:
                        file_log_content = file_log_path.read_text(encoding="utf-8", errors="replace")
                        log_handle.write(file_log_content)
                        log_handle.write("\n")
                        if xml_generated:
                            log_handle.write(f"  Result: SUCCESS (returncode={rc})\n")
                        else:
                            log_handle.write(f"  Result: CRASH - NO XML (returncode={rc})\n")
                        log_handle.write(f"{'=' * 80}\n\n")
                        log_handle.flush()
                    except Exception:
                        pass

            if not xml_generated:
                print(f"    WARNING: No XML for {test_name} (likely crashed)")

            return (test_file, rc, xml_generated, test_name)

        # Execute files in parallel
        with ThreadPoolExecutor(max_workers=isolation_parallel) as executor:
            futures = {}
            for idx, test_file in enumerate(planned_tests, 1):
                future = executor.submit(run_single_file_isolated, test_file, idx)
                futures[future] = test_file

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

    # Identify missing files (crashed)
    missing_files = []
    for test_file, info in executed_files.items():
        if not info["xml_generated"]:
            missing_files.append(test_file)

    if missing_files:
        result_module.save_missing_files_file(str(report_dir), shard, missing_files, shard_type)
        print(f"\nWARNING: {len(missing_files)} files did not generate XML (likely crashed)")

    elapsed = monotonic() - start

    if worst_returncode != 0:
        print(f"\n{shard_type.capitalize()} shard tests completed with errors (returncode: {worst_returncode})")
    else:
        print(f"\n{shard_type.capitalize()} shard tests completed successfully")

    return worst_returncode, elapsed, missing_files


# ==============================================================================
# CLI
# ==============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run PyTorch NPU tests for a shard via per-file isolation"
    )
    parser.add_argument("--shard", type=int, required=True, help="Shard number (1-indexed)")
    parser.add_argument("--num-shards", type=int, required=True, help="Total number of shards")
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["distributed", "regular"],
        default="regular",
        help="Test type",
    )
    parser.add_argument("--test-dir", type=str, required=True, help="Path to PyTorch test directory")
    parser.add_argument("--disabled-testcases", type=str, help="Path to disabled_testcases.json")
    parser.add_argument("--case-paths-config", type=str, help="Path to case_paths_ci.yml")
    parser.add_argument("--report-dir", type=str, default="test-reports", help="Directory for reports")
    parser.add_argument("--timeout", type=int, default=600, help="Per-test timeout")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", type=int, default=2, help="Parallel workers")
    return parser.parse_args()


def main():
    """Main entry point."""
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

    # Load modules
    discover_module = load_discover_module(script_dir)
    result_module = load_parse_test_results_module(script_dir)

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
    info["parallel_procs"] = args.parallel
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
    print(f"Parallel workers: {args.parallel}")
    print("Execution mode: per-file isolation")
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
    missing_files = []
    if planned_tests:
        returncode, duration, missing_files = run_tests_via_pytest(
            planned_tests,
            args.shard,
            test_dir,
            report_dir,
            env_updates,
            args.timeout,
            args.verbose,
            args.parallel,
            shard_type,
            result_module,
        )
        info["per_file_isolation"] = True
        info["effective_parallel"] = args.parallel
        info["missing_files_count"] = len(missing_files)
    else:
        print("No test files assigned to this shard after file-level filtering.")
        returncode = 0
        duration = 0.0

    # ==========================================================================
    # Parse results (via parse_test_results module)
    # ==========================================================================
    stats, log_metrics = result_module.parse_shard_results(
        report_dir=report_dir,
        shard=args.shard,
        shard_type=shard_type,
        returncode=returncode,
        duration=duration,
        missing_files=missing_files,
    )

    # Update info with parsed results
    info["junit_generated"] = bool(stats.get("junit_generated", False))
    info["junit_xml_files"] = int(stats.get("junit_xml_files", 0))
    info["zero_item_test_files"] = int(log_metrics.get("zero_item_test_files", 0))
    info["startup_failures"] = int(log_metrics.get("startup_failures", 0))
    info["import_failures"] = int(log_metrics.get("import_failures", 0))
    info["test_failures"] = int(log_metrics.get("test_failures", 0))

    # ==========================================================================
    # Generate reports
    # ==========================================================================
    result_module.save_info_file(str(report_dir), args.shard, info, shard_type)
    result_module.save_stats_file(str(report_dir), args.shard, stats, shard_type)
    result_module.print_stats_summary(args.shard, stats, shard_type)

    sys.exit(stats.get("returncode", 1))


if __name__ == "__main__":
    main()