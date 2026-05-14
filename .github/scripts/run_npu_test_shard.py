#!/usr/bin/env python3
"""
Run PyTorch NPU tests via per-case isolation pytest execution.

This script executes pre-collected test cases or specified test files
with per-case subprocess isolation for crash safety.

Execution modes:
    - Pre-collected cases (--cases-json): Execute cases from JSON file
    - Custom test files (--test-files): Execute specified test files

Each case runs in its own pytest subprocess for isolation:
    - NPU kernel crashes won't cascade to other cases
    - Results recorded in cases.json file

Test types:
    - distributed: Serial execution (one case at a time)
    - regular: Concurrent execution (multiple workers)

Usage:
    # Pre-collected cases mode (primary usage):
    python run_npu_test_shard.py \
        --cases-json distributed_cases_shard_1.json \
        --test-dir /path/to/pytorch/test \
        --disabled-testcases /path/to/disabled_testcases.json \
        --report-dir test-reports \
        --timeout 1200 \
        --max-workers 64 \
        --verbose

    # Custom test files mode:
    python run_npu_test_shard.py \
        --test-files test_meta.py,test_nn.py \
        --test-dir /path/to/pytorch/test \
        --disabled-testcases /path/to/disabled_testcases.json \
        --report-dir test-reports \
        --timeout 1200 \
        --max-workers 4 \
        --verbose

Note: Shard discovery mode (--shard/--num-shards/--test-type) has been removed.
      Use collect_all_cases.py for case discovery and sharding.
"""

import argparse
import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Queue
from time import monotonic
from typing import Dict, List, Tuple

import collect_all_cases


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
class CaseExecutionTask:
    """Task for concurrent case execution."""
    case_idx: int
    nodeid: str
    test_file: str
    file_idx: int


@dataclasses.dataclass
class ConcurrentExecutionConfig:
    """Configuration for concurrent execution."""
    max_workers: int = 4
    per_case_timeout: int = 1200
    verbose: bool = False


# ==============================================================================
# Case Log Saving Functions
# ==============================================================================


def sanitize_nodeid_for_filename(nodeid: str) -> str:
    """
    Convert nodeid to a safe filename.

    Replaces special characters with underscores and truncates if too long.
    Invalid characters for NTFS/filesystems: " : < > | * ? \r \n
    """
    # Replace special characters (including NTFS-invalid chars)
    safe_name = nodeid.replace("::", "_").replace("/", "_").replace("\\", "_")
    safe_name = safe_name.replace("(", "_").replace(")", "_").replace("[", "_").replace("]", "_")
    # NTFS-invalid characters that GitHub Actions artifact upload rejects
    safe_name = safe_name.replace("<", "_lt_").replace(">", "_gt_")
    safe_name = safe_name.replace('"', "_quot_").replace("|", "_pipe_")
    safe_name = safe_name.replace("*", "_star_").replace("?", "_q_")
    safe_name = safe_name.replace(":", "_colon_")
    safe_name = safe_name.replace(" ", "_")
    safe_name = safe_name.replace(".", "_")

    # Remove leading underscores and collapse multiple underscores
    while safe_name.startswith("_"):
        safe_name = safe_name[1:]
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")

    # Truncate if too long (max 200 chars)
    if len(safe_name) > 200:
        safe_name = safe_name[:200]

    return safe_name or "unknown_case"


def save_case_log(
    report_dir: Path,
    shard: int,
    shard_type: str,
    nodeid: str,
    case_idx: int,
    status: str,
    stdout: str,
    stderr: str,
    duration: float,
    returncode: int,
    command: str,
) -> Path:
    """
    Save complete execution log for all test cases.

    Creates a dedicated log file containing:
    - Case metadata (nodeid, status, duration, returncode)
    - Full stdout and stderr output
    - Execution command

    Returns:
        Path to the saved log file
    """
    # Create cases log directory
    cases_logs_dir = report_dir / "cases_logs"
    cases_logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate safe filename
    safe_name = sanitize_nodeid_for_filename(nodeid)
    prefix = "dist" if shard_type == "distributed" else "reg"
    log_filename = f"{prefix}-{shard}_{case_idx}_{safe_name}.log"
    log_path = cases_logs_dir / log_filename

    # Write log content
    content_lines = [
        "=" * 80,
        f"CASE LOG",
        "=" * 80,
        f"Shard: {prefix}-{shard}",
        f"Case Index: {case_idx}",
        f"Nodeid: {nodeid}",
        f"Status: {status}",
        f"Duration: {duration:.2f}s",
        f"Return Code: {returncode}",
        f"Command: {command}",
        "=" * 80,
        "",
        "STDOUT:",
        "-" * 80,
        stdout or "(empty)",
        "",
        "STDERR:",
        "-" * 80,
        stderr or "(empty)",
        "",
        "=" * 80,
    ]

    log_path.write_text("\n".join(content_lines), encoding="utf-8")
    return log_path


class ConcurrentResultAggregator:
    """Thread-safe result aggregator for concurrent execution."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cases_list: List[Dict] = []
        self._worst_returncode: int = 0
        self._passed_count: int = 0
        self._failed_count: int = 0
        self._error_count: int = 0
        self._skipped_count: int = 0
        self._timeout_count: int = 0
        self._total_cases: int = 0

    def add_case_result(self, case_result: Dict) -> None:
        """Thread-safe add case result."""
        with self._lock:
            self._cases_list.append(case_result)
            self._total_cases += 1

            status = case_result.get("status", "error")
            if status == "passed":
                self._passed_count += 1
            elif status == "failed":
                self._failed_count += 1
            elif status == "skipped":
                self._skipped_count += 1
            elif status == "timeout":
                self._timeout_count += 1
            else:
                # error
                self._error_count += 1

            # Track worst returncode (ignore skipped)
            rc = case_result.get("returncode", 1)
            if rc != 0:
                if self._worst_returncode == 0:
                    self._worst_returncode = rc

    def get_sorted_cases(self) -> List[Dict]:
        """Get cases sorted by case_idx."""
        with self._lock:
            return sorted(self._cases_list, key=lambda x: x.get("case_idx", 0))

    def get_summary(self) -> Dict:
        """Get execution summary."""
        with self._lock:
            return {
                "total_cases": self._total_cases,
                "passed_count": self._passed_count,
                "failed_count": self._failed_count,
                "error_count": self._error_count,
                "skipped_count": self._skipped_count,
                "timeout_count": self._timeout_count,
                "worst_returncode": self._worst_returncode,
            }


class ProgressTracker:
    """Thread-safe progress tracker with real-time output."""

    def __init__(self, total_tasks: int):
        self._total_tasks = total_tasks
        self._completed_tasks = 0
        self._lock = threading.Lock()
        self._start_time = monotonic()

    def mark_completed(self, nodeid: str, status: str, duration: float) -> None:
        """Mark task completed and print progress."""
        with self._lock:
            self._completed_tasks += 1
            elapsed = monotonic() - self._start_time
            progress_pct = (self._completed_tasks / self._total_tasks) * 100

            # Status indicator
            status_icon = {
                "passed": "[PASS]",
                "failed": "[FAIL]",
                "error": "[ERR]",
                "timeout": "[TIME]",
                "skipped": "[SKIP]",
            }.get(status, "[?]")

            # Truncate nodeid for display
            display_nodeid = nodeid[:60] + "..." if len(nodeid) > 60 else nodeid

            print(f"[{self._completed_tasks}/{self._total_tasks}] {progress_pct:.1f}% "
                  f"{status_icon} {display_nodeid} ({duration:.1f}s) "
                  f"[elapsed: {elapsed:.0f}s]", flush=True)


# ==============================================================================
# JUnit XML Parsing for Accurate Status Detection
# ==============================================================================


def parse_junit_xml_status(xml_file: Path) -> Dict:
    """
    解析 JUnit XML 报告，获取测试状态。

    Args:
        xml_file: JUnit XML 文件路径

    Returns:
        Dict: {"status": "passed" | "skipped" | "failed" | "error" | "no_xml", "message": str}
    """
    if not xml_file.exists():
        return {"status": "no_xml", "message": "XML file not generated"}

    try:
        tree = ET.parse(str(xml_file))
        root = tree.getroot()

        for testcase in root.iter("testcase"):
            result = {"status": "passed", "message": ""}

            # Check <skipped>
            skipped_elem = testcase.find("skipped")
            if skipped_elem is not None:
                result["status"] = "skipped"
                result["message"] = skipped_elem.get("message", "")
                return result

            # Check <failure>
            failure_elem = testcase.find("failure")
            if failure_elem is not None:
                result["status"] = "failed"
                result["message"] = failure_elem.get("message", "")
                return result

            # Check <error>
            error_elem = testcase.find("error")
            if error_elem is not None:
                result["status"] = "error"
                result["message"] = error_elem.get("message", "")
                return result

            # No failure/error/skipped = passed
            return result

        return {"status": "error", "message": "No testcase in XML"}

    except Exception:
        return {"status": "no_xml", "message": "XML parse failed"}


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
# Concurrent Case Execution
# ==============================================================================


def run_single_case_concurrent(
    task: CaseExecutionTask,
    test_dir: Path,
    merged_env: Dict[str, str],
    config: ConcurrentExecutionConfig,
    result_aggregator: ConcurrentResultAggregator,
    progress_tracker: ProgressTracker,
    log_queue: Queue,
    report_dir: Path,
    shard: int,
    shard_type: str,
) -> Dict:
    """
    Execute a single test case in subprocess (for concurrent execution).

    This function runs in ThreadPoolExecutor threads. Each call spawns
    an independent subprocess for the test case. Core dumps and crashes
    in the subprocess do NOT affect the main Python process or other
    concurrent tasks.

    CRITICAL: This function must catch ALL exceptions and return a result
    dict. It should NEVER raise exceptions to ThreadPoolExecutor level.

    Args:
        task: Case execution task with nodeid and metadata
        test_dir: PyTorch test directory
        merged_env: Environment variables
        config: Execution configuration
        result_aggregator: Thread-safe result collector
        progress_tracker: Thread-safe progress tracker
        log_queue: Queue for log messages

    Returns:
        Dict with case result (never raises exception)
    """
    start_time = monotonic()
    original_nodeid = task.nodeid
    case_nodeid = task.nodeid

    # Strip test/ prefix for pytest execution
    if case_nodeid.startswith("test/"):
        case_nodeid = case_nodeid[5:]

    # Generate XML file path with descriptive name
    prefix = "dist" if shard_type == "distributed" else "reg"
    safe_case_name = sanitize_nodeid_for_filename(original_nodeid)
    xml_filename = f"{prefix}-{shard}_{task.case_idx}_{safe_case_name}.xml"
    xml_file = report_dir / "junit_xmls" / xml_filename

    command = [
        sys.executable,
        "-m",
        "pytest",
        "--color=no",
        "-ra",
        "--tb=short",
        case_nodeid,
        f"--junitxml={xml_file}",
        "--junit-prefix=",
    ]

    if config.per_case_timeout > 0:
        command.append(f"--timeout={config.per_case_timeout}")

    if config.verbose:
        command.append("-vv")
    else:
        command.append("-v")

    command_str = " ".join(command)

    # Build per-case environment with test file directory in PYTHONPATH
    # This enables imports of sibling modules (e.g., 'from model_registry import MLPModule')
    case_env = merged_env.copy()
    test_file = task.test_file
    if test_file.startswith("test/"):
        test_file_rel = test_file[5:]
    else:
        test_file_rel = test_file

    test_file_path = Path(test_file_rel)
    test_file_dir = test_dir / test_file_path.parent

    existing_pythonpath = case_env.get("PYTHONPATH", "")
    case_env["PYTHONPATH"] = str(test_file_dir) + (":" + existing_pythonpath if existing_pythonpath else "")

    # Print start log to stdout (before execution)
    # Truncate nodeid for display
    display_nodeid = original_nodeid[:70] + "..." if len(original_nodeid) > 70 else original_nodeid
    print(f"[{task.case_idx}] Starting: {display_nodeid}", flush=True)

    # Log start
    log_queue.put({
        "type": "case_start",
        "case_idx": task.case_idx,
        "nodeid": original_nodeid,
        "file": task.test_file,
        "command": command_str,
    })

    # Execute subprocess - CRITICAL: catch ALL exceptions
    try:
        result = subprocess.run(
            command,
            cwd=str(test_dir),
            env=case_env,  # Use per-case environment with test file directory in PYTHONPATH
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=config.per_case_timeout + 30,  # Extra buffer
        )

        duration = monotonic() - start_time
        returncode = result.returncode

        # Parse JUnit XML for status
        # - Has XML: use XML status
        # - No XML: error
        xml_result = parse_junit_xml_status(xml_file)
        xml_status = xml_result.get("status")

        if xml_status == "no_xml":
            # No XML → error
            status = "error"
            message = xml_result.get("message")
        else:
            # Has XML → use XML status
            status = xml_status
            message = xml_result.get("message", "")

        # Save logs for all cases
        save_case_log(
            report_dir=report_dir,
            shard=shard,
            shard_type=shard_type,
            nodeid=original_nodeid,
            case_idx=task.case_idx,
            status=status,
            stdout=result.stdout,
            stderr=result.stderr,
            duration=duration,
            returncode=returncode,
            command=command_str,
        )

        case_result = {
            "nodeid": original_nodeid,
            "status": status,
            "duration": duration,
            "returncode": returncode,
            "message": message,
            "command": command_str,
            "file": task.test_file,
            "case_idx": task.case_idx,
        }

    except subprocess.TimeoutExpired:
        # Timeout → no XML, status = timeout
        duration = monotonic() - start_time
        status = "timeout"
        case_result = {
            "nodeid": original_nodeid,
            "status": status,
            "duration": duration,
            "returncode": -1,
            "message": f"Timeout after {config.per_case_timeout}s",
            "command": command_str,
            "file": task.test_file,
            "case_idx": task.case_idx,
        }

        # Save log for timeout
        save_case_log(
            report_dir=report_dir,
            shard=shard,
            shard_type=shard_type,
            nodeid=original_nodeid,
            case_idx=task.case_idx,
            status=status,
            stdout="(process timed out, no output captured)",
            stderr="(process timed out, no output captured)",
            duration=duration,
            returncode=-1,
            command=command_str,
        )

    except Exception as e:
        # Any other exception - return result, don't raise
        duration = monotonic() - start_time
        case_result = {
            "nodeid": original_nodeid,
            "status": "error",
            "duration": duration,
            "returncode": 1,
            "message": f"Unexpected error: {str(e)[:200]}",
            "command": command_str,
            "file": task.test_file,
            "case_idx": task.case_idx,
        }

        # Save error case log
        save_case_log(
            report_dir=report_dir,
            shard=shard,
            shard_type=shard_type,
            nodeid=original_nodeid,
            case_idx=task.case_idx,
            status="error",
            stdout="(exception occurred before execution)",
            stderr=str(e),
            duration=duration,
            returncode=1,
            command=command_str,
        )

    # Log finish
    log_queue.put({
        "type": "case_finish",
        "case_idx": task.case_idx,
        "nodeid": original_nodeid,
        "status": case_result["status"],
        "duration": case_result["duration"],
        "message": case_result["message"][:200] if case_result["message"] else "",
    })

    # Update aggregator (thread-safe)
    result_aggregator.add_case_result(case_result)

    # Update progress (thread-safe)
    progress_tracker.mark_completed(original_nodeid, case_result["status"], duration)

    return case_result


def log_writer_thread(log_queue: Queue, log_file: Path, stop_event: threading.Event) -> None:
    """
    Background thread for writing logs.

    Ensures thread-safe log file writes while concurrent tasks run.
    """
    with log_file.open("w", encoding="utf-8") as log_handle:
        while not stop_event.is_set() or not log_queue.empty():
            try:
                log_entry = log_queue.get(timeout=0.5)
            except:
                continue

            if log_entry.get("type") == "header":
                log_handle.write(log_entry.get("content", ""))
                log_handle.flush()
            elif log_entry.get("type") == "case_start":
                log_handle.write(f"\n[{log_entry['case_idx']}] {log_entry['nodeid']}\n")
                log_handle.write(f"  File: {log_entry.get('file', '')}\n")
                log_handle.write(f"  Command: {log_entry.get('command', '')}\n")
                log_handle.flush()
            elif log_entry.get("type") == "case_finish":
                status_str = log_entry.get("status", "")
                duration_str = f"{log_entry.get('duration', 0):.2f}s"
                log_handle.write(f"  Status: {status_str}, Duration: {duration_str}\n")
                if log_entry.get("message"):
                    log_handle.write(f"  Message: {log_entry['message']}\n")
                log_handle.flush()
            elif log_entry.get("type") == "summary":
                log_handle.write(log_entry.get("content", ""))
                log_handle.flush()


def run_tests_with_tasks_concurrent(
    tasks: List[CaseExecutionTask],
    shard: int,
    test_dir: Path,
    report_dir: Path,
    env_updates: Dict[str, str],
    timeout: int,
    verbose: bool,
    shard_type: str,
    max_workers: int,
    result_module,
    quick_test: int = None,
) -> Tuple[int, float, List[Dict]]:
    """
    Execute pre-collected test cases with concurrent per-case isolation.

    This function takes CaseExecutionTask objects directly (pre-collected cases)
    and executes them concurrently without the file-level case collection phase.

    Args:
        tasks: List of CaseExecutionTask objects (pre-collected cases)
        shard: Shard number
        test_dir: PyTorch test directory
        report_dir: Report output directory
        env_updates: Environment variable updates
        timeout: Per-case timeout in seconds
        verbose: Verbose output
        shard_type: "distributed" or "regular"
        max_workers: Maximum concurrent subprocesses
        result_module: parse_test_results module
        quick_test: Maximum number of cases to execute (None = all cases)

    Returns:
        Tuple of (worst_returncode, duration, cases_list_sorted)
    """
    start = monotonic()
    log_file = result_module.get_shard_log_file(report_dir, shard, shard_type)

    # Create junit_xmls directory for XML reports
    junit_xml_dir = report_dir / "junit_xmls"
    junit_xml_dir.mkdir(parents=True, exist_ok=True)

    merged_env = os.environ.copy()
    merged_env.update(env_updates)

    config = ConcurrentExecutionConfig(
        max_workers=max_workers,
        per_case_timeout=timeout,
        verbose=verbose,
    )

    # Thread-safe result aggregator
    result_aggregator = ConcurrentResultAggregator()

    # Log queue and writer thread
    log_queue = Queue()
    stop_event = threading.Event()
    log_thread = threading.Thread(
        target=log_writer_thread,
        args=(log_queue, log_file, stop_event),
        daemon=True,
    )

    # Write log header
    log_queue.put({
        "type": "header",
        "content": (
            "=" * 80 + "\n"
            f"Pre-collected cases concurrent execution ({shard_type} shard)\n"
            "=" * 80 + "\n"
            f"Total cases: {len(tasks)}\n"
            f"Max concurrent workers: {max_workers}\n"
            "Execution mode: concurrent subprocess, each case isolated\n"
            "=" * 80 + "\n\n"
        ),
    })

    log_thread.start()

    # Quick test: limit number of cases to execute
    if quick_test and len(tasks) > quick_test:
        tasks = tasks[:quick_test]
        print(f"\nQuick test mode: executing only {quick_test} cases", flush=True)

    print(f"\n{'=' * 80}", flush=True)
    print(f"Pre-collected cases: {len(tasks)} cases", flush=True)
    print(f"Execution mode: {max_workers} workers concurrent, each case in subprocess", flush=True)
    print(f"{'=' * 80}\n", flush=True)

    total_cases = len(tasks)
    print(f"Phase 1: Executing {total_cases} pre-collected cases...", flush=True)

    # Phase 2: Concurrent execution via ThreadPoolExecutor
    progress_tracker = ProgressTracker(total_cases)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                run_single_case_concurrent,
                task,
                test_dir,
                merged_env,
                config,
                result_aggregator,
                progress_tracker,
                log_queue,
                report_dir,
                shard,
                shard_type,
            ): task
            for task in tasks
        }

        # Wait for completion (as_completed gives results as they finish)
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                # Result already collected in aggregator
                _ = future.result()
            except Exception as e:
                # Should never happen (run_single_case_concurrent catches all)
                # But as safety, create error result
                case_result = {
                    "nodeid": task.nodeid,
                    "status": "error",
                    "duration": 0.0,
                    "returncode": 1,
                    "message": f"Future error: {str(e)[:200]}",
                    "file": task.test_file,
                    "case_idx": task.case_idx,
                }
                result_aggregator.add_case_result(case_result)
                progress_tracker.mark_completed(task.nodeid, "error", 0.0)

    # Stop log thread
    elapsed = monotonic() - start
    summary = result_aggregator.get_summary()

    log_queue.put({
        "type": "summary",
        "content": (
            f"\n{'=' * 80}\n"
            f"Summary: {summary['total_cases']} cases executed\n"
            f"  Passed: {summary['passed_count']}\n"
            f"  Failed: {summary['failed_count']}\n"
            f"  Errors: {summary['error_count']}\n"
            f"  Timeout: {summary['timeout_count']}\n"
            f"  Skipped: {summary['skipped_count']}\n"
            f"  Duration: {elapsed:.2f}s\n"
            f"  Concurrent workers: {max_workers}\n"
            f"{'=' * 80}\n"
        ),
    })

    stop_event.set()
    log_thread.join(timeout=5)

    # Print final summary
    print(f"\n{'=' * 80}", flush=True)
    print(f"Summary: {summary['total_cases']} cases executed", flush=True)
    print(f"  Passed: {summary['passed_count']}", flush=True)
    print(f"  Failed: {summary['failed_count']}", flush=True)
    print(f"  Errors: {summary['error_count']}", flush=True)
    print(f"  Timeout: {summary['timeout_count']}", flush=True)
    print(f"  Skipped: {summary['skipped_count']}", flush=True)
    print(f"  Duration: {elapsed:.2f}s", flush=True)
    print(f"{'=' * 80}", flush=True)

    return summary["worst_returncode"], elapsed, result_aggregator.get_sorted_cases()


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
        description="Run PyTorch NPU tests via per-case isolation pytest execution"
    )
    parser.add_argument("--test-files", type=str, help="Comma-separated test file paths to run directly (e.g., 'test_meta.py,test_nn.py')")
    parser.add_argument("--cases-json", type=str, help="Path to pre-collected cases JSON file")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to PyTorch test directory")
    parser.add_argument("--disabled-testcases", type=str, help="Path to disabled_testcases.json")
    parser.add_argument("--report-dir", type=str, default="test-reports", help="Directory for reports")
    parser.add_argument("--timeout", type=int, default=1200, help="Per-case timeout in seconds (default: 1200 = 20 minutes)")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent workers for regular tests (default: 4). Each worker runs one pytest subprocess.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick-test", type=int, default=None, help="Quick test mode: execute only N cases for fast verification (default: None, run all cases)")
    args = parser.parse_args()

    # Validate required arguments: must specify either --test-files or --cases-json
    if not args.test_files and not args.cases_json:
        parser.error("Either --test-files or --cases-json must be specified")

    # Validate max_workers
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.max_workers > 128:
        print(f"WARNING: --max-workers={args.max_workers} is very high, may cause resource contention")

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
        print(f"Execution mode: concurrent ({args.max_workers} workers, per-case subprocess isolation)")
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

        # Execute tests (custom mode uses concurrent execution by default)
        cases_list = []
        if planned_tests:
            # Phase 1: Collect all test cases using collect_all_cases module
            print("\nPhase 1: Collecting test cases...")
            error_log_dir = report_dir / "collection_errors"
            collected_cases = collect_all_cases.collect_all_cases(
                planned_tests,
                test_dir,
                error_log_dir,
                parallel=16,
            )

            # Apply quick_test limit if specified
            if args.quick_test and len(collected_cases) > args.quick_test:
                collected_cases = collected_cases[:args.quick_test]
                print(f"  Quick test mode: using only {args.quick_test} cases")

            total_cases = len(collected_cases)
            print(f"\nPhase 2: Executing {total_cases} cases with {args.max_workers} workers")

            # Build CaseExecutionTask list
            tasks = []
            for i, case in enumerate(collected_cases, 1):
                tasks.append(CaseExecutionTask(
                    case_idx=i,
                    nodeid=case["nodeid"],
                    test_file=case["file"],
                    file_idx=0,  # Not needed for pre-collected cases
                ))

            # Phase 2: Execute cases using run_tests_with_tasks_concurrent
            returncode, duration, cases_list = run_tests_with_tasks_concurrent(
                tasks,
                shard,
                test_dir,
                report_dir,
                env_updates,
                args.timeout,
                args.verbose,
                shard_type,
                args.max_workers,
                result_module,
                args.quick_test,
            )
            info["per_case_isolation"] = True
            info["concurrent_workers"] = args.max_workers
            info["returncode"] = returncode
            info["duration"] = duration
        else:
            returncode = 0
            duration = 0.0

        # Build cases.json data
        passed_count = sum(1 for c in cases_list if c["status"] == "passed")
        failed_count = sum(1 for c in cases_list if c["status"] == "failed")
        error_count = sum(1 for c in cases_list if c["status"] == "error")
        timeout_count = sum(1 for c in cases_list if c["status"] == "timeout")
        skipped_count = sum(1 for c in cases_list if c["status"] == "skipped")

        cases_data = {
            "shard": shard,
            "shard_type": shard_type,
            "execution_mode": "concurrent",
            "concurrent_workers": args.max_workers,
            "total_cases": len(cases_list),
            "passed": passed_count,
            "failed": failed_count,
            "errors": error_count,
            "timeout": timeout_count,
            "skipped": skipped_count,
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
            "timeout": timeout_count,
            "duration": duration,
            "returncode": returncode,
            "per_case_isolation": True,
        }

        result_module.save_stats_file(str(report_dir), shard, stats, shard_type)

        # Print summary
        result_module.print_stats_summary(shard, stats, shard_type)

        # Exit with 0 to allow step to succeed and report generation to proceed
        # The actual test results are recorded in cases.json
        sys.exit(0)

    # ==========================================================================
    # Mode: Pre-collected cases JSON execution
    # ==========================================================================
    if args.cases_json:
        print("=" * 80)
        print("Pre-collected Cases Execution Mode")
        print("=" * 80)

        cases_file = Path(args.cases_json).resolve()
        if not cases_file.exists():
            raise FileNotFoundError(f"Cases JSON file not found: {cases_file}")

        cases_data = json.loads(cases_file.read_text(encoding="utf-8"))

        shard = cases_data["shard"]
        num_shards = cases_data["num_shards"]
        shard_type = cases_data.get("test_type", "regular")
        planned_cases = cases_data["cases"]
        total_cases = len(planned_cases)

        print(f"Cases JSON: {cases_file}")
        print(f"Shard: {shard}/{num_shards}")
        print(f"Test type: {shard_type}")
        print(f"Total cases: {total_cases}")
        print(f"Test directory: {test_dir}")

        # Execution mode based on test_type
        if shard_type == "distributed":
            print(f"Execution mode: SERIAL (per-case subprocess isolation)")
        else:
            print(f"Execution mode: CONCURRENT ({args.max_workers} workers, per-case subprocess isolation)")

        if args.disabled_testcases:
            disabled_count = result_module.load_disabled_testcases_count(args.disabled_testcases)
            print(f"Disabled testcase entries: {disabled_count}")

        print(f"\n{'=' * 80}\n")

        # Create info dict for cases-json mode
        info = result_module.create_shard_info(shard, num_shards, timestamp)
        info["selection_mode"] = "cases_json"
        info["shard_type"] = shard_type
        info["cases_json_file"] = str(cases_file)
        info["total_cases"] = total_cases
        info["per_case_isolation"] = True
        if args.disabled_testcases:
            info["disabled_count"] = result_module.load_disabled_testcases_count(args.disabled_testcases)

        # Clean old files
        clean_existing_junit_xml(report_dir)
        remove_existing_file(result_module.get_shard_log_file(report_dir, shard, shard_type))

        # Build execution env
        env_updates = build_execution_env(
            test_dir, script_dir, args.disabled_testcases, shard, shard_type
        )

        # Convert cases to CaseExecutionTask format
        tasks = []
        for i, case in enumerate(planned_cases, 1):
            tasks.append(CaseExecutionTask(
                case_idx=i,
                nodeid=case["nodeid"],
                test_file=case.get("file", ""),
                file_idx=0,
            ))

        # Execute tests based on shard_type
        cases_list = []
        if tasks:
            # Determine execution mode and worker count
            if shard_type == "distributed":
                # Distributed: serial execution (1 worker)
                effective_workers = 1
                print(f"\nExecution mode: SERIAL (distributed tests require sequential execution)")
            else:
                # Regular: concurrent execution
                effective_workers = args.max_workers
                print(f"\nExecution mode: CONCURRENT ({effective_workers} workers)")

            # Execute tasks directly using the new function
            returncode, duration, cases_list = run_tests_with_tasks_concurrent(
                tasks,
                shard,
                test_dir,
                report_dir,
                env_updates,
                args.timeout,
                args.verbose,
                shard_type,
                effective_workers,
                result_module,
                args.quick_test,
            )
            info["execution_mode"] = "serial" if effective_workers == 1 else "concurrent"
            info["concurrent_workers"] = effective_workers

            info["returncode"] = returncode
            info["duration"] = duration
        else:
            print("No cases to execute.")
            returncode = 0
            duration = 0.0

        # Build cases.json data
        passed_count = sum(1 for c in cases_list if c["status"] == "passed")
        failed_count = sum(1 for c in cases_list if c["status"] == "failed")
        error_count = sum(1 for c in cases_list if c["status"] == "error")
        timeout_count = sum(1 for c in cases_list if c["status"] == "timeout")
        skipped_count = sum(1 for c in cases_list if c["status"] == "skipped")

        output_cases_data = {
            "shard": shard,
            "shard_type": shard_type,
            "execution_mode": info.get("execution_mode", "unknown"),
            "concurrent_workers": info.get("concurrent_workers", 1),
            "total_cases": len(cases_list),
            "passed": passed_count,
            "failed": failed_count,
            "errors": error_count,
            "timeout": timeout_count,
            "skipped": skipped_count,
            "duration": duration,
            "cases": cases_list,
        }

        # Save cases.json
        result_module.save_cases_file(str(report_dir), shard, output_cases_data, shard_type)

        # Save info and stats
        result_module.save_info_file(str(report_dir), shard, info, shard_type)

        stats = {
            "total": len(cases_list),
            "passed": passed_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "errors": error_count,
            "timeout": timeout_count,
            "duration": duration,
            "returncode": returncode,
            "per_case_isolation": True,
        }

        result_module.save_stats_file(str(report_dir), shard, stats, shard_type)

        # Print summary
        result_module.print_stats_summary(shard, stats, shard_type)

        # Exit with 0 to allow step to succeed and report generation to proceed
        # The actual test results are recorded in cases.json
        sys.exit(0)

    # No valid mode specified (should not reach here due to argument validation)
    print("ERROR: Either --test-files or --cases-json must be specified")
    sys.exit(1)


if __name__ == "__main__":
    main()