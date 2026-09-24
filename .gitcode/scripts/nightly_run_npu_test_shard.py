#!/usr/bin/env python3
"""
Run PyTorch NPU tests via pytest.main() batch execution.

This script executes pre-collected test cases or specified test files
using pytest.main() within worker subprocesses for efficient batch execution.

Execution modes:
    - Pre-collected cases (--cases-json): Execute cases from JSON file
    - Custom test files (--test-files): Execute specified test files

Each worker subprocess runs pytest.main() for multiple same-file cases:
    - Cases are sorted by test file and grouped into batches (max 100 per batch)
    - pytest.main() avoids per-case subprocess startup overhead
    - Worker subprocesses provide crash isolation between batches
    - Coredump detection and automatic retry for affected cases
    - Results recorded in cases.json file

Test types:
    - distributed: Serial execution (one batch at a time)
    - regular: Concurrent execution (multiple batch workers)

Usage:
    # Pre-collected cases mode (primary usage):
    python nightly_run_npu_test_shard.py \
        --cases-json distributed_cases_shard_1.json \
        --test-dir /path/to/pytorch/test \
        --disabled-testcases /path/to/disabled_testcases.json \
        --report-dir test-reports \
        --timeout 1200 \
        --max-workers 64 \
        --verbose

    # Custom test files mode:
    python nightly_run_npu_test_shard.py \
        --test-files test_meta.py,test_nn.py \
        --test-dir /path/to/pytorch/test \
        --disabled-testcases /path/to/disabled_testcases.json \
        --report-dir test-reports \
        --timeout 1200 \
        --max-workers 4 \
        --verbose

Note: Shard discovery mode (--shard/--num-shards/--test-type) has been removed.
      Use nightly_collect_all_cases.py for case discovery and sharding.
"""

import argparse
import contextlib
import dataclasses
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from time import monotonic, sleep
from typing import Dict, List, Optional, Tuple

import collect_all_cases


# ==============================================================================
# NPU Device Detection
# ==============================================================================


def get_npu_device_count() -> int:
    """
    Detect NPU device count via libascend_hal.so.

    Returns the number of available NPU devices. Falls back to 8 if detection fails.
    """
    try:
        from ctypes import byref, c_int, CDLL
        ascend_hal = CDLL("libascend_hal.so")
        dev_count = c_int(-1)
        rc = ascend_hal.drvGetDevNum(byref(dev_count))
        if rc == 0 and dev_count.value > 0:
            return dev_count.value
    except OSError:
        print("Warning: Failed to load libascend_hal.so, using default 8 NPU devices")
    except AttributeError:
        print("Warning: drvGetDevNum not found in libascend_hal.so, using default 8 NPU devices")
    return 8  # Default: typical node has 8 NPU cards


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

    # Truncate if too long (max 200 chars for filesystem compatibility)
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
    npu_device_id: Optional[str] = None,
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
        "CASE LOG",
        "=" * 80,
        f"Shard: {prefix}-{shard}",
        f"Case Index: {case_idx}",
        f"Nodeid: {nodeid}",
        f"Status: {status}",
        f"Duration: {duration:.2f}s",
        f"Return Code: {returncode}",
        f"Command: {command}",
    ]
    if npu_device_id is not None:
        content_lines.append(f"NPU Device: {npu_device_id}")
    content_lines.extend([
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
    ])

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

            # Track worst returncode (largest non-zero value)
            # Negative returncodes (signal crashes) have larger absolute values
            rc = case_result.get("returncode", 1)
            if rc != 0:
                # Keep the "worst" returncode: max of current worst and new rc
                # This captures both high positive codes and severe crashes (negative)
                self._worst_returncode = max(self._worst_returncode, rc)

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
                skip_type = skipped_elem.get("type", "")
                if skip_type == "pytest.xfail":
                    result["status"] = "passed"
                    result["message"] = "xfailed: expected failure"
                    return result
                result["status"] = "skipped"
                attr_msg = skipped_elem.get("message", "")
                text_msg = (skipped_elem.text or "").strip()
                result["message"] = attr_msg + ("\n" + text_msg if text_msg else "")
                return result

            # Check <failure>
            failure_elem = testcase.find("failure")
            if failure_elem is not None:
                result["status"] = "failed"
                attr_msg = failure_elem.get("message", "")
                text_msg = (failure_elem.text or "").strip()
                result["message"] = attr_msg + ("\n" + text_msg if text_msg else "")
                return result

            # Check <error>
            error_elem = testcase.find("error")
            if error_elem is not None:
                result["status"] = "error"
                attr_msg = error_elem.get("message", "")
                text_msg = (error_elem.text or "").strip()
                result["message"] = attr_msg + ("\n" + text_msg if text_msg else "")
                return result

            # No failure/error/skipped = passed
            return result

        return {"status": "error", "message": "No testcase in XML"}

    except Exception:
        return {"status": "no_xml", "message": "XML parse failed"}


# ==============================================================================
# Case Batching Functions
# ==============================================================================


def sort_and_batch_tasks(
    tasks: List[CaseExecutionTask],
    max_cases_per_batch: int = 100,
) -> List[List[CaseExecutionTask]]:
    """
    Sort tasks by test_file then nodeid, group into same-file batches <= max_cases_per_batch.

    This ensures:
    - All cases in a batch share the same test file (required for safe pytest.main() reuse)
    - No batch exceeds max_cases_per_batch (process restart boundary)
    - Cases within each file are ordered by nodeid for deterministic execution
    """
    if not tasks:
        return []

    sorted_tasks = sorted(tasks, key=lambda t: (t.test_file, t.nodeid))
    batches = []
    i = 0
    while i < len(sorted_tasks):
        current_file = sorted_tasks[i].test_file
        batch = []
        while (
            i < len(sorted_tasks)
            and sorted_tasks[i].test_file == current_file
            and len(batch) < max_cases_per_batch
        ):
            batch.append(sorted_tasks[i])
            i += 1
        batches.append(batch)
    return batches


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
# Log Writer Thread
# ==============================================================================


def log_writer_thread(log_queue: Queue, log_file: Path, stop_event: threading.Event) -> None:
    """
    Background thread for writing logs.

    Ensures thread-safe log file writes while concurrent tasks run.
    """
    with log_file.open("w", encoding="utf-8") as log_handle:
        while not stop_event.is_set() or not log_queue.empty():
            try:
                log_entry = log_queue.get(timeout=0.5)
            except Empty:
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
    quick_test: int = None, cards_per_test: int = 0,
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

    # Detect NPU device count and allocate devices
    # distributed default: no ASCEND_RT_VISIBLE_DEVICES → use all cards (serial, 1 batch at a time)
    # distributed + cards_per_test > 0: allocate card ranges → concurrent batches
    if shard_type.startswith("distributed"):
        if cards_per_test > 0:
            num_npu_devices = get_npu_device_count()
            max_workers = max(1, num_npu_devices // cards_per_test)
            print(f"NPU device allocation: {num_npu_devices} devices, "
                  f"{cards_per_test} cards/test → {max_workers} concurrent batches")
        else:
            num_npu_devices = None
            max_workers = 1
            print("NPU device allocation: DISABLED (distributed test uses all devices)")
    else:
        num_npu_devices = get_npu_device_count()
        max_workers = max(1, max_workers)  # 保底
        print(f"NPU device allocation: {num_npu_devices} devices, "
              f"max {max_workers} workers (round-robin)")

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
            f"Pre-collected cases batch execution ({shard_type} shard)\n"
            "=" * 80 + "\n"
            f"Total cases: {len(tasks)}\n"
            f"Max concurrent workers: {max_workers}\n"
            "Execution mode: pytest.main() per case, batched by file (max 100/batch)\n"
            "=" * 80 + "\n\n"
        ),
    })

    log_thread.start()

    # Quick test: limit number of cases to execute
    if quick_test and len(tasks) > quick_test:
        tasks = tasks[:quick_test]
        print(f"\nQuick test mode: executing only {quick_test} cases", flush=True)

    total_cases = len(tasks)

    # Sort and batch tasks: group same-file cases, max 100 per batch
    batches = sort_and_batch_tasks(tasks, max_cases_per_batch=50)

    print(f"\n{'=' * 80}", flush=True)
    print(f"Pre-collected cases: {total_cases} cases", flush=True)
    print(f"Execution mode: {max_workers} workers concurrent, "
          f"{len(batches)} batches (max 1 same-file cases per batch, pytest.main() per case)", flush=True)
    print(f"{'=' * 80}\n", flush=True)

    # Print batch summary
    for bi, b in enumerate(batches):
        display_file = b[0].test_file
        if display_file.startswith("test/"):
            display_file = display_file[5:]
        print(f"  Batch {bi}: {len(b)} cases from {display_file}")

    print(f"\nPhase: Executing {total_cases} pre-collected cases in {len(batches)} batches...", flush=True)

    progress_tracker = ProgressTracker(total_cases)

    # Push case_start log entries for all cases (preserves log format)
    for task in tasks:
        display_nodeid = task.nodeid[:70] + "..." if len(task.nodeid) > 70 else task.nodeid
        log_queue.put({
            "type": "case_start",
            "case_idx": task.case_idx,
            "nodeid": task.nodeid,
            "file": task.test_file,
            "command": f"pytest.main(['{task.nodeid}', '--junitxml=...'])",
        })

    # Execute batches via ThreadPoolExecutor.
    # 按设备组分串行队列：同一组卡(相同取模余数)的 batch 必须串行，否则某组卡上
    # 的 batch 先跑完时，空闲 worker 会立即调度到同组卡的下一个 batch，造成同卡
    # 并发争抢(显存/算力冲突、超时/结果不稳定)。
    dev_groups = 0
    if num_npu_devices is not None:
        dev_groups = max(1, num_npu_devices // cards_per_test) if cards_per_test > 0 else num_npu_devices
    group_locks = [threading.Lock() for _ in range(dev_groups)]

    def _execute_one(batch, batch_id):
        # Calculate device ID:
        #   cards_per_test > 0 → allocate range (e.g. devices 0,1 for 2-card test)
        #   cards_per_test = 0, regular → round-robin single device
        #   cards_per_test = 0, distributed → None (all cards visible)
        if cards_per_test > 0 and num_npu_devices is not None:
            start_device = (batch_id * cards_per_test) % num_npu_devices
            end_device = start_device + cards_per_test - 1
            device_id = ",".join(str(d) for d in range(start_device, end_device + 1))
        elif num_npu_devices is not None:
            device_id = str(batch_id % num_npu_devices)
        else:
            device_id = None

        worker_args = (
            batch, batch_id, test_dir, report_dir, merged_env, timeout, verbose,
            shard, shard_type, device_id, result_aggregator, progress_tracker, log_queue,
        )
        if dev_groups:
            with group_locks[batch_id % dev_groups]:
                _execute_worker_batch(*worker_args)
        else:
            _execute_worker_batch(*worker_args)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch_id, batch in enumerate(batches):
            future = executor.submit(_execute_one, batch, batch_id)
            futures.append((future, batch_id))

        # Check for exceptions
        for future, batch_id in futures:
            try:
                future.result()
            except Exception as e:
                print(f"  ERROR: Batch {batch_id} execution failed: {str(e)[:200]}", flush=True)

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
    disabled_methods_file: str = "",
) -> Dict[str, str]:
    """Build environment variables for test execution."""
    repo_root = test_dir.parent
    pythonpath_parts = [str(script_dir)]

    torch_path = load_installed_torch_root()
    if torch_path:
        pythonpath_parts.append(torch_path)

    pythonpath_parts.extend([str(test_dir)])

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    updates = {
        "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        "PYTORCH_TEST_NPU": "1",
        # 测试文件均显式 import torch_npu, 无需 autoload; 关闭它避免
        # import torch 时 torch_npu 被重复加载, 导致 torch/__init__.py 二次执行、
        # "triton" namespace 被注册两次而报 "Only a single TORCH_LIBRARY" 错误。
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "1",
        "NO_TD": "1",
        "PYTHONUNBUFFERED": "1",
        # "PYTORCH_TESTING_DEVICE_ONLY_FOR": 'privateuse1,cpu'
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

    # Same wiring as DISABLED_TESTS_FILE, but method-level: worker strips the
    # [param] suffix from each nodeid for matching, so one entry covers all variants
    if disabled_methods_file:
        updates["DISABLED_TEST_METHODS_FILE"] = os.path.abspath(disabled_methods_file)

    return updates


# ==============================================================================
# Worker Process (pytest.main() batch execution)
# ==============================================================================


def _build_batch_input_json(
    batch: List[CaseExecutionTask],
    batch_id: int,
    test_dir: Path,
    report_dir: Path,
    env_updates: Dict[str, str],
    timeout: int,
    verbose: bool,
    shard: int,
    shard_type: str,
    npu_device_id: Optional[str],
) -> Dict:
    """Build the JSON input dict for a worker subprocess."""
    return {
        "batch_id": batch_id,
        "test_dir": str(test_dir),
        "report_dir": str(report_dir),
        "env_updates": env_updates,
        "timeout": timeout,
        "verbose": verbose,
        "shard": shard,
        "shard_type": shard_type,
        "npu_device_id": npu_device_id,
        "cases": [
            {
                "case_idx": t.case_idx,
                "nodeid": t.nodeid,
                "test_file": t.test_file,
            }
            for t in batch
        ],
    }


def _execute_worker_batch(
    batch: List[CaseExecutionTask],
    batch_id: int,
    test_dir: Path,
    report_dir: Path,
    merged_env: Dict[str, str],
    timeout: int,
    verbose: bool,
    shard: int,
    shard_type: str,
    npu_device_id: Optional[str],
    result_aggregator: ConcurrentResultAggregator,
    progress_tracker: ProgressTracker,
    log_queue: Queue,
) -> None:
    """
    Execute one batch in a worker subprocess using pytest.main().

    Spawns a subprocess that calls pytest.main() for each case in the batch.
    Reads stdout JSON lines for real-time progress updates.
    No retries: on coredump or idle timeout, the first unreported case is
    marked as error/timeout and a new worker is started for the remaining
    cases. Every case gets exactly one execution chance.
    Never raises — all errors become case_result entries in the aggregator.
    """
    script_path = Path(__file__).resolve()
    batch_input_file = report_dir / f"batch_input_{batch_id}.json"

    remaining_cases = list(batch)
    completed_nodeids = set()
    batch_input = _build_batch_input_json(
        batch, batch_id, test_dir, report_dir,
        {},  # env_updates already merged by caller
        timeout, verbose, shard, shard_type, npu_device_id,
    )

    while remaining_cases:
        batch_input["cases"] = [
            {
                "case_idx": t.case_idx,
                "nodeid": t.nodeid,
                "test_file": t.test_file,
            }
            for t in remaining_cases
        ]
        batch_input_file.write_text(json.dumps(batch_input, indent=2), encoding="utf-8")

        attempt_completed = set()

        try:
            worker_cmd = [
                sys.executable, "-u", str(script_path),
                "--worker", str(batch_input_file),
                "--test-dir", str(test_dir),
            ]

            proc = subprocess.Popen(
                worker_cmd,
                cwd=str(test_dir),
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            last_output_time = monotonic()

            def _read_stdout():
                nonlocal last_output_time
                if proc.stdout:
                    for line in proc.stdout:
                        last_output_time = monotonic()
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            case_result = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        nodeid = case_result.get("nodeid", "")
                        status = case_result.get("status", "error")
                        duration = case_result.get("duration", 0.0)

                        full_result = {
                            "nodeid": nodeid,
                            "status": status,
                            "duration": duration,
                            "returncode": int(case_result.get("returncode", 1)),
                            "message": case_result.get("message", ""),
                            "command": case_result.get("command", ""),
                            "file": case_result.get("file", ""),
                            "case_idx": int(case_result.get("case_idx", 0)),
                        }

                        result_aggregator.add_case_result(full_result)
                        progress_tracker.mark_completed(nodeid, status, duration)
                        log_queue.put({
                            "type": "case_finish",
                            "case_idx": full_result["case_idx"],
                            "nodeid": nodeid,
                            "status": status,
                            "duration": duration,
                            "message": case_result.get("message", "")[:200],
                        })
                        attempt_completed.add(nodeid)

            reader_thread = threading.Thread(target=_read_stdout, daemon=True)
            reader_thread.start()

            idle_timeout = timeout + 30
            timeout_occurred = False

            while True:
                returncode = proc.poll()
                if returncode is not None:
                    reader_thread.join(timeout=10)
                    break

                if monotonic() - last_output_time > idle_timeout:
                    timeout_occurred = True
                    hung_duration = monotonic() - last_output_time
                    print(
                        f"  [Batch {batch_id}] Idle timeout ({hung_duration:.0f}s "
                        f"without output), killing worker...",
                        flush=True,
                    )
                    proc.kill()
                    try:
                        returncode = proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        returncode = -9
                    reader_thread.join(timeout=10)
                    break

                sleep(0.5)

            completed_nodeids.update(attempt_completed)
            not_reported = [
                t for t in remaining_cases
                if t.nodeid not in attempt_completed
            ]

            if timeout_occurred:
                if not_reported:
                    hung_case = not_reported[0]
                    timeout_result = {
                        "nodeid": hung_case.nodeid,
                        "status": "timeout",
                        "duration": hung_duration,
                        "returncode": -1,
                        "message": f"Case hung (no output for {hung_duration:.0f}s)",
                        "command": "",
                        "file": hung_case.test_file,
                        "case_idx": hung_case.case_idx,
                    }
                    result_aggregator.add_case_result(timeout_result)
                    progress_tracker.mark_completed(
                        hung_case.nodeid, "timeout", hung_duration
                    )
                    completed_nodeids.add(hung_case.nodeid)
                    remaining_cases = not_reported[1:]
                else:
                    remaining_cases = []

                if remaining_cases:
                    print(
                        f"  [Batch {batch_id}] Continuing with "
                        f"{len(remaining_cases)} remaining cases...",
                        flush=True,
                    )
                continue

            if returncode < 0:
                signal_num = -returncode
                try:
                    signal_name = signal.Signals(signal_num).name
                except (ValueError, AttributeError):
                    signal_name = f"signal {signal_num}"
                print(
                    f"  [Batch {batch_id}] Worker coredump ({signal_name})",
                    flush=True,
                )

                if not_reported:
                    crashed_case = not_reported[0]
                    error_result = {
                        "nodeid": crashed_case.nodeid,
                        "status": "error",
                        "duration": 0.0,
                        "returncode": returncode,
                        "message": f"Worker killed by signal ({signal_name})",
                        "command": "",
                        "file": crashed_case.test_file,
                        "case_idx": crashed_case.case_idx,
                    }
                    result_aggregator.add_case_result(error_result)
                    progress_tracker.mark_completed(
                        crashed_case.nodeid, "error", 0.0
                    )
                    completed_nodeids.add(crashed_case.nodeid)
                    remaining_cases = not_reported[1:]
                else:
                    remaining_cases = []

                if remaining_cases:
                    print(
                        f"  [Batch {batch_id}] Continuing with "
                        f"{len(remaining_cases)} remaining cases...",
                        flush=True,
                    )
                continue

            # Normal exit: all cases processed
            if not attempt_completed:
                results_file = report_dir / f"batch_results_{batch_id}.json"
                if results_file.exists():
                    try:
                        fallback_results = json.loads(
                            results_file.read_text(encoding="utf-8")
                        )
                        for cr in fallback_results:
                            full_result = {
                                "nodeid": cr.get("nodeid", ""),
                                "status": cr.get("status", "error"),
                                "duration": cr.get("duration", 0.0),
                                "returncode": int(cr.get("returncode", 1)),
                                "message": cr.get("message", ""),
                                "command": cr.get("command", ""),
                                "file": cr.get("file", ""),
                                "case_idx": int(cr.get("case_idx", 0)),
                            }
                            result_aggregator.add_case_result(full_result)
                            progress_tracker.mark_completed(
                                full_result["nodeid"],
                                full_result["status"],
                                full_result["duration"],
                            )
                            completed_nodeids.add(full_result["nodeid"])
                    except (json.JSONDecodeError, OSError):
                        pass

            remaining = [
                t for t in batch if t.nodeid not in completed_nodeids
            ]
            if remaining:
                print(
                    f"  [Batch {batch_id}] {len(remaining)} cases missing "
                    f"results (normal exit), marking as error",
                    flush=True,
                )
                for task in remaining:
                    error_result = {
                        "nodeid": task.nodeid,
                        "status": "error",
                        "duration": 0.0,
                        "returncode": 1,
                        "message": "No result produced (worker exited normally)",
                        "command": "",
                        "file": task.test_file,
                        "case_idx": task.case_idx,
                    }
                    result_aggregator.add_case_result(error_result)
                    progress_tracker.mark_completed(
                        task.nodeid, "error", 0.0
                    )
            break

        except Exception as e:
            print(
                f"  [Batch {batch_id}] Worker execution failed: {str(e)[:200]}",
                flush=True,
            )
            for task in remaining_cases:
                if task.nodeid not in completed_nodeids:
                    error_result = {
                        "nodeid": task.nodeid,
                        "status": "error",
                        "duration": 0.0,
                        "returncode": 1,
                        "message": f"Worker failure: {str(e)[:200]}",
                        "command": "",
                        "file": task.test_file,
                        "case_idx": task.case_idx,
                    }
                    result_aggregator.add_case_result(error_result)
                    progress_tracker.mark_completed(task.nodeid, "error", 0.0)
            break

    # Cleanup temp file
    batch_input_file.unlink(missing_ok=True)
    results_file = report_dir / f"batch_results_{batch_id}.json"
    results_file.unlink(missing_ok=True)


def _worker_main(worker_input_file: str) -> None:
    """
    Worker entry point. Called via:
        python nightly_run_npu_test_shard.py --worker <batch_input.json>

    Reads batch input, runs each case via pytest.main() sequentially,
    prints one JSON line per case to stdout, writes batch_results file,
    then calls os._exit(0). Never returns.
    """
    import time as time_mod

    import pytest

    with open(worker_input_file, encoding="utf-8") as f:
        batch_input = json.load(f)

    cases = batch_input["cases"]
    test_dir = Path(batch_input["test_dir"])
    report_dir = Path(batch_input["report_dir"])
    env_updates = batch_input.get("env_updates", {})
    timeout = batch_input.get("timeout", 1200)
    verbose = batch_input.get("verbose", False)
    shard = batch_input.get("shard", 0)
    shard_type = batch_input.get("shard_type", "regular")
    batch_id = batch_input.get("batch_id", 0)
    npu_device_id = batch_input.get("npu_device_id", None)

    # Apply environment
    for key, value in env_updates.items():
        os.environ[key] = value
    if npu_device_id is not None:
        # npu_device_id can be "3" (single card) or "0,1" (card range)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(npu_device_id)

    # Change to test directory
    os.chdir(str(test_dir))

    # Ensure junit_xmls directory exists
    junit_xml_dir = report_dir / "junit_xmls"
    junit_xml_dir.mkdir(parents=True, exist_ok=True)

    # Determine PYTHONPATH from first case (all cases in batch are same-file)
    if cases:
        first_case = cases[0]
        test_file_rel = first_case["test_file"]
        if test_file_rel.startswith("test/"):
            test_file_rel = test_file_rel[5:]
        test_file_dir = test_dir / Path(test_file_rel).parent
        existing = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = str(test_file_dir) + (":" + existing if existing else "")

    # Case-level blacklist: full nodeid exact match (dict for O(1) lookup with reason)
    disabled_testcases = {}
    dt_file = os.environ.get("DISABLED_TESTS_FILE", "")
    if dt_file and os.path.isfile(dt_file):
        try:
            with open(dt_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                disabled_testcases = {k: str(v) for k, v in data.items() if not k.startswith("_")}
        except (json.JSONDecodeError, OSError):
            disabled_testcases = {}

    # Method-level blacklist: strip [param] suffix, startswith match
    disabled_methods = {}
    dm_file = os.environ.get("DISABLED_TEST_METHODS_FILE", "")
    if dm_file and os.path.isfile(dm_file):
        try:
            with open(dm_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                disabled_methods = {k: str(v) for k, v in data.items() if not k.startswith("_")}
        except (json.JSONDecodeError, OSError):
            disabled_methods = {}

    all_results = []

    for case in cases:
        original_nodeid = case["nodeid"]
        stripped_nodeid = original_nodeid.split("[", 1)[0]
        skip_reason = None
        for method_key, reason in disabled_methods.items():
            if stripped_nodeid.startswith(method_key):
                skip_reason = reason
                break
        if skip_reason is None:
            skip_reason = disabled_testcases.get(original_nodeid)
        if skip_reason is not None:
            case_result = {
                "case_idx": case["case_idx"],
                "nodeid": original_nodeid,
                "status": "skipped",
                "duration": 0.0,
                "returncode": 0,
                "message": f"disabled by method blacklist: {skip_reason}",
                "command": "",
                "file": case["test_file"],
            }
            all_results.append(case_result)
            print(json.dumps(case_result, ensure_ascii=False), flush=True)
            continue

        case_nodeid = original_nodeid
        if case_nodeid.startswith("test/"):
            case_nodeid = case_nodeid[5:]

        # Generate XML filename
        prefix = "dist" if shard_type == "distributed" else "reg"
        safe_name = sanitize_nodeid_for_filename(original_nodeid)
        xml_filename = f"{prefix}-{shard}_{case['case_idx']}_{safe_name}.xml"
        xml_file = junit_xml_dir / xml_filename

        # Build pytest args
        pytest_args = [
            "--color=no",
            "-ra",
            "--tb=short",
            case_nodeid,
            f"--junitxml={xml_file}",
        ]
        if timeout > 0:
            pytest_args.append(f"--timeout={timeout}")
        if verbose:
            pytest_args.append("-vv")
        else:
            pytest_args.append("-v")

        command_str = " ".join([sys.executable, "-m", "pytest"] + pytest_args)

        # Log start to stdout (for parent visibility)
        display_nodeid = (
            original_nodeid[:70] + "..."
            if len(original_nodeid) > 70
            else original_nodeid
        )
        print(f"[{case['case_idx']}] Starting: {display_nodeid}", flush=True)

        # Capture stdout/stderr
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        start_time = time_mod.monotonic()

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                try:
                    returncode = pytest.main(args=pytest_args)
                    if not isinstance(returncode, int):
                        returncode = int(returncode) if returncode is not None else 1
                except SystemExit as e:
                    returncode = int(e.code) if e.code is not None else 1
        except Exception as e:
            returncode = -1
            print(f"  Fatal worker error: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr, flush=True)

        duration = time_mod.monotonic() - start_time

        captured_stdout = stdout_buf.getvalue()
        captured_stderr = stderr_buf.getvalue()

        # Parse JUnit XML for status
        xml_result = parse_junit_xml_status(xml_file)
        if xml_result["status"] == "no_xml":
            status = "error"
            message = xml_result.get("message", "")
        else:
            status = xml_result["status"]
            message = xml_result.get("message", "")

        # Save case log
        save_case_log(
            report_dir=report_dir,
            shard=shard,
            shard_type=shard_type,
            nodeid=original_nodeid,
            case_idx=case["case_idx"],
            status=status,
            stdout=captured_stdout,
            stderr=captured_stderr,
            duration=duration,
            returncode=returncode,
            command=command_str,
            npu_device_id=npu_device_id,
        )

        case_result = {
            "case_idx": case["case_idx"],
            "nodeid": original_nodeid,
            "status": status,
            "duration": duration,
            "returncode": returncode,
            "message": message,
            "command": command_str,
            "file": case["test_file"],
        }
        all_results.append(case_result)

        # Print JSON line to stdout (parent reads in real-time)
        print(json.dumps(case_result, ensure_ascii=False), flush=True)

    # Write batch results file as fallback
    results_file = report_dir / f"batch_results_{batch_id}.json"
    try:
        results_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    except OSError:
        pass

    # Flush and exit (os._exit avoids pytest atexit handlers)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _run_tasks(collected_cases: List[Dict], shard: int,
               test_dir: Path, report_dir: Path, script_dir: Path,
               shard_type: str, args, result_module) -> Tuple[int, float, List[Dict]]:
    """
    Build tasks from collected cases, prepare env, execute, and return results.

    Centralises the shared "prepare → execute" pipeline used by both --test-files
    and --cases-json modes.  Execution mode (serial / concurrent / cards-per-test)
    is decided inside run_tests_with_tasks_concurrent — not here.
    """
    tasks = [CaseExecutionTask(case_idx=i, nodeid=c["nodeid"], test_file=c["file"])
             for i, c in enumerate(collected_cases, 1)]

    clean_existing_junit_xml(report_dir)
    result_module.get_shard_log_file(report_dir, shard, shard_type).unlink(missing_ok=True)

    env_updates = build_execution_env(
        test_dir, script_dir, args.disabled_testcases, shard, shard_type,
        disabled_methods_file=args.disabled_methods,
    )

    return run_tests_with_tasks_concurrent(
        tasks, shard, test_dir, report_dir, env_updates,
        args.timeout, args.verbose, shard_type, args.max_workers,
        result_module, args.quick_test,
        cards_per_test=args.cards_per_test,
    )


def save_results_and_summary(
    result_module,
    report_dir: Path,
    shard: int,
    shard_type: str,
    cases_list: List[Dict],
    duration: float,
    returncode: int,
    info: Dict,
    execution_mode: Optional[str] = None,
    concurrent_workers: Optional[int] = None,
    has_distributed_files: Optional[bool] = None,
) -> None:
    """
    Save results and print summary.

    This function handles the common result processing logic:
    - Calculate statistics (passed, failed, errors, etc.)
    - Build cases_data and stats dicts
    - Save cases.json, info, stats files
    - Print summary
    """
    # Calculate statistics
    passed_count = sum(1 for c in cases_list if c["status"] == "passed")
    failed_count = sum(1 for c in cases_list if c["status"] == "failed")
    error_count = sum(1 for c in cases_list if c["status"] == "error")
    timeout_count = sum(1 for c in cases_list if c["status"] == "timeout")
    skipped_count = sum(1 for c in cases_list if c["status"] == "skipped")

    # Build cases.json data
    cases_data = {
        "shard": shard,
        "shard_type": shard_type,
        "execution_mode": execution_mode or info.get("execution_mode", "unknown"),
        "concurrent_workers": concurrent_workers or info.get("concurrent_workers", 1),
        "total_cases": len(cases_list),
        "passed": passed_count,
        "failed": failed_count,
        "errors": error_count,
        "timeout": timeout_count,
        "skipped": skipped_count,
        "duration": duration,
        "cases": cases_list,
    }
    if has_distributed_files is not None:
        cases_data["has_distributed_files"] = has_distributed_files

    # Save cases.json
    result_module.save_cases_file(str(report_dir), shard, cases_data, shard_type)

    # Save info file
    info["returncode"] = returncode
    info["duration"] = duration
    result_module.save_info_file(str(report_dir), shard, info, shard_type)

    # Build and save stats
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
    if execution_mode:
        stats["execution_mode"] = execution_mode
    if concurrent_workers:
        stats["concurrent_workers"] = concurrent_workers
    if has_distributed_files is not None:
        stats["has_distributed_files"] = has_distributed_files

    result_module.save_stats_file(str(report_dir), shard, stats, shard_type)

    # Print summary
    result_module.print_stats_summary(shard, stats, shard_type)


def clean_existing_junit_xml(report_dir: Path) -> None:
    """Clean existing JUnit XML files."""
    if not report_dir.exists():
        return
    for xml_file in report_dir.rglob("*.xml"):
        xml_file.unlink(missing_ok=True)


# ==============================================================================
# Test Files Input Parser
# ==============================================================================


def has_distributed_test_files(test_files: List[str]) -> bool:
    """
    Check if any test file is a distributed test.

    Distributed tests are identified by path starting with "test/distributed/".

    Args:
        test_files: List of test file paths (e.g., ["test/test_meta.py", "test/distributed/test_ddp.py"])

    Returns:
        True if any file is a distributed test, False otherwise
    """
    for f in test_files:
        if f.startswith("test/distributed/"):
            return True
    return False


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
    parser.add_argument(
        "--test-files",
        type=str,
        help="Comma-separated test file paths to run directly (e.g., 'test_meta.py,test_nn.py')",
    )
    parser.add_argument("--cases-json", type=str, help="Path to pre-collected cases JSON file")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to PyTorch test directory")
    parser.add_argument("--disabled-testcases", type=str, help="Path to disabled_testcases.json")
    parser.add_argument(
        "--disabled-methods",
        type=str,
        help="Path to disabled_test_methods.json (method-level blacklist, covers all parametrized variants)",
    )
    parser.add_argument("--report-dir", type=str, default="test-reports", help="Directory for reports")
    parser.add_argument("--timeout", type=int, default=1200, help="Per-case timeout in seconds (default: 1200 = 20 minutes)")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent workers for regular tests (default: 4). Each worker handles one batch of cases.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--quick-test",
        type=int,
        default=None,
        help="Quick test mode: execute only N cases for fast verification "
        "(default: None, run all cases)",
    )
    parser.add_argument("--cards-per-test", type=int, default=0,
                        help="Cards required per distributed test (default: 0 = no limit, use all cards). "
                             "When >0, allocates card ranges and enables concurrent batch execution. "
                             "e.g. --cards-per-test 2 on 16-card machine → 8 concurrent batches.")
    parser.add_argument("--worker", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Validate required arguments: must specify either --test-files or --cases-json
    # Skip validation in --worker mode (worker only needs --test-dir for path setup)
    if not args.worker and not args.test_files and not args.cases_json:
        parser.error("Either --test-files or --cases-json must be specified")

    # Validate max_workers
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.max_workers > 128:
        print(f"WARNING: --max-workers={args.max_workers} is very high, may cause resource contention")

    return args


def _run_mode_test_files(args, test_dir: Path, report_dir: Path, script_dir: Path,
                         result_module, timestamp: str) -> None:
    """Execute --test-files mode: parse file names, collect cases online, execute, save."""
    planned_tests = parse_test_files_input(args.test_files, test_dir)
    shard, num_shards = 1, 1

    has_distributed = has_distributed_test_files(planned_tests)
    shard_type = "distributed" if has_distributed else "regular"

    print(f"Test files specified: {len(planned_tests)}")
    print(f"Test directory: {test_dir}")
    print(f"Test type: {shard_type}")
    if has_distributed:
        distributed_files = [f for f in planned_tests if f.startswith("test/distributed/")]
        print(f"  Distributed files: {len(distributed_files)}")
        for df in distributed_files:
            print(f"    - {strip_test_prefix_and_suffix(df)}")
    if args.disabled_testcases:
        print(f"Disabled testcase entries: {result_module.load_disabled_testcases_count(args.disabled_testcases)}")
    if args.disabled_methods:
        print(f"Disabled method entries: {result_module.load_disabled_testcases_count(args.disabled_methods)}")
    print(f"\n{'=' * 80}\n")

    for index, target in enumerate(planned_tests, 1):
        print(f"  [{index:03d}] {strip_test_prefix_and_suffix(target)}"
              f"{' [distributed]' if target.startswith('test/distributed/') else ''}")

    info = result_module.create_shard_info(shard, num_shards, timestamp)
    info.update({
        "selection_mode": "custom_files", "shard_type": shard_type,
        "shard_files": len(planned_tests), "total_files": len(planned_tests),
        "selected_test_files": len(planned_tests), "has_distributed_files": has_distributed,
    })
    if args.disabled_testcases:
        info["disabled_count"] = result_module.load_disabled_testcases_count(args.disabled_testcases)

    result_module.save_test_plan_file(str(report_dir), shard, planned_tests, shard_type)

    cases_list = []
    returncode = duration = 0
    if planned_tests:
        print("\nPhase 1: Collecting test cases...")
        error_log_dir = report_dir / "collection_errors"
        collected_cases = collect_all_cases.collect_all_cases(
            planned_tests, test_dir, error_log_dir, parallel=16,
        )
        if args.quick_test and len(collected_cases) > args.quick_test:
            collected_cases = collected_cases[:args.quick_test]
            print(f"  Quick test mode: using only {args.quick_test} cases")

        returncode, duration, cases_list = _run_tasks(
            collected_cases, shard, test_dir, report_dir, script_dir,
            shard_type, args, result_module,
        )
        returncode = returncode or 0
        duration = duration or 0.0

    save_results_and_summary(
        result_module=result_module, report_dir=report_dir,
        shard=shard, shard_type=shard_type,
        cases_list=cases_list, duration=duration, returncode=returncode,
        info=info, has_distributed_files=has_distributed,
    )


def _run_mode_cases_json(args, test_dir: Path, report_dir: Path, script_dir: Path,
                         result_module, timestamp: str) -> None:
    """Execute --cases-json mode: read pre-collected shard JSON, execute, save."""
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

    # Execution mode preview (actual mode decided inside run_tests_with_tasks_concurrent)
    if shard_type.startswith("distributed"):
        if args.cards_per_test > 0:
            print(f"Execution mode: CONCURRENT ({args.cards_per_test} cards/test, "
                  f"batches auto-scheduled on {args.cards_per_test}-card groups)")
        else:
            print("Execution mode: SERIAL (1 batch at a time, all cards visible)")
    else:
        print(f"Execution mode: CONCURRENT ({args.max_workers} workers, "
              f"round-robin single-card, pytest.main() per case)")

    if args.disabled_testcases:
        print(f"Disabled testcase entries: {result_module.load_disabled_testcases_count(args.disabled_testcases)}")
    if args.disabled_methods:
        print(f"Disabled method entries: {result_module.load_disabled_testcases_count(args.disabled_methods)}")
    print(f"\n{'=' * 80}\n")

    info = result_module.create_shard_info(shard, num_shards, timestamp)
    info.update({
        "selection_mode": "cases_json", "shard_type": shard_type,
        "cases_json_file": str(cases_file), "total_cases": total_cases,
        "per_case_isolation": True,
    })
    if args.disabled_testcases:
        info["disabled_count"] = result_module.load_disabled_testcases_count(args.disabled_testcases)
    if args.cards_per_test > 0:
        info["cards_per_test"] = args.cards_per_test

    cases_list = []
    returncode = duration = 0
    if planned_cases:
        returncode, duration, cases_list = _run_tasks(
            planned_cases, shard, test_dir, report_dir, script_dir,
            shard_type, args, result_module,
        )
        returncode = returncode or 0
        duration = duration or 0.0
    else:
        print("No cases to execute.")

    save_results_and_summary(
        result_module=result_module, report_dir=report_dir,
        shard=shard, shard_type=shard_type,
        cases_list=cases_list, duration=duration, returncode=returncode,
        info=info,
    )


def main():
    """Main entry point."""
    args = parse_args()

    # Worker mode dispatch
    if args.worker:
        _worker_main(args.worker)
        return  # _worker_main calls os._exit(0), unreachable

    # Resolve paths
    test_dir = Path(args.test_dir).resolve()
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    script_dir = Path(__file__).resolve().parent
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load modules
    result_module = load_parse_test_results_module(script_dir)

    timestamp = datetime.now().isoformat()

    # ==========================================================================
    # Dispatch: --test-files (PR 增量) vs --cases-json (日流水线全量)
    # ==========================================================================
    if args.test_files:
        print("=" * 80)
        print("Custom Test Files Execution Mode")
        print("=" * 80)
        _run_mode_test_files(args, test_dir, report_dir, script_dir, result_module, timestamp)
        sys.exit(0)

    if args.cases_json:
        print("=" * 80)
        print("Pre-collected Cases Execution Mode")
        print("=" * 80)
        _run_mode_cases_json(args, test_dir, report_dir, script_dir, result_module, timestamp)
        sys.exit(0)

    # No valid mode specified (should not reach here due to argument validation)
    print("ERROR: Either --test-files or --cases-json must be specified")
    sys.exit(1)


if __name__ == "__main__":
    main()
