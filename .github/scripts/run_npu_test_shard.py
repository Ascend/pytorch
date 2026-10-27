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
import contextlib
import dataclasses
import faulthandler
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty, Full
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
    # Control characters (newlines etc. from parametrized test names) would
    # otherwise end up in junit/case-log filenames and break artifact upload.
    safe_name = "".join(
        "_" if (ord(c) < 32 or ord(c) == 127 or c in "\x85\u2028\u2029") else c
        for c in safe_name
    )

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
    npu_device_id: Optional[int] = None,
    filename_suffix: str = "",
) -> Path:
    """
    Save complete execution log for all test cases.

    Creates a dedicated log file containing:
    - Case metadata (nodeid, status, duration, returncode)
    - Full stdout and stderr output
    - Execution command

    filename_suffix appends to the base filename so parent-side crash logs
    never overwrite a log the worker itself already wrote for the case.

    Returns:
        Path to the saved log file
    """
    # Create cases log directory
    cases_logs_dir = report_dir / "cases_logs"
    cases_logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate safe filename
    safe_name = sanitize_nodeid_for_filename(nodeid)
    prefix = {"distributed": "dist", "core": "core", "tensor": "tensor",
              "graph": "graph", "others": "others",
              "regular": "reg", "custom": "custom"}.get(shard_type, "reg")
    log_filename = f"{prefix}-{shard}_{case_idx}_{safe_name}{filename_suffix}.log"
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
    ])

    log_path.write_text("\n".join(content_lines), encoding="utf-8")
    return log_path


# ==============================================================================
# Worker Crash Forensics (parent-side pipe capture)
# ==============================================================================
#
# When a worker dies mid-case (SIGABRT/SIGSEGV coredump) or hangs until the
# idle watchdog kills it, the worker's own pytest output — captured in an
# in-process StringIO — is lost with the process. The crash evidence (C++
# abort messages, CANN FATAL output, native stack traces, pytest's
# fatal-signal dumps) flows to fd 1/2, which the parent merges into the
# worker's stdout pipe. The reader used to drop those lines as noise; now
# the last NOISE_TAIL_LINES raw pipe lines are kept per worker instance
# and written into the affected case's log (same cases_logs/ format), so
# the standard test-reports artifact chain carries the crash scene.
#
# The SIGUSR1 all-thread stack dump requested by the parent before killing
# a hung worker goes to a dedicated per-batch file instead (see
# _worker_main / _read_worker_stack_dump): a worker with a large thread
# pool (OMP/interop threads on a many-core NPU host) can produce more
# dump lines than the bounded pipe tail, and faulthandler prints the main
# (hung) thread FIRST — exactly the lines a bounded tail would evict.

NOISE_TAIL_LINES = 600                # bounded crash-log size (~600KB worst)
NOISE_TAIL_LINE_MAX_CHARS = 1000
HEARTBEAT_DUMP_WAIT_SECONDS = 3.0     # drain time for the SIGUSR1 stack dump


def _is_case_start_marker(line: str) -> bool:
    """True for the worker's own '[<case_idx>] Starting: <nodeid>' marker lines."""
    return line.startswith("[") and "] Starting: " in line


def _save_crash_case_log(
    report_dir: Path,
    shard: int,
    shard_type: str,
    task: CaseExecutionTask,
    status: str,
    duration: float,
    returncode: int,
    message: str,
    worker_state: Dict,
    stack_dump: str = "",
) -> None:
    """
    Write a case log for a case whose worker died or was killed mid-run.

    The worker writes its own per-case log only after a case finishes; when
    it crashes or the idle watchdog kills it, that log never materializes.
    This parent-side log reuses the cases_logs format and carries:
    - the bounded tail of the worker's stdout pipe (native abort/CANN
      output, pytest fatal-signal dumps)
    - optionally the complete faulthandler all-thread stack dump that the
      parent requested via SIGUSR1 before killing a hung worker

    Never overwrites an existing log: if the worker somehow already wrote
    one for this case (e.g. its JSON result line was lost in transport
    before a later hang), the parent-side copy gets a '_parent' suffix so
    the worker's own evidence survives.
    """
    with worker_state["lock"]:
        tail = list(worker_state["noise"])

    if tail:
        stdout_section = (
            f"{message}\n"
            f"[parent-side pipe capture: last {len(tail)} lines of worker "
            f"stdout (fd1+fd2 merged)]\n" + "\n".join(tail)
        )
    else:
        stdout_section = f"{message}\n(no worker output captured on the pipe)"

    if stack_dump:
        stdout_section += (
            "\n\n[faulthandler all-thread stack dump, triggered via SIGUSR1 "
            "before the kill]\n" + stack_dump
        )

    prefix = {"distributed": "dist", "core": "core", "tensor": "tensor",
              "graph": "graph", "others": "others",
              "regular": "reg", "custom": "custom"}.get(shard_type, "reg")
    safe_name = sanitize_nodeid_for_filename(task.nodeid)
    base_log = (report_dir / "cases_logs" /
                f"{prefix}-{shard}_{task.case_idx}_{safe_name}.log")
    filename_suffix = "_parent" if base_log.exists() else ""

    save_case_log(
        report_dir=report_dir,
        shard=shard,
        shard_type=shard_type,
        nodeid=task.nodeid,
        case_idx=task.case_idx,
        status=status,
        stdout=stdout_section,
        stderr="(worker died before pytest teardown; evidence is in the stdout tail above)",
        duration=duration,
        returncode=returncode,
        command="",
        npu_device_id=worker_state.get("device_id"),
        filename_suffix=filename_suffix,
    )


def _read_worker_stack_dump(report_dir: Path, batch_id: int) -> str:
    """
    Read and remove the current worker attempt's SIGUSR1 stack dump file.

    The worker registers faulthandler against this dedicated file (see
    _worker_main); the parent reads it right after killing the hung worker
    group. Each worker restart reopens the file in write mode, so the
    content always belongs to the attempt that just died. Returns "" when
    no dump was produced (e.g. the signal never got delivered).
    """
    dump_path = report_dir / "stack_dumps" / f"batch_{batch_id}_stack.txt"
    try:
        content = dump_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    try:
        dump_path.unlink()
    except OSError:
        pass
    return content


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


class NonBlockingConsole:
    """
    Queued stdout wrapper whose callers can never block on console output.

    A single daemon drain thread owns all real printing; every caller only
    does a bounded put_nowait. When the step's stdout stops draining (wedged
    tee / runner log pipe — the failure mode where a direct print blocks
    forever, historically while holding the progress-tracker lock and
    freezing every reader thread), exactly one thread blocks and the rest
    of the machinery (result aggregation, worker supervision, disk
    forensics) keeps running. Overflow messages are dropped and counted,
    with a note emitted once draining resumes.

    Progress lines routed here are also written to the on-disk shard log by
    the log writer thread, so a wedge costs live console output only, never
    data.
    """

    def __init__(self, maxsize: int = 2000):
        self._queue: Queue = Queue(maxsize=maxsize)
        self._dropped = 0
        self._reported = 0   # high-water mark of drops covered by a delivered note
        self._dropped_lock = threading.Lock()
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._drain, daemon=True)

    def start(self) -> None:
        self._started.set()
        self._thread.start()

    def print(self, msg: str) -> None:
        """Enqueue a line for printing. Never blocks; drops on overflow."""
        # Note emission is atomic under the lock: two racing threads must
        # not both deliver a note for the same span and both advance
        # _reported past _dropped.
        with self._dropped_lock:
            unreported = self._dropped - self._reported
            if unreported:
                try:
                    self._queue.put_nowait(
                        f"[console] {unreported} message(s) dropped while "
                        f"stdout was stalled"
                    )
                    # Note delivered; mark its span as reported. Drops that
                    # race in after this snapshot stay unreported for the
                    # next note.
                    self._reported += unreported
                except Full:
                    pass  # the note itself was dropped; retried next print
        self._try_enqueue(msg)

    def stats(self) -> Dict:
        """Drop/pending counters for stall reporting."""
        with self._dropped_lock:
            dropped = self._dropped
        return {"dropped": dropped, "pending": self._queue.qsize()}

    def finish(self, timeout: float) -> bool:
        """
        Stop the drain loop and wait for the queue to flush.

        Returns True when the drain thread exited (everything flushed).
        False means it is still blocked in a write after the budget —
        stdout is wedged and remaining queued messages will be lost.
        """
        self._started.clear()
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def _try_enqueue(self, msg: str) -> bool:
        """Bounded enqueue; returns False (and counts the drop) when full.
        Never blocks. (The drop-note path in print() enqueues directly so
        it can stay atomic under _dropped_lock.)"""
        try:
            self._queue.put_nowait(msg)
            return True
        except Full:
            with self._dropped_lock:
                self._dropped += 1
            return False

    def _drain(self) -> None:
        while True:
            try:
                msg = self._queue.get(timeout=0.5)
            except Empty:
                if not self._started.is_set():
                    return  # stopped and nothing pending
                continue
            try:
                print(msg, flush=True)
            except Exception:
                # stdout closed/unwritable: keep draining silently so the
                # queue does not fill with undeliverable messages.
                pass


# ==============================================================================
# Parent-Side Global Stall Watchdog
# ==============================================================================
#
# Safety net for stall modes the per-worker idle watchdogs cannot act on
# (e.g. a wedged stdout freezing readers on a shared lock, an unknown
# future deadlock). If NO case completes anywhere for
# PARENT_STALL_THRESHOLD_SECONDS while the run is active, dump every
# parent thread's stack to the report dir — paired with the worker-side
# SIGUSR1 dumps this gives the full causal chain. Disk-only, never prints.

PARENT_STALL_THRESHOLD_SECONDS = 1800   # > measured worst inter-completion gap (~938s in an 11h run)
PARENT_STALL_CHECK_INTERVAL_SECONDS = 60


def _parent_stall_watchdog(
    progress_tracker: "ProgressTracker",
    report_dir: Path,
    stop_event: threading.Event,
) -> None:
    """Daemon: snapshot all parent thread stacks while global progress stalls."""
    last_count = -1
    last_change = monotonic()
    while not stop_event.is_set():
        sleep(PARENT_STALL_CHECK_INTERVAL_SECONDS)
        if stop_event.is_set():
            break
        count = progress_tracker.get_completed()
        if count != last_count:
            last_count = count
            last_change = monotonic()
            continue
        stalled_for = monotonic() - last_change
        if stalled_for < PARENT_STALL_THRESHOLD_SECONDS:
            continue
        dump_path = report_dir / f"parent_stall_{int(monotonic())}.txt"
        try:
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write(
                    f"Global progress stalled for {stalled_for:.0f}s "
                    f"(completed={count}, no completion anywhere). "
                    f"All parent thread stacks follow.\n"
                    f"{'=' * 80}\n"
                )
                faulthandler.dump_traceback(file=f)
        except OSError:
            pass
        # Snapshot once per threshold period while the stall persists.
        last_change = monotonic()


class ProgressTracker:
    """Thread-safe progress tracker with real-time output."""

    def __init__(self, total_tasks: int, console: NonBlockingConsole):
        self._total_tasks = total_tasks
        self._completed_tasks = 0
        self._lock = threading.Lock()
        self._start_time = monotonic()
        self._console = console

    def get_completed(self) -> int:
        """Read-only completed count (never blocks)."""
        with self._lock:
            return self._completed_tasks

    def mark_completed(self, nodeid: str, status: str, duration: float) -> None:
        """Mark task completed and enqueue a progress line.

        The console enqueue never blocks, so this lock is never held across
        a potentially blocking operation (the old direct print here could
        block forever on a wedged stdout while holding the lock, freezing
        every reader thread and, through them, every worker).
        """
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

            self._console.print(
                f"[{self._completed_tasks}/{self._total_tasks}] {progress_pct:.1f}% "
                f"{status_icon} {nodeid} ({duration:.1f}s) "
                f"[elapsed: {elapsed:.0f}s]"
            )


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


def _recover_result_from_junit_xml(
    report_dir: Path,
    shard: int,
    shard_type: str,
    timeout: int,
    task: CaseExecutionTask,
) -> Optional[Dict]:
    """
    Rebuild a case result dict from the JUnit XML the worker left on disk.

    Used when a worker's stdout JSON result line is lost or corrupted in
    transport. pytest writes the XML before the worker prints the result
    line, so any case that actually ran has an XML on disk.

    Returns None when no XML exists for the case (case never ran).
    """
    junit_dir = report_dir / "junit_xmls"
    prefix = {"distributed": "dist", "core": "core", "tensor": "tensor",
              "graph": "graph", "others": "others",
              "regular": "reg", "custom": "custom"}.get(shard_type, "reg")
    safe_name = sanitize_nodeid_for_filename(task.nodeid)
    xml_file = junit_dir / f"{prefix}-{shard}_{task.case_idx}_{safe_name}.xml"
    if not xml_file.exists():
        # Fallback: locate by (prefix, shard, case_idx), unique per case
        candidates = sorted(
            junit_dir.glob(f"{prefix}-{shard}_{task.case_idx}_*.xml")
        )
        if not candidates:
            return None
        xml_file = candidates[0]

    xml_result = parse_junit_xml_status(xml_file)
    if xml_result["status"] == "no_xml":
        return None

    status = xml_result["status"]
    message = xml_result.get("message", "")

    # Extract duration (testcase time preferred, testsuite time fallback)
    duration = 0.0
    try:
        tree = ET.parse(str(xml_file))
        for testcase in tree.getroot().iter("testcase"):
            duration = float(testcase.get("time", 0) or 0)
            break
        if duration == 0.0:
            for suite in tree.getroot().iter("testsuite"):
                duration = float(suite.get("time", 0) or 0)
                break
    except Exception:
        duration = 0.0

    returncode = 0 if status in ("passed", "skipped") else 1

    # Reconstruct the command the worker ran (mirrors _worker_main)
    nodeid_arg = task.nodeid
    if nodeid_arg.startswith("test/"):
        nodeid_arg = nodeid_arg[5:]
    command_str = (
        f"{sys.executable} -m pytest --color=no -ra --tb=short "
        f"{nodeid_arg} --junitxml={xml_file}"
    )
    if timeout > 0:
        command_str += f" --timeout={timeout}"
    command_str += " -vv"

    return {
        "nodeid": task.nodeid,
        "status": status,
        "duration": duration,
        "returncode": returncode,
        "message": message,
        "command": command_str,
        "file": task.test_file,
        "case_idx": task.case_idx,
        "recovered": "junit_xml",
    }


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
    quick_test: Optional[int] = None,
    console: Optional[NonBlockingConsole] = None,
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
        console: NonBlockingConsole for all mid-run output. One is created
            (and leaked to the process lifetime) when not supplied; callers
            that want the end-of-run stdout health check should create one,
            pass it here, and use finish() on it afterwards.

    Returns:
        Tuple of (worst_returncode, duration, cases_list_sorted)
    """
    start = monotonic()
    log_file = result_module.get_shard_log_file(report_dir, shard, shard_type)

    if console is None:
        console = NonBlockingConsole()
        console.start()

    # Create junit_xmls directory for XML reports
    junit_xml_dir = report_dir / "junit_xmls"
    junit_xml_dir.mkdir(parents=True, exist_ok=True)

    merged_env = os.environ.copy()
    merged_env.update(env_updates)

    # Detect NPU device count and allocate devices
    # distributed tests do not set ASCEND_RT_VISIBLE_DEVICES to allow using all devices
    if shard_type == "distributed":
        num_npu_devices = None
        console.print("NPU device allocation: DISABLED (distributed test uses all devices)")
    else:
        num_npu_devices = get_npu_device_count()
        console.print(f"NPU device allocation: {num_npu_devices} devices detected (round-robin)")

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

    log_thread.start()

    # Quick test: limit number of cases to execute
    if quick_test and len(tasks) > quick_test:
        tasks = tasks[:quick_test]
        console.print(f"Quick test mode: executing only {quick_test} cases")

    # Write log header (after the quick_test truncation so the logged total
    # matches what actually runs)
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

    total_cases = len(tasks)

    # Sort and batch tasks: group same-file cases, max 100 per batch
    batches = sort_and_batch_tasks(tasks, max_cases_per_batch=100)

    console.print(f"{'=' * 80}")
    console.print(f"Pre-collected cases: {total_cases} cases")
    console.print(f"Execution mode: {max_workers} workers concurrent, "
                  f"{len(batches)} batches (max 100 same-file cases per batch, pytest.main() per case)")
    console.print(f"{'=' * 80}")

    # Print batch summary
    for bi, b in enumerate(batches):
        display_file = b[0].test_file
        if display_file.startswith("test/"):
            display_file = display_file[5:]
        console.print(f"  Batch {bi}: {len(b)} cases from {display_file}")

    console.print(f"Phase: Executing {total_cases} pre-collected cases in {len(batches)} batches...")

    progress_tracker = ProgressTracker(total_cases, console)

    # Parent-side global-stall watchdog (see _parent_stall_watchdog): started
    # only after the tracker exists, stopped as soon as the executor drains.
    stall_stop = threading.Event()
    stall_thread = threading.Thread(
        target=_parent_stall_watchdog,
        args=(progress_tracker, report_dir, stall_stop),
        daemon=True,
    )
    stall_thread.start()

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

    # Execute batches via ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch_id, batch in enumerate(batches):
            # Calculate device ID (round-robin by batch_id)
            if num_npu_devices is not None:
                device_id = batch_id % num_npu_devices
            else:
                device_id = None

            future = executor.submit(
                _execute_worker_batch,
                batch,
                batch_id,
                test_dir,
                report_dir,
                merged_env,
                timeout,
                verbose,
                shard,
                shard_type,
                device_id,
                result_aggregator,
                progress_tracker,
                log_queue,
                console,
            )
            futures.append((future, batch_id))

        # Check for exceptions
        for future, batch_id in futures:
            try:
                future.result()
            except Exception as e:
                console.print(f"  ERROR: Batch {batch_id} execution failed: {str(e)[:200]}")

    # Stop the stall watchdog and log thread
    stall_stop.set()
    stall_thread.join(timeout=2)

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
    console.print(f"{'=' * 80}")
    console.print(f"Summary: {summary['total_cases']} cases executed")
    console.print(f"  Passed: {summary['passed_count']}")
    console.print(f"  Failed: {summary['failed_count']}")
    console.print(f"  Errors: {summary['error_count']}")
    console.print(f"  Timeout: {summary['timeout_count']}")
    console.print(f"  Skipped: {summary['skipped_count']}")
    console.print(f"  Duration: {elapsed:.2f}s")
    console.print(f"{'=' * 80}")

    return summary["worst_returncode"], elapsed, result_aggregator.get_sorted_cases()


def build_execution_env(
    test_dir: Path,
    script_dir: Path,
    disabled_testcases_file: str,
    shard: int,
    shard_type: str,
    device_env: str = "privateuse1",
) -> Dict[str, str]:
    """Build environment variables for test execution."""
    repo_root = test_dir.parent
    pythonpath_parts = [str(script_dir)]

    torch_path = load_installed_torch_root()
    if torch_path:
        pythonpath_parts.append(torch_path)

    pythonpath_parts.extend([str(repo_root), str(test_dir)])

    # 分布式用例目录需显式加入 PYTHONPATH：这两个目录下的用例 import 同目录
    # 的 helper 模块（如 model_registry/schedule_registry），需保证其所在目录在 sys.path 中。
    pythonpath_parts.extend(
        [
            str(test_dir / "distributed" / "pipelining"),
            str(test_dir / "distributed" / "local_tensor_tutorial_examples"),
        ]
    )

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    updates = {
        "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        "PYTORCH_TEST_NPU": "1",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "1",
        "NO_TD": "1",
        "PYTHONUNBUFFERED": "1",
        # Only PYTORCH_TESTING_DEVICE_ONLY_FOR is set here. Do NOT set
        # PYTORCH_TESTING_DEVICE_FOR_CUSTOM: it must match the collection
        # environment, which does not set it either (setting it renames
        # instantiated classes and breaks nodeid parity).
        "PYTORCH_TESTING_DEVICE_ONLY_FOR": device_env,
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
    npu_device_id: Optional[int],
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


# ==============================================================================
# NPU Task Queue Poisoning & Hardware Error Detection
# ==============================================================================
#
# When an aclnn operator fails fatally (operator bug like aclnnRepeat 0-dim,
# CANN OOM, or hardware AiCore exception), the NPU device context becomes
# poisoned. The NPU task queue (NPUQueue.cpp) transitions to CAN_EXIT state:
# all subsequent operators become silent no-ops that produce garbage data
# (uninitialized device memory). This poisons every remaining test case in
# the same worker process.
#
# Detection: check each case's output for fatal signatures. The poisoning
# case itself throws a RuntimeError containing one of these strings. After
# detection, the worker exits with a special exit code; the parent restarts
# a fresh worker for remaining cases.
#
# Two categories of fatal signatures:
#   1. "The process exits for this inner error":
#      NPUQueue ERROR_EXIT throw path (operator bugs, OOM). NOT produced
#      by hardware errors (UCE/ECC go through deviceErrorMap).
#   2. Hardware errors (EZ9999, EE9999, vector core exception):
#      AiCore faults that corrupt device state beyond recovery.

NPU_QUEUE_FATAL_EXIT_CODE = 70

# Layer 1: Fatal error signatures (string matching)
#
# Two categories:
#   A. NPUQueue ERROR_EXIT throw path — produces "The process exits for
#      this inner error" (covers operator bugs like aclnnRepeat, CANN OOM).
#   B. deviceErrorMap throw path (NPUQueue.cpp:175-183 ThrowDeviceError) —
#      produces device error labels for hardware faults. These bypass
#      ERROR_EXIT and go directly to UCE_EXIT / HBM_ECC_EXIT / etc.,
#      so they do NOT contain the "process exits" signature.
#
# Note: EZ9999 / EE9999 / EZ1009 are CANN runtime error codes that appear
# in the exception message alongside the deviceErrorMap labels. They are
# kept for additional robustness but are partially redundant with category B.
NPU_QUEUE_FATAL_SIGNATURES = [
    # A. NPUQueue ERROR_EXIT (operator bugs, OOM)
    "The process exits for this inner error",
    # B. deviceErrorMap labels (hardware faults, NPUQueue.cpp:175-183)
    "UCE ERROR",
    "HBM MULTI BIT ECC ERROR",
    "SUSPECT MEM ERROR",
    "HCCS LINK ERROR",
    "HCCL OP RETRY FAILED",
    "SUSPECT REMOTE ERROR",
    # C. CANN runtime error codes (redundant with B, kept for robustness)
    "EZ9999",
    "EE9999",
    "EZ1009",
]


def _check_fatal_npu_error(
    status: str,
    message: str,
    stdout: str,
    stderr: str,
) -> bool:
    """Layer 1: Check if a case result contains a known fatal NPU error signature."""
    if status not in ("failed", "error"):
        return False
    combined = (message or "") + "\n" + (stdout or "") + "\n" + (stderr or "")
    return any(sig in combined for sig in NPU_QUEUE_FATAL_SIGNATURES)


def _check_npu_poisoned() -> bool:
    """
    Layer 2: Probe NPU device health by running a trivial computation.

    When the NPU task queue is in CAN_EXIT state (poisoned by a prior fatal
    error), all operators become silent no-ops — NPUQueue.cpp:573-596 Enqueue()
    returns without executing, and output tensors contain uninitialized device
    memory. This probe creates a tensor with a known value and verifies the
    result. If the queue is poisoned, the result will be garbage. If the
    device is in error state, the sync will throw.

    This catches poisoning paths that signature matching misses (e.g. new
    CANN error formats, SUSPECT REMOTE ERROR without EZ9999 prefix, etc.).

    Cost: ~1ms per call. Only invoked on failed/error cases where Layer 1
    did not match, so zero overhead for passing tests.
    """
    try:
        import torch
        # torch.ones on NPU calls fill_ -> EXEC_NPU_CMD(aclnnInplaceFillScalar)
        # -> Enqueue -> no-op if CAN_EXIT. sum() similarly no-op.
        # .item() triggers MakeSureQueueEmpty (sync), which does NOT throw
        # in CAN_EXIT state (runtime_error left empty, NPUQueue.cpp:313-315),
        # so it returns garbage from uninitialized device memory.
        probe = torch.ones(4, device="npu")
        return probe.sum().item() != 4.0
    except Exception:
        # Any exception (sync timeout, device error, OOM) means the device
        # context is unhealthy and the worker should be restarted.
        return True


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
    npu_device_id: Optional[int],
    result_aggregator: ConcurrentResultAggregator,
    progress_tracker: ProgressTracker,
    log_queue: Queue,
    console: NonBlockingConsole,
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

        attempt_completed = set()
        # Set when THIS attempt's poll loop exits: tells the attempt's reader
        # thread to stop processing further lines (reconciliation owns result
        # recovery from disk from that point; a stale reader must not keep
        # aggregating into the shared trackers).
        reader_done = threading.Event()
        # Reset per-attempt watchdog/forensics state; assigned after Popen.
        # Stays None if the worker never started (Popen failure path).
        worker_state = None

        try:
            # Inside the try on purpose: a full disk raises OSError here, and
            # the handler below turns it into per-case error results (keeping
            # cases.json complete) instead of escaping with the batch's cases
            # missing from the aggregator entirely.
            batch_input_file.write_text(json.dumps(batch_input, indent=2), encoding="utf-8")

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
                # Own session/process group: lets the idle watchdog and crash
                # cleanup take down the whole worker tree (worker + leaked
                # grandchildren such as inductor compile workers) with one
                # os.killpg(), instead of leaving orphans holding the stdout
                # pipe open and the NPU devices busy.
                start_new_session=True,
            )

            # Watchdog + forensics state, fresh per worker instance. A fresh
            # container per Popen is essential: reader threads of previously
            # killed workers may keep draining leaked-grandchild output
            # forever, and must never be able to refresh THIS worker's idle
            # timer or pollute its forensics.
            worker_state = {
                "last_heartbeat": monotonic(),
                # Bounded tail of raw pipe lines = crash/hang scene
                "noise": deque(maxlen=NOISE_TAIL_LINES),
                "lock": threading.Lock(),
                "device_id": npu_device_id,
            }

            # Late-binding guard: the reader must capture THIS attempt's
            # objects BY VALUE. A plain closure shares the enclosing cells,
            # so after a restart rebinds proc / worker_state /
            # attempt_completed, a still-running old reader (alive because a
            # grandchild that escaped the killed process group keeps the old
            # pipe open) would read the OLD pipe but write into the NEW
            # worker's noise deque, refresh its heartbeat, and pollute its
            # attempt set. Default arguments bind at def time, isolating
            # each attempt.
            def _read_stdout(
                proc=proc,
                worker_state=worker_state,
                attempt_completed=attempt_completed,
                reader_done=reader_done,
            ):
                if proc.stdout:
                    for raw_line in proc.stdout:
                        if reader_done.is_set():
                            # Attempt is over (worker exited or was killed):
                            # late lines are dropped; the reconciliation
                            # passes recover results from on-disk evidence.
                            return
                        line = raw_line.rstrip("\n")
                        if not line:
                            continue
                        # Keep every raw line (truncated) as crash evidence.
                        with worker_state["lock"]:
                            worker_state["noise"].append(
                                line[:NOISE_TAIL_LINE_MAX_CHARS]
                            )

                        line = line.strip()
                        if not line:
                            continue

                        # Only structured worker output refreshes the idle
                        # watchdog: a JSON result line (case finished) or the
                        # worker's own "[N] Starting: ..." marker (case
                        # started). Native C/C++ output and grandchild
                        # processes (e.g. inductor compile workers) write
                        # straight to fd 1/2 — that noise must NOT keep a
                        # hung worker alive.
                        try:
                            case_result = json.loads(line)
                        except json.JSONDecodeError:
                            if _is_case_start_marker(line):
                                worker_state["last_heartbeat"] = monotonic()
                            continue
                        if not isinstance(case_result, dict) or "nodeid" not in case_result:
                            # JSON-looking noise (e.g. grandchild output)
                            continue
                        worker_state["last_heartbeat"] = monotonic()

                        nodeid = case_result.get("nodeid", "")
                        status = case_result.get("status", "error")
                        duration = case_result.get("duration", 0.0)
                        message = case_result.get("message", "")

                        full_result = {
                            "nodeid": nodeid,
                            "status": status,
                            "duration": duration,
                            "returncode": int(case_result.get("returncode", 1)),
                            "message": message,
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
                            "message": message[:200],
                        })
                        attempt_completed.add(nodeid)

            reader_thread = threading.Thread(target=_read_stdout, daemon=True)
            reader_thread.start()

            idle_timeout = timeout + 30
            timeout_occurred = False

            while True:
                returncode = proc.poll()
                if returncode is not None:
                    # Worker gone (normal exit, os._exit, or coredump): any
                    # surviving process-group member is a leaked grandchild
                    # (compile workers, mp children) still holding the pipe
                    # or NPU devices — take the whole group down before
                    # reporting/restarting. Kill before joining the reader
                    # so the pipe EOFs and the reader thread can exit.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    reader_thread.join(timeout=10)
                    # After the join: a still-alive reader is stuck on a
                    # pipe held outside the killed group. Flag the attempt
                    # closed so it exits at its next line instead of
                    # double-counting results the reconciliation will
                    # already have recovered.
                    reader_done.set()
                    break

                if monotonic() - worker_state["last_heartbeat"] > idle_timeout:
                    timeout_occurred = True
                    hung_duration = monotonic() - worker_state["last_heartbeat"]
                    # Recovery ordering is deliberate and MUST NOT start
                    # with console output: a wedged stdout (full pipe behind
                    # a stuck tee/runner) blocks a leading print forever and
                    # smothers the entire recovery chain. Act first, notify
                    # last:
                    # 1) ask the worker for an all-thread stack dump
                    #    (faulthandler registered in _worker_main; works even
                    #    when the GIL is held by a hung native call — exactly
                    #    where pytest-timeout's SIGALRM method cannot fire —
                    #    and it writes straight to the stack_dumps file on
                    #    disk, immune to both pytest fd capture and stdout
                    #    state)
                    # 2) take down the whole process group (worker + leaked
                    #    grandchildren still holding the pipe or devices)
                    # 3) reap the worker, join the reader
                    # 4) notify the console (best effort, never required)
                    try:
                        proc.send_signal(signal.SIGUSR1)
                    except (ProcessLookupError, OSError):
                        pass
                    sleep(HEARTBEAT_DUMP_WAIT_SECONDS)
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()
                    try:
                        returncode = proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        returncode = -9
                    reader_thread.join(timeout=10)
                    reader_done.set()
                    console.print(
                        f"  [Batch {batch_id}] Idle timeout ({hung_duration:.0f}s "
                        f"without worker progress); stack dump requested, "
                        f"worker group killed"
                    )
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
                    hang_msg = (
                        f"Case hung (no worker progress for "
                        f"{hung_duration:.0f}s); case log carries the worker "
                        f"stdout tail and the faulthandler stack dump"
                    )
                    timeout_result = {
                        "nodeid": hung_case.nodeid,
                        "status": "timeout",
                        "duration": hung_duration,
                        "returncode": -9,
                        "message": hang_msg,
                        "command": "",
                        "file": hung_case.test_file,
                        "case_idx": hung_case.case_idx,
                    }
                    # Record order: aggregate -> persist forensics to disk ->
                    # progress bookkeeping. The crash log must be on disk
                    # before any call that could block on a wedged stdout.
                    result_aggregator.add_case_result(timeout_result)
                    _save_crash_case_log(
                        report_dir, shard, shard_type, hung_case,
                        "timeout", hung_duration, -9, hang_msg, worker_state,
                        stack_dump=_read_worker_stack_dump(report_dir, batch_id),
                    )
                    progress_tracker.mark_completed(
                        hung_case.nodeid, "timeout", hung_duration
                    )
                    completed_nodeids.add(hung_case.nodeid)
                    remaining_cases = not_reported[1:]
                else:
                    remaining_cases = []

                if remaining_cases:
                    console.print(
                        f"  [Batch {batch_id}] Continuing with "
                        f"{len(remaining_cases)} remaining cases..."
                    )
                continue

            # NPU task queue poisoning or hardware device error: worker
            # detected the fatal condition and exited cleanly between
            # cases (os._exit(NPU_QUEUE_FATAL_EXIT_CODE)). Unlike coredump,
            # no case was in progress — restart for ALL unreported cases
            # without sacrificing any.
            if returncode == NPU_QUEUE_FATAL_EXIT_CODE:
                console.print(
                    f"  [Batch {batch_id}] NPU fatal error detected by worker,"
                    f" restarting for {len(not_reported)} remaining cases..."
                )
                remaining_cases = not_reported
                if remaining_cases:
                    console.print(
                        f"  [Batch {batch_id}] Continuing with "
                        f"{len(remaining_cases)} remaining cases..."
                    )
                continue

            if returncode < 0:
                signal_num = -returncode
                try:
                    signal_name = signal.Signals(signal_num).name
                except (ValueError, AttributeError):
                    signal_name = f"signal {signal_num}"

                if not_reported:
                    crashed_case = not_reported[0]
                    crash_msg = (
                        f"Worker killed by signal ({signal_name}); case log "
                        f"carries the worker stdout tail captured by the parent"
                    )
                    error_result = {
                        "nodeid": crashed_case.nodeid,
                        "status": "error",
                        "duration": 0.0,
                        "returncode": returncode,
                        "message": crash_msg,
                        "command": "",
                        "file": crashed_case.test_file,
                        "case_idx": crashed_case.case_idx,
                    }
                    # Record order: aggregate -> persist forensics to disk ->
                    # progress bookkeeping -> notify (see the idle-timeout
                    # path for the rationale).
                    result_aggregator.add_case_result(error_result)
                    _save_crash_case_log(
                        report_dir, shard, shard_type, crashed_case,
                        "error", 0.0, returncode, crash_msg, worker_state,
                    )
                    progress_tracker.mark_completed(
                        crashed_case.nodeid, "error", 0.0
                    )
                    completed_nodeids.add(crashed_case.nodeid)
                    remaining_cases = not_reported[1:]
                else:
                    remaining_cases = []

                console.print(
                    f"  [Batch {batch_id}] Worker coredump ({signal_name})"
                )
                if remaining_cases:
                    console.print(
                        f"  [Batch {batch_id}] Continuing with "
                        f"{len(remaining_cases)} remaining cases..."
                    )
                continue

            # Normal exit: all cases processed
            # A worker reports one JSON result line per case via stdout,
            # but that channel is shared with native (C++) output from the
            # test process, so a line can be corrupted and silently dropped
            # by the reader. Before synthesizing errors, reconcile every
            # missing case against on-disk evidence, in order:
            #   1. batch_results_{batch_id}.json — written by the worker
            #      just before exit, holds the authoritative per-case dict
            #   2. junit_xmls/{prefix}-{shard}_{case_idx}_*.xml — written
            #      by pytest for every case that actually ran
            # Only cases with no evidence anywhere become synthetic errors.
            results_file = report_dir / f"batch_results_{batch_id}.json"
            fallback_results = []
            if results_file.exists():
                try:
                    fallback_results = json.loads(
                        results_file.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError):
                    fallback_results = []

            remaining = [
                t for t in batch if t.nodeid not in completed_nodeids
            ]

            # Pass 1: recover missing results from the worker's
            # batch results file
            if remaining and fallback_results:
                fallback_by_nodeid = {
                    cr.get("nodeid", ""): cr for cr in fallback_results
                }
                still_missing = []
                recovered = 0
                for task in remaining:
                    cr = fallback_by_nodeid.get(task.nodeid)
                    if cr is None:
                        still_missing.append(task)
                        continue
                    full_result = {
                        "nodeid": cr.get("nodeid", task.nodeid),
                        "status": cr.get("status", "error"),
                        "duration": cr.get("duration", 0.0),
                        "returncode": int(cr.get("returncode", 1)),
                        "message": cr.get("message", ""),
                        "command": cr.get("command", ""),
                        "file": cr.get("file", task.test_file),
                        "case_idx": int(cr.get("case_idx", task.case_idx)),
                        "recovered": "batch_results",
                    }
                    result_aggregator.add_case_result(full_result)
                    progress_tracker.mark_completed(
                        full_result["nodeid"],
                        full_result["status"],
                        full_result["duration"],
                    )
                    completed_nodeids.add(task.nodeid)
                    recovered += 1
                if recovered:
                    console.print(
                        f"  [Batch {batch_id}] Recovered {recovered} case "
                        f"result(s) from batch results file "
                        f"(stdout result line lost)"
                    )
                remaining = still_missing

            # Pass 2: recover missing results from JUnit XML files on disk
            if remaining:
                still_missing = []
                recovered = 0
                for task in remaining:
                    full_result = _recover_result_from_junit_xml(
                        report_dir, shard, shard_type, timeout, task
                    )
                    if full_result is None:
                        still_missing.append(task)
                        continue
                    result_aggregator.add_case_result(full_result)
                    progress_tracker.mark_completed(
                        full_result["nodeid"],
                        full_result["status"],
                        full_result["duration"],
                    )
                    completed_nodeids.add(task.nodeid)
                    recovered += 1
                if recovered:
                    console.print(
                        f"  [Batch {batch_id}] Recovered {recovered} case "
                        f"result(s) from JUnit XML files "
                        f"(stdout result line lost)"
                    )
                remaining = still_missing

            # Pass 3: no on-disk evidence — synthetic error
            if remaining:
                console.print(
                    f"  [Batch {batch_id}] {len(remaining)} cases missing "
                    f"results (normal exit, no on-disk evidence), "
                    f"marking as error"
                )
                for task in remaining:
                    error_result = {
                        "nodeid": task.nodeid,
                        "status": "error",
                        "duration": 0.0,
                        "returncode": 1,
                        "message": (
                            "No result produced (worker exited normally; "
                            "no junit xml or batch results found)"
                        ),
                        "command": "",
                        "file": task.test_file,
                        "case_idx": task.case_idx,
                        "synthetic": True,
                    }
                    result_aggregator.add_case_result(error_result)
                    progress_tracker.mark_completed(
                        task.nodeid, "error", 0.0
                    )
            break

        except Exception as e:
            # Kill first (a wedged stdout must not block cleanup), then
            # record, then notify. worker_state is only set after a
            # successful Popen, so proc is bound whenever it is not None.
            if worker_state is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            for task in remaining_cases:
                if task.nodeid not in completed_nodeids:
                    failure_msg = f"Worker failure: {str(e)[:200]}"
                    error_result = {
                        "nodeid": task.nodeid,
                        "status": "error",
                        "duration": 0.0,
                        "returncode": 1,
                        "message": failure_msg,
                        "command": "",
                        "file": task.test_file,
                        "case_idx": task.case_idx,
                    }
                    result_aggregator.add_case_result(error_result)
                    if worker_state is not None:
                        _save_crash_case_log(
                            report_dir, shard, shard_type, task,
                            "error", 0.0, 1, failure_msg, worker_state,
                        )
                    progress_tracker.mark_completed(task.nodeid, "error", 0.0)
            console.print(
                f"  [Batch {batch_id}] Worker execution failed: {str(e)[:200]}"
            )
            break

    # Cleanup temp file
    batch_input_file.unlink(missing_ok=True)
    results_file = report_dir / f"batch_results_{batch_id}.json"
    results_file.unlink(missing_ok=True)
    # Remove the stack-dump scratch file (normally already consumed by the
    # idle-timeout path; this covers the no-hang / header-only leftovers).
    (report_dir / "stack_dumps" / f"batch_{batch_id}_stack.txt").unlink(missing_ok=True)


def _worker_main(worker_input_file: str) -> None:
    """
    Worker entry point. Called via:
        python run_npu_test_shard.py --worker <batch_input.json>

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

    # Forensics: the parent sends SIGUSR1 right before killing a hung
    # worker; faulthandler then dumps all thread stacks — it works even
    # when the GIL is held by a stuck native call (CANN sync, C++
    # deadlock), exactly where pytest-timeout's SIGALRM method cannot
    # fire. The dump goes to a dedicated per-batch file rather than the
    # stdout pipe: (1) a worker with a large thread pool (OMP/interop
    # threads on a many-core NPU host) can produce more dump lines than
    # the parent's bounded pipe tail, and faulthandler prints the main
    # (hung) thread FIRST — exactly the lines a bounded tail would evict;
    # (2) a plain file fd also sidesteps pytest's fd-level capture, which
    # dup2's a temp file over fd 2 and would swallow dumps registered
    # against the default sys.stderr. The parent reads the file after the
    # kill (_read_worker_stack_dump); each worker restart reopens it in
    # write mode, so the content always belongs to the current attempt.
    try:
        dump_dir = report_dir / "stack_dumps"
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_file = open(
            dump_dir / f"batch_{batch_id}_stack.txt", "w", encoding="utf-8"
        )
        dump_file.write(f"# SIGUSR1 stack dump target, worker pid {os.getpid()}\n")
        dump_file.flush()
        faulthandler.register(signal.SIGUSR1, file=dump_file, all_threads=True)
    except (ValueError, OSError):
        pass

    # Apply environment
    for key, value in env_updates.items():
        os.environ[key] = value
    if npu_device_id is not None:
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

    all_results = []

    for case in cases:
        original_nodeid = case["nodeid"]
        case_nodeid = original_nodeid
        if case_nodeid.startswith("test/"):
            case_nodeid = case_nodeid[5:]

        # Generate XML filename
        prefix = {"distributed": "dist", "core": "core", "tensor": "tensor",
                  "graph": "graph", "others": "others",
                  "regular": "reg", "custom": "custom"}.get(shard_type, "reg")
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
        except BaseException as e:
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

        # Check for NPU fatal errors — exit worker immediately to trigger
        # process-level restart for remaining cases.
        if status in ("failed", "error"):
            poisoned = _check_fatal_npu_error(
                status, message, captured_stdout, captured_stderr
            )
            if not poisoned:
                poisoned = _check_npu_poisoned()
            if poisoned:
                # NPU fatal: save result and exit
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
                print(json.dumps(case_result, ensure_ascii=False), flush=True)
                print(
                    f"[{case['case_idx']}] NPU fatal error detected, "
                    f"exiting worker (code {NPU_QUEUE_FATAL_EXIT_CODE}) "
                    f"to trigger restart for remaining cases",
                    flush=True,
                )
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(NPU_QUEUE_FATAL_EXIT_CODE)

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


def _write_console_stall_report(report_dir: Path, console: NonBlockingConsole) -> None:
    """Record that the step's stdout stopped draining before process exit."""
    stats = console.stats()
    try:
        (report_dir / "CONSOLE_STALLED.txt").write_text(
            "Step stdout stopped draining: the console drain thread is still\n"
            "blocked in a write after the run finished. All test results are\n"
            "complete on disk (see cases.json / stats / info files); only live\n"
            "console output was affected.\n"
            f"dropped={stats['dropped']} pending={stats['pending']}\n",
            encoding="utf-8",
        )
    except OSError:
        pass


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
    quiet: bool = False,
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

    # Print summary (skipped when stdout is known-wedged: print_stats_summary
    # prints directly and would hang the step after results are already safe
    # on disk)
    if not quiet:
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
        help="Maximum concurrent workers for regular tests (default: 4). Each worker handles one batch of cases.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick-test", type=int, default=None, help="Quick test mode: execute only N cases for fast verification (default: None, run all cases)")
    parser.add_argument(
        "--device-env",
        default="privateuse1",
        help="Comma-separated device types exported as "
             "PYTORCH_TESTING_DEVICE_ONLY_FOR during execution "
             "(default: privateuse1). Must match the value used at collection "
             "time so collected nodeids exist when tests run.",
    )
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

        # Check for distributed test files: if any exist, run ALL cases as
        # distributed (serial, no NPU binding). Otherwise run as regular
        # (concurrent, NPU round-robin binding).
        has_distributed = has_distributed_test_files(planned_tests)
        if has_distributed:
            shard_type = "distributed"
            effective_workers = 1
            execution_mode = "serial"
        else:
            shard_type = "regular"
            effective_workers = args.max_workers
            execution_mode = "concurrent"

        print(f"Test files specified: {len(planned_tests)}")
        print(f"Test directory: {test_dir}")
        print(f"Test type: {shard_type}")
        print(f"Execution mode: {execution_mode} ({effective_workers} workers, pytest.main() per case, batched by file)")
        if has_distributed:
            distributed_files = [f for f in planned_tests if f.startswith("test/distributed/")]
            print(f"  Distributed files: {len(distributed_files)}")
            for df in distributed_files:
                print(f"    - {strip_test_prefix_and_suffix(df)}")
        if args.disabled_testcases:
            disabled_count = result_module.load_disabled_testcases_count(args.disabled_testcases)
            print(f"Disabled testcase entries: {disabled_count}")
        print(f"\n{'=' * 80}\n")

        for index, target in enumerate(planned_tests, 1):
            display_name = strip_test_prefix_and_suffix(target)
            is_dist = target.startswith("test/distributed/")
            dist_marker = " [distributed]" if is_dist else ""
            print(f"  [{index:03d}] {display_name}{dist_marker}")

        # Create info dict for custom mode
        info = result_module.create_shard_info(shard, num_shards, timestamp)
        info["selection_mode"] = "custom_files"
        info["shard_type"] = shard_type
        info["shard_files"] = len(planned_tests)
        info["total_files"] = len(planned_tests)
        info["selected_test_files"] = len(planned_tests)
        info["has_distributed_files"] = has_distributed
        info["execution_mode"] = execution_mode
        if args.disabled_testcases:
            info["disabled_count"] = result_module.load_disabled_testcases_count(args.disabled_testcases)

        # Save test plan
        result_module.save_test_plan_file(str(report_dir), shard, planned_tests, shard_type)

        # Clean old files
        clean_existing_junit_xml(report_dir)
        result_module.get_shard_log_file(report_dir, shard, shard_type).unlink(missing_ok=True)

        # Build execution env
        env_updates = build_execution_env(
            test_dir, script_dir, args.disabled_testcases, shard, shard_type,
            args.device_env,
        )

        # Execute tests (custom mode: auto-detect distributed files for execution mode)
        cases_list = []
        if planned_tests:
            # Phase 1: Collect all test cases using collect_all_cases module
            print("\nPhase 1: Collecting test cases...")
            error_log_dir = report_dir / "collection_errors"
            # Per-file JSONL nodeid dumps are transient channel state; a
            # temp dir keeps them out of the uploaded test-reports artifact.
            with tempfile.TemporaryDirectory(prefix="nodeid_dumps_") as dump_root:
                collected_cases, _nodeid_meta = collect_all_cases.collect_all_cases(
                    planned_tests,
                    test_dir,
                    error_log_dir,
                    dump_dir=Path(dump_root),
                    parallel=16,  # 16 parallel collectors balance speed vs resource usage
                )

            # Apply quick_test limit if specified
            if args.quick_test and len(collected_cases) > args.quick_test:
                collected_cases = collected_cases[:args.quick_test]
                print(f"  Quick test mode: using only {args.quick_test} cases")

            total_cases = len(collected_cases)
            print(f"\nPhase 2: Executing {total_cases} cases with {effective_workers} workers")

            # Build CaseExecutionTask list
            tasks = []
            for i, case in enumerate(collected_cases, 1):
                tasks.append(CaseExecutionTask(
                    case_idx=i,
                    nodeid=case["nodeid"],
                    test_file=case["file"],
                ))

            # Phase 2: Execute cases using run_tests_with_tasks_concurrent
            # Use effective_workers (1 for distributed files, args.max_workers otherwise)
            # Note: quick_test already applied above, pass None to avoid redundant check
            console = NonBlockingConsole()
            console.start()
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
                None,  # quick_test already applied above
                console=console,
            )
            info["per_case_isolation"] = True
            info["concurrent_workers"] = effective_workers
            info["returncode"] = returncode
            info["duration"] = duration
        else:
            returncode = 0
            duration = 0.0
            console = None

        # Stdout health gate: if the console drain thread is still blocked in
        # a write here, stdout is wedged — persist everything quietly and
        # exit hard instead of hanging on the final direct-print summary for
        # the remaining job-timeout budget. Results are already on disk and
        # artifacts upload via the if: always() steps.
        if console is not None and not console.finish(timeout=30):
            _write_console_stall_report(report_dir, console)
            save_results_and_summary(
                result_module=result_module,
                report_dir=report_dir,
                shard=shard,
                shard_type=shard_type,
                cases_list=cases_list,
                duration=duration,
                returncode=returncode,
                info=info,
                execution_mode=execution_mode,
                concurrent_workers=effective_workers,
                has_distributed_files=has_distributed,
                quiet=True,
            )
            os._exit(0)

        # Save results and print summary
        save_results_and_summary(
            result_module=result_module,
            report_dir=report_dir,
            shard=shard,
            shard_type=shard_type,
            cases_list=cases_list,
            duration=duration,
            returncode=returncode,
            info=info,
            execution_mode=execution_mode,
            concurrent_workers=effective_workers,
            has_distributed_files=has_distributed,
        )

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
            print(f"Execution mode: SERIAL (pytest.main() per case, batched by file)")
        else:
            print(f"Execution mode: CONCURRENT ({args.max_workers} workers, pytest.main() per case, batched by file)")

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
        result_module.get_shard_log_file(report_dir, shard, shard_type).unlink(missing_ok=True)

        # Build execution env
        env_updates = build_execution_env(
            test_dir, script_dir, args.disabled_testcases, shard, shard_type,
            args.device_env,
        )

        # Convert cases to CaseExecutionTask format
        tasks = []
        for i, case in enumerate(planned_cases, 1):
            tasks.append(CaseExecutionTask(
                case_idx=i,
                nodeid=case["nodeid"],
                test_file=case.get("file", ""),
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
            console = NonBlockingConsole()
            console.start()
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
                console=console,
            )
            info["execution_mode"] = "serial" if effective_workers == 1 else "concurrent"
            info["concurrent_workers"] = effective_workers

        else:
            print("No cases to execute.")
            returncode = 0
            duration = 0.0
            console = None

        # Stdout health gate: if the console drain thread is still blocked in
        # a write here, stdout is wedged — persist everything quietly and
        # exit hard instead of hanging on the final direct-print summary for
        # the remaining job-timeout budget. Results are already on disk and
        # artifacts upload via the if: always() steps.
        if console is not None and not console.finish(timeout=30):
            _write_console_stall_report(report_dir, console)
            save_results_and_summary(
                result_module=result_module,
                report_dir=report_dir,
                shard=shard,
                shard_type=shard_type,
                cases_list=cases_list,
                duration=duration,
                returncode=returncode,
                info=info,
                quiet=True,
            )
            os._exit(0)

        # Save results and print summary
        save_results_and_summary(
            result_module=result_module,
            report_dir=report_dir,
            shard=shard,
            shard_type=shard_type,
            cases_list=cases_list,
            duration=duration,
            returncode=returncode,
            info=info,
        )

        # Exit with 0 to allow step to succeed and report generation to proceed
        # The actual test results are recorded in cases.json
        sys.exit(0)

    # No valid mode specified (should not reach here due to argument validation)
    print("ERROR: Either --test-files or --cases-json must be specified")
    sys.exit(1)


if __name__ == "__main__":
    main()
