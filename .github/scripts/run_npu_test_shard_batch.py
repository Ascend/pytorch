#!/usr/bin/env python3
"""
Run PyTorch NPU tests via merged-batch pytest execution.

Simplified batch executor used by the manual build-and-test pipeline
(pytorch_ci_trigger_manual.yml -> _test.yml -> _test-category.yml with
batch_execution: true). Reuses the batch planning and reporting helpers
from run_npu_test_shard.py, which stays untouched for all other flows.

Execution model (differs from run_npu_test_shard.py on purpose):
    - Same-file cases are grouped into batches (max 100 per batch, sorted
      by nodeid — sort_and_batch_tasks from run_npu_test_shard)
    - Each batch runs in ONE pytest.main() call with all nodeids plus a
      single batch-level --junitxml, so the expensive module collection
      (test_meta.py alone instantiates 34k+ items) happens once per batch
      instead of once per case (~10x faster on collection-bound shards)
    - Each batch is one worker subprocess (crash isolation boundary:
      a crashed batch marks ALL its cases as error, never restarts)
    - No NPU poisoner checks and no multi-layer result recovery: the
      manual pipeline runs whitelisted cases; failures are root-caused
      manually, so a poisoned device surfaces as subsequent failures

Timeout layers:
    1. pytest-timeout (signal method, default): a single case exceeding
       --timeout is marked failed and the batch continues
    2. Batch idle watchdog (parent side): no worker stdout for
       --timeout + 120s (covers collection phase + one silent case) ->
       kill the worker, mark the whole batch as error

Outputs (schema-identical to run_npu_test_shard.py):
    - shard_<prefix>-<n>_cases.json / _stats.json / _info.json
    - junit_xmls/<prefix>-<n>_batch_<id>.xml  (one per batch)
    - cases_logs/<prefix>-<n>_batch_<id>.log  (one per batch, full pytest output)

Usage:
    python run_npu_test_shard_batch.py \
        --cases-json core_cases_shard_1.json \
        --test-dir /path/to/pytorch/test \
        --report-dir test-reports \
        --max-workers 32 \
        --timeout 1200 \
        --verbose
"""

import argparse
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
from time import monotonic, sleep
from typing import Dict, List, Optional, Tuple

import run_npu_test_shard as runner


# Batch size and watchdog margins. The idle watchdog must tolerate the
# collection phase plus one long silent case (pytest -v only prints a line
# when a case finishes), hence the +120s margin over the per-case timeout.
MAX_CASES_PER_BATCH = 100
IDLE_TIMEOUT_MARGIN = 120

SHARD_PREFIXES = {
    "distributed": "dist", "core": "core", "tensor": "tensor",
    "graph": "graph", "others": "others",
    "regular": "reg", "custom": "custom",
}


def _prefix_for(shard_type: str) -> str:
    return SHARD_PREFIXES.get(shard_type, "reg")


def _batch_xml_path(report_dir: Path, shard: int, shard_type: str, batch_id: int) -> Path:
    return report_dir / "junit_xmls" / f"{_prefix_for(shard_type)}-{shard}_batch_{batch_id:04d}.xml"


def _batch_log_path(report_dir: Path, shard: int, shard_type: str, batch_id: int) -> Path:
    return report_dir / "cases_logs" / f"{_prefix_for(shard_type)}-{shard}_batch_{batch_id:04d}.log"


# ==============================================================================
# Batch JUnit XML Parsing
# ==============================================================================


def _result_from_testcase(testcase) -> Dict:
    """
    Derive {status, message} from one <testcase> element.

    Mirrors the semantics of runner.parse_junit_xml_status (which only
    handles single-case XML files): xfail skips count as passed,
    <failure> as failed, <error> as error, plain <skipped> as skipped.
    """
    skipped = testcase.find("skipped")
    if skipped is not None:
        if skipped.get("type", "") == "pytest.xfail":
            return {"status": "passed", "message": "xfailed: expected failure"}
        msg = skipped.get("message", "") or ""
        text = (skipped.text or "").strip()
        if text:
            msg = f"{msg}\n{text}" if msg else text
        return {"status": "skipped", "message": msg}

    failure = testcase.find("failure")
    if failure is not None:
        msg = failure.get("message", "") or ""
        text = (failure.text or "").strip()
        if text:
            msg = f"{msg}\n{text}" if msg else text
        return {"status": "failed", "message": msg}

    error = testcase.find("error")
    if error is not None:
        msg = error.get("message", "") or ""
        text = (error.text or "").strip()
        if text:
            msg = f"{msg}\n{text}" if msg else text
        return {"status": "error", "message": msg}

    return {"status": "passed", "message": ""}


def _expected_classname(file_part: str, classes: List[str]) -> str:
    """
    Build the junit classname pytest derives from a nodeid.

    pytest's junitxml mangles classname as the file path relative to the
    rootdir (dirs and module name joined by dots) plus the class segments,
    e.g. "test/test_meta.py::TestX::test_a" -> "test.test_meta.TestX".
    The rootdir is normally the pytorch repo root (pytest.ini there), so
    the classname carries the "test." prefix even though the nodeids we
    pass to pytest are relative to the test dir. Both variants are
    registered by the caller to stay robust either way.
    """
    dotted = file_part
    if dotted.endswith(".py"):
        dotted = dotted[:-3]
    dotted = dotted.replace("/", ".").replace("\\", ".")
    if classes:
        dotted += "." + ".".join(classes)
    return dotted


def _match_batch_results(
    xml_file: Path,
    batch: List[runner.CaseExecutionTask],
) -> Dict[str, Dict]:
    """
    Match <testcase> elements of a batch XML back to batch cases.

    Builds a {(classname, name) -> task} index from the batch nodeids
    (same file per batch, unique nodeids -> unique (classname, name)),
    then maps every testcase to its case. Returns {original_nodeid:
    {status, message, duration}}; cases absent from the XML are simply
    missing from the dict.
    """
    index: Dict[Tuple[str, str], runner.CaseExecutionTask] = {}
    for task in batch:
        parts = task.nodeid.split("::")
        file_part, classes, name = parts[0], parts[1:-1], parts[-1]
        stripped_file = file_part[5:] if file_part.startswith("test/") else file_part
        for cls in {
            _expected_classname(file_part, classes),
            _expected_classname(stripped_file, classes),
        }:
            index[(cls, name)] = task

    results: Dict[str, Dict] = {}
    tree = ET.parse(str(xml_file))
    for testcase in tree.getroot().iter("testcase"):
        key = (testcase.get("classname", ""), testcase.get("name", ""))
        task = index.get(key)
        if task is None:
            continue
        result = _result_from_testcase(testcase)
        result["duration"] = float(testcase.get("time", 0) or 0)
        results[task.nodeid] = result
    return results


# ==============================================================================
# NPU Canary Plugin (per-case poisoning diagnostics)
# ==============================================================================
#
# Merged-batch workers run many cases in ONE pytest process. When an aclnn
# operator fails fatally (e.g. MirrorPad/TopKV2 tiling failure on boundary
# inputs), the NPU task queue (NPUQueue.cpp) transitions to CAN_EXIT:
# subsequent operators become silent no-ops that return garbage from
# uninitialized device memory (see run_npu_test_shard.py::_check_npu_poisoned
# for the per-case runner's equivalent probe). The batch runner has no
# per-process isolation, so one such failure poisons every remaining case in
# the batch while the poisoning case itself often still reports PASSED.
#
# This plugin probes process health after EVERY case so the poisoning case
# can be identified by name. Diagnostics only: it prints and records
# findings, never aborts the batch.
#
# Output:
#   - stdout: one "[NPU-CANARY]" line per detection (goes to the batch log)
#   - report_dir/npu_canary_<shard_type><shard>_b<batch_id>.jsonl: records
#
# Disable with NPU_CANARY=0.


class NpuCanaryPlugin:
    """Probe NPU process health around each test case (diagnostics only)."""

    def __init__(self, report_dir, shard, shard_type, batch_id):
        self.report_dir = Path(report_dir)
        self.shard = shard
        self.shard_type = shard_type
        self.batch_id = batch_id
        self.enabled = os.environ.get("NPU_CANARY", "1") != "0"
        self.poisoned = False
        self._torch = None
        self._marker_path = self.report_dir / (
            f"npu_canary_{shard_type}{shard}_b{batch_id}.jsonl"
        )

    def _probe(self):
        """Return True if the process NPU state is still healthy.

        Mirrors run_npu_test_shard.py::_check_npu_poisoned: a trivial
        computation whose result is verified. In CAN_EXIT state operators
        are silent no-ops, so the result is garbage (or the sync throws).
        """
        if self._torch is None:
            import torch

            self._torch = torch
        try:
            probe = self._torch.ones(4, device="npu")
            return probe.sum().item() == 4.0
        except Exception:
            return False

    def _record(self, nodeid, mode):
        rec = {
            "nodeid": nodeid,
            "mode": mode,  # "poisoner" | "victim"
            "pid": os.getpid(),
            "shard": f"{self.shard_type}{self.shard}",
            "batch_id": self.batch_id,
            "time": datetime.now().isoformat(),
        }
        label = (
            "suspected poisoner (process healthy before this case)"
            if mode == "poisoner"
            else "victim (process already poisoned before this case)"
        )
        print(
            f"[NPU-CANARY] {mode.upper()} | {nodeid} | pid {rec['pid']} "
            f"| batch {self.batch_id} | {label}",
            flush=True,
        )
        try:
            with open(self._marker_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            print(f"[NPU-CANARY] marker write failed: {e}", flush=True)

    # ---- pytest hooks ------------------------------------------------------
    def pytest_runtest_setup(self, item):
        # Pre-case probe: a failure here means the process was already
        # poisoned before this case started — this case is a victim, not
        # the source.
        if not self.enabled or self.poisoned:
            return
        if not self._probe():
            self.poisoned = True
            self._record(item.nodeid, "victim")

    def pytest_runtest_teardown(self, item, nextitem):
        # Post-case probe: a failure here means the case that just finished
        # poisoned the process (or poisoning slipped past an earlier probe).
        if not self.enabled or self.poisoned:
            return
        if not self._probe():
            self.poisoned = True
            self._record(item.nodeid, "poisoner")


# ==============================================================================
# Worker Process (one pytest.main() per batch)
# ==============================================================================


def _worker_batch_main(worker_input_file: str) -> None:
    """
    Worker entry point. Called via:
        python run_npu_test_shard_batch.py --worker-batch <batch_input.json>

    Runs the whole batch in ONE pytest.main() call with a batch-level
    --junitxml. stdout is NOT redirected: pytest's -v output goes to the
    pipe so the parent can log it and feed the idle watchdog. Exits with
    the pytest return code (os._exit skips pytest atexit handlers).
    """
    import pytest

    with open(worker_input_file, encoding="utf-8") as f:
        batch_input = json.load(f)

    cases = batch_input["cases"]
    test_dir = Path(batch_input["test_dir"])
    report_dir = Path(batch_input["report_dir"])
    timeout = batch_input.get("timeout", 1200)
    verbose = batch_input.get("verbose", False)
    shard = batch_input.get("shard", 0)
    shard_type = batch_input.get("shard_type", "regular")
    batch_id = batch_input.get("batch_id", 0)

    # Change to test directory
    os.chdir(str(test_dir))

    # Ensure junit_xmls directory exists
    junit_xml_dir = report_dir / "junit_xmls"
    junit_xml_dir.mkdir(parents=True, exist_ok=True)

    # Determine PYTHONPATH from first case (all cases in batch are same-file)
    if cases:
        test_file_rel = cases[0]["test_file"]
        if test_file_rel.startswith("test/"):
            test_file_rel = test_file_rel[5:]
        test_file_dir = test_dir / Path(test_file_rel).parent
        existing = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = str(test_file_dir) + (":" + existing if existing else "")

    nodeids = []
    for case in cases:
        nodeid = case["nodeid"]
        nodeids.append(nodeid[5:] if nodeid.startswith("test/") else nodeid)

    xml_file = _batch_xml_path(report_dir, shard, shard_type, batch_id)

    pytest_args = [
        "--color=no",
        "-ra",
        "--tb=short",
        *nodeids,
        f"--junitxml={xml_file}",
    ]
    if timeout > 0:
        pytest_args.append(f"--timeout={timeout}")
    pytest_args.append("-vv" if verbose else "-v")

    display_file = cases[0]["test_file"] if cases else "?"
    print(f"[batch {batch_id}] starting: {len(nodeids)} cases from {display_file}", flush=True)

    try:
        canary = NpuCanaryPlugin(
            report_dir=report_dir,
            shard=shard,
            shard_type=shard_type,
            batch_id=batch_id,
        )
        returncode = pytest.main(args=pytest_args, plugins=[canary])
        if not isinstance(returncode, int):
            returncode = int(returncode) if returncode is not None else 1
    except SystemExit as e:
        returncode = int(e.code) if isinstance(e.code, int) and e.code is not None else 1
    except BaseException as e:
        print(
            f"[batch {batch_id}] fatal worker error: {type(e).__name__}: {str(e)[:200]}",
            file=sys.stderr, flush=True,
        )
        returncode = 1

    print(f"[batch {batch_id}] pytest finished with rc={returncode}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(returncode if returncode >= 0 else 1)


# ==============================================================================
# Parent: Batch Execution
# ==============================================================================


def _batch_command_str(
    batch: List[runner.CaseExecutionTask],
    xml_file: Path,
    timeout: int,
    verbose: bool,
) -> str:
    """Human-readable reconstruction of the batch pytest command."""
    nodeids = [t.nodeid[5:] if t.nodeid.startswith("test/") else t.nodeid for t in batch]
    shown = " ".join(nodeids[:2])
    if len(nodeids) > 2:
        shown += f" ... (+{len(nodeids) - 2} more)"
    cmd = (
        f"{sys.executable} -m pytest --color=no -ra --tb=short {shown} "
        f"--junitxml={xml_file}"
    )
    if timeout > 0:
        cmd += f" --timeout={timeout}"
    cmd += " -vv" if verbose else " -v"
    return f"{cmd} [merged batch, {len(nodeids)} cases]"


def _batch_error_result(
    task: runner.CaseExecutionTask,
    message: str,
    returncode: int,
) -> Dict:
    return {
        "nodeid": task.nodeid,
        "status": "error",
        "duration": 0.0,
        "returncode": returncode,
        "message": message,
        "command": "",
        "file": task.test_file,
        "case_idx": task.case_idx,
    }


def _execute_batch(
    batch: List[runner.CaseExecutionTask],
    batch_id: int,
    total_batches: int,
    test_dir: Path,
    report_dir: Path,
    merged_env: Dict[str, str],
    timeout: int,
    verbose: bool,
    shard: int,
    shard_type: str,
    device_id: Optional[int],
    result_aggregator: runner.ConcurrentResultAggregator,
    progress_tracker: runner.ProgressTracker,
    shard_log_lock: threading.Lock,
    shard_log_file: Path,
) -> None:
    """
    Execute one batch in a worker subprocess, then aggregate results.

    - Worker exits normally (any rc >= 0): parse the batch XML and match
      every case; cases missing from the XML (invalid nodeid, session
      interrupted) become error results.
    - Worker crashed (signal) or hung (idle watchdog kill): the whole
      batch is marked error — no restart, no retry, by design.
    Never raises — all failures become error results in the aggregator.
    """
    script_path = Path(__file__).resolve()
    batch_input_file = report_dir / f"batch_input_{batch_id}.json"
    xml_file = _batch_xml_path(report_dir, shard, shard_type, batch_id)
    log_file = _batch_log_path(report_dir, shard, shard_type, batch_id)

    batch_input_file.write_text(json.dumps({
        "batch_id": batch_id,
        "test_dir": str(test_dir),
        "report_dir": str(report_dir),
        "timeout": timeout,
        "verbose": verbose,
        "shard": shard,
        "shard_type": shard_type,
        "cases": [
            {"case_idx": t.case_idx, "nodeid": t.nodeid, "test_file": t.test_file}
            for t in batch
        ],
    }, indent=2), encoding="utf-8")

    env = dict(merged_env)
    if device_id is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)

    worker_cmd = [
        sys.executable, "-u", str(script_path),
        "--worker-batch", str(batch_input_file),
        "--test-dir", str(test_dir),
    ]

    display_file = batch[0].test_file
    if display_file.startswith("test/"):
        display_file = display_file[5:]
    started = monotonic()

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log_handle:
        log_handle.write(
            f"{'=' * 80}\n"
            f"BATCH LOG (merged batch execution)\n"
            f"{'=' * 80}\n"
            f"Shard: {_prefix_for(shard_type)}-{shard}  Batch: {batch_id + 1}/{total_batches}\n"
            f"File: {batch[0].test_file}\n"
            f"Cases: {len(batch)} (case_idx {batch[0].case_idx}..{batch[-1].case_idx})\n"
            f"NPU Device: {device_id if device_id is not None else 'all (distributed)'}\n"
            f"Command: {_batch_command_str(batch, xml_file, timeout, verbose)}\n"
            f"{'=' * 80}\n\n"
        )
        log_handle.flush()

        try:
            proc = subprocess.Popen(
                worker_cmd,
                cwd=str(test_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as e:
            log_handle.write(f"Failed to spawn worker: {e}\n")
            _record_results(
                batch, result_aggregator, progress_tracker,
                [_batch_error_result(t, f"Worker spawn failed: {str(e)[:200]}", 1) for t in batch],
            )
            batch_input_file.unlink(missing_ok=True)
            return

        last_output_time = monotonic()
        write_lock = threading.Lock()

        def _read_stdout():
            nonlocal last_output_time
            if proc.stdout:
                for line in proc.stdout:
                    last_output_time = monotonic()
                    with write_lock:
                        log_handle.write(line)
                        log_handle.flush()

        reader_thread = threading.Thread(target=_read_stdout, daemon=True)
        reader_thread.start()

        idle_timeout = timeout + IDLE_TIMEOUT_MARGIN
        watchdog_killed = False
        while True:
            returncode = proc.poll()
            if returncode is not None:
                reader_thread.join(timeout=10)
                break

            if monotonic() - last_output_time > idle_timeout:
                watchdog_killed = True
                silent = monotonic() - last_output_time
                log_handle.write(
                    f"\n[watchdog] idle timeout ({silent:.0f}s without output), "
                    f"killing worker...\n"
                )
                log_handle.flush()
                proc.kill()
                try:
                    returncode = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    returncode = -9
                reader_thread.join(timeout=10)
                break

            sleep(0.5)

        log_handle.write(
            f"\n{'=' * 80}\n"
            f"Worker exit code: {returncode}"
            f"{' (idle watchdog kill)' if watchdog_killed else ''}\n"
            f"Batch wall time: {monotonic() - started:.2f}s\n"
            f"{'=' * 80}\n"
        )

    duration = monotonic() - started
    command_str = _batch_command_str(batch, xml_file, timeout, verbose)
    results: List[Dict] = []

    if returncode < 0 or watchdog_killed:
        # Worker crashed or hung: the batch XML is unusable/absent (pytest
        # writes --junitxml at session end only). Mark the whole batch as
        # error — no restart, no per-case rescue, by design.
        if watchdog_killed:
            reason = (
                f"Batch aborted: idle timeout (no worker output for "
                f"{duration:.0f}s, limit {idle_timeout}s); all {len(batch)} "
                f"cases in this batch marked failed"
            )
            rc_for_result = -1
        else:
            try:
                signal_name = signal.Signals(-returncode).name
            except (ValueError, AttributeError):
                signal_name = f"signal {-returncode}"
            reason = (
                f"Batch aborted: worker killed by {signal_name}; "
                f"all {len(batch)} cases in this batch marked failed"
            )
            rc_for_result = returncode
        results = [_batch_error_result(t, reason, rc_for_result) for t in batch]
    else:
        # Worker exited normally (pytest rc may still be 1/2/3/4/5 — e.g.
        # an invalid nodeid aborts collection with rc=4 before any test
        # runs). The batch XML is the source of truth: cases present in
        # it get their real result; missing ones are errors.
        try:
            matched = _match_batch_results(xml_file, batch)
        except (ET.ParseError, OSError) as e:
            matched = None
            parse_error = str(e)[:200]

        for task in batch:
            found = (matched or {}).get(task.nodeid)
            if found is None:
                if matched is None:
                    message = (
                        f"Batch XML unreadable (pytest rc={returncode}): {parse_error}"
                    )
                else:
                    message = (
                        f"Not present in batch XML (pytest rc={returncode}); "
                        f"case never ran (invalid nodeid or session interrupted)"
                    )
                results.append(_batch_error_result(task, message, 1))
                continue
            results.append({
                "nodeid": task.nodeid,
                "status": found["status"],
                "duration": found["duration"],
                "returncode": 0 if found["status"] in ("passed", "skipped") else 1,
                "message": found["message"],
                "command": command_str,
                "file": task.test_file,
                "case_idx": task.case_idx,
            })

    _record_results(batch, result_aggregator, progress_tracker, results)

    # Batch summary line (console + shard log)
    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "timeout": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary_line = (
        f"[Batch {batch_id + 1}/{total_batches}] {display_file}: "
        f"{counts['passed']} passed, {counts['failed']} failed, "
        f"{counts['error']} error, {counts['skipped']} skipped "
        f"— {duration:.1f}s"
        f"{' [WATCHDOG KILL]' if watchdog_killed else ''}"
        f"{' [WORKER CRASH]' if returncode < 0 and not watchdog_killed else ''}"
    )
    print(summary_line, flush=True)
    with shard_log_lock:
        shard_log_file.write(summary_line + "\n")
        shard_log_file.flush()

    batch_input_file.unlink(missing_ok=True)


def _record_results(
    batch: List[runner.CaseExecutionTask],
    result_aggregator: runner.ConcurrentResultAggregator,
    progress_tracker: runner.ProgressTracker,
    results: List[Dict],
) -> None:
    for result in results:
        result_aggregator.add_case_result(result)
        progress_tracker.mark_completed(
            result["nodeid"], result["status"], result["duration"]
        )


def run_batch_tests_concurrent(
    tasks: List[runner.CaseExecutionTask],
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
) -> Tuple[int, float, List[Dict]]:
    """
    Execute pre-collected cases as merged batches.

    Returns (worst_returncode, duration, cases_list_sorted) — the same
    contract as runner.run_tests_with_tasks_concurrent.
    """
    start = monotonic()
    log_file = result_module.get_shard_log_file(report_dir, shard, shard_type)

    junit_xml_dir = report_dir / "junit_xmls"
    junit_xml_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "cases_logs").mkdir(parents=True, exist_ok=True)

    merged_env = os.environ.copy()
    merged_env.update(env_updates)

    # Device allocation: distributed tests use all devices; regular tests
    # round-robin batches across NPU devices.
    if shard_type == "distributed":
        num_npu_devices = None
        print("NPU device allocation: DISABLED (distributed test uses all devices)")
    else:
        num_npu_devices = runner.get_npu_device_count()
        print(f"NPU device allocation: {num_npu_devices} devices detected (round-robin)")

    result_aggregator = runner.ConcurrentResultAggregator()
    progress_tracker = runner.ProgressTracker(len(tasks))
    shard_log_lock = threading.Lock()

    if quick_test and len(tasks) > quick_test:
        tasks = tasks[:quick_test]
        print(f"\nQuick test mode: executing only {quick_test} cases", flush=True)

    total_cases = len(tasks)
    batches = runner.sort_and_batch_tasks(tasks, max_cases_per_batch=MAX_CASES_PER_BATCH)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as shard_log_file:
        shard_log_file.write(
            f"{'=' * 80}\n"
            f"Merged batch execution ({shard_type} shard)\n"
            f"{'=' * 80}\n"
            f"Total cases: {total_cases}\n"
            f"Max concurrent workers: {max_workers}\n"
            f"Batch size: {MAX_CASES_PER_BATCH} same-file cases per batch "
            f"(one pytest.main() per batch)\n"
            f"Timeouts: per-case {timeout}s (pytest-timeout signal method) + "
            f"batch idle watchdog {timeout + IDLE_TIMEOUT_MARGIN}s\n"
            f"{'=' * 80}\n\n"
        )

        print(f"\n{'=' * 80}", flush=True)
        print(f"Pre-collected cases: {total_cases} cases", flush=True)
        print(
            f"Execution mode: MERGED BATCH — {len(batches)} batches "
            f"(max {MAX_CASES_PER_BATCH} same-file cases per batch, "
            f"one pytest.main() per batch), {max_workers} workers concurrent",
            flush=True,
        )
        print(f"{'=' * 80}\n", flush=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for batch_id, batch in enumerate(batches):
                if num_npu_devices is not None:
                    device_id = batch_id % num_npu_devices
                else:
                    device_id = None

                future = executor.submit(
                    _execute_batch,
                    batch,
                    batch_id,
                    len(batches),
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
                    shard_log_lock,
                    shard_log_file,
                )
                futures.append((future, batch_id))

            for future, batch_id in futures:
                try:
                    future.result()
                except Exception as e:
                    print(
                        f"  ERROR: Batch {batch_id} execution failed: {str(e)[:200]}",
                        flush=True,
                    )

        elapsed = monotonic() - start
        summary = result_aggregator.get_summary()

        summary_text = (
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
        )
        shard_log_file.write(summary_text)
        shard_log_file.flush()

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


# ==============================================================================
# CLI
# ==============================================================================


def parse_args():
    """Parse command line arguments (CLI-compatible with run_npu_test_shard.py)."""
    parser = argparse.ArgumentParser(
        description="Run PyTorch NPU tests via merged-batch pytest execution "
                    "(one pytest.main() per same-file batch)"
    )
    parser.add_argument("--cases-json", type=str, help="Path to pre-collected cases JSON file")
    parser.add_argument("--test-dir", type=str, required=True, help="PyTorch test directory")
    parser.add_argument("--report-dir", type=str, default="test-reports", help="Directory for reports")
    parser.add_argument("--timeout", type=int, default=1200, help="Per-case timeout in seconds (default: 1200 = 20 minutes)")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum concurrent batch workers (default: 4)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick-test", type=int, default=None, help="Quick test mode: execute only N cases")
    parser.add_argument(
        "--device-env",
        default="privateuse1",
        help="Comma-separated device types exported as "
             "PYTORCH_TESTING_DEVICE_ONLY_FOR during execution "
             "(default: privateuse1). Must match the value used at collection "
             "time so collected nodeids exist when tests run.",
    )
    parser.add_argument("--worker-batch", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.worker_batch and not args.cases_json:
        parser.error("--cases-json must be specified")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")

    return args


def main():
    """Main entry point."""
    args = parse_args()

    # Worker mode dispatch
    if args.worker_batch:
        _worker_batch_main(args.worker_batch)
        return  # _worker_batch_main calls os._exit, unreachable

    # Resolve paths
    test_dir = Path(args.test_dir).resolve()
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    script_dir = Path(__file__).resolve().parent
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    result_module = runner.load_parse_test_results_module(script_dir)
    timestamp = datetime.now().isoformat()

    print("=" * 80)
    print("Pre-collected Cases Execution Mode (merged batch)")
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

    # Distributed tests run serially (one batch at a time)
    if shard_type == "distributed":
        effective_workers = 1
        print(f"Execution mode: SERIAL (distributed tests require sequential execution)")
    else:
        effective_workers = args.max_workers
        print(f"Execution mode: CONCURRENT ({effective_workers} workers, merged batch)")

    print(f"\n{'=' * 80}\n")

    info = result_module.create_shard_info(shard, num_shards, timestamp)
    info["selection_mode"] = "cases_json"
    info["shard_type"] = shard_type
    info["cases_json_file"] = str(cases_file)
    info["total_cases"] = total_cases
    info["per_case_isolation"] = False
    info["batch_execution"] = True

    # Clean old files
    runner.clean_existing_junit_xml(report_dir)
    result_module.get_shard_log_file(report_dir, shard, shard_type).unlink(missing_ok=True)

    # Build execution env (same environment as run_npu_test_shard.py)
    env_updates = runner.build_execution_env(
        test_dir, script_dir, None, shard, shard_type, args.device_env,
    )

    # Convert cases to CaseExecutionTask format
    tasks = []
    for i, case in enumerate(planned_cases, 1):
        tasks.append(runner.CaseExecutionTask(
            case_idx=i,
            nodeid=case["nodeid"],
            test_file=case.get("file", ""),
        ))

    cases_list = []
    if tasks:
        returncode, duration, cases_list = run_batch_tests_concurrent(
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
        info["execution_mode"] = "merged_batch"
        info["concurrent_workers"] = effective_workers
    else:
        print("No cases to execute.")
        returncode = 0
        duration = 0.0

    runner.save_results_and_summary(
        result_module=result_module,
        report_dir=report_dir,
        shard=shard,
        shard_type=shard_type,
        cases_list=cases_list,
        duration=duration,
        returncode=returncode,
        info=info,
        execution_mode="merged_batch",
        concurrent_workers=effective_workers,
    )

    # Exit with 0 to allow the step to succeed and report generation to
    # proceed. The actual test results are recorded in cases.json.
    sys.exit(0)


if __name__ == "__main__":
    main()
