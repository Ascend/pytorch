#!/usr/bin/env python3
"""
Run NPU tests at file-level granularity with crash recovery.

Executes test files (not individual cases) via pytest, using
StepcurrentPlugin (--sc/--scs) to skip previously crashed test cases
on retry. Supports two execution modes derived from npu_count and
devices_per_proc:

  - concurrent: multiple device groups, ProcessPoolExecutor
  - serial:     single device group (devices_per_proc == npu_count)

Device groups follow the same convention as v2's DevicePool:
  - devices_per_proc=1: each group is a single card "0", "1", ...
  - devices_per_proc=8: one group "0,1,2,3,4,5,6,7" (all cards)

Usage:
    python run_npu_test_file.py \
        --files-json cases-shards/core_files_shard_1.json \
        --test-dir pytorch/test \
        --report-dir test-reports \
        --npu-count 8 \
        --devices-per-proc 1 \
        --timeout 1800 \
        --case-timeout 300 \
        --shard-type core \
        --shard 1
"""

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, List, Tuple


# ==============================================================================
# Signal Mapping (for core dump / crash detection)
# ==============================================================================

_SIGNAL_MAP = {
    -1: "SIGHUP", -2: "SIGINT", -3: "SIGQUIT", -4: "SIGILL",
    -5: "SIGTRAP", -6: "SIGABRT", -7: "SIGBUS", -8: "SIGFPE",
    -9: "SIGKILL", -10: "SIGUSR1", -11: "SIGSEGV", -12: "SIGUSR2",
    -13: "SIGPIPE", -14: "SIGALRM", -15: "SIGTERM",
}


# ==============================================================================
# Device Groups (same logic as v2 npu_scheduler.py DevicePool)
# ==============================================================================


def build_device_groups(npu_count: int, devices_per_proc: int) -> List[str]:
    """Build device group strings for ASCEND_RT_VISIBLE_DEVICES.

    Each device group is a string suitable for ASCEND_RT_VISIBLE_DEVICES.
    Number of groups = npu_count // devices_per_proc.

    Example:
        npu_count=8, devices_per_proc=1 → ["0","1","2","3","4","5","6","7"]
        npu_count=8, devices_per_proc=8 → ["0,1,2,3,4,5,6,7"]
    """
    groups = max(1, npu_count // devices_per_proc)
    available = []
    for i in range(groups):
        if devices_per_proc == 1:
            available.append(str(i))
        else:
            start = i * devices_per_proc
            available.append(
                ",".join(str(start + j) for j in range(devices_per_proc))
            )
    return available


# ==============================================================================
# Test Execution
# ==============================================================================


def run_single_file(
    test_file: str,
    test_dir: Path,
    report_dir: Path,
    device_group: str,
    timeout: int,
    case_timeout: int,
    progress: str = "",
) -> Dict:
    """Run a single test file via pytest with NPU poisoning recovery.

    On NPU poisoning (exit code 70), the poisoned test case is identified
    via a marker file written by npu_poisoning_plugin, marked as "poisoned"
    in results, and the JUnit XML is patched so that StepcurrentPlugin's
    --scs skips it on retry.  Normal test failures are NOT retried.

    Args:
        device_group: ASCEND_RT_VISIBLE_DEVICES value, e.g. "0" or "0,1,2,3,4,5,6,7"
        progress: optional progress prefix e.g. "[5/123]" for log output
    """
    file_path = test_dir.parent / test_file
    if not file_path.exists():
        return {
            "file": test_file,
            "status": "file_not_found",
            "return_code": -2,
            "message": "Test file not found on disk",
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
            "elapsed": 0,
            "cases": [],
        }

    # Bind this process to the assigned device group
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device_group
    start_time = time.time()

    # Set up NPU environment
    os.environ.setdefault("PYTORCH_TESTING_DEVICE_ONLY_FOR", "privateuse1")
    os.environ.setdefault("PYTORCH_TESTING_DEVICE_FOR_CUSTOM", "privateuse1")

    junit_dir = report_dir / "junit_xmls"
    junit_dir.mkdir(parents=True, exist_ok=True)

    safe = _safe_name(test_file)
    poisoned_marker = report_dir / f"{safe}_poisoned_case.txt"
    poisoned_case_nodeid = None  # set after first poisoning, used for retry

    max_attempts = 2  # only used for NPU poisoning retries
    all_passed = 0
    all_failed = 0
    all_errors = 0
    all_skipped = 0
    cases_detail = []

    for attempt in range(1, max_attempts + 1):
        junit_file = junit_dir / f"{safe}_attempt{attempt}.xml"

        cmd = [
            sys.executable,
            "-m", "pytest",
            str(file_path),
            "-p", "no:xdist",
            "-p", "npu_poisoning_plugin",
            "-p", "timeout",
            f"--timeout={case_timeout}",
            f"--junit-xml={junit_file}",
            "-v",
            "--tb=short",
            "--hw-classification", "ACCELERATOR",
        ]

        # On retry: use StepcurrentPlugin to skip already-passed (and
        # now-also-poisoned) cases.  The poisoned case was patched in
        # the previous attempt's JUnit XML to appear as "skipped" so
        # that --scs will exclude it.
        if attempt > 1:
            prev_junit = junit_dir / f"{safe}_attempt{attempt - 1}.xml"
            if prev_junit.exists():
                cmd.extend(["--scs", str(prev_junit)])
            else:
                cmd.append("--sc")

        # Pass the poisoned-case marker path to the plugin via env
        run_env = os.environ.copy()
        run_env["NPU_POISONED_CASE_FILE"] = str(poisoned_marker)
        # Clean up stale marker from a previous (different) file run
        if poisoned_marker.exists():
            poisoned_marker.unlink()

        try:
            # Stream pytest output to terminal in real-time.
            # Print the full command so the exact invocation is logged.
            cmd_str = " ".join(cmd)
            pfx = f"{progress} " if progress else ""
            print(f"{pfx}[device {device_group}] [{test_file}] "
                  f"Command: {cmd_str}", flush=True)
            proc = subprocess.run(
                cmd,
                cwd=str(test_dir.parent),
                timeout=timeout,
                env=run_env,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return {
                "file": test_file,
                "status": "timeout",
                "return_code": -1,
                "message": f"File-level timeout after {timeout}s",
                "passed": all_passed,
                "failed": all_failed,
                "errors": all_errors + 1,
                "skipped": all_skipped,
                "elapsed": elapsed,
                "cases": cases_detail,
            }

        # ── Core dump / signal death ──────────────────────────────
        if proc.returncode < 0:
            signal_name = _SIGNAL_MAP.get(proc.returncode,
                                          f"SIGNAL({abs(proc.returncode)})")
            partial_passed, partial_failed, partial_errors, partial_skipped, \
                partial_cases = _parse_junit(junit_file)
            elapsed = time.time() - start_time
            return {
                "file": test_file,
                "status": "crashed",
                "return_code": proc.returncode,
                "message": f"{signal_name}: process killed by signal",
                "passed": partial_passed,
                "failed": partial_failed,
                "errors": partial_errors + 1,
                "skipped": partial_skipped,
                "elapsed": elapsed,
                "cases": partial_cases,
            }

        # ── NPU poisoning (exit code 70) ──────────────────────────
        if proc.returncode == 70:
            # Read which case caused the poisoning
            nodeid, reason = _read_poisoned_marker(poisoned_marker)
            if nodeid:
                poisoned_case_nodeid = nodeid
                print(f"  [WARN] NPU poisoning by case: {nodeid} "
                      f"({reason})", file=sys.stderr)
            else:
                print(f"  [WARN] NPU poisoning detected on device group "
                      f"{device_group} for {test_file} (attempt {attempt})",
                      file=sys.stderr)

            # Parse partial JUnit (cases completed before the crash)
            partial_passed, partial_failed, partial_errors, partial_skipped, \
                partial_cases = _parse_junit(junit_file)

            # Mark the poisoned case with a flag (keeps original status,
            # preserving passed+failed+errors+skipped = total_cases).
            # Also patch the on-disk JUnit XML so that --scs skips it on retry.
            if poisoned_case_nodeid:
                partial_cases = _mark_case_poisoned_in_list(
                    partial_cases, poisoned_case_nodeid)
                _mark_junit_case_poisoned(junit_file, poisoned_case_nodeid)

            if attempt == 1:
                all_passed = partial_passed
                all_failed = partial_failed
                all_errors = partial_errors
                all_skipped = partial_skipped
                cases_detail = partial_cases
            else:
                all_passed += partial_passed
                all_failed = partial_failed
                all_errors = partial_errors
                all_skipped += partial_skipped
                cases_detail = _merge_cases(cases_detail, partial_cases)

            if attempt < max_attempts:
                time.sleep(5)
                continue
            break

        # ── Normal completion (no more retry for regular failures) ─
        file_passed, file_failed, file_errors, file_skipped, file_cases = \
            _parse_junit(junit_file)

        if attempt == 1:
            all_passed = file_passed
            all_failed = file_failed
            all_errors = file_errors
            all_skipped = file_skipped
            cases_detail = file_cases
        else:
            # Retry after NPU poisoning: accumulate passed/skipped,
            # replace failed/errors with the re-run results
            all_passed += file_passed
            all_failed = file_failed
            all_errors = file_errors
            all_skipped += file_skipped
            cases_detail = _merge_cases(cases_detail, file_cases)

        # Do NOT retry for normal failures — always stop after a clean run
        break

    elapsed = time.time() - start_time

    # Merge JUnit XMLs across attempts
    _merge_junit_xmls(junit_dir, safe, max_attempts)

    # Reconstruct final counts from the merged cases_detail.  This
    # guarantees that passed+failed+errors+skipped == len(cases_detail)
    # regardless of counting drift across retry attempts (e.g. poisoned
    # cases skipped via --scs are preserved in cases_detail but would
    # otherwise be lost from the per-attempt accumulators).
    final_passed = sum(1 for c in cases_detail if c.get("status") == "passed")
    final_failed = sum(1 for c in cases_detail if c.get("status") == "failed")
    final_errors = sum(1 for c in cases_detail if c.get("status") == "error")
    final_skipped = sum(1 for c in cases_detail if c.get("status") == "skipped")

    return {
        "file": test_file,
        "status": "completed",
        "return_code": 0 if final_failed == 0 and final_errors == 0 else 1,
        "message": "",
        "passed": final_passed,
        "failed": final_failed,
        "errors": final_errors,
        "skipped": final_skipped,
        "elapsed": elapsed,
        "cases": cases_detail,
    }


def _parse_junit(junit_path: Path) -> Tuple[int, int, int, int, List[Dict]]:
    """Parse pytest JUnit XML to extract per-case results."""
    passed = 0
    failed = 0
    errors = 0
    skipped = 0
    cases = []

    if not junit_path.exists():
        return 0, 0, 0, 0, cases

    try:
        tree = ET.parse(str(junit_path))
        root = tree.getroot()
        for testcase in root.iter("testcase"):
            case_name = testcase.get("name", "")
            class_name = testcase.get("classname", "")
            case_time = float(testcase.get("time", 0))

            case_info = {
                "name": case_name,
                "classname": class_name,
                "time": case_time,
                "status": "passed",
            }

            if testcase.find("failure") is not None:
                failed += 1
                case_info["status"] = "failed"
                failure = testcase.find("failure")
                case_info["message"] = failure.get("message", "")[:500]
            elif testcase.find("error") is not None:
                errors += 1
                case_info["status"] = "error"
                error = testcase.find("error")
                case_info["message"] = error.get("message", "")[:500]
            elif testcase.find("skipped") is not None:
                skipped += 1
                case_info["status"] = "skipped"
                skip = testcase.find("skipped")
                case_info["message"] = skip.get("message", "")[:200]
            else:
                passed += 1

            cases.append(case_info)
    except ET.ParseError:
        pass

    return passed, failed, errors, skipped, cases


def _merge_junit_xmls(junit_dir: Path, safe_name: str, max_attempts: int):
    """Merge JUnit XMLs from multiple attempts into a single file.

    Deduplicates by (classname, name): later attempts overwrite earlier ones
    because retry with --scs only re-runs the failed/error subset.
    """
    merged = junit_dir / f"{safe_name}.xml"
    by_key: Dict[str, ET.Element] = {}

    for attempt in range(1, max_attempts + 1):
        fpath = junit_dir / f"{safe_name}_attempt{attempt}.xml"
        if not fpath.exists():
            continue
        try:
            tree = ET.parse(str(fpath))
            for tc in tree.getroot().iter("testcase"):
                key = f"{tc.get('classname', '')}::{tc.get('name', '')}"
                by_key[key] = tc  # later attempt wins
        except ET.ParseError:
            continue

    if not by_key:
        return

    root = ET.Element("testsuite", name="pytest", tests=str(len(by_key)))
    for tc in by_key.values():
        root.append(tc)

    merged.write_text(
        ET.tostring(root, encoding="unicode"), encoding="utf-8"
    )


def _safe_name(test_file: str) -> str:
    """Convert test file path to a safe filename."""
    return test_file.replace("/", "_").replace("\\", "_").replace(".py", "")


def _merge_cases(prev_cases: List[Dict], new_cases: List[Dict]) -> List[Dict]:
    """Merge case details from two attempts. Later attempt wins on status.

    Used when retry with --scs only produces results for the re-run subset.
    Cases present in new_cases replace their counterparts in prev_cases;
    cases only in prev_cases are kept as-is.
    """
    merged = {c.get("classname", "") + "::" + c.get("name", ""): c for c in prev_cases}
    for c in new_cases:
        key = c.get("classname", "") + "::" + c.get("name", "")
        merged[key] = c  # later attempt wins
    return list(merged.values())


def _read_poisoned_marker(marker_path: Path) -> Tuple[str, str]:
    """Read the poisoned-case marker file written by npu_poisoning_plugin.

    Returns (nodeid, reason) or ("", "") if the file doesn't exist.
    """
    if not marker_path.exists():
        return "", ""
    try:
        lines = marker_path.read_text(encoding="utf-8").strip().split("\n", 1)
        nodeid = lines[0].strip() if lines else ""
        reason = lines[1].strip() if len(lines) > 1 else ""
        return nodeid, reason
    except OSError:
        return "", ""


def _nodeid_to_junit_key(nodeid: str) -> Tuple[str, str]:
    """Convert a pytest nodeid to a JUnit XML (classname, name) pair.

    Examples:
        'test/nn/test_foo.py::test_func'
            → ('test.nn.test_foo', 'test_func')
        'test/nn/test_foo.py::TestClass::test_method[x]'
            → ('test.nn.test_foo.TestClass', 'test_method[x]')
    """
    parts = nodeid.split("::")
    if not parts:
        return "", ""
    file_part = parts[0].replace("/", ".").replace(".py", "")
    if len(parts) >= 3:
        classname = file_part + "." + parts[1]
        name = parts[-1]
    elif len(parts) == 2:
        classname = file_part
        name = parts[1]
    else:
        classname = file_part
        name = ""
    return classname, name


def _mark_case_poisoned_in_list(cases: List[Dict], nodeid: str) -> List[Dict]:
    """Add a 'poisoned' flag to the case that triggered NPU poisoning.

    The case keeps its original JUnit status (passed/failed/error).
    The poisoned flag is orthogonal — it does not affect the four
    status counters; the caller (run_single_file) is responsible for
    maintaining the passed+failed+errors+skipped = len(cases) invariant
    by reconstructing final counts from the merged cases list.
    """
    target_cn, target_name = _nodeid_to_junit_key(nodeid)
    result = []
    for c in cases:
        if c.get("classname") == target_cn and c.get("name") == target_name:
            c = dict(c)  # shallow copy
            c["poisoned"] = True
            if not c.get("message"):
                c["message"] = f"NPU poisoning detected: {nodeid}"
        result.append(c)
    return result


def _mark_junit_case_poisoned(junit_path: Path, nodeid: str):
    """Edit the JUnit XML on disk: change the poisoned case's <failure>/<error>
    to <skipped type='npu_poisoned'> so that StepcurrentPlugin's --scs will
    skip it on retry.
    """
    if not junit_path.exists():
        return
    target_cn, target_name = _nodeid_to_junit_key(nodeid)
    try:
        tree = ET.parse(str(junit_path))
        root = tree.getroot()
        for tc in root.iter("testcase"):
            if tc.get("classname") == target_cn and tc.get("name") == target_name:
                # Remove failure/error children
                for tag in ("failure", "error"):
                    elem = tc.find(tag)
                    if elem is not None:
                        tc.remove(elem)
                # Already has a skipped element (shouldn't, but be safe)
                skip_elem = tc.find("skipped")
                if skip_elem is not None:
                    skip_elem.set("type", "npu_poisoned")
                    skip_elem.set("message", f"NPU poisoning: {nodeid}")
                else:
                    skip_elem = ET.SubElement(tc, "skipped")
                    skip_elem.set("type", "npu_poisoned")
                    skip_elem.set("message", f"NPU poisoning: {nodeid}")
                junit_path.write_text(
                    ET.tostring(root, encoding="unicode"), encoding="utf-8")
                break
    except ET.ParseError:
        pass


def _run_one_file_worker(args_tuple: Tuple) -> Dict:
    """Worker for ProcessPoolExecutor.

    Acquires a device group from the shared queue before running,
    releases it after completion.  This prevents multiple processes
    from contending for the same NPU card(s).

    Args tuple: (test_file, test_dir, report_dir, device_queue, timeout, case_timeout,
                 started_counter, total_files)
    """
    test_file, test_dir, report_dir, device_queue, timeout, case_timeout, \
        started_counter, total_files = args_tuple
    device_group = device_queue.get()  # blocks until a device group is free
    with started_counter.get_lock():
        started_counter.value += 1
        progress = f"[{started_counter.value}/{total_files}]"
    try:
        return run_single_file(test_file, test_dir, report_dir, device_group,
                               timeout, case_timeout, progress=progress)
    finally:
        device_queue.put(device_group)  # return device group to pool


# ==============================================================================
# Main Orchestrator
# ==============================================================================


def write_jsonl(report_dir: Path, prefix: str, shard: int,
                results: List[Dict], test_type: str, runner: str) -> Path:
    """Write shard results as JSONL — the only output format.

    Line 1: shard summary
    Lines 2+: per-file records with case details

    Format is compatible with v2's generate_shard_jsonl.py output.
    """
    total_passed = sum(r.get("passed", 0) for r in results)
    total_failed = sum(r.get("failed", 0) for r in results)
    total_errors = sum(r.get("errors", 0) for r in results)
    total_skipped = sum(r.get("skipped", 0) for r in results)

    summary = {
        "shard": shard,
        "shard_type": test_type,
        "execution_mode": "file_level_upstream",
        "runner": runner,
        "total_files": len(results),
        "total_cases": total_passed + total_failed + total_errors + total_skipped,
        "passed": total_passed,
        "failed": total_failed,
        "errors": total_errors,
        "skipped": total_skipped,
    }

    path = report_dir / f"shard_{prefix}-{shard}_cases.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        for r in results:
            cases_out = []
            for c in r.get("cases", []):
                classname = c.get("classname", "")
                name = c.get("name", "")
                case_obj = {
                    "nodeid": f"{classname}::{name}" if classname else name,
                    "status": c.get("status", "unknown"),
                    "duration": c.get("time", 0),
                    "message": c.get("message", ""),
                }
                # Poisoned flag is orthogonal to status
                if c.get("poisoned"):
                    case_obj["poisoned"] = True
                cases_out.append(case_obj)

            record = {
                "test_file": r.get("file", ""),
                "duration": r.get("elapsed"),
                "return_code": r.get("return_code", 0),
                "message": r.get("message", ""),
                "cases": cases_out,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path


def main():
    args = parse_args()

    test_dir = Path(args.test_dir).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    # ── Classify + shard test files in-memory (no artifact round-trip) ──
    # Reuses shard_test_files.py functions that are already on PYTHONPATH.
    from shard_test_files import load_categories_config, scan_all_test_files, \
        classify_files, split_round_robin

    config = load_categories_config(args.categories_config)
    exclude = config.get("exclude", [])
    categories = config.get("categories", {})

    # Scan all test_*.py under test_dir (e.g. pytorch/test/), classify,
    # shard — same as collect step.  Paths are relative to test_dir.parent
    # (e.g. "test/nn/test_foo.py"), which matches the whitelist config and
    # run_single_file uses test_dir.parent / test_file for resolution.
    all_files = scan_all_test_files(test_dir)

    classified = classify_files(all_files, categories, exclude)
    category_files = classified.get(args.category, [])
    shards = split_round_robin(category_files, args.num_shards)
    files = shards[args.shard - 1] if args.shard <= len(shards) else []

    test_type = args.category
    total_files = len(files)

    # Build device groups (same logic as v2 DevicePool)
    npu_count = args.npu_count
    devices_per_proc = args.devices_per_proc
    device_groups = build_device_groups(npu_count, devices_per_proc)
    num_workers = len(device_groups)

    print("=" * 80)
    print(f"NPU Test File Runner (v3) — {test_type} Shard {args.shard}")
    print("=" * 80)
    print(f"Files: {total_files}")
    print(f"NPU count: {npu_count}")
    print(f"Devices per proc: {devices_per_proc}")
    print(f"Device groups: {device_groups}")
    print(f"Concurrency: {num_workers} workers")
    print(f"File timeout: {args.timeout}s")
    print(f"Case timeout: {args.case_timeout}s")
    print()

    start_time = time.time()
    results = []

    if num_workers == 1:
        # Serial mode: single device group (e.g. distributed with devices_per_proc=8)
        print("Execution mode: SERIAL (1 device group, all cards visible)")
        for i, f in enumerate(files, 1):
            group_idx = 0  # only one group
            result = run_single_file(
                f, test_dir, report_dir, device_groups[group_idx],
                args.timeout, args.case_timeout,
                progress=f"[{i}/{total_files}]",
            )
            results.append(result)
            p, fe, e, s = result.get("passed", 0), result.get("failed", 0), result.get("errors", 0), result.get("skipped", 0)
            print(f"  [{i}/{total_files}] {f} ({p} passed, {fe} failed, {e} errors, {s} skipped)")
    else:
        # Concurrent mode: managed device queue prevents NPU card contention.
        # Each worker acquires a device group from the queue before running,
        # and releases it after completion.  The queue enforces mutual exclusion:
        # at most num_workers files run concurrently, each on a distinct device group.
        manager = Manager()
        device_queue = manager.Queue()
        for g in device_groups:
            device_queue.put(g)

        print(f"Execution mode: CONCURRENT ({num_workers} device groups, queue-based scheduling)")
        started_counter = manager.Value('i', 0)
        worker_args = [
            (f, test_dir, report_dir, device_queue,
             args.timeout, args.case_timeout, started_counter, total_files)
            for f in files
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_run_one_file_worker, wa): wa[0]
                for wa in worker_args
            }
            completed = 0
            for future in as_completed(futures):
                test_file = futures[future]
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"  [ERROR] {test_file}: {e}", file=sys.stderr)
                    results.append({
                        "file": test_file,
                        "status": "worker_error",
                        "return_code": -1,
                        "message": f"Worker exception: {str(e)[:200]}",
                        "passed": 0, "failed": 0, "errors": 1,
                        "skipped": 0, "elapsed": 0, "cases": [],
                    })
                p, fe, e, s = result.get("passed", 0), result.get("failed", 0), result.get("errors", 0), result.get("skipped", 0)
                print(f"  [{completed}/{total_files}] {test_file} ({p} passed, {fe} failed, {e} errors, {s} skipped)")

        manager.shutdown()

    elapsed_total = time.time() - start_time

    # Aggregate statistics
    total_passed = sum(r.get("passed", 0) for r in results)
    total_failed = sum(r.get("failed", 0) for r in results)
    total_errors = sum(r.get("errors", 0) for r in results)
    total_skipped = sum(r.get("skipped", 0) for r in results)
    total_cases = total_passed + total_failed + total_errors + total_skipped

    # Derive runner name from NPU count for summary metadata
    prefix_map = {
        "core": "core", "tensor": "tensor",
        "distributed": "dist", "graph": "graph", "others": "others",
    }
    prefix = prefix_map.get(test_type, "reg")
    runner_label = args.runner or f"linux-aarch64-a3-{npu_count}"

    # Write JSONL output (replaces old JSON format)
    result_path = write_jsonl(report_dir, prefix, args.shard, results,
                              test_type, runner_label)

    # Print summary
    print()
    print("=" * 80)
    print(f"Shard {args.shard} Complete")
    print("=" * 80)
    print(f"  Files: {total_files}")
    print(f"  Cases: {total_cases} ({total_passed} passed, {total_failed} failed, "
          f"{total_errors} errors, {total_skipped} skipped)")
    print(f"  Elapsed: {elapsed_total:.0f}s ({elapsed_total / 60:.1f}min)")
    print(f"  Report: {result_path}")

    if total_failed > 0 or total_errors > 0:
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NPU tests at file-level with crash recovery"
    )
    parser.add_argument("--categories-config", required=True,
                        help="Path to whitelist YAML (classification rules)")
    parser.add_argument("--category", required=True,
                        help="Test category name (core/tensor/distributed/graph/others)")
    parser.add_argument("--num-shards", type=int, required=True,
                        help="Total number of shards for this category")
    parser.add_argument("--test-dir", required=True,
                        help="PyTorch test/ directory")
    parser.add_argument("--report-dir", default="test-reports",
                        help="Output directory for test reports")
    parser.add_argument("--npu-count", type=int, required=True,
                        help="Number of NPU cards (derived from runner label)")
    parser.add_argument("--devices-per-proc", type=int, required=True,
                        help="NPU cards per pytest process")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-file timeout in seconds")
    parser.add_argument("--case-timeout", type=int, default=300,
                        help="Per-test-case timeout in seconds")
    parser.add_argument("--shard", type=int, default=1,
                        help="Shard index (1-based)")
    parser.add_argument("--runner", default="",
                        help="Runner label for this shard (e.g. linux-aarch64-a3-8)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
