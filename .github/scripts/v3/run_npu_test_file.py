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
) -> Dict:
    """Run a single test file via pytest with crash recovery.

    Uses StepcurrentPlugin to track completed test cases, so that
    retries after NPU crashes skip already-passed cases.

    Args:
        device_group: ASCEND_RT_VISIBLE_DEVICES value, e.g. "0" or "0,1,2,3,4,5,6,7"
    """
    file_path = test_dir / test_file
    if not file_path.exists():
        return {
            "file": test_file,
            "status": "file_not_found",
            "passed": 0,
            "failed": 0,
            "errors": 0,
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

    case_log_dir = report_dir / "cases_logs"
    case_log_dir.mkdir(parents=True, exist_ok=True)

    max_attempts = 2
    all_passed = 0
    all_failed = 0
    all_errors = 0
    all_skipped = 0
    cases_detail = []

    for attempt in range(1, max_attempts + 1):
        junit_file = junit_dir / f"{_safe_name(test_file)}_attempt{attempt}.xml"

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

        # Use StepcurrentPlugin to skip previously crashed cases on retry
        if attempt > 1:
            prev_junit = junit_dir / f"{_safe_name(test_file)}_attempt{attempt - 1}.xml"
            if prev_junit.exists():
                cmd.extend(["--scs", str(prev_junit)])
            else:
                cmd.append("--sc")

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(test_dir.parent),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return {
                "file": test_file,
                "status": "timeout",
                "passed": all_passed,
                "failed": all_failed,
                "errors": all_errors + 1,
                "skipped": all_skipped,
                "elapsed": elapsed,
                "cases": cases_detail,
            }

        # Check for NPU poisoning (exit code 70).
        # Still parse the partial JUnit — StepcurrentPlugin may have
        # recorded passed cases before the crash.  Those results must
        # carry forward to the retry.
        if proc.returncode == 70:
            print(f"  [WARN] NPU poisoning detected on device group {device_group} "
                  f"for {test_file} (attempt {attempt})", file=sys.stderr)
            # Parse partial JUnit to capture cases completed before the crash
            partial_passed, partial_failed, partial_errors, partial_skipped, partial_cases = \
                _parse_junit(junit_file)
            if attempt == 1:
                all_passed = partial_passed
                all_failed = partial_failed
                all_errors = partial_errors
                all_skipped = partial_skipped
                cases_detail = partial_cases
            else:
                # Retry: partial results replace failed/errors from previous attempt
                all_passed += partial_passed
                all_failed = partial_failed
                all_errors = partial_errors
                all_skipped += partial_skipped
                cases_detail = _merge_cases(cases_detail, partial_cases)
            if attempt < max_attempts:
                time.sleep(5)
                continue
            # Last attempt — poisoning after partial JUnit already parsed.
            # Break to avoid falling into the normal parse below which would
            # double-count all_passed/all_skipped from the same JUnit file.
            break

        # Parse JUnit XML for case results.
        # With --scs on retry, the JUnit only contains re-run cases
        # (previously-passed cases are skipped by StepcurrentPlugin).
        # So we must accumulate passed/skipped, not overwrite.
        file_passed, file_failed, file_errors, file_skipped, file_cases = \
            _parse_junit(junit_file)

        if attempt == 1:
            # Baseline: first run captures everything
            all_passed = file_passed
            all_failed = file_failed
            all_errors = file_errors
            all_skipped = file_skipped
            cases_detail = file_cases
        else:
            # Retry with --scs: only re-ran failed/error cases.
            #   passed  → accumulate (failed→passed conversions from prev attempt)
            #   failed  → replace  (still failing after retry)
            #   errors  → replace  (still erroring after retry)
            #   skipped → accumulate
            all_passed += file_passed
            all_failed = file_failed
            all_errors = file_errors
            all_skipped += file_skipped
            cases_detail = _merge_cases(cases_detail, file_cases)

        if file_failed == 0 and file_errors == 0:
            break

        if attempt < max_attempts:
            print(f"  [RETRY] {test_file}: {file_failed} failed, {file_errors} errors "
                  f"→ attempt {attempt + 1}/{max_attempts}", file=sys.stderr)

    elapsed = time.time() - start_time

    # Merge JUnit XMLs across attempts
    _merge_junit_xmls(junit_dir, _safe_name(test_file), max_attempts)

    return {
        "file": test_file,
        "status": "completed",
        "passed": all_passed,
        "failed": all_failed,
        "errors": all_errors,
        "skipped": all_skipped,
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


def _run_one_file_worker(args_tuple: Tuple) -> Dict:
    """Worker for ProcessPoolExecutor.

    Acquires a device group from the shared queue before running,
    releases it after completion.  This prevents multiple processes
    from contending for the same NPU card(s).

    Args tuple: (test_file, test_dir, report_dir, device_queue, timeout, case_timeout)
    """
    test_file, test_dir, report_dir, device_queue, timeout, case_timeout = args_tuple
    device_group = device_queue.get()  # blocks until a device group is free
    try:
        return run_single_file(test_file, test_dir, report_dir, device_group,
                               timeout, case_timeout)
    finally:
        device_queue.put(device_group)  # return device group to pool


# ==============================================================================
# Main Orchestrator
# ==============================================================================


def main():
    args = parse_args()

    # Load files JSON
    files_json = Path(args.files_json)
    if not files_json.exists():
        print(f"ERROR: Files JSON not found: {files_json}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(files_json.read_text(encoding="utf-8"))
    files = data.get("files", [])
    test_type = data.get("test_type", args.shard_type)
    total_files = len(files)

    test_dir = Path(args.test_dir).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

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
            print(f"  [{i}/{total_files}] {f} (ASCEND_RT_VISIBLE_DEVICES={device_groups[group_idx]})")
            result = run_single_file(
                f, test_dir, report_dir, device_groups[group_idx],
                args.timeout, args.case_timeout,
            )
            results.append(result)
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
        worker_args = [
            (f, test_dir, report_dir, device_queue,
             args.timeout, args.case_timeout)
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
                        "passed": 0, "failed": 0, "errors": 1,
                        "skipped": 0, "elapsed": 0, "cases": [],
                    })
                print(f"  [{completed}/{total_files}] {test_file}")

        manager.shutdown()

    elapsed_total = time.time() - start_time

    # Aggregate statistics
    total_passed = sum(r.get("passed", 0) for r in results)
    total_failed = sum(r.get("failed", 0) for r in results)
    total_errors = sum(r.get("errors", 0) for r in results)
    total_skipped = sum(r.get("skipped", 0) for r in results)
    total_cases = total_passed + total_failed + total_errors + total_skipped

    # Save shard result JSON
    prefix_map = {
        "core": "core", "tensor": "tensor",
        "distributed": "dist", "graph": "graph", "others": "others",
    }
    prefix = prefix_map.get(test_type, "reg")

    shard_result = {
        "shard": args.shard,
        "test_type": test_type,
        "total_files": total_files,
        "total_cases": total_cases,
        "passed": total_passed,
        "failed": total_failed,
        "errors": total_errors,
        "skipped": total_skipped,
        "elapsed": elapsed_total,
        "files": [],
    }

    for r in results:
        shard_result["files"].append({
            "file": r.get("file", ""),
            "status": r.get("status", "unknown"),
            "passed": r.get("passed", 0),
            "failed": r.get("failed", 0),
            "errors": r.get("errors", 0),
            "skipped": r.get("skipped", 0),
            "elapsed": r.get("elapsed", 0),
        })

    result_path = report_dir / f"shard_{prefix}-{args.shard}_cases.json"
    result_path.write_text(
        json.dumps(shard_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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
    parser.add_argument("--files-json", required=True,
                        help="Path to {category}_files_shard_{n}.json")
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
    parser.add_argument("--shard-type", default="regular",
                        help="Test category name")
    parser.add_argument("--shard", type=int, default=1,
                        help="Shard index (1-based)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
