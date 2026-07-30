#!/usr/bin/env python3
"""
Retry failed test cases in independent subprocesses.

This is a step-level retry mechanism that runs AFTER the initial test
execution completes. It reads the cases.json produced by run_npu_test_shard.py,
extracts all failed/error/timeout cases, and re-executes each one in a
brand new subprocess (python -m pytest <nodeid>).

Key design decisions:
  - Each retried case gets a completely fresh process: no poison_pill
    contamination, no pytest state leakage, no NPU device context pollution.
  - Distributed tests: serial execution (1 case at a time, all NPU devices).
  - Regular tests: concurrent execution (ThreadPoolExecutor, each case in
    its own subprocess with round-robin NPU device allocation).
  - Retry results replace the original entries in cases.json.
  - Original failure info is preserved in the "retry_history" field.
  - If a case passes on retry, its final status is "passed".

Usage:
    python retry_failed_cases.py \
        --cases-json test-reports/shard_dist-2_cases.json \
        --test-dir pytorch/test \
        --report-dir test-reports \
        --shard 2 \
        --shard-type distributed \
        --timeout 1200 \
        --verbose
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import run_npu_test_shard as runner


def load_cases_json(cases_file: Path) -> Optional[Dict]:
    """Load cases.json, return None if not found or invalid."""
    if not cases_file.exists():
        print(f"Warning: Cases file not found: {cases_file}", file=sys.stderr)
        return None
    try:
        with open(cases_file, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not parse {cases_file}: {e}", file=sys.stderr)
        return None


def extract_failed_cases(cases_data: Dict) -> List[Dict]:
    """Extract failed/error/timeout cases from cases data."""
    return [
        c for c in cases_data.get("cases", [])
        if c.get("status") in ("failed", "error", "timeout")
    ]


def run_single_case_retry(
    case: Dict,
    test_dir: Path,
    report_dir: Path,
    shard: int,
    shard_type: str,
    timeout: int,
    verbose: bool,
    npu_device_count: int,
    retry_index: int,
) -> Dict:
    """
    Run one failed case in a brand new subprocess.

    Each case gets its own `python -m pytest <nodeid>` subprocess with a
    clean process state. This avoids poison_pill contamination, pytest
    session leakage, and NPU device context pollution.

    Returns a case result dict that replaces the original entry.
    """
    original_nodeid = case["nodeid"]
    case_idx = case["case_idx"]

    # Strip "test/" prefix for pytest target
    nodeid = original_nodeid[5:] if original_nodeid.startswith("test/") else original_nodeid

    # XML filename with _retry suffix to avoid overwriting original
    prefix = {"distributed": "dist", "core": "core", "tensor": "tensor",
              "graph": "graph", "others": "others",
              "regular": "reg", "custom": "custom"}.get(shard_type, "reg")
    safe_name = runner.sanitize_nodeid_for_filename(original_nodeid)
    junit_xml_dir = report_dir / "junit_xmls"
    junit_xml_dir.mkdir(parents=True, exist_ok=True)
    xml_file = junit_xml_dir / f"{prefix}-{shard}_{case_idx}_{safe_name}_retry.xml"

    # Build pytest command
    pytest_cmd = [
        sys.executable, "-u", "-m", "pytest",
        "--color=no",
        "-ra",
        "--tb=short",
        nodeid,
        f"--junitxml={xml_file}",
        f"--timeout={timeout}",
    ]
    if verbose:
        pytest_cmd.append("-vv")
    else:
        pytest_cmd.append("-v")

    command_str = " ".join(pytest_cmd)

    # Build environment
    env = os.environ.copy()
    script_dir = Path(__file__).resolve().parent
    env_updates = runner.build_execution_env(
        test_dir, script_dir, "", shard, shard_type
    )
    env.update(env_updates)

    # NPU device allocation: regular tests use round-robin, distributed use all
    npu_device_id = None
    if shard_type == "regular" and npu_device_count > 0:
        npu_device_id = retry_index % npu_device_count
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(npu_device_id)

    # Run subprocess
    start_time = time.monotonic()
    try:
        proc = subprocess.run(
            pytest_cmd,
            cwd=str(test_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout + 60,
        )
        returncode = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired:
        returncode = -1
        stdout = ""
        stderr = f"Retry timed out after {timeout + 60}s"
    except Exception as e:
        returncode = -1
        stdout = ""
        stderr = f"Retry execution error: {type(e).__name__}: {e}"

    duration = time.monotonic() - start_time

    # Parse JUnit XML for status
    xml_result = runner.parse_junit_xml_status(xml_file)
    if xml_result["status"] == "no_xml":
        status = "error"
        message = xml_result.get("message", "")
    else:
        status = xml_result["status"]
        message = xml_result.get("message", "")

    # Build retry_history preserving the original failure info
    original_failure = {
        "attempt": 1,
        "status": case.get("status", "unknown"),
        "message": (case.get("message", "") or "")[:1000],
        "duration": case.get("duration", 0),
        "returncode": case.get("returncode", 0),
    }

    # Save case log with _retry suffix
    runner.save_case_log(
        report_dir=report_dir,
        shard=shard,
        shard_type=shard_type,
        nodeid=original_nodeid,
        case_idx=case_idx,
        status=status,
        stdout=stdout,
        stderr=stderr,
        duration=duration,
        returncode=returncode,
        command=command_str,
        npu_device_id=npu_device_id,
        retry_count=1,
        retry_history=[original_failure],
        suffix="_retry",
    )

    # Build result dict (replaces original in cases.json)
    result = {
        "nodeid": original_nodeid,
        "status": status,
        "duration": duration,
        "returncode": returncode,
        "message": message,
        "command": command_str,
        "file": case.get("file", ""),
        "case_idx": case_idx,
        "retry_history": [original_failure],
    }
    return result


def merge_and_recalc(
    cases_data: Dict,
    retry_results: Dict[int, Dict],
    retry_duration: float,
) -> Dict:
    """Merge retry results into cases_data and recalculate statistics."""
    cases = cases_data.get("cases", [])

    # Replace original cases with retry results
    for i, case in enumerate(cases):
        if case.get("case_idx") in retry_results:
            cases[i] = retry_results[case["case_idx"]]

    # Recalculate stats
    cases_data["passed"] = sum(1 for c in cases if c.get("status") == "passed")
    cases_data["failed"] = sum(1 for c in cases if c.get("status") == "failed")
    cases_data["errors"] = sum(1 for c in cases if c.get("status") == "error")
    cases_data["timeout"] = sum(1 for c in cases if c.get("status") == "timeout")
    cases_data["skipped"] = sum(1 for c in cases if c.get("status") == "skipped")

    # Add retry duration to total
    cases_data["duration"] = cases_data.get("duration", 0) + retry_duration

    return cases_data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Retry failed test cases in independent subprocesses"
    )
    parser.add_argument(
        "--cases-json", required=True,
        help="Path to shard_*_cases.json from initial test run",
    )
    parser.add_argument(
        "--test-dir", required=True,
        help="Path to PyTorch test directory",
    )
    parser.add_argument(
        "--report-dir", required=True,
        help="Directory for test reports (same as initial run)",
    )
    parser.add_argument(
        "--shard", type=int, required=True,
        help="Shard number",
    )
    parser.add_argument(
        "--shard-type", required=True,
        choices=["distributed", "regular", "core", "tensor", "graph", "others", "custom"],
        help="Test type / category name (affects concurrency and NPU allocation)",
    )
    parser.add_argument(
        "--timeout", type=int, default=1200,
        help="Per-case timeout in seconds (default: 1200)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Max concurrent workers for regular tests (default: 32). "
             "Distributed tests always run serially.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    cases_file = Path(args.cases_json)
    cases_data = load_cases_json(cases_file)
    if cases_data is None:
        print("No cases data to retry, exiting.")
        return 0

    failed_cases = extract_failed_cases(cases_data)
    if not failed_cases:
        print("No failed cases to retry.")
        return 0

    test_dir = Path(args.test_dir).resolve()
    if not test_dir.is_dir():
        print(f"Error: Test directory not found: {test_dir}", file=sys.stderr)
        return 1

    report_dir = Path(args.report_dir).resolve()

    # Determine concurrency
    if args.shard_type == "distributed":
        max_workers = 1
    else:
        max_workers = args.max_workers or 32

    # Detect NPU device count for non-distributed tests (round-robin device allocation)
    npu_device_count = 0
    if args.shard_type != "distributed":
        npu_device_count = runner.get_npu_device_count()

    print(f"{'=' * 80}")
    print(f"Retrying {len(failed_cases)} failed/error/timeout cases")
    print(f"Shard: {args.shard} ({args.shard_type})")
    if max_workers == 1:
        print(f"Execution mode: SERIAL (each case in independent subprocess)")
    else:
        print(f"Execution mode: CONCURRENT ({max_workers} workers, "
              f"each case in independent subprocess)")
    if npu_device_count > 0:
        print(f"NPU devices: {npu_device_count} (round-robin allocation)")
    print(f"Timeout: {args.timeout}s per case")
    print(f"{'=' * 80}\n", flush=True)

    retry_results: Dict[int, Dict] = {}
    retry_start = time.monotonic()

    if max_workers == 1:
        # Serial execution (distributed tests)
        for i, case in enumerate(failed_cases):
            nodeid_short = case["nodeid"][:70]
            print(f"[{i + 1}/{len(failed_cases)}] Retrying: {nodeid_short}", flush=True)
            result = run_single_case_retry(
                case, test_dir, report_dir, args.shard, args.shard_type,
                args.timeout, args.verbose, npu_device_count, i,
            )
            retry_results[case["case_idx"]] = result
            status_icon = {
                "passed": "PASS",
                "failed": "FAIL",
                "error": "ERR",
                "timeout": "TIME",
                "skipped": "SKIP",
            }.get(result["status"], "?")
            print(f"  [{status_icon}] {result['status']:8s} "
                  f"({result['duration']:.1f}s)", flush=True)
    else:
        # Concurrent execution (regular tests)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, case in enumerate(failed_cases):
                future = executor.submit(
                    run_single_case_retry,
                    case, test_dir, report_dir, args.shard, args.shard_type,
                    args.timeout, args.verbose, npu_device_count, i,
                )
                futures[future] = case

            for future in as_completed(futures):
                case = futures[future]
                try:
                    result = future.result()
                    retry_results[case["case_idx"]] = result
                    status_icon = {
                        "passed": "PASS",
                        "failed": "FAIL",
                        "error": "ERR",
                        "timeout": "TIME",
                        "skipped": "SKIP",
                    }.get(result["status"], "?")
                    print(f"  [{status_icon}] {result['status']:8s} "
                          f"{case['nodeid'][:70]} ({result['duration']:.1f}s)",
                          flush=True)
                except Exception as e:
                    print(f"  [ERR]  error    {case['nodeid'][:70]}: {e}",
                          flush=True)

    retry_duration = time.monotonic() - retry_start

    # Merge retry results into cases.json and recalculate stats
    cases_data = merge_and_recalc(cases_data, retry_results, retry_duration)
    with open(cases_file, "w", encoding="utf-8") as f:
        json.dump(cases_data, f, indent=2, ensure_ascii=False)

    # Summary
    passed_after = sum(
        1 for r in retry_results.values() if r["status"] == "passed"
    )
    still_failing = sum(
        1 for r in retry_results.values()
        if r["status"] in ("failed", "error", "timeout")
    )

    print(f"\n{'=' * 80}")
    print(f"Retry Summary:")
    print(f"  Retried:           {len(failed_cases)} cases")
    print(f"  Passed after retry: {passed_after}")
    print(f"  Still failing:     {still_failing}")
    print(f"  Retry duration:    {retry_duration:.1f}s")
    print(f"\nFinal stats: "
          f"{cases_data['passed']} passed, "
          f"{cases_data['failed']} failed, "
          f"{cases_data['errors']} errors, "
          f"{cases_data['timeout']} timeout, "
          f"{cases_data['skipped']} skipped, "
          f"{cases_data.get('total_cases', 0)} total")
    print(f"{'=' * 80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
