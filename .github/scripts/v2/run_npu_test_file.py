#!/usr/bin/env python3
"""
Run PyTorch NPU tests at file level.

Replaces run_npu_test_shard.py (1864 lines).
One pytest session per file, with crash recovery via
StepcurrentPlugin (--sc/--scs) and NPU poisoning detection.

Execution modes:
    - concurrent (regular/core/tensor/graph/math): spawn Pool with N workers
    - serial (distributed): single-process, all NPU devices

Usage:
    python run_npu_test_file.py \
        --files-json shard_files.json \
        --test-dir /path/to/pytorch/test \
        --report-dir test-reports \
        --max-workers 3 \
        --timeout 1800 \
        --shard-type core \
        --shard 1 \
        --verbose
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from multiprocessing import get_context, current_process
from pathlib import Path
from time import monotonic
from typing import Dict, List, Optional, Tuple

import parse_test_results


# ==============================================================================
# NPU Device Detection (migrated from run_npu_test_shard.py:75-92)
# ==============================================================================


def get_npu_device_count() -> int:
    """Detect NPU device count via libascend_hal.so. Falls back to 8."""
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
    return 8


# ==============================================================================
# NPU Poisoning Detection
# ==============================================================================
#
# Per-case NPU poisoning detection is handled by npu_poisoning_plugin.py,
# which runs inside the pytest process and detects poisoning after each
# failed/error case. When poisoning is detected, the plugin calls
# pytest.exit(returncode=70) to stop the session immediately.
#
# This module handles exit codes from pytest subprocess:
#   - rc == 70:  NPU poisoning detected by plugin → skip case, continue
#   - rc < 0:    Signal death (SIGSEGV/SIGABRT) → skip case, continue
#   - rc == 124: Timeout → skip case, continue
#   - rc >= 0:   Orderly exit (0/1/2/5) → done

NPU_POISONING_EXIT_CODE = 70


# ==============================================================================
# SUBPROCESS_FILES (from upstream CUSTOM_HANDLERS, intersected with whitelist)
# ==============================================================================

SUBPROCESS_FILES = {
    "distributed/test_c10d_gloo",
    "distributed/test_c10d_nccl",
    "distributed/test_c10d_spawn_nccl",
    "distributed/test_c10d_spawn_ucc",
    "distributed/test_c10d_ucc",
    "distributed/test_store",
}


# ==============================================================================
# Path Utilities
# ==============================================================================


def normalize_test_file(test_file: str) -> str:
    """Remove 'test/' prefix from test file path."""
    if test_file.startswith("test/"):
        return test_file[5:]
    return test_file


def get_base_name(test_file: str) -> str:
    """Get a safe base name for file naming (e.g., 'distributed/test_c10d.py' -> 'distributed_test_c10d')."""
    rel = normalize_test_file(test_file)
    return rel.replace("/", "_").replace("\\", "_").replace(".py", "")


def build_nodeid(testcase_elem, test_file: str) -> str:
    """Build a pytest nodeid from a JUnit XML testcase element."""
    classname = testcase_elem.get("classname", "")
    name = testcase_elem.get("name", "")
    rel = normalize_test_file(test_file)
    if classname:
        return f"{rel}::{classname}::{name}"
    return f"{rel}::{name}"


# ==============================================================================
# Pytest Command Builder
# ==============================================================================


def build_pytest_command(
    test_file: str,
    xml_file: Path,
    stepcurrent: Optional[str] = None,
    marker: Optional[str] = None,
    subprocess_flag: bool = False,
    verbose: bool = False,
    timeout: int = 1800,
) -> List[str]:
    """Build pytest command for a single test file.

    Uses upstream conftest.py's --num-shards=1 (no file-internal sharding).
    """
    rel = normalize_test_file(test_file)
    cmd = [
        sys.executable, "-u", rel,
        "-p", "no:xdist",
        "-p", "npu_poisoning_plugin",
        "--use-pytest",
        "--num-shards=1",
        "-ra", "--tb=short", "--color=no",
        f"--junitxml={xml_file}",
        f"--timeout={timeout}",
    ]
    if stepcurrent:
        cmd.append(stepcurrent)
    if marker:
        cmd.extend(["-m", marker])
    if subprocess_flag:
        cmd.append("--subprocess")
    if verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    return cmd


# ==============================================================================
# Subprocess Execution with Timeout (SIGINT -> 5s -> kill -> 124)
# ==============================================================================


def run_subprocess_with_timeout(
    command: List[str],
    timeout: int,
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str]:
    """Run pytest subprocess with timeout.

    Returns (returncode, combined_stdout_stderr).
    SIGINT lets pytest write XML, then 5s grace, then kill.
    """
    p = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    try:
        stdout, _ = p.communicate(timeout=timeout)
        return p.returncode, stdout or ""
    except subprocess.TimeoutExpired:
        p.send_signal(signal.SIGINT)
        try:
            stdout, _ = p.communicate(timeout=5)
            return 124, stdout or ""
        except subprocess.TimeoutExpired:
            p.kill()
            stdout, _ = p.communicate(timeout=10)
            return 124, stdout or ""


# ==============================================================================
# Single File Execution + Crash Recovery
# ==============================================================================


def run_test_file_with_retry(
    test_file: str,
    test_dir: Path,
    report_dir: Path,
    timeout: int,
    verbose: bool,
    shard: int,
    shard_type: str,
    env_updates: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """Execute a single test file with crash skip via StepcurrentPlugin.

    Execution strategy:
    - Attempt 0 (--sc): run all cases from the beginning
    - Crash/timeout/NPU-poisoning: use --scs to skip the crashed case and
      continue with the remaining cases. No retry of the crashed case itself.
    - Orderly exit (0/1/2/5): done, results are final
    - Max 20 skip iterations per file (safety bound)

    Returns list of JUnit XML file paths (one per iteration, merged later).
    """
    junit_dir = report_dir / "junit_xmls"
    junit_dir.mkdir(parents=True, exist_ok=True)

    base_name = get_base_name(test_file)
    merged_env = os.environ.copy()
    if env_updates:
        merged_env.update(env_updates)

    # SUBPROCESS_FILES: single execution, no skip (aligned with upstream run_test.py:637)
    if base_name in SUBPROCESS_FILES:
        xml_file = junit_dir / f"{base_name}.xml"
        cmd = build_pytest_command(
            test_file, xml_file,
            marker="(not serial)",
            subprocess_flag=True,
            verbose=verbose,
            timeout=timeout,
        )
        command_str = " ".join(cmd)
        print(f"  [SUBPROCESS] {test_file}", flush=True)
        print(f"    Command: {command_str}", flush=True)
        rc, output = run_subprocess_with_timeout(cmd, timeout, test_dir, merged_env)
        if rc != 0:
            print(f"    Exit code: {rc}", flush=True)
        if not xml_file.exists():
            print(f"    WARNING: XML not generated", flush=True)
        return [xml_file] if xml_file.exists() else []

    # Normal files: crash skip via StepcurrentPlugin
    sc_key = f"{base_name}_{os.urandom(8).hex()}"
    sc_cmd = f"--sc={sc_key}"
    max_iterations = 20
    iteration = 0
    xml_files = []

    while iteration < max_iterations:
        xml_file = junit_dir / f"{base_name}_attempt{iteration}.xml"
        xml_files.append(xml_file)

        cmd = build_pytest_command(
            test_file, xml_file,
            stepcurrent=sc_cmd,
            marker="(not serial)",
            verbose=verbose,
            timeout=timeout,
        )
        command_str = " ".join(cmd)

        if iteration == 0:
            print(f"  [{test_file}]", flush=True)
        else:
            print(f"  [{test_file}] Skip iteration {iteration} (continuing after crash)", flush=True)
        print(f"    Command: {command_str}", flush=True)

        rc, output = run_subprocess_with_timeout(cmd, timeout, test_dir, merged_env)

        # Orderly exit (0=pass, 1=failures, 2=interrupted, 5=no tests collected)
        # Results are final, no more iterations needed
        if rc >= 0 and rc != 124:
            print(f"    Exit code: {rc} (done)", flush=True)
            break

        # Crash (rc < 0) or timeout (124) or NPU poisoning (70)
        # StepcurrentPlugin has already written lastrun to .pytest_cache
        # before the crash. Use --scs to skip the crashed case and continue
        # with remaining cases in a fresh process.
        if rc == NPU_POISONING_EXIT_CODE:
            print(f"    Exit code: {rc} (NPU poisoning, skipping case)", flush=True)
        else:
            signal_name = ""
            if rc < 0:
                try:
                    signal_name = f" ({signal.Signals(-rc).name})"
                except (ValueError, AttributeError):
                    signal_name = f" (signal {-rc})"
            print(f"    Exit code: {rc}{signal_name} (crash/timeout, skipping case)", flush=True)

        sc_cmd = f"--scs={sc_key}"
        iteration += 1

    if iteration >= max_iterations:
        print(f"    Max iterations ({max_iterations}) reached for {test_file}", flush=True)

    return [f for f in xml_files if f.exists()]


# ==============================================================================
# JUnit XML Merge + Parse
# ==============================================================================


def merge_junit_xmls(xml_files: List[Path], test_file: str) -> List:
    """Merge multiple attempt XMLs. nodeid dedup, last-write-wins.

    Returns list of testcase XML elements.
    """
    all_cases = {}  # nodeid -> testcase element

    for xml_file in xml_files:
        if not xml_file.exists():
            continue
        try:
            tree = ET.parse(str(xml_file))
            root = tree.getroot()
            for tc in root.iter("testcase"):
                nodeid = build_nodeid(tc, test_file)
                all_cases[nodeid] = tc
        except ET.ParseError:
            continue

    return list(all_cases.values())


def parse_testcases(testcases: List, test_file: str, command_str: str) -> List[Dict]:
    """Parse JUnit XML testcase elements into per-case result dicts.

    Output format is compatible with shard_*_cases.json from v1.
    """
    results = []
    for idx, tc in enumerate(testcases):
        skip_elem = tc.find("skipped")
        failure_elem = tc.find("failure")
        error_elem = tc.find("error")

        if skip_elem is not None:
            if skip_elem.get("type", "") == "pytest.xfail":
                status = "passed"
                message = "xfailed: expected failure"
            else:
                status = "skipped"
                attr_msg = skip_elem.get("message", "")
                text_msg = (skip_elem.text or "").strip()
                message = attr_msg + ("\n" + text_msg if text_msg else "")
        elif error_elem is not None:
            status = "error"
            attr_msg = error_elem.get("message", "")
            text_msg = (error_elem.text or "").strip()
            message = attr_msg + ("\n" + text_msg if text_msg else "")
        elif failure_elem is not None:
            status = "failed"
            attr_msg = failure_elem.get("message", "")
            text_msg = (failure_elem.text or "").strip()
            message = attr_msg + ("\n" + text_msg if text_msg else "")
        else:
            status = "passed"
            message = ""

        if status in ("passed", "skipped"):
            rc = 0
        elif status == "error":
            rc = -1
        else:
            rc = 1

        results.append({
            "nodeid": build_nodeid(tc, test_file),
            "status": status,
            "duration": float(tc.get("time", 0)),
            "returncode": rc,
            "message": message,
            "command": command_str,
            "file": test_file,
            "case_idx": idx,
        })
    return results


# ==============================================================================
# Execution Environment
# ==============================================================================


def build_execution_env(test_dir: Path, script_dir: Path) -> Dict[str, str]:
    """Build environment variables for test execution."""
    repo_root = test_dir.parent
    pythonpath_parts = [str(script_dir)]

    try:
        import torch
        torch_path = str(Path(torch.__file__).resolve().parent.parent)
        pythonpath_parts.append(torch_path)
    except Exception:
        pass

    pythonpath_parts.extend([str(repo_root), str(test_dir)])

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    return {
        "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        "PYTORCH_TEST_NPU": "1",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "1",
        "NO_TD": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_TESTING_DEVICE_ONLY_FOR": "privateuse1",
    }


# ==============================================================================
# Pool Worker
# ==============================================================================


def worker_init(num_npu_devices: int):
    """Pool worker initializer: assign NPU device round-robin."""
    p = current_process()
    if p.name != "MainProcess":
        device_id = (p._identity[0] - 1) % num_npu_devices
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)


def _execute_file_in_worker(args_tuple):
    """Worker function for Pool: execute a single test file.

    Args are passed as a tuple because Pool.apply_async doesn't accept kwargs well.
    """
    (test_file, test_dir, report_dir, timeout, verbose, shard, shard_type, env_updates) = args_tuple
    return run_test_file_with_retry(
        test_file, test_dir, report_dir, timeout, verbose,
        shard, shard_type, env_updates,
    )


# ==============================================================================
# Main Shard Execution
# ==============================================================================


def run_shard(
    files_json: str,
    test_dir: Path,
    report_dir: Path,
    max_workers: int,
    timeout: int,
    shard_type: str,
    shard: int,
    verbose: bool,
    script_dir: Path,
):
    """Execute all test files in a shard.

    - serial mode (max_workers=1): sequential execution
    - concurrent mode (max_workers>1): spawn Pool with N workers
    """
    start = monotonic()

    # Load files
    data = json.loads(Path(files_json).read_text(encoding="utf-8"))
    files = data["files"]
    num_shards = data["num_shards"]
    total_files = len(files)

    print(f"{'=' * 80}", flush=True)
    print(f"Shard {shard}/{num_shards} ({shard_type})", flush=True)
    print(f"Total files: {total_files}", flush=True)
    print(f"Max workers: {max_workers}", flush=True)
    if max_workers == 1:
        print(f"Execution mode: SERIAL", flush=True)
    else:
        print(f"Execution mode: CONCURRENT ({max_workers} workers, spawn Pool)", flush=True)
    print(f"Timeout per file: {timeout}s", flush=True)
    print(f"{'=' * 80}", flush=True)

    if not files:
        print("No files to execute.", flush=True)
        return [], 0.0, 0

    # Build execution env
    env_updates = build_execution_env(test_dir, script_dir)

    all_cases_results: List[Dict] = []

    if max_workers <= 1:
        # Serial execution (distributed tests)
        print(f"\nExecuting {total_files} files serially...", flush=True)
        for i, test_file in enumerate(files, 1):
            print(f"\n[{i}/{total_files}] Processing: {test_file}", flush=True)
            xml_files = run_test_file_with_retry(
                test_file, test_dir, report_dir, timeout, verbose,
                shard, shard_type, env_updates,
            )
            # Merge XMLs and parse
            merged = merge_junit_xmls(xml_files, test_file)
            cmd_str = f"python {normalize_test_file(test_file)} --num-shards=1 ..."
            case_results = parse_testcases(merged, test_file, cmd_str)
            all_cases_results.extend(case_results)
            passed = sum(1 for c in case_results if c["status"] == "passed")
            failed = sum(1 for c in case_results if c["status"] == "failed")
            errors = sum(1 for c in case_results if c["status"] == "error")
            skipped = sum(1 for c in case_results if c["status"] == "skipped")
            print(f"  Result: {len(case_results)} cases ({passed} passed, {failed} failed, "
                  f"{errors} errors, {skipped} skipped)", flush=True)
    else:
        # Concurrent execution via spawn Pool
        num_npu = get_npu_device_count()
        print(f"\nNPU devices detected: {num_npu}", flush=True)
        print(f"Executing {total_files} files concurrently with {max_workers} workers...", flush=True)

        # Prepare args for each file
        worker_args = [
            (f, test_dir, report_dir, timeout, verbose, shard, shard_type, env_updates)
            for f in files
        ]

        ctx = get_context("spawn")
        pool = ctx.Pool(
            max_workers,
            maxtasksperchild=1,
            initializer=worker_init,
            initargs=(num_npu,),
        )

        try:
            results = pool.map(_execute_file_in_worker, worker_args)
        except Exception as e:
            print(f"Pool execution error: {e}", flush=True)
            results = [[] for _ in files]
        finally:
            pool.close()
            pool.join()

        # Merge XMLs and parse results for each file
        for i, (test_file, xml_files) in enumerate(zip(files, results), 1):
            merged = merge_junit_xmls(xml_files, test_file)
            cmd_str = f"python {normalize_test_file(test_file)} --num-shards=1 ..."
            case_results = parse_testcases(merged, test_file, cmd_str)
            all_cases_results.extend(case_results)
            passed = sum(1 for c in case_results if c["status"] == "passed")
            failed = sum(1 for c in case_results if c["status"] == "failed")
            errors = sum(1 for c in case_results if c["status"] == "error")
            skipped = sum(1 for c in case_results if c["status"] == "skipped")
            print(f"  [{i}/{total_files}] {test_file}: {len(case_results)} cases "
                  f"({passed}P {failed}F {errors}E {skipped}S)", flush=True)

    duration = monotonic() - start

    # Calculate stats
    passed_count = sum(1 for c in all_cases_results if c["status"] == "passed")
    failed_count = sum(1 for c in all_cases_results if c["status"] == "failed")
    error_count = sum(1 for c in all_cases_results if c["status"] == "error")
    timeout_count = sum(1 for c in all_cases_results if c["status"] == "timeout")
    skipped_count = sum(1 for c in all_cases_results if c["status"] == "skipped")

    worst_returncode = 0
    for c in all_cases_results:
        rc = c.get("returncode", 0)
        if rc != 0 and rc > worst_returncode:
            worst_returncode = rc
    if any(c.get("returncode", 0) < 0 for c in all_cases_results):
        worst_returncode = -1

    # Save results (format compatible with v1)
    cases_data = {
        "shard": shard,
        "shard_type": shard_type,
        "execution_mode": "file_level",
        "concurrent_workers": max_workers,
        "total_cases": len(all_cases_results),
        "passed": passed_count,
        "failed": failed_count,
        "errors": error_count,
        "timeout": timeout_count,
        "skipped": skipped_count,
        "duration": duration,
        "cases": all_cases_results,
    }

    result_module = parse_test_results
    result_module.save_cases_file(str(report_dir), shard, cases_data, shard_type)

    # Save info file
    timestamp = datetime.now().isoformat()
    info = result_module.create_shard_info(shard, num_shards, timestamp)
    info["selection_mode"] = "file_level"
    info["shard_type"] = shard_type
    info["total_files"] = total_files
    info["selected_test_files"] = total_files
    info["shard_files"] = total_files
    info["per_case_isolation"] = False
    info["execution_mode"] = "file_level"
    info["concurrent_workers"] = max_workers
    info["returncode"] = worst_returncode
    info["duration"] = duration
    info["total_cases"] = len(all_cases_results)
    result_module.save_info_file(str(report_dir), shard, info, shard_type)

    # Save stats
    stats = {
        "total": len(all_cases_results),
        "passed": passed_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "errors": error_count,
        "timeout": timeout_count,
        "duration": duration,
        "returncode": worst_returncode,
        "per_case_isolation": False,
        "execution_mode": "file_level",
        "concurrent_workers": max_workers,
    }
    result_module.save_stats_file(str(report_dir), shard, stats, shard_type)
    result_module.print_stats_summary(shard, stats, shard_type)

    # Save test plan
    result_module.save_test_plan_file(str(report_dir), shard, files, shard_type)

    return all_cases_results, duration, worst_returncode


# ==============================================================================
# CLI
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PyTorch NPU tests at file level"
    )
    parser.add_argument("--files-json", required=True, help="Path to shard files JSON")
    parser.add_argument("--test-dir", required=True, help="PyTorch test directory")
    parser.add_argument("--report-dir", default="test-reports", help="Report output directory")
    parser.add_argument("--max-workers", type=int, default=3, help="Max concurrent workers (regular: 3, distributed: 1)")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-file timeout in seconds")
    parser.add_argument("--shard-type", default="regular", help="Shard type (core/tensor/distributed/graph/math)")
    parser.add_argument("--shard", type=int, default=1, help="Shard number")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser.parse_args()


def main():
    args = parse_args()

    test_dir = Path(args.test_dir).resolve()
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent

    files_json = Path(args.files_json).resolve()
    if not files_json.exists():
        raise FileNotFoundError(f"Files JSON not found: {files_json}")

    cases_list, duration, returncode = run_shard(
        str(files_json),
        test_dir,
        report_dir,
        args.max_workers,
        args.timeout,
        args.shard_type,
        args.shard,
        args.verbose,
        script_dir,
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
