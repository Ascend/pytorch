#!/usr/bin/env python3
"""
Run a shard of patched upstream PyTorch tests by reusing upstream test/run_test.py
for test selection, sharding, and special-handler execution.
"""

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
import signal
import sys
import traceback
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Dict, List, Tuple


SPECIAL_TEST_FILE_ALIASES = {
    "test_autoload_enable": "test_autoload.py",
    "test_autoload_disable": "test_autoload.py",
    "test_cpp_extensions_aot_ninja": "test_cpp_extensions_aot.py",
    "test_cpp_extensions_aot_no_ninja": "test_cpp_extensions_aot.py",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run PyTorch NPU tests for a shard")
    parser.add_argument("--shard", type=int, required=True, help="Shard number (1-indexed)")
    parser.add_argument("--num-shards", type=int, required=True, help="Total number of shards")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to PyTorch test directory")
    parser.add_argument("--disabled-testcases", type=str, help="Path to disabled_testcases.json")
    parser.add_argument("--report-dir", type=str, default="test-reports", help="Directory for test reports")
    parser.add_argument("--timeout", type=int, default=600, help="Reserved for compatibility")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser.parse_args()


def load_disabled_testcases(json_file: str) -> int:
    if json_file and os.path.exists(json_file):
        with open(json_file, encoding="utf-8") as f:
            return len(json.load(f))
    return 0


def discover_raw_test_files(test_dir: str) -> List[str]:
    test_path = Path(test_dir)
    files = []
    for test_file in test_path.rglob("test_*.py"):
        rel_path = test_file.relative_to(test_path)
        files.append(str(rel_path).replace("\\", "/"))
    return sorted(files)


def parse_junit_xml(xml_file: str) -> Dict:
    stats = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "duration": 0.0,
    }

    if not os.path.exists(xml_file):
        return stats

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for testsuite in root.iter("testsuite"):
            stats["total"] += int(testsuite.get("tests", 0))
            stats["failed"] += int(testsuite.get("failures", 0))
            stats["skipped"] += int(testsuite.get("skipped", 0))
            stats["errors"] += int(testsuite.get("errors", 0))
            stats["duration"] += float(testsuite.get("time", 0))
        stats["passed"] = stats["total"] - stats["failed"] - stats["skipped"] - stats["errors"]
    except Exception as exc:
        print(f"Warning: Failed to parse XML report {xml_file}: {exc}")

    return stats


def aggregate_junit_stats(report_roots: List[Path]) -> Dict:
    totals = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "duration": 0.0,
    }

    seen_files = set()
    for report_root in report_roots:
        if not report_root.exists():
            continue
        for xml_file in report_root.rglob("*.xml"):
            try:
                resolved = str(xml_file.resolve())
            except OSError:
                resolved = str(xml_file)
            if resolved in seen_files:
                continue
            seen_files.add(resolved)

            stats = parse_junit_xml(str(xml_file))
            for key in totals:
                totals[key] += stats[key]
    return totals


def create_empty_stats() -> Dict:
    return {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "duration": 0.0,
        "junit_generated": False,
        "junit_xml_files": 0,
        "zero_item_test_files": 0,
        "startup_failures": 0,
        "import_failures": 0,
        "test_failures": 0,
    }


def create_shard_info(shard: int, num_shards: int, timestamp: str) -> Dict:
    return {
        "shard": shard,
        "num_shards": num_shards,
        "selection_mode": "upstream_run_test",
        "total_files": 0,
        "upstream_selected_tests": 0,
        "upstream_selected_file_tests": 0,
        "upstream_unhandled_tests": 0,
        "shard_files": 0,
        "excluded_test_files": 0,
        "disabled_count": 0,
        "disabled_count_matched": 0,
        "disabled_count_deselected": 0,
        "junit_generated": False,
        "junit_xml_files": 0,
        "zero_item_test_files": 0,
        "startup_failures": 0,
        "import_failures": 0,
        "test_failures": 0,
        "timestamp": timestamp,
    }


def finalize_stats(base_stats: Dict, returncode: int, duration: float, error_message: str = "") -> Dict:
    stats = dict(base_stats)
    stats["duration"] = max(float(stats.get("duration", 0.0)), duration)
    if returncode != 0:
        stats["returncode"] = returncode
        if returncode < 0:
            signal_num = abs(returncode)
            try:
                signal_name = signal.Signals(signal_num).name
            except ValueError:
                signal_name = f"SIG{signal_num}"
            stats["crashed"] = True
            stats["crash_signal"] = signal_name
        if stats.get("total", 0) == 0:
            stats["errors"] = max(stats.get("errors", 0), 1)
            stats["incomplete"] = True
        if error_message:
            stats["error_message"] = error_message
    else:
        stats["returncode"] = 0
    return stats


def save_stats_file(report_dir: str, shard: int, stats: Dict) -> str:
    os.makedirs(report_dir, exist_ok=True)
    stats_file = os.path.join(report_dir, f"shard_{shard}_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats_file


def save_info_file(report_dir: str, shard: int, info: Dict) -> str:
    os.makedirs(report_dir, exist_ok=True)
    info_file = os.path.join(report_dir, f"shard_{shard}_info.json")
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info_file


def save_test_plan_file(report_dir: str, shard: int, planned_tests: List[str]) -> str:
    os.makedirs(report_dir, exist_ok=True)
    plan_file = os.path.join(report_dir, f"shard_{shard}_planned_test_files.txt")
    with open(plan_file, "w", encoding="utf-8") as f:
        for target in planned_tests:
            f.write(f"{target}\n")
    return plan_file


def save_excluded_test_files_file(report_dir: str, shard: int, test_targets: List[str]) -> str:
    os.makedirs(report_dir, exist_ok=True)
    excluded_file = os.path.join(report_dir, f"shard_{shard}_excluded_test_files.txt")
    with open(excluded_file, "w", encoding="utf-8") as f:
        for target in test_targets:
            f.write(f"{target}\n")
    return excluded_file


def save_unhandled_upstream_tests_file(report_dir: str, shard: int, test_targets: List[str]) -> str:
    os.makedirs(report_dir, exist_ok=True)
    unhandled_file = os.path.join(report_dir, f"shard_{shard}_unhandled_upstream_tests.txt")
    with open(unhandled_file, "w", encoding="utf-8") as f:
        for target in test_targets:
            f.write(f"{target}\n")
    return unhandled_file


def get_disabled_testcases_report_file(report_dir: str, shard: int) -> str:
    return os.path.join(report_dir, f"shard_{shard}_disabled_testcases.json")


def load_disabled_testcases_report(report_dir: str, shard: int) -> Dict:
    report_file = get_disabled_testcases_report_file(report_dir, shard)
    if not os.path.exists(report_file):
        return {
            "disabled_count_matched": 0,
            "disabled_count_deselected": 0,
        }

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "disabled_count_matched": data.get("disabled_count_matched", 0),
            "disabled_count_deselected": data.get("disabled_count_deselected", 0),
        }
    except Exception as exc:
        print(f"Warning: Failed to read disabled testcase report: {exc}")
        return {
            "disabled_count_matched": 0,
            "disabled_count_deselected": 0,
        }


def print_stats_summary(shard: int, stats: Dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"Test Results for Shard {shard}")
    print(f"{'=' * 60}")
    print(f"Total:  {stats['total']}")
    print(f"Passed: {stats['passed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print(f"Duration: {stats['duration']:.2f}s")
    print(f"{'=' * 60}")


@contextmanager
def patched_argv(fake_argv: List[str]):
    original_argv = sys.argv[:]
    sys.argv = fake_argv
    try:
        yield
    finally:
        sys.argv = original_argv


@contextmanager
def prepend_sys_path(paths: List[str]):
    original_sys_path = sys.path[:]
    for path in reversed(paths):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path = original_sys_path


@contextmanager
def patched_environ(updates: Dict[str, str]):
    original_env = os.environ.copy()
    os.environ.update(updates)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_env)


@contextmanager
def patched_cwd(path: Path):
    original_cwd = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(original_cwd))


def build_execution_env(test_dir: Path, script_dir: Path, disabled_testcases_file: str, report_dir: str, shard: int) -> Dict[str, str]:
    import torch

    repo_root = test_dir.parent
    torch_path = str(Path(torch.__file__).parent.parent)
    pythonpath_parts = [torch_path, str(repo_root), str(test_dir), str(script_dir)]
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    updates = {
        "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        "PYTORCH_TEST_NPU": "1",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "1",
        "NO_TD": "1",
        "PYTEST_ADDOPTS": os.environ.get("PYTEST_ADDOPTS", ""),
    }
    if disabled_testcases_file:
        updates["NPU_DISABLED_TESTCASES_JSON"] = os.path.abspath(disabled_testcases_file)
        updates["NPU_DISABLED_TESTCASES_REPORT"] = os.path.abspath(
            get_disabled_testcases_report_file(report_dir, shard)
        )
    return updates


def import_upstream_run_test_module(test_dir: Path, script_dir: Path, env_updates: Dict[str, str]):
    run_test_path = test_dir / "run_test.py"
    repo_root = test_dir.parent
    spec = importlib.util.spec_from_file_location("upstream_run_test", run_test_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load upstream run_test.py from {run_test_path}")

    with patched_environ(env_updates), prepend_sys_path([str(repo_root), str(test_dir), str(script_dir)]), patched_cwd(test_dir):
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    original_get_pytest_args = module.get_pytest_args

    def wrapped_get_pytest_args(options, is_cpp_test=False, is_distributed_test=False):
        args = list(original_get_pytest_args(options, is_cpp_test=is_cpp_test, is_distributed_test=is_distributed_test))
        if not is_cpp_test:
            args.extend(["-p", "pytest_disabled_testcases_plugin"])
        return args

    module.get_pytest_args = wrapped_get_pytest_args
    module.IS_CI = False
    return module


def build_options(module, shard: int, num_shards: int, verbose: bool):
    fake_argv = [
        "run_test.py",
        "--shard",
        str(shard),
        str(num_shards),
        "--continue-through-error",
        "--pipe-logs",
        "--enable-timeout",
    ]
    if verbose:
        fake_argv.append("-v")

    with patched_argv(fake_argv):
        options = module.parse_args()

    options.enable_td = False
    options.upload_artifacts_while_running = False
    options.coverage = False
    options.dry_run = False
    options.pytest = False
    return options


def compute_selection_plan(module, options, test_dir: Path) -> Tuple[List[str], List[object], List[str], List[str], int]:
    selected_tests = module.get_selected_tests(options)
    raw_tests = [module.TestRun(test_name) for test_name in selected_tests]
    test_file_times = module.load_test_file_times()
    test_class_times = module.load_test_class_times()
    _, sharded_tests = module.do_sharding(
        options,
        raw_tests,
        test_file_times,
        test_class_times,
        sort_by_time=True,
    )

    raw_test_files = discover_raw_test_files(str(test_dir))
    selected_backing_files = set()
    unhandled_tests = []
    custom_handlers = getattr(module, "CUSTOM_HANDLERS", {})
    cpp_tests = set(getattr(module, "CPP_TESTS", []))
    for test_name in selected_tests:
        relative_file = f"{test_name}.py"
        if (test_dir / relative_file).exists():
            selected_backing_files.add(relative_file)
            continue
        alias_file = SPECIAL_TEST_FILE_ALIASES.get(test_name)
        if alias_file and (test_dir / alias_file).exists():
            selected_backing_files.add(alias_file)
            continue
        if test_name in custom_handlers or test_name in cpp_tests:
            continue
        unhandled_tests.append(test_name)

    excluded_test_files = sorted(
        relative_file for relative_file in raw_test_files if relative_file not in selected_backing_files
    )
    return selected_tests, sharded_tests, excluded_test_files, sorted(unhandled_tests), len(selected_backing_files)


def clean_test_reports_dir(test_reports_dir: Path) -> None:
    if test_reports_dir.exists():
        shutil.rmtree(test_reports_dir)
    test_reports_dir.mkdir(parents=True, exist_ok=True)


def clean_existing_junit_xml(report_dir: Path) -> None:
    if not report_dir.exists():
        return
    for xml_file in report_dir.rglob("*.xml"):
        xml_file.unlink(missing_ok=True)


def collect_junit_xml_files(report_dir: Path, candidate_roots: List[Path]) -> List[Path]:
    copied_files = []
    seen_sources = set()
    report_dir.mkdir(parents=True, exist_ok=True)

    for candidate_root in candidate_roots:
        if not candidate_root.exists():
            continue
        for xml_file in candidate_root.rglob("*.xml"):
            try:
                resolved = str(xml_file.resolve())
            except OSError:
                resolved = str(xml_file)
            if resolved in seen_sources:
                continue
            seen_sources.add(resolved)

            destination = report_dir / xml_file.name
            if destination.exists():
                if destination.resolve() == xml_file.resolve():
                    copied_files.append(destination)
                    continue
                stem = destination.stem
                suffix = destination.suffix
                with tempfile.NamedTemporaryFile(
                    dir=report_dir,
                    prefix=f"{stem}_",
                    suffix=suffix,
                    delete=False,
                ) as tmp_file:
                    destination = Path(tmp_file.name)
            shutil.copy2(xml_file, destination)
            copied_files.append(destination)

    return copied_files


def classify_test_segment(segment: str) -> str | None:
    normalized = segment.lower()
    if " was successful" in normalized:
        return None
    if "modulenotfounderror:" in normalized or "importerror:" in normalized:
        return "import"
    if (
        "error: unrecognized arguments:" in normalized
        or "no stepcurrent file found" in normalized
        or "running 0 items in this shard:" in normalized and "failed!" in normalized
        or "valueerror: an empty op_list was passed to @ops" in normalized
        or "filenotfounderror:" in normalized and ".xml" in normalized
    ):
        return "startup"
    if " failed!" in normalized:
        return "test"
    return None


def analyze_shard_log(log_file: Path) -> Dict:
    metrics = {
        "zero_item_test_files": 0,
        "startup_failures": 0,
        "import_failures": 0,
        "test_failures": 0,
    }

    if not log_file.exists():
        return metrics

    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return metrics

    metrics["zero_item_test_files"] = content.count("Running 0 items in this shard:")

    current_lines: List[str] = []
    for line in content.splitlines():
        if line.startswith("Running ") and " ... [" in line:
            if current_lines:
                category = classify_test_segment("\n".join(current_lines))
                if category == "startup":
                    metrics["startup_failures"] += 1
                elif category == "import":
                    metrics["import_failures"] += 1
                elif category == "test":
                    metrics["test_failures"] += 1
            current_lines = [line]
            continue
        if current_lines:
            current_lines.append(line)

    if current_lines:
        category = classify_test_segment("\n".join(current_lines))
        if category == "startup":
            metrics["startup_failures"] += 1
        elif category == "import":
            metrics["import_failures"] += 1
        elif category == "test":
            metrics["test_failures"] += 1

    return metrics


def find_shard_log(report_dir: Path, shard: int) -> Path | None:
    direct_path = report_dir / f"test_shard_{shard}.log"
    if direct_path.exists():
        return direct_path

    matches = sorted(report_dir.rglob(f"test_shard_{shard}.log"))
    if matches:
        return matches[0]
    return None


def run_upstream_shard(
    module,
    options,
    sharded_tests,
    shard: int,
    test_dir: Path,
    report_dir: Path,
    env_updates: Dict[str, str],
) -> Tuple[int, Dict, str, Dict]:
    failures = []
    error_message = ""
    start = monotonic()

    with patched_environ(env_updates), patched_cwd(test_dir):
        try:
            module.run_tests(sharded_tests, str(test_dir), options, failures)
            returncode = 0 if not failures else 1
            if failures:
                error_message = "\n".join(failure.message for failure in failures[:20])
        except Exception as exc:
            returncode = 1
            error_message = f"Unexpected error while running upstream run_test shard: {exc}\n{traceback.format_exc()}"

    duration = monotonic() - start
    candidate_report_roots = [report_dir, test_dir / "test-reports"]
    copied_xml_files = collect_junit_xml_files(report_dir, candidate_report_roots)
    stats = aggregate_junit_stats(candidate_report_roots)
    stats["junit_generated"] = bool(copied_xml_files)
    stats["junit_xml_files"] = len(copied_xml_files)

    shard_log = find_shard_log(report_dir, shard)
    log_metrics = analyze_shard_log(shard_log) if shard_log else {
        "zero_item_test_files": 0,
        "startup_failures": 0,
        "import_failures": 0,
        "test_failures": 0,
    }
    stats.update(log_metrics)
    stats = finalize_stats(stats or create_empty_stats(), returncode, duration, error_message)
    if copied_xml_files:
        print(f"Collected {len(copied_xml_files)} JUnit XML file(s) into {report_dir}")
    else:
        print(f"Warning: No JUnit XML files found under: {', '.join(str(path) for path in candidate_report_roots)}")
    return returncode, stats, error_message, log_metrics


def main():
    args = parse_args()
    timestamp = datetime.now().isoformat()
    info = create_shard_info(args.shard, args.num_shards, timestamp)
    info["disabled_count"] = load_disabled_testcases(args.disabled_testcases)

    test_dir = Path(args.test_dir).resolve()
    script_dir = Path(__file__).resolve().parent
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    env_updates = build_execution_env(test_dir, script_dir, args.disabled_testcases, str(report_dir), args.shard)
    module = import_upstream_run_test_module(test_dir, script_dir, env_updates)
    options = build_options(module, args.shard, args.num_shards, args.verbose)
    selected_tests, sharded_tests, excluded_test_files, unhandled_tests, selected_file_tests = compute_selection_plan(
        module,
        options,
        test_dir,
    )

    planned_tests = [test.name for test in sharded_tests]
    info["total_files"] = len(discover_raw_test_files(str(test_dir)))
    info["upstream_selected_tests"] = len(selected_tests)
    info["upstream_selected_file_tests"] = selected_file_tests
    info["upstream_unhandled_tests"] = len(unhandled_tests)
    info["shard_files"] = len(planned_tests)
    info["excluded_test_files"] = len(excluded_test_files)

    save_test_plan_file(str(report_dir), args.shard, planned_tests)
    save_excluded_test_files_file(str(report_dir), args.shard, excluded_test_files)
    save_unhandled_upstream_tests_file(str(report_dir), args.shard, unhandled_tests)

    print(f"\n{'=' * 60}")
    print("PyTorch NPU Upstream run_test Shard Runner")
    print(f"{'=' * 60}")
    print(f"Shard: {args.shard}/{args.num_shards}")
    print(f"Test directory: {test_dir}")
    print(f"Selected tests: {len(selected_tests)}")
    print(f"Tests in shard: {len(planned_tests)}")
    print(f"{'=' * 60}\n")
    for index, target in enumerate(planned_tests, 1):
        print(f"  [{index:03d}] {target}")

    test_reports_dir = test_dir / "test-reports"
    clean_existing_junit_xml(report_dir)
    clean_test_reports_dir(test_reports_dir)

    _, stats, _, log_metrics = run_upstream_shard(
        module,
        options,
        sharded_tests,
        args.shard,
        test_dir,
        report_dir,
        env_updates,
    )
    info["junit_generated"] = bool(stats.get("junit_generated", False))
    info["junit_xml_files"] = int(stats.get("junit_xml_files", 0))
    info["zero_item_test_files"] = int(log_metrics.get("zero_item_test_files", 0))
    info["startup_failures"] = int(log_metrics.get("startup_failures", 0))
    info["import_failures"] = int(log_metrics.get("import_failures", 0))
    info["test_failures"] = int(log_metrics.get("test_failures", 0))
    info.update(load_disabled_testcases_report(str(report_dir), args.shard))

    save_info_file(str(report_dir), args.shard, info)
    save_stats_file(str(report_dir), args.shard, stats)
    print_stats_summary(args.shard, stats)
    sys.exit(stats.get("returncode", 1))


if __name__ == "__main__":
    main()
