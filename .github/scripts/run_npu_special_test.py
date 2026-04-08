#!/usr/bin/env python3
"""Run special upstream-selected tests that require custom handling."""

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
from time import monotonic


SPECIAL_TEST_GROUPS = {
    "test_autoload_enable": "autoload",
    "test_autoload_disable": "autoload",
    "test_cpp_extensions_aot_ninja": "cpp_extension",
    "test_cpp_extensions_aot_no_ninja": "cpp_extension",
    "distributed/test_distributed_spawn": "distributed",
    "distributed/algorithms/quantization/test_quantization": "distributed",
    "distributed/test_c10d_common": "distributed",
    "distributed/test_c10d_nccl": "distributed",
    "distributed/test_c10d_spawn_nccl": "distributed",
    "distributed/test_store": "distributed",
    "distributed/test_pg_wrapper": "distributed",
}


def sanitize_test_name(test_name: str) -> str:
    return test_name.replace('/', '__')


def parse_args():
    parser = argparse.ArgumentParser(description="Run a torch-npu special test")
    parser.add_argument("--test-name", required=True, choices=sorted(SPECIAL_TEST_GROUPS))
    parser.add_argument("--repo-root", required=True, help="Prepared upstream repository root")
    parser.add_argument("--report-dir", default="special-test-reports")
    parser.add_argument("--timeout", type=int, default=7200)
    return parser.parse_args()


def build_python_env(test_dir: Path) -> dict:
    env = os.environ.copy()
    import torch

    torch_path = Path(torch.__file__).parent.parent
    script_dir = Path(__file__).resolve().parent
    pythonpath_parts = [str(torch_path), str(test_dir), str(script_dir)]
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["PYTORCH_TEST_NPU"] = "1"
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = env.get("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
    return env


def run_command(cmd, cwd: Path, env: dict, log_path: Path, timeout: int) -> tuple[int, float, str]:
    start = monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        output = result.stdout or ""
        with open(log_path, "a", encoding="utf-8") as log_file:
            if log_path.exists() and log_path.stat().st_size > 0:
                log_file.write("\n")
            log_file.write(output)
        note = ""
        if result.returncode < 0:
            signal_num = abs(result.returncode)
            try:
                note = signal.Signals(signal_num).name
            except ValueError:
                note = f"SIG{signal_num}"
        return result.returncode, monotonic() - start, note
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        with open(log_path, "a", encoding="utf-8") as log_file:
            if log_path.exists() and log_path.stat().st_size > 0:
                log_file.write("\n")
            log_file.write((output or "") + "\nTimed out\n")
        return 124, monotonic() - start, f"Timed out after {timeout}s"


def find_install_directory(install_root: Path) -> Path:
    for root, directories, _ in os.walk(install_root):
        for directory in directories:
            if "-packages" in directory:
                return Path(root) / directory
    raise RuntimeError(f"Failed to locate site-packages under {install_root}")


def run_autoload_test(test_name: str, repo_root: Path, report_dir: Path, timeout: int) -> tuple[int, float, str, str]:
    test_dir = repo_root / "test"
    env = build_python_env(test_dir)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1" if test_name.endswith("enable") else "0"
    log_path = report_dir / f"{sanitize_test_name(test_name)}.log"
    returncode, duration, note = run_command(
        [sys.executable, "test_autoload.py"],
        cwd=test_dir,
        env=env,
        log_path=log_path,
        timeout=timeout,
    )
    return returncode, duration, note, str(log_path)


def run_cpp_extension_test(test_name: str, repo_root: Path, report_dir: Path, timeout: int) -> tuple[int, float, str, str]:
    use_ninja = test_name.endswith("_ninja")
    test_dir = repo_root / "test"
    cpp_extensions_dir = test_dir / "cpp_extensions"
    cpp_extensions_test_dir = cpp_extensions_dir / "test"
    build_dir = cpp_extensions_dir / "build"
    install_dir = cpp_extensions_dir / "install"
    log_path = report_dir / f"{sanitize_test_name(test_name)}.log"
    env = build_python_env(test_dir)
    env["USE_NINJA"] = "1" if use_ninja else "0"

    import torch
    from torch.utils import cpp_extension

    if use_ninja:
        cpp_extension.verify_ninja_availability()

    if build_dir.exists():
        shutil.rmtree(build_dir)
    if install_dir.exists():
        shutil.rmtree(install_dir)

    returncode, duration, note = run_command(
        [sys.executable, "setup.py", "install", "--root", "./install"],
        cwd=cpp_extensions_dir,
        env=env,
        log_path=log_path,
        timeout=timeout,
    )
    if returncode != 0:
        return returncode, duration, note, str(log_path)

    installed_site_packages = find_install_directory(install_dir)
    generated_test = cpp_extensions_test_dir / f"{test_name}.py"
    shutil.copyfile(cpp_extensions_test_dir / "test_cpp_extensions_aot.py", generated_test)

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(installed_site_packages), existing_pythonpath] if existing_pythonpath else [str(installed_site_packages)]
    )

    try:
        test_returncode, test_duration, test_note = run_command(
            [sys.executable, generated_test.name],
            cwd=cpp_extensions_test_dir,
            env=env,
            log_path=log_path,
            timeout=timeout,
        )
        return test_returncode, duration + test_duration, test_note, str(log_path)
    finally:
        if generated_test.exists():
            generated_test.unlink()


def run_distributed_test(test_name: str, repo_root: Path, report_dir: Path, timeout: int) -> tuple[int, float, str, str]:
    del timeout

    ascend_repo_root = Path(__file__).resolve().parents[2]
    npu_run_test_path = ascend_repo_root / 'test' / 'run_test.py'
    spec = importlib.util.spec_from_file_location('torch_npu_run_test', npu_run_test_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to load test runner from {npu_run_test_path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    log_path = report_dir / f"{sanitize_test_name(test_name)}.log"
    start = monotonic()
    previous_cwd = Path.cwd()
    previous_env = os.environ.copy()

    options = SimpleNamespace(
        additional_unittest_args=[],
        verbose=1,
        init_method=0,
    )

    test_directory = str(repo_root / 'test')
    os.chdir(str(ascend_repo_root))

    try:
        from contextlib import redirect_stdout, redirect_stderr

        with open(log_path, 'w', encoding='utf-8') as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
            err_msg = module.run_test_module(test_name, test_directory, options)
        returncode = 0 if err_msg is None else 1
        note = err_msg or ''
        return returncode, monotonic() - start, note, str(log_path)
    finally:
        os.chdir(str(previous_cwd))
        os.environ.clear()
        os.environ.update(previous_env)


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.test_name.startswith("test_autoload"):
        returncode, duration, note, log_file = run_autoload_test(
            args.test_name,
            repo_root,
            report_dir,
            args.timeout,
        )
    elif SPECIAL_TEST_GROUPS[args.test_name] == 'distributed':
        returncode, duration, note, log_file = run_distributed_test(
            args.test_name,
            repo_root,
            report_dir,
            args.timeout,
        )
    else:
        returncode, duration, note, log_file = run_cpp_extension_test(
            args.test_name,
            repo_root,
            report_dir,
            args.timeout,
        )

    status = "PASSED" if returncode == 0 else "FAILED"
    if returncode == 124:
        status = "TIMEOUT"

    result = {
        "name": args.test_name,
        "group": SPECIAL_TEST_GROUPS[args.test_name],
        "status": status,
        "returncode": returncode,
        "duration": duration,
        "note": note,
        "log_file": Path(log_file).name,
    }
    result_path = report_dir / f"special_test_{sanitize_test_name(args.test_name)}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if returncode == 0 else 1)


if __name__ == "__main__":
    main()