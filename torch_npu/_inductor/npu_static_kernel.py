__all__ = []

import os
import subprocess
import datetime
from pathlib import Path

import torch_npu
from .config import log

_uninstall_path = None


def safe_resolve_output_dir(build_dir: str):
    base_dir = Path.cwd().resolve()
    if build_dir is not None:
        if "\x00" in build_dir:
            raise ValueError("build_dir contains null byte")

        candidate = Path(build_dir)
        if ".." in candidate.parts:
            raise ValueError("build_dir must not contain '..'")

        script_dir = candidate if candidate.is_absolute() else base_dir / candidate

        cur = Path(script_dir.anchor)
        for part in script_dir.parts[1:]:
            cur = cur / part
            if cur.exists() and cur.is_symlink():
                raise ValueError(f"symlink detected in path: {cur}")

        try:
            script_dir = script_dir.resolve(strict=False)
        except Exception as e:
            raise ValueError(f"cannot resolve path {script_dir}: {e}")
    else:
        script_dir = base_dir

    timestamp = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}_{os.getpid()}"
    result_root = script_dir / f"{timestamp}_kernel_aot_optimization_build_outputs"

    try:
        result_root.mkdir(exist_ok=True)
    except (PermissionError, OSError) as e:
        raise RuntimeError(f"failed to create output directory {result_root}: {e}") from e

    return result_root


class AclopDumpContext:
    def __init__(self, save_path: str):
        self.save_path = save_path

    def __enter__(self):
        torch_npu._C._aclop_start_dmp(self.save_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch_npu._C._aclop_stop_dump()


def save_uninstall_info(filename: str):
    global _uninstall_path
    latest = Path(os.environ["ASCEND_HOME_PATH"])
    root = latest.parent
    pattern = f"*/opp/static_kernel/ai_core/{filename}/uninstall.sh"
    match = next(root.glob(pattern), None)
    if match is None:
        _uninstall_path = None
        log.debug(f"can not find uninstall path, pattern: {pattern}")
    else:
        _uninstall_path = str(match)


class StaticKernelCompiler:
    def __init__(self, build_dir=None):
        self.result_root = safe_resolve_output_dir(build_dir)
        log.debug(f"StaticKernelCompiler initialized. Build directory: {self.result_root}")

    def __enter__(self):
        log.info(f"Starting operator dump...")
        torch_npu._C._aclop_start_dump(str(self.result_root))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch_npu._C._aclop_stop_dump()
        log.info(f"Stopping operator dump.")

        if exc_type:
            log.error(f"An exception occurred during model execution: {exc_val}")
            log.info(f"Skipping static kernel compilation due to the error.")
            return

        log.info(f"Starting static kernel compilation process...")
        debug_dirs = [d for d in self.result_root.iterdir() if d.is_dir() and d.name.endswith("_debug")]
        if not debug_dirs:
            log.error(f"Can not find json of ops, skipping op_compiler.")
            return

        debug_dir = max(debug_dirs, key=lambda d: d.stat().st_mtime)
        json_files = list(debug_dir.glob("*.json"))
        if not json_files:
            log.error(f"No json files in {debug_dir}, skipping op_compiler.")
            return

        cmd = [
            "op_compiler",
            "-p", str(debug_dir),
            "-v", torch_npu.npu.get_device_name(),
            "-l", "info",
            "-j", "4",
            "-o", str(self.result_root),
        ]
        try:
            log.debug(f"Executing op_compiler command: {' '.join(cmd)}")
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            log.debug(f"op_compiler execution successful, msg: {res.stdout}")
        except subprocess.CalledProcessError as e:
            log.error(f"op_compiler execution failed, msg: {e.stderr}")
            return

        for run_pkg in self.result_root.glob("*.run"):
            filename = run_pkg.name
            try:
                log.info(f"Installing static kernel package: {filename}")
                result = subprocess.run([str(run_pkg)], check=True, capture_output=True, text=True)
                log.info(f"{filename} install successful, msg: {result.stdout}")
                save_uninstall_info(filename[:-4])
                torch_npu.npu._aclnn_reselect_static_kernel()
            except subprocess.CalledProcessError as e:
                log.error(f" {filename} install failed, msg: {e.stderr}")


def uninstall_static_kernel():
    global _uninstall_path
    if not _uninstall_path:
        log.debug(f"uninstall_path is none, skip uninstall static kernel")
        return

    try:
        result = subprocess.run(
            [_uninstall_path],
            check=True,
            capture_output=True,
            text=True,
        )
        log.debug(f"{_uninstall_path} uninstall success, msg: \n{result.stdout}")
    except subprocess.CalledProcessError as e:
        log.error(f"{_uninstall_path} uninstall failed, msg: \n{e.stderr}")
    finally:
        _uninstall_path = None