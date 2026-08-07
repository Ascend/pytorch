#!/usr/bin/env python3
"""
NPU test scheduler — per-file execution with device-pool binding.

Replaces the inline bash scheduling loop in test-npu.sh.  Launches one
``run_test.py -i <file>`` invocation per test file, with concurrency
controlled by a ``ThreadPoolExecutor`` and NPU device binding via a
thread-safe device pool (mirrors ``run_test.py``'s Pool slot pattern).

Usage:
    python3 npu_scheduler.py \\
        --expected-files /tmp/files.txt \\
        --npu-count 8 \\
        --devices-per-proc 1 \\
        --output-log /tmp/combined.log \\
        --pytorch-root /path/to/pytorch
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ==============================================================================
# Device Pool — mirrors run_test.py Pool slot pattern:
#   fixed N slots, acquire → use → release, no wrap-around conflict.
# ==============================================================================


class DevicePool:
    """Thread-safe pool of NPU device ids.

    Each device group is a string suitable for ``ASCEND_RT_VISIBLE_DEVICES``.
    ``acquire()`` blocks until a device is available; ``release()`` returns it.
    """

    def __init__(self, npu_count: int, devices_per_proc: int):
        groups = max(1, npu_count // devices_per_proc)
        self._available: list[str] = []
        for i in range(groups):
            if devices_per_proc == 1:
                self._available.append(str(i))
            else:
                start = i * devices_per_proc
                self._available.append(
                    ",".join(str(start + j) for j in range(devices_per_proc))
                )
        self._cv = threading.Condition()

    def acquire(self) -> str:
        """Block until a device is free, then return its id string."""
        with self._cv:
            while not self._available:
                self._cv.wait()
            return self._available.pop()

    def release(self, device: str) -> None:
        """Return a device to the pool."""
        with self._cv:
            self._available.append(device)
            self._cv.notify()


# ==============================================================================
# Single-file execution
# ==============================================================================


def run_one(pytorch_root: str, test_file: str, device: str) -> dict:
    """Execute ``run_test.py -i <test_file>`` on a pre-assigned NPU device.

    Returns:
        dict with keys: test_file, device, rc, duration, stdout, stderr
    """
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = device

    start = time.time()
    try:
        proc = subprocess.run(
            [
                sys.executable,
                f"{pytorch_root}/test/run_test.py",
                "-i", test_file,
                "--hw-classification", "ACCELERATOR",
                "--continue-through-error",
                "--verbose",
            ],
            capture_output=True,
            text=True,
            timeout=3600,  # per-file兜底超时; run_test.py 自带细粒度超时
            env=env,
            cwd=pytorch_root,
        )
        rc = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = -1
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\n[timeout] Killed after {exc.timeout}s"
    elapsed = time.time() - start

    return {
        "test_file": test_file,
        "device": device,
        "rc": rc,
        "duration": elapsed,
        "stdout": stdout,
        "stderr": stderr,
    }


# ==============================================================================
# Worker wrapper — acquires device from pool, runs test, releases device.
# ==============================================================================


def _run_with_device(args, pool: DevicePool, test_file: str) -> dict:
    """Wrapper that acquires a device, runs the test, and releases the device."""
    device = pool.acquire()
    try:
        return run_one(args.pytorch_root, test_file, device)
    finally:
        pool.release(device)


# ==============================================================================
# Scheduler
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NPU test scheduler — per-file run_test.py with device pool"
    )
    parser.add_argument("--expected-files", required=True,
                        help="File listing test module paths (one per line)")
    parser.add_argument("--npu-count", type=int, required=True,
                        help="Total NPU cards available")
    parser.add_argument("--devices-per-proc", type=int, required=True,
                        help="NPU cards visible to each pytest process")
    parser.add_argument("--output-log", required=True,
                        help="Path for combined run_test.py stdout/stderr")
    parser.add_argument("--pytorch-root", required=True,
                        help="Path to pytorch source repository")
    args = parser.parse_args()

    # Read test file list
    with open(args.expected_files) as fh:
        test_files = [line.strip() for line in fh if line.strip()]

    num_workers = args.npu_count // args.devices_per_proc
    total = len(test_files)

    print(f"[scheduler] {total} files, {num_workers} workers, "
          f"{args.npu_count} NPU cards, "
          f"{args.devices_per_proc} device(s)/proc", flush=True)

    if total == 0:
        print("[scheduler] No files to run", flush=True)
        Path(args.output_log).write_text("")
        return

    pool = DevicePool(args.npu_count, args.devices_per_proc)

    passed = 0
    failed = 0

    with open(args.output_log, "w") as log_fh:
        # max_workers = num_workers is correct: when a worker finishes,
        # it releases its device and its thread picks up the next task,
        # which then acquires the just-released device. No deadlock.
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_map = {
                executor.submit(_run_with_device, args, pool, f): f
                for f in test_files
            }

            for future in as_completed(future_map):
                result = future.result()
                f = result["test_file"]
                device = result["device"]
                rc = result["rc"]
                duration = result["duration"]
                stdout = result["stdout"]
                stderr = result["stderr"]

                # Write to combined log
                log_fh.write(f"=== [{f}] device={device} start ===\n")
                if stdout:
                    log_fh.write(stdout)
                if stderr:
                    log_fh.write(stderr)
                log_fh.write(f"=== [{f}] device={device} done "
                             f"(rc={rc}, {duration:.1f}s) ===\n")
                log_fh.flush()

                # Real-time progress
                if rc == 0:
                    passed += 1
                else:
                    failed += 1
                done = passed + failed
                remaining = total - done
                print(f"[{done}/{total}] {f}: rc={rc} ({duration:.1f}s) "
                      f"device={device} "
                      f"({passed}P {failed}F {remaining} left)", flush=True)

    print(f"\n[scheduler] Done: {passed} passed, {failed} failed, {total} total",
          flush=True)

    # Exit non-zero if any file failed, so test-npu.sh can capture it.
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
