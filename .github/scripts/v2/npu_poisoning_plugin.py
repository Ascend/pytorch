#!/usr/bin/env python3
"""
NPU poisoning detection pytest plugin.

Provides per-case NPU poisoning detection inside the pytest process,
replacing the post-process detection that was too late (file-level
execution would silently fail all remaining cases after poisoning).

Two-layer detection (same as v1 run_npu_test_shard.py):
    Layer 1: signature matching on case output (fast, zero overhead for passing)
    Layer 2: probe computation (~1ms, catches unknown patterns)

NPU device binding:
    On pytest startup (pytest_configure), each pytest process acquires an
    exclusive NPU card via an atomic file-lock counter.  ASCEND_RT_VISIBLE_DEVICES
    is set before any torch import, so device_type_test instantiation only sees
    the bound card.  The caller (test-npu.sh) must ensure NUM_PROCS matches the
    modulo range used in _assign_npu_device().

When poisoning is detected, the plugin calls pytest.exit(returncode=70)
to stop the session immediately, preventing cascading failures across
remaining test cases. The upstream run_test.py sees the non-zero exit
and reports the test file as failed.

Usage:
    This plugin is auto-loaded via -p npu_poisoning_plugin when the
    PYTHONPATH includes the v2 scripts directory. No manual registration
    needed — pytest's -p flag handles it.
"""

import fcntl
import os
import signal as _signal

import pytest


# ==============================================================================
# NPU Fatal Error Signatures (migrated from run_npu_test_shard.py:800-814)
# ==============================================================================

NPU_POISONING_EXIT_CODE = 70

NPU_POISONING_SIGNATURES = [
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


# ==============================================================================
# Module-level poisoning state
# ==============================================================================

_poisoned = False
_poisoning_reason = ""


# ==============================================================================
# Detection Functions
# ==============================================================================


def _check_fatal_npu_error(message: str, stdout: str, stderr: str) -> bool:
    """Layer 1: Check if case output contains a known fatal NPU error signature.

    Fast string matching, zero overhead for passing cases.
    """
    combined = (message or "") + "\n" + (stdout or "") + "\n" + (stderr or "")
    return any(sig in combined for sig in NPU_POISONING_SIGNATURES)


def _check_npu_poisoned() -> bool:
    """Layer 2: Probe NPU device health by running a trivial computation.

    When the NPU task queue is in CAN_EXIT state (poisoned by a prior fatal
    error), all operators become silent no-ops — Enqueue() returns without
    executing, output tensors contain uninitialized device memory.

    This probe creates a tensor with a known value and verifies the result.
    If the queue is poisoned, the result will be garbage. If the device is
    in error state, the sync will throw.

    Cost: ~1ms per call. Only invoked on failed/error cases where Layer 1
    did not match, so zero overhead for passing tests.

    A 10-second alarm guards the probe: if the NPU stream is hung (e.g.
    MakeSureQueueEmpty blocks indefinitely in CAN_EXIT state), the probe
    is interrupted and the device is considered poisoned. This prevents
    pytest-timeout from killing the entire session via internal error.
    """
    try:
        import torch

        old_handler = _signal.signal(_signal.SIGALRM, _signal.SIG_IGN)
        _signal.alarm(10)
        try:
            probe = torch.ones(4, device="npu")
            result = probe.sum().item()
        finally:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, old_handler)

        if result != 4.0:
            print(f"  [NPU POISONING] Probe failed: torch.ones(4).sum() = {result} (expected 4.0)", flush=True)
            return True
        return False
    except Exception as e:
        print(f"  [NPU POISONING] Probe raised exception: {type(e).__name__}: {e}", flush=True)
        return True


# ==============================================================================
# NPU Device Binding (fcntl file-lock counter)
# ==============================================================================

_COUNTER_FILE = "/tmp/npu_device_counter.lock"

# Number of NPU devices per pytest process.  Must match NUM_PROCS set by the
# caller (test-npu.sh) so that sum(devices_per_proc × NUM_PROCS) ≤ total NPU
# cards.  Default: 1 card per process with 8 concurrent processes on an 8-card
# machine.
_NPU_DEVICES_PER_PROC = int(os.environ.get("NPU_DEVICES_PER_PROC", "1"))


def _assign_npu_device():
    """Atomically assign NPU device(s) to this pytest process via file-lock counter.

    Called from pytest_configure() BEFORE any test collection (and therefore
    before any ``import torch`` inside test files).  The fcntl file lock ensures
    only one process increments the counter at a time.

    Device assignment is ``(count % num_groups) * devices_per_proc`` where
    ``num_groups = total_cards / devices_per_proc``.

    With NPU_COUNT=8, devices_per_proc=1: count 0→card0, 1→card1, ..., 7→card7.
    With NPU_COUNT=16, devices_per_proc=2: count 0→(0,1), ..., 7→(14,15).
    """
    fd = os.open(_COUNTER_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.lseek(fd, 0, os.SEEK_SET)
        data = os.read(fd, 10)
        count = int(data) if data.strip() else 0
        new_value = str(count + 1).encode()
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, new_value)
        os.ftruncate(fd, len(new_value))

        devices_per_proc = _NPU_DEVICES_PER_PROC
        npu_count = int(os.environ.get("NPU_COUNT", "8"))
        num_groups = max(1, npu_count // devices_per_proc)
        group_id = count % num_groups
        start = group_id * devices_per_proc

        if devices_per_proc == 1:
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(start)
        else:
            ids = ",".join(str(start + i) for i in range(devices_per_proc))
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ids
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ==============================================================================
# Pytest Hooks
# ==============================================================================

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Bind this pytest process to a dedicated NPU card before collection starts.

    Runs before any conftest or test file is imported, so ``torch.npu.device_count()``
    sees the restricted device set.
    """
    _assign_npu_device()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """After each test case completes, check for NPU poisoning.

    Only checks failed or error cases — passing cases have zero overhead.
    Layer 1 (signature) runs first; if no match, Layer 2 (probe) runs.
    """
    global _poisoned, _poisoning_reason

    outcome = yield
    report = outcome.get_result()

    # Only check after the "call" phase (actual test execution, not setup/teardown)
    if report.when != "call":
        return

    # Only check failed or error cases — passing cases are clean
    if report.passed:
        return

    # Gather case output for Layer 1 signature matching
    message = str(report.longrepr) if report.longrepr else ""
    stdout = ""
    stderr = ""

    # pytest's capfd/capsys captures per-case output
    # The captured output is available via item._pytest_capture
    capstdout = getattr(item.config, "_capstdout", None)
    if capstdout is not None:
        try:
            stdout = capstdout.getvalue()
        except Exception:
            pass

    # Layer 1: signature matching (fast)
    if _check_fatal_npu_error(message, stdout, stderr):
        _poisoned = True
        _poisoning_reason = f"Layer 1 signature match in {item.nodeid}"
        print(f"  [NPU POISONING] Detected by Layer 1 (signature) in {item.nodeid}", flush=True)
        return

    # Layer 2: probe computation (~1ms, only if Layer 1 didn't match)
    if _check_npu_poisoned():
        _poisoned = True
        _poisoning_reason = f"Layer 2 probe failed after {item.nodeid}"
        print(f"  [NPU POISONING] Detected by Layer 2 (probe) after {item.nodeid}", flush=True)
        return


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Before each test: if poisoning was detected, stop the session immediately.

    This hook runs before every test case. If the previous case triggered
    poisoning detection (via pytest_runtest_makereport), we immediately
    exit the pytest session with exit code 70.

    This prevents all subsequent cases from running with a poisoned NPU
    device context, avoiding silent failures (false positives).
    """
    global _poisoned

    if _poisoned:
        print(f"  [NPU POISONING] Stopping session before {item.nodeid} "
              f"(reason: {_poisoning_reason})", flush=True)
        # pytest.exit raises SessionInterrupt, which triggers pytest_sessionfinish
        # The returncode=70 is propagated to the parent process
        pytest.exit(f"NPU poisoning detected: {_poisoning_reason}", returncode=NPU_POISONING_EXIT_CODE)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Final safety check: if poisoning was detected but exit didn't fire.

    This handles edge cases where poisoning was detected on the very last
    test case (no subsequent pytest_runtest_protocol to trigger exit).
    """
    global _poisoned

    if _poisoned:
        print(f"  [NPU POISONING] Detected during teardown (reason: {_poisoning_reason})", flush=True)
        # Force exit — this runs after sessionfinish, so we need os._exit
        # to ensure the returncode is propagated correctly
        import os
        os._exit(NPU_POISONING_EXIT_CODE)
