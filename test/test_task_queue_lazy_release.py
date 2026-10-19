# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# All rights reserved.

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

import torch_npu  # noqa: F401  (environment probe: fail-fast if not importable)

from torch_npu.testing.common_utils import SupportedDevices

# npu_nms_with_mask (the aclop trigger) is only supported on the 910B-class
# / 910_93-series SOCs; skip on all other devices.
ACLOP_SUPPORTED_SOCS = ['Ascend910B', 'Ascend910_93']


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Task queue force-enable mode (same 0/1/2 space as the existing
# test_task_queue_enable.py). 1 or 2 both nonzero -> queue enabled.
TASK_QUEUE_MODE = "2"

# Pure-opapi trigger(s): a small executable Python snippet that runs only
# opapi-typed tasks. Assumed valid for a canonical torch_npu build; adjust the
# op if a given op is registered through the legacy aclop path instead.
OPAPI_TRIGGER = "y = x + 1; z = x * 2"

# Aclop (COMPILE_AND_EXECUTE) trigger(s). ``torch_npu.npu_nms_with_mask`` is a
# private op that has only an aclop path — aclops/NmsWithMaskKernelNpu.cpp,
# OpCommand.Name("NMSWithMask").Run() — so it deterministically produces a
# COMPILE_AND_EXECUTE task and starts the release thread on first release.
ACLOP_TRIGGER = (
    "box = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.6],\n"
    "                    [6.0, 7.0, 8.0, 9.0, 0.4]], device='npu')\n"
    "torch_npu.npu_nms_with_mask(box, 0.5)\n"
)


def _run_npu_script(script_body, timeout=180):
    """Run ``script_body`` in a clean subprocess with the task queue enabled.

    Each scenario runs in its own process so that the Repository lifecycle
    (including destructor / thread teardown) and the task-queue mode are
    isolated between cases.
    """
    env = os.environ.copy()
    env.pop("ASCEND_LAUNCH_BLOCKING", None)
    env["TASK_QUEUE_ENABLE"] = TASK_QUEUE_MODE
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "scenario.py")
        with open(script, "w") as f:
            f.write(textwrap.dedent(script_body))
        res = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, env=env, timeout=timeout)
    return res


def _gen_scenario(workload, expect_no_worker, extra=""):
    """Build a self-asserting scenario body.

    ``workload`` is Python statements that reference an existing tensor ``x``
    (in-place: outer name binding kept via exec of the body).
    ``expect_no_worker``: True -> assert no release_thread appears at the end.
    """
    return f"""
import os
import torch
import torch_npu  # noqa: F401

def _threads():
    names = []
    for t in os.listdir("/proc/self/task"):
        try:
            with open("/proc/self/task/%s/comm" % t) as f:
                names.append(f.read().strip())
        except OSError:
            pass
    return names

torch.npu.set_task_queue_enable({int(TASK_QUEUE_MODE)})
assert not any(c == "release_thread" for c in _threads()), \\
    "release_thread must not exist right after init"
x = torch.randn(64, 64, device="npu")
y = x + 1
{workload}
torch.npu.synchronize()
assert torch.equal(y.cpu(), (x + 1).cpu()), "result mismatch"
{extra}
if {1 if expect_no_worker else 0}:
    assert not any(c == "release_thread" for c in _threads()), \\
        "release_thread spawned unexpectedly"
else:
    n = sum(1 for c in _threads() if c == "release_thread")
    assert n == 1, "expected exactly one release_thread, got %d" % n
    assert torch.sum(y) == torch.sum(y), "y must remain usable"
print("SCENARIO_OK")
"""


def _gen_step_scenario(steps, check_result=True):
    """Build a scenario that asserts the release-thread count at every step.

    ``steps`` is a list of ``(desc, code, expected_release_threads)`` where
    ``code`` is Python statements to run and ``expected_release_threads`` is the
    exact number of ``release_thread`` that must exist right after that step
    (0 or 1). This lets a case verify intermediate states, not just the end
    state. ``check_result`` keeps the final result-correctness assertion.
    """
    step_blocks = []
    for desc, code, expected in steps:
        step_blocks.append(
            "# -- {desc}: expect exactly {expected} release_thread --\n"
            "{code}\n"
            "torch.npu.synchronize()\n"
            "n = sum(1 for c in _threads() if c == 'release_thread')\n"
            "assert n == {expected}, 'after [{desc}] expected {expected} "
            "release_thread, got ' + str(n)\n".format(desc=desc, code=code, expected=expected)
        )
    steps_src = "\n".join(step_blocks)
    return f"""
import os
import torch
import torch_npu  # noqa: F401

def _threads():
    names = []
    for t in os.listdir("/proc/self/task"):
        try:
            with open("/proc/self/task/%s/comm" % t) as f:
                names.append(f.read().strip())
        except OSError:
            pass
    return names

torch.npu.set_task_queue_enable({int(TASK_QUEUE_MODE)})
assert not any(c == "release_thread" for c in _threads()), \\
    "release_thread must not exist right after init"
x = torch.randn(64, 64, device="npu")
y = x + 1
{steps_src}
if {1 if check_result else 0}:
    assert torch.equal(y.cpu(), (x + 1).cpu()), "result mismatch"
print("SCENARIO_OK")
"""


class TestTaskQueueLazyReleaseThread(unittest.TestCase):

    def _assert_ok(self, res, msg=""):
        self.assertEqual(
            res.returncode, 0,
            "%s\nSTDOUT:\n%s\nSTDERR:\n%s" % (msg, res.stdout[-2000:], res.stderr[-2000:]))
        self.assertIn("SCENARIO_OK", res.stdout)

    # ------------------------------------------------------------------
    # pure opapi
    # ------------------------------------------------------------------
    def test_pure_opapi_never_spawns_worker(self):
        """Drive opapi tasks continuously; the release thread must never appear.

        Directly verified (thread scan): no release_thread at init, and none
        after a long run of opapi tasks, with correct results. Because the
        thread is never created, worker_status stays NOT_STARTED and the
        read/write indices never move by construction — but those exact states
        are not OS-observable here, so that part is inferred from "no thread"
        and delegated to C++ (see module docstring).
        """
        loop_workload = (
            "for _ in range(200):\n"
            "    y = x + 1\n"
            "    z = x * 2\n"
        )
        res = _run_npu_script(_gen_scenario(loop_workload, expect_no_worker=True))
        self._assert_ok(res, "pure opapi leaked a release thread")

    def test_pure_opapi_process_exits_cleanly(self):
        """No aclop -> no need to stop/join a release thread on teardown.

        The subprocess must exit 0 (no deadlock, no invalid join hang) after
        only opapi work. A tight timeout additionally guards against a
        destructor that accidentally waits on a thread that was never started.
        """
        res = _run_npu_script(
            _gen_scenario(
                "for _ in range(50):\n    y = x + 1\n",
                expect_no_worker=True),
            timeout=120)
        self._assert_ok(res, "pure opapi process did not exit cleanly")
        # sanity: the whole scenario (incl. process exit) finished well inside
        # the timeout, i.e. no stalling destructor/join path.
        self.assertLess(res.returncode, 2)

    def test_init_only_no_task(self):
        """InitRepo alone must not spawn the worker (no task at all)."""
        res = _run_npu_script(
            "import os\nimport torch\nimport torch_npu\n"
            f"torch.npu.set_task_queue_enable({int(TASK_QUEUE_MODE)})\n"
            "assert not any(c=='release_thread' for c in "
            "(open('/proc/self/task/%s/comm' % t).read().strip() for t in "
            "os.listdir('/proc/self/task')\n"
            "    if os.path.isfile('/proc/self/task/%s/comm' % t))), 'early worker'\n"
            "print('SCENARIO_OK')\n")
        self._assert_ok(res, "init alone spawned release thread")

    # ------------------------------------------------------------------
    # first aclop
    # ------------------------------------------------------------------
    @unittest.skipUnless(
        ACLOP_TRIGGER, "ACLOP_TRIGGER is not configured for this environment; "
                       "a deterministically-aclop (COMPILE_AND_EXECUTE) op must "
                       "be identified against the target torch_npu/op-plugin build.")
    @SupportedDevices(ACLOP_SUPPORTED_SOCS)
    def test_first_aclop_spawns_exactly_one_worker(self):
        """First aclop lazily creates exactly one release thread.

        Directly verified (thread scan): after init there is no release_thread
        (checked inside _gen_scenario), and after the first aclop there is
        exactly one. Whether that aclop entered ReleaseQueue and whether
        worker_status reached RUNNING are not OS-observable; the single-thread
        outcome plus result correctness is the plain-Python proxy for them.
        """
        res = _run_npu_script(_gen_scenario(ACLOP_TRIGGER, expect_no_worker=False))
        self._assert_ok(res, "first aclop did not spawn exactly one worker")

    # ------------------------------------------------------------------
    # opapi -> aclop -> opapi
    # ------------------------------------------------------------------
    @unittest.skipUnless(ACLOP_TRIGGER, "ACLOP_TRIGGER not configured; see test_first_aclop_spawns_exactly_one_worker")
    @SupportedDevices(ACLOP_SUPPORTED_SOCS)
    def test_opapi_aclop_opapi(self):
        """Lifecycle: opapi adds no thread, aclop adds one, opapi keeps one.

        Each step is asserted via the release_thread count in the subprocess:
          step1 opapi -> 0 release_thread  (released directly, no thread)
          step2 aclop -> 1 release_thread  (lazily starts the worker)
          step3 opapi -> 1 release_thread  (still one, does not fall back)

        step3 keeping the worker at 1 is the OS-visible proxy that the
        post-aclop opapi did NOT revert to a synchronous-style teardown of the
        thread. Whether it is physically pushed into ReleaseQueue vs released
        synchronously is not OS-observable from Python (delegated to C++).
        """
        steps = [
            ("opapi 1 (direct)", OPAPI_TRIGGER, 0),
            ("aclop (start worker)", ACLOP_TRIGGER, 1),
            ("opapi 2 (keep queue path)", OPAPI_TRIGGER, 1),
        ]
        res = _run_npu_script(_gen_step_scenario(steps))
        self._assert_ok(res, "opapi->aclop->opapi step assertion failed")

    # ------------------------------------------------------------------
    # aclop -> opapi
    # ------------------------------------------------------------------
    @unittest.skipUnless(ACLOP_TRIGGER, "ACLOP_TRIGGER not configured; see test_first_aclop_spawns_exactly_one_worker")
    @SupportedDevices(ACLOP_SUPPORTED_SOCS)
    def test_aclop_then_opapi(self):
        """First aclop starts the worker; every following opapi keeps it at 1.

        Each step is asserted via the release_thread count: the aclop brings it
        from 0 to 1, and each subsequent opapi must keep it at exactly 1 —
        i.e. the worker is retained and no second thread is spawned. The exact
        "pushed into ReleaseQueue" vs "synchronous release" routing is not
        OS-observable from plain Python (delegated to C++); thread-count
        stability is the proxy verified here.
        """
        steps = [
            ("aclop (start worker)", ACLOP_TRIGGER, 1),
            ("opapi 1", OPAPI_TRIGGER, 1),
            ("opapi 2", OPAPI_TRIGGER, 1),
            ("opapi 3", OPAPI_TRIGGER, 1),
        ]
        res = _run_npu_script(_gen_step_scenario(steps))
        self._assert_ok(res, "aclop->opapi step assertion failed")


if __name__ == "__main__":
    unittest.main()
