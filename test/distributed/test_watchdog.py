import os
import sys
import tempfile
import time
import unittest
from datetime import timedelta

import torch
import torch.distributed as c10d
import torch.distributed.run as launch
from torch.testing._internal.common_distributed import MultiProcessTestCase

import torch_npu  # noqa: F401
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


class ElasticLaunchTest(TestCase):
    @skipIfUnsupportMultiNPU(2)
    def test_npu_watchdog_timeout(self):
        try:
            error = None
            launch.main(
                [
                    "--nproc-per-node=2",
                    path("watchdog/watchdog_base.py"),
                ]
            )
        except Exception as e:
            error = e
        finally:
            if error is None:
                raise RuntimeError("Test case fail")


    @skipIfUnsupportMultiNPU(2)
    def test_npu_watchdog_quick_exit(self):
        start_time = time.time()
        try:
            launch.main(
                [
                    "--nproc-per-node=2",
                    path("watchdog/watchdog_quick_exit.py"),
                ]
            )
        except Exception:
            print("Program fail and exit")

        end_time = time.time()
        execution_time = end_time - start_time
        if execution_time > 120:
            print(f"Execution time using time.time(): {execution_time} seconds")
            raise RuntimeError("Test case fail")


# Works time out in the watchdog after PG_TIMEOUT_S seconds.
PG_TIMEOUT_S = 10
# Rank 0 stalls STALL_S seconds before joining the second collective, so
# rank 1's pending work hits the timeout window before healing.
STALL_S = 20


class Mode0WatchdogTest(MultiProcessTestCase):
    # TORCH_HCCL_ASYNC_ERROR_HANDLING=0: works are still enqueued and
    # monitored by the watchdog without abort/teardown.
    def setUp(self):
        super().setUp()
        # Skip before spawning so insufficient hardware shows as a visible
        # skipped test.
        if not torch.npu.is_available() or torch.npu.device_count() < self.world_size:
            raise unittest.SkipTest("Multi-NPU 2 condition not satisfied")
        # Set before ranks create their process groups (inherited at spawn);
        # restore afterwards so other tests are unaffected.
        self._prev_handling = os.environ.get("TORCH_HCCL_ASYNC_ERROR_HANDLING")
        self.addCleanup(self._restore_handling_env)
        os.environ["TORCH_HCCL_ASYNC_ERROR_HANDLING"] = "0"
        self._spawn_processes()

    def _restore_handling_env(self):
        if self._prev_handling is None:
            os.environ.pop("TORCH_HCCL_ASYNC_ERROR_HANDLING", None)
        else:
            os.environ["TORCH_HCCL_ASYNC_ERROR_HANDLING"] = self._prev_handling

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self) -> int:
        return 2

    @property
    def local_device(self):
        return torch.device("npu", self.rank)

    def _init_pg(self, timeout_s):
        torch.npu.set_device(self.rank)
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="hccl",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
            timeout=timedelta(seconds=timeout_s),
        )

    def test_mode0_timeout_heal(self):
        self._init_pg(PG_TIMEOUT_S)
        tensor = torch.ones(1, device=self.local_device)

        # First collective completes on both ranks.
        c10d.all_reduce(tensor)
        self.assertEqual(tensor, torch.full((1,), 2.0, device=self.local_device))

        if self.rank == 0:
            # Stall before joining the second collective.
            time.sleep(STALL_S)
            c10d.all_reduce(tensor)
            self.assertEqual(tensor, torch.full((1,), 4.0, device=self.local_device))
        else:
            # Guard goal: the pending work is monitored, the timeout is
            # logged without teardown, and the work heals once rank 0 joins.
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stderr_file:
                # The watchdog thread writes to fd 2, not sys.stderr.
                saved_stderr = os.dup(2)
                os.dup2(stderr_file.fileno(), 2)
                try:
                    work = c10d.all_reduce(tensor, async_op=True)
                    time.sleep(STALL_S)
                    work.wait()
                finally:
                    os.dup2(saved_stderr, 2)
                    os.close(saved_stderr)
                stderr_file.seek(0)
                stderr_text = stderr_file.read()
            if "Watchdog caught collective operation timeout" not in stderr_text:
                # Surface the capture before the assertion below loses it.
                sys.stderr.write(stderr_text)
            self.assertIn("Watchdog caught collective operation timeout", stderr_text)
            self.assertEqual(tensor, torch.full((1,), 4.0, device=self.local_device))

        c10d.destroy_process_group()


class BlockingWaitTest(MultiProcessTestCase):
    # TORCH_HCCL_BLOCKING_WAIT=1: works are not enqueued to the watchdog and
    # work.wait() busy-waits on the host thread until completion.
    def setUp(self):
        super().setUp()
        # Skip before spawning so insufficient hardware shows as a visible
        # skipped test.
        if not torch.npu.is_available() or torch.npu.device_count() < self.world_size:
            raise unittest.SkipTest("Multi-NPU 2 condition not satisfied")
        # Set before ranks create their process groups (inherited at spawn);
        # restore afterwards so other tests are unaffected.
        self._prev_blocking_wait = os.environ.get("TORCH_HCCL_BLOCKING_WAIT")
        self.addCleanup(self._restore_blocking_wait_env)
        os.environ["TORCH_HCCL_BLOCKING_WAIT"] = "1"
        self._spawn_processes()

    def _restore_blocking_wait_env(self):
        if self._prev_blocking_wait is None:
            os.environ.pop("TORCH_HCCL_BLOCKING_WAIT", None)
        else:
            os.environ["TORCH_HCCL_BLOCKING_WAIT"] = self._prev_blocking_wait

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self) -> int:
        return 2

    @property
    def local_device(self):
        return torch.device("npu", self.rank)

    def _init_pg(self, timeout_s):
        torch.npu.set_device(self.rank)
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="hccl",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
            timeout=timedelta(seconds=timeout_s),
        )

    def test_blocking_wait_not_enqueued(self):
        self._init_pg(PG_TIMEOUT_S)
        tensor = torch.ones(1, device=self.local_device)

        # Sync collective completes through the host-side busy wait.
        c10d.all_reduce(tensor)
        self.assertEqual(tensor, torch.full((1,), 2.0, device=self.local_device))

        if self.rank == 0:
            time.sleep(STALL_S)
            c10d.all_reduce(tensor)
            self.assertEqual(tensor, torch.full((1,), 4.0, device=self.local_device))
        else:
            # Guard goal: the pending work is not enqueued, so the watchdog
            # stays silent while the collective itself still heals.
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stderr_file:
                # The watchdog thread writes to fd 2, not sys.stderr.
                saved_stderr = os.dup(2)
                os.dup2(stderr_file.fileno(), 2)
                try:
                    c10d.all_reduce(tensor, async_op=True)
                    time.sleep(STALL_S)
                finally:
                    os.dup2(saved_stderr, 2)
                    os.close(saved_stderr)
                stderr_file.seek(0)
                stderr_text = stderr_file.read()
            if "Watchdog caught collective operation timeout" in stderr_text:
                # Surface the capture before the assertion below loses it.
                sys.stderr.write(stderr_text)
            self.assertNotIn("Watchdog caught collective operation timeout", stderr_text)
            # A wait() here would self-time out at the PG timeout; a device
            # sync proves the pending collective healed once rank 0 joined.
            torch.npu.synchronize()
            self.assertEqual(tensor, torch.full((1,), 4.0, device=self.local_device))

        c10d.destroy_process_group()


if __name__ == "__main__":
    run_tests()
