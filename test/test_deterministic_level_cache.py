# -*- coding: utf-8 -*-
"""
测试目的：验证确定性算法级别缓存机制的功能正确性（OpParamMaker.cpp 中的
          g_applied_deterministic_level 全局缓存 + tls_applied_deterministic_level
          线程级缓存）
"""
import threading

import torch
import torch_npu  # noqa: F401
from torch_npu.testing.common_utils import SupportedDevices

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestDeterministicLevelCache(TestCase):
    """Functional tests for the deterministic level snapshot cache on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name,
            "npu",
            f"Expected device 'npu', got '{self.device_name}'",
        )
        self.device = torch.device(self.device_name)
        self.original_det = torch.are_deterministic_algorithms_enabled()
        self.original_level = torch_npu._C._npu_get_deterministic_level()
        self.errors = []

    def tearDown(self):
        torch.use_deterministic_algorithms(self.original_det)
        torch_npu._C._npu_set_deterministic_level(self.original_level)
        # Re-launch one op so the restored state is re-applied (and both cache
        # markers of this thread are refreshed) before the next test runs.
        self._run_op_cycle()
        torch.npu.synchronize()
        super().tearDown()

    def _run_op_cycle(self):
        """Launches a small op batch on the current thread and returns the results.

        Every NPU op launch goes through ApplyDeterministicSnapshot, so this is
        the Python-side entry into the code under test.
        """
        a = torch.arange(8, dtype=torch.float32, device=self.device).reshape(2, 4)
        b = a * 2 + 1
        added = a + b
        summed = torch.sum(added)
        return a, b, added, summed

    def _assert_op_results(self, results):
        a, b, added, summed = results
        expected_added = a.cpu() + b.cpu()
        expected_sum = expected_added.sum()
        self.assertEqual(added.cpu(), expected_added)
        self.assertEqual(summed.cpu(), expected_sum)

    @staticmethod
    def _set_deterministic(enabled):
        torch.use_deterministic_algorithms(enabled)
        torch_npu._C._npu_set_deterministic_level(1 if enabled else 0)

    def test_deterministic_toggle_cycles_keep_op_results_correct(self):
        # Alternating effective_level 1 <-> 0 switches ApplyDeterministicSnapshot
        # between the locked apply path and the cache fast path. The cycle must
        # keep the ACL state consistent: every op launch succeeds and results
        # stay numerically correct.
        for i in range(5):
            for enabled in (True, False):
                self._set_deterministic(enabled)
                self.assertEqual(
                    torch.are_deterministic_algorithms_enabled(), enabled,
                    f"toggle failed at iteration {i} (enabled={enabled})",
                )
                results = self._run_op_cycle()
                self._assert_op_results(results)
        torch.npu.synchronize()

    def test_repeated_zero_snapshot_stays_consistent(self):
        # Fast-path regression: once both cache markers hold 0, repeated level-0
        # launches must stay stable (the snapshot skip condition must not break
        # the state machine).
        self._set_deterministic(False)
        for _ in range(100):
            results = self._run_op_cycle()
        self._assert_op_results(results)
        torch.npu.synchronize()

    def test_thread_local_guard_clears_stale_thread_context(self):
        # Regression for the thread-local guard: a worker thread that applied a
        # non-zero level before determinism got disabled must not be
        # short-circuited by the global cache when it launches ops later,
        # otherwise its context keeps the stale level
        # (AclrtCtxSetSysParamOpt takes effect per context).
        stage1_done = threading.Event()
        determinism_disabled = threading.Event()
        worker_done = threading.Event()

        def worker():
            try:
                # Stage 1: launch ops with determinism on, so this thread
                # applies level 1 on its own context (tls marker becomes 1).
                results = self._run_op_cycle()
                self._assert_op_results(results)
                stage1_done.set()

                determinism_disabled.wait(timeout=60)
                # Stage 2: the main thread already disabled determinism
                # (global cache == 0, main thread marker == 0). Launching ops
                # here must NOT be skipped for this thread: the tls guard
                # forces the locked path and clears the stale context level.
                results = self._run_op_cycle()
                self._assert_op_results(results)
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"worker thread failed: {e!r}")
            finally:
                worker_done.set()

        self._set_deterministic(True)
        thread = threading.Thread(target=worker)
        thread.start()

        self.assertTrue(stage1_done.wait(timeout=60), "worker stage 1 timed out")

        # Disable determinism on the main thread: its own launches reset the
        # global cache and the main-thread marker to 0.
        self._set_deterministic(False)
        results = self._run_op_cycle()
        self._assert_op_results(results)
        determinism_disabled.set()

        self.assertTrue(worker_done.wait(timeout=60), "worker stage 2 timed out")
        thread.join(timeout=60)
        self.assertEqual(self.errors, [])
        torch.npu.synchronize()

    def test_fresh_threads_apply_level_after_global_disable(self):
        # A freshly created thread has tls marker -1. When the global cache
        # already holds 0, its first launches must still reach the locked path
        # (the skip condition requires the per-thread marker to be 0 as well),
        # so every new thread's context is explicitly synchronized.
        self._set_deterministic(True)
        self._run_op_cycle()

        self._set_deterministic(False)
        self._run_op_cycle()

        for i in range(3):
            thread = threading.Thread(
                target=self._run_worker_once, args=(i,), name=f"fresh-thread-{i}"
            )
            thread.start()
            thread.join(timeout=60)
            self.assertFalse(thread.is_alive(), f"thread {i} timed out")

        self.assertEqual(self.errors, [])
        torch.npu.synchronize()

    def _run_worker_once(self, index):
        try:
            results = self._run_op_cycle()
            self._assert_op_results(results)
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"fresh thread {index} failed: {e!r}")

    def test_deterministic_level_state_consistency(self):
        # The requested level must round-trip through the bindings and survive
        # op launches (the snapshot cache must not corrupt the requested state).
        for level in (0, 1, 2, 0):
            torch_npu._C._npu_set_deterministic_level(level)
            torch.use_deterministic_algorithms(level != 0)
            self.assertEqual(torch_npu._C._npu_get_deterministic_level(), level)
            results = self._run_op_cycle()
            self._assert_op_results(results)
        torch.npu.synchronize()

    def test_set_deterministic_level_syncs_conflicting_torch_state(self):
        # torch_npu.npu.set_deterministic_level reconciles a conflicting
        # torch.use_deterministic_algorithms state (with a warning) and must
        # leave both sides consistent afterwards.
        import warnings

        torch.use_deterministic_algorithms(False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch_npu.npu.set_deterministic_level(1)
        self.assertEqual(torch_npu._C._npu_get_deterministic_level(), 1)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        results = self._run_op_cycle()
        self._assert_op_results(results)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch_npu.npu.set_deterministic_level(0)
        self.assertEqual(torch_npu._C._npu_get_deterministic_level(), 0)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        results = self._run_op_cycle()
        self._assert_op_results(results)
        torch.npu.synchronize()

    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_concurrent_launches_with_deterministic_toggle(self):
        # Stress guard: several threads launch ops concurrently while the main
        # thread toggles the deterministic switch. The recursive mutex + the
        # tls markers must keep this race-free: no crash, no deadlock, and all
        # threads observe correct results.
        self._set_deterministic(True)
        base_a = torch.arange(8, dtype=torch.float32, device=self.device).reshape(2, 4)
        base_b = base_a * 2 + 1
        expected = base_a.cpu() + base_b.cpu()

        stop = threading.Event()

        def launcher():
            while not stop.is_set():
                try:
                    out = base_a + base_b
                    if not torch.allclose(out.cpu(), expected):
                        self.errors.append("concurrent launch produced wrong result")
                        return
                except Exception as e:  # noqa: BLE001
                    self.errors.append(f"concurrent launch failed: {e!r}")
                    return

        threads = [threading.Thread(target=launcher, name=f"launcher-{i}") for i in range(3)]
        for thread in threads:
            thread.start()

        try:
            for enabled in (False, True, False, True, False):
                self._set_deterministic(enabled)
                torch.npu.synchronize()
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=60)
                self.assertFalse(thread.is_alive(), "launcher thread timed out")

        self.assertEqual(self.errors, [])
        torch.npu.synchronize()


if __name__ == "__main__":
    run_tests()
