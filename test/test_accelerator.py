# Owner(s): ["module: tests"]

import gc
import sys

import torch
from torch.testing._internal.common_utils import NoTest, run_tests, TestCase


if not torch.accelerator.is_available():
    print("No available accelerator detected, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811
    sys.exit()


class TestAccelerator(TestCase):
    def test_current_accelerator(self):
        self.assertTrue(torch.accelerator.is_available())
        accelerators = ["cuda", "xpu", "mps", "npu"]
        for accelerator in accelerators:
            if torch.get_device_module(accelerator).is_available():
                self.assertEqual(
                    torch.accelerator.current_accelerator().type, accelerator
                )
                self.assertIsNone(torch.accelerator.current_accelerator().index)
                with self.assertRaisesRegex(
                    ValueError, "doesn't match the current accelerator"
                ):
                    torch.accelerator.set_device_index("cpu")

    def test_generic_stream_behavior(self):
        s1 = torch.Stream()
        s2 = torch.Stream()
        torch.accelerator.set_stream(s1)
        self.assertEqual(torch.accelerator.current_stream(), s1)
        event = torch.Event()
        a = torch.randn(1000)
        b = torch.randn(1000)
        c = a + b
        torch.accelerator.set_stream(s2)
        self.assertEqual(torch.accelerator.current_stream(), s2)
        a_acc = a.to(torch.accelerator.current_accelerator(), non_blocking=True)
        b_acc = b.to(torch.accelerator.current_accelerator(), non_blocking=True)
        torch.accelerator.set_stream(s1)
        self.assertEqual(torch.accelerator.current_stream(), s1)
        event.record(s2)
        event.synchronize()
        c_acc = a_acc + b_acc
        event.record(s2)
        torch.accelerator.synchronize()
        self.assertTrue(event.query())
        self.assertEqual(c_acc.cpu(), c)

    def test_current_stream_query(self):
        s = torch.accelerator.current_stream()
        self.assertEqual(torch.accelerator.current_stream(s.device), s)
        self.assertEqual(torch.accelerator.current_stream(s.device.index), s)
        self.assertEqual(torch.accelerator.current_stream(str(s.device)), s)
        other_device = torch.device("cpu")
        with self.assertRaisesRegex(
            ValueError, "doesn't match the current accelerator"
        ):
            torch.accelerator.current_stream(other_device)

    def test_stream_context_manager(self):
        prev_stream = torch.accelerator.current_stream()
        with torch.Stream() as s:
            self.assertEqual(torch.accelerator.current_stream(), s)
        self.assertEqual(torch.accelerator.current_stream(), prev_stream)

    def test_pin_memory_on_non_blocking_copy(self):
        t_acc = torch.randn(100).to(torch.accelerator.current_accelerator())
        t_host = t_acc.to("cpu", non_blocking=True)
        torch.accelerator.synchronize()
        self.assertTrue(t_host.is_pinned())
        self.assertEqual(t_acc.cpu(), t_host)

    def test_generic_event_behavior(self):
        event1 = torch.Event(enable_timing=False)
        event2 = torch.Event(enable_timing=False)
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)

        event1 = torch.Event(enable_timing=True)
        event2 = torch.Event(enable_timing=True)
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be recorded before calculating elapsed time",
        ):
            event1.elapsed_time(event2)

        # check default value of enable_timing: False
        event1 = torch.Event()
        event2 = torch.Event()
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)

    def test_event_elapsed_time(self):
        start_event = torch.Event(enable_timing=True)
        end_event = torch.Event(enable_timing=True)
        start_event.record()
        x = torch.randn(1000, 1000, device='npu')
        y = torch.randn(1000, 1000, device='npu')
        z = torch.matmul(x, y)
        end_event.record()
        torch.npu.synchronize()
        ms = start_event.elapsed_time(end_event)
        self.assertGreater(ms, 0)

    def test_device_context_manager(self):
        prev_device = torch.accelerator.current_device_index()
        with torch.accelerator.device_index(None):
            self.assertEqual(torch.accelerator.current_device_index(), prev_device)
        self.assertEqual(torch.accelerator.current_device_index(), prev_device)
        with torch.accelerator.device_index(0):
            self.assertEqual(torch.accelerator.current_device_index(), 0)
        self.assertEqual(torch.accelerator.current_device_index(), prev_device)

    def test_stream_context_manager_reentrance(self):
        prev_stream = torch.accelerator.current_stream()
        s0 = torch.Stream()
        with s0, s0:
            self.assertEqual(torch.accelerator.current_stream(), s0)
        self.assertEqual(torch.accelerator.current_stream(), prev_stream)
        s1 = torch.Stream()
        with s0:
            self.assertEqual(torch.accelerator.current_stream(), s0)
            with s1:
                self.assertEqual(torch.accelerator.current_stream(), s1)
                with s0:
                    self.assertEqual(torch.accelerator.current_stream(), s0)
        self.assertEqual(torch.accelerator.current_stream(), prev_stream)

    def test_memory_interfaces_smoke(self):
        """All 10 torch.accelerator.memory APIs exist and return correct types."""
        acc = torch.accelerator.current_accelerator()
        tmp = torch.randn(10, device=acc)

        # void-return APIs: exist and don't throw
        torch.accelerator.empty_cache()
        torch.accelerator.memory.empty_host_cache()
        torch.accelerator.reset_peak_memory_stats()
        torch.accelerator.reset_accumulated_memory_stats()

        # dict-return APIs
        self.assertIsInstance(torch.accelerator.memory_stats(), dict)

        # int-return APIs
        for fn in [
            torch.accelerator.memory_allocated,
            torch.accelerator.max_memory_allocated,
            torch.accelerator.memory_reserved,
            torch.accelerator.max_memory_reserved,
        ]:
            self.assertIsInstance(fn(), int)

        # tuple-return API
        free_mem, total_mem = torch.accelerator.get_memory_info()
        self.assertIsInstance(free_mem, int)
        self.assertIsInstance(total_mem, int)

        del tmp
        gc.collect()
        torch.accelerator.empty_cache()

    def test_memory_stats(self):
        acc = torch.accelerator.current_accelerator()
        tmp = torch.randn(100, device=acc)
        del tmp
        gc.collect()
        self.assertTrue(torch._C._accelerator_isAllocatorInitialized())
        torch.accelerator.empty_cache()

        pool_type = ["all", "small_pool", "large_pool"]
        metric_type = ["peak", "current", "allocated", "freed"]
        stats_type = [
            "allocated_bytes",
            "reserved_bytes",
            "active_bytes",
            "requested_bytes",
        ]
        mem_stats = torch.accelerator.memory_stats()
        expected_stats = [
            f"{st}.{pt}.{mt}"
            for st in stats_type
            for pt in pool_type
            for mt in metric_type
        ]
        missing_stats = [stat for stat in expected_stats if stat not in mem_stats]
        self.assertEqual(
            len(missing_stats),
            0,
            f"Missing expected memory statistics: {missing_stats}",
        )

        self.assertIn("num_alloc_retries", mem_stats)
        self.assertIn("num_ooms", mem_stats)
        self.assertIn("num_sync_all_streams", mem_stats)
        self.assertIn("num_device_alloc", mem_stats)
        self.assertIn("num_device_free", mem_stats)

    def test_memory_allocated_reserved(self):
        acc = torch.accelerator.current_accelerator()
        tmp = torch.randn(100, device=acc)
        del tmp
        gc.collect()
        self.assertTrue(torch._C._accelerator_isAllocatorInitialized())
        torch.accelerator.empty_cache()

        prev_allocated = torch.accelerator.memory_allocated()
        prev_reserved = torch.accelerator.memory_reserved()
        prev_max_allocated = torch.accelerator.max_memory_allocated()
        prev_max_reserved = torch.accelerator.max_memory_reserved()
        self.assertGreaterEqual(prev_allocated, 0)
        self.assertGreaterEqual(prev_reserved, 0)
        self.assertGreater(prev_max_allocated, 0)
        self.assertGreater(prev_max_reserved, 0)
        tmp = torch.ones(256, device=acc)
        self.assertGreater(torch.accelerator.memory_allocated(), prev_allocated)
        self.assertGreaterEqual(torch.accelerator.memory_reserved(), prev_reserved)
        del tmp
        gc.collect()
        torch.accelerator.empty_cache()
        torch.accelerator.reset_peak_memory_stats()
        self.assertEqual(torch.accelerator.memory_allocated(), prev_allocated)
        self.assertEqual(torch.accelerator.memory_reserved(), prev_reserved)

    def test_memory_stats_active_bytes(self):
        acc = torch.accelerator.current_accelerator()
        tmp = torch.empty(100, device=acc)
        del tmp
        gc.collect()
        self.assertTrue(torch._C._accelerator_isAllocatorInitialized())
        torch.accelerator.empty_cache()
        torch.accelerator.reset_accumulated_memory_stats()
        torch.accelerator.reset_peak_memory_stats()
        self.assertEqual(torch.accelerator.memory_stats()["active_bytes.all.freed"], 0)

        prev_max_allocated = torch.accelerator.max_memory_allocated()
        prev_max_reserved = torch.accelerator.max_memory_reserved()
        prev_active_current = torch.accelerator.memory_stats()[
            "active_bytes.all.current"
        ]
        self.assertEqual(prev_max_allocated, 0)
        self.assertEqual(prev_active_current, 0)

        tmp = torch.randn(256, device=acc)
        active_delta = (
            torch.accelerator.memory_stats()["active_bytes.all.current"]
            - prev_active_current
        )
        self.assertGreater(active_delta, 0)
        del tmp
        gc.collect()
        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()
        self.assertEqual(
            torch.accelerator.memory_stats()["active_bytes.all.current"],
            prev_active_current,
        )
        torch.accelerator.reset_peak_memory_stats()
        self.assertEqual(torch.accelerator.max_memory_allocated(), prev_max_allocated)
        self.assertEqual(torch.accelerator.max_memory_reserved(), prev_max_reserved)

    def test_get_memory_info(self):
        """getMemoryInfo is an independent aclrtGetMemInfo + NPUGuard path."""
        free_bytes, total_bytes = torch.accelerator.get_memory_info()
        self.assertGreaterEqual(free_bytes, 0)
        self.assertGreaterEqual(total_bytes, 0)
        self.assertGreater(total_bytes, free_bytes)

    def test_device_capability_supported_dtypes(self):
        try:
            caps = torch.accelerator.get_device_capability()
        except RuntimeError:
            self.skipTest("Backend doesn't support get_device_capability")
        supported_dtypes = caps["supported_dtypes"]
        self.assertIsInstance(supported_dtypes, set)
        self.assertGreater(len(supported_dtypes), 0)

    def test_memory_stats_consistency_with_npu(self):
        """accelerator memory_stats must match torch.npu.memory_stats exactly."""
        acc_stats = torch.accelerator.memory_stats()
        npu_stats = torch.npu.memory_stats()

        for key in acc_stats:
            self.assertIn(key, npu_stats, f"Key '{key}' missing from npu stats")
            self.assertEqual(
                acc_stats[key],
                npu_stats[key],
                f"Mismatch for key '{key}': accelerator={acc_stats[key]}, "
                f"npu={npu_stats[key]}",
            )

        self.assertEqual(
            torch.accelerator.memory_allocated(),
            torch.npu.memory_allocated(),
        )
        self.assertEqual(
            torch.accelerator.max_memory_allocated(),
            torch.npu.max_memory_allocated(),
        )
        self.assertEqual(
            torch.accelerator.memory_reserved(),
            torch.npu.memory_reserved(),
        )
        self.assertEqual(
            torch.accelerator.max_memory_reserved(),
            torch.npu.max_memory_reserved(),
        )


if __name__ == "__main__":
    run_tests()
