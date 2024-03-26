# Owner(s): ["module: cuda"]

import collections
import contextlib
import ctypes
import io
import gc
import queue
import sys
import tempfile
import threading
import unittest
from itertools import repeat, chain
from typing import NamedTuple
import torch
import torch.cuda.comm as comm

from torch.nn.parallel import scatter_gather
import torch_npu
import torch_npu.testing
from torch_npu.testing.common_utils import get_cycles_per_ms
from torch.testing._internal.common_utils import (
    IS_JETSON,
    IS_REMOTE_GPU,
    IS_SANDCASTLE,
    NoTest,
    TEST_PRIVATEUSE1,
    TestCase,
    instantiate_parametrized_tests,
    run_tests,
    skipCUDANonDefaultStreamIf,
    skipIfRocm,
)
from torch.testing._internal.common_cuda import _create_scaling_case, _create_scaling_models_optimizers

TEST_MULTINPU = TEST_PRIVATEUSE1 and torch_npu.npu.device_count() >= 2

if not TEST_PRIVATEUSE1:
    print('NPU not available, skipping tests', file=sys.stderr)
    TestCase = NoTest  # noqa: F811


class TestNpuMultiNpu(TestCase):
    FIFTY_MIL_CYCLES = 50000000

    def _check_memory_stat_consistency(self):
        snapshot = torch_npu.npu.memory_snapshot()

        expected_each_device = collections.defaultdict(lambda: collections.defaultdict(int))

        for segment in snapshot:
            expandable = segment["is_expandable"]
            expected = expected_each_device[segment["device"]]
            pool_str = segment["segment_type"] + "_pool"

            if not expandable:
                expected["segment.all.current"] += 1
                expected["segment." + pool_str + ".current"] += 1

            expected["allocated_bytes.all.current"] += segment["allocated_size"]
            expected["allocated_bytes." + pool_str + ".current"] += segment["allocated_size"]

            expected["reserved_bytes.all.current"] += segment["total_size"]
            expected["reserved_bytes." + pool_str + ".current"] += segment["total_size"]

            expected["active_bytes.all.current"] += segment["active_size"]
            expected["active_bytes." + pool_str + ".current"] += segment["active_size"]

            expected["requested_bytes.all.current"] += segment["requested_size"]
            expected["requested_bytes." + pool_str + ".current"] += segment["requested_size"]

            sum_requested = 0
            is_split = len(segment["blocks"]) > 1
            for block in segment["blocks"]:
                if block["state"] == "active_allocated":
                    expected["allocation.all.current"] += 1
                    expected["allocation." + pool_str + ".current"] += 1

                if block["state"].startswith("active_"):
                    sum_requested += block["requested_size"]
                    expected["active.all.current"] += 1
                    expected["active." + pool_str + ".current"] += 1

                if block["state"] == "inactive" and is_split and not expandable:
                    expected["inactive_split.all.current"] += 1
                    expected["inactive_split." + pool_str + ".current"] += 1
                    expected["inactive_split_bytes.all.current"] += block["size"]
                    expected["inactive_split_bytes." + pool_str + ".current"] += block["size"]

            self.assertEqual(sum_requested, segment["requested_size"])

        for device, expected in expected_each_device.items():
            stats = torch_npu.npu.memory_stats(device)
            for k, v in expected.items():
                self.assertEqual(v, stats[k])

    def test_cuda_synchronize(self):
        torch_npu.npu.synchronize()
        torch_npu.npu.synchronize('npu')
        torch_npu.npu.synchronize('npu:0')
        torch_npu.npu.synchronize(0)
        torch_npu.npu.synchronize(torch.device('npu:0'))

        if TEST_MULTINPU:
            torch_npu.npu.synchronize('npu:1')
            torch_npu.npu.synchronize(1)
            torch_npu.npu.synchronize(torch.device('npu:1'))

        with self.assertRaisesRegex(ValueError, "Expected a npu device, but"):
            torch_npu.npu.synchronize(torch.device("cpu"))

        with self.assertRaisesRegex(ValueError, "Expected a npu device, but"):
            torch_npu.npu.synchronize("cpu")

    @staticmethod
    def _test_memory_stats_generator(self, device=None, N=35):
        if device is None:
            device = torch_npu.npu.current_device()

        m0 = torch_npu.npu.memory_allocated(device)
        last_m_arr = [torch_npu.npu.memory_allocated(device)]
        max_m_arr = [torch_npu.npu.max_memory_allocated(device)]
        last_r_arr = [torch_npu.npu.memory_reserved(device)]
        max_r_arr = [torch_npu.npu.max_memory_reserved(device)]

        def alloc(*size):
            with torch_npu.npu.device(device):
                # NOTE: do **not** use methods that can have additional
                #       memory overhead, e.g., inplace random sampling methods.
                #       they can leave some memory occupied even after being
                #       deallocated, e.g., initialized RNG state, causing some
                #       memory checks below to fail.
                return torch_npu.npu.FloatTensor(*size)

        def assert_change(comp=1, empty_cache=False, reset_peak=False):
            # comp > 0: increased
            # comp = 0: equal
            # comp < 0: decreased
            new_m = torch_npu.npu.memory_allocated(device)
            new_max_m = torch_npu.npu.max_memory_allocated(device)
            if comp > 0:
                self.assertGreater(new_m, last_m_arr[0])
            elif comp < 0:
                self.assertLess(new_m, last_m_arr[0])
            else:
                self.assertEqual(new_m, last_m_arr[0])
            self.assertLessEqual(new_m, new_max_m)
            self.assertGreaterEqual(new_max_m, max_m_arr[0])
            last_m_arr[0] = new_m
            max_m_arr[0] = new_max_m

            new_r = torch_npu.npu.memory_reserved(device)
            new_max_r = torch_npu.npu.max_memory_reserved(device)
            # emptying cache may happen (due to allocation or empty_cache), so
            # we can't assert new_c >= last_c
            self.assertLessEqual(new_r, new_max_r)
            self.assertGreaterEqual(new_max_r, max_r_arr[0])
            last_r_arr[0] = new_r
            max_r_arr[0] = new_max_r

            if empty_cache:
                torch_npu.npu.empty_cache()
                new_r = torch_npu.npu.memory_reserved(device)
                new_max_r = torch_npu.npu.max_memory_reserved(device)
                self.assertLessEqual(new_r, last_r_arr[0])
                self.assertLessEqual(new_r, new_max_r)
                self.assertEqual(new_max_r, max_r_arr[0])
                last_r_arr[0] = new_r

            if reset_peak:
                torch_npu.npu.reset_peak_memory_stats(device)
                self.assertEqual(torch_npu.npu.memory_allocated(device), last_m_arr[0])
                self.assertEqual(torch_npu.npu.max_memory_allocated(device), last_m_arr[0])
                max_m_arr[0] = last_m_arr[0]
                self.assertEqual(torch_npu.npu.memory_reserved(device), last_r_arr[0])
                self.assertEqual(torch_npu.npu.max_memory_reserved(device), last_r_arr[0])
                max_r_arr[0] = last_r_arr[0]

        assert_change(0)
        assert_change(0, reset_peak=True)
        assert_change(0, empty_cache=True)
        assert_change(0, reset_peak=True)
        assert_change(0)
        yield

        tensors1 = [alloc(1), alloc(10, 20), alloc(200, 300, 2000)]
        m1 = torch_npu.npu.memory_allocated(device)
        assert_change(1)
        yield

        tensors2 = []

        for i in range(1, int(N / 2) + 1):
            # small ones
            tensors2.append(alloc(i, i * 4))
            assert_change(1)
            yield

        for i in range(5, int(N / 2) + 5):
            # large ones
            tensors2.append(alloc(i, i * 7, i * 9, i * 11))
            assert_change(1, reset_peak=(i % 2 == 0))
            yield

        tensors2.append(alloc(0, 0, 0))
        assert_change(0)
        yield

        permute = []
        for i in torch.randperm(len(tensors2)):
            permute.append(tensors2[i])
            assert_change(0)
            yield

        del tensors2
        assert_change(0)
        yield
        tensors2 = permute
        assert_change(0)
        yield
        del permute
        assert_change(0, reset_peak=True)
        yield

        for i in range(int(N / 2)):
            x = tensors2[i].numel()
            del tensors2[i]
            assert_change(-x)  # in case that tensors2[i] is empty
            yield

        for i in range(2, int(2 * N / 3) + 2):
            tensors2.append(alloc(i, i * 3, i * 8))
            assert_change(1)
            yield

        del tensors2
        assert_change(-1, reset_peak=True)
        assert_change(0)
        self.assertEqual(torch_npu.npu.memory_allocated(device), m1)
        yield True

        del tensors1
        assert_change(-1, reset_peak=True)
        self.assertEqual(torch_npu.npu.memory_allocated(device), m0)

        # test empty_cache and reset_peak
        assert_change(0, empty_cache=True)
        assert_change(0, reset_peak=True)

    def test_memory_stats(self):
        gc.collect()
        torch_npu.npu.empty_cache()
        for _ in self._test_memory_stats_generator(self):
            self._check_memory_stat_consistency()

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_memory_stats_multigpu(self):
        # advance a generator with a end flag
        def advance(gen, end):
            if not end:
                try:
                    next(gen)
                except StopIteration:
                    end = True
            return end

        # interlace
        torch_npu.npu.empty_cache()
        gen0 = self._test_memory_stats_generator(self, device='npu:0', N=35)
        gen1 = self._test_memory_stats_generator(self, device=torch.device('npu:1'), N=35)
        end0 = end1 = False
        while not (end0 and end1):
            end0 = advance(gen0, end0)
            end1 = advance(gen1, end1)

        # semi-random order
        torch_npu.npu.empty_cache()
        gen0 = self._test_memory_stats_generator(self, device=0, N=35)
        gen1 = self._test_memory_stats_generator(self, device=torch.device('npu:1'), N=35)
        end0 = end1 = False

        while not (end0 and end1):
            end0 = advance(gen0, end0)
            if not end0:
                gen1_max_times = torch.LongTensor(1).random_(0, 3)[0]
            else:
                gen1_max_times = torch.inf
            t = 0
            while t < gen1_max_times and not end1:
                end1 = advance(gen1, end1)
                t += 1

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_autogpu(self):
        x = torch.randn(5, 5).npu()
        y = torch.randn(5, 5).npu()
        self.assertEqual(x.get_device(), 0)
        self.assertEqual(x.get_device(), 0)
        with torch_npu.npu.device(1):
            z = torch.randn(5, 5).npu()
            self.assertEqual(z.get_device(), 1)
            q = x.add(y)
            self.assertEqual(q.get_device(), 0)
            w = torch.randn(5, 5).npu()
            self.assertEqual(w.get_device(), 1)
            self.assertEqual(y.npu().get_device(), 1)
        z = z.npu()
        self.assertEqual(z.get_device(), 0)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_new(self):
        x = torch.randn(3, 3).npu()
        self.assertEqual(x.new([0, 1, 2]).get_device(), 0)
        self.assertEqual(x.new([0, 1, 2], device=1).get_device(), 1)

        with torch_npu.npu.device(1):
            self.assertEqual(x.new([0, 1, 2]).get_device(), 0)
            self.assertEqual(x.new([0, 1, 2], device=1).get_device(), 1)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_copy_device(self):
        x = torch.randn(5, 5).npu()
        with torch_npu.npu.device(1):
            y = x.npu()
            self.assertEqual(y.get_device(), 1)
            self.assertIs(y.npu(), y)
            z = y.npu(0)
            self.assertEqual(z.get_device(), 0)
            self.assertIs(z.npu(0), z)

        x = torch.randn(5, 5)
        with torch_npu.npu.device(1):
            y = x.npu()
            self.assertEqual(y.get_device(), 1)
            self.assertIs(y.npu(), y)
            z = y.npu(0)

            self.assertEqual(z.get_device(), 0)
            self.assertIs(z.npu(0), z)

    def _test_copy_sync_current_stream(self, x, y):
        x_plus_one = x + 1
        s0 = torch_npu.npu.Stream(device=x.device)
        s1 = torch_npu.npu.Stream(device=y.device)
        s2 = torch_npu.npu.Stream(device=x.device)
        s3 = torch_npu.npu.Stream(device=y.device)

        # same dst stream different src streams
        with torch_npu.npu.stream(s0):
            torch_npu.npu._sleep(TestNpuMultiNpu.FIFTY_MIL_CYCLES)
            with torch_npu.npu.stream(s1):
                y.copy_(x_plus_one)

        with torch_npu.npu.stream(s2), torch_npu.npu.stream(s1):
            y.copy_(x)

        s1.synchronize()
        # The copy() is synchronized on the current streams of both src and dst.
        # In the above test, the _sleep() op on s0 will not block the copy() on
        # s2, but both copies are synchronized on s1 in the dst device. Hence,
        # x is copied to y after x_plus_one is copied to y. If x and y are on
        # the same device, both copy() ops are synchronized on s1.
        self.assertEqual(y, x)

        # same src stream different dst streams
        with torch_npu.npu.stream(s1):
            torch_npu.npu._sleep(TestNpuMultiNpu.FIFTY_MIL_CYCLES)
            with torch_npu.npu.stream(s0):
                y.copy_(x_plus_one)

        with torch_npu.npu.stream(s3), torch_npu.npu.stream(s0):
            y.copy_(x)

        s0.synchronize()
        # Similarly, both copy() ops are synchronized on s0.
        self.assertEqual(y, x)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_copy_streams(self):
        d0 = torch.device('npu:0')
        x0 = torch.zeros(5, 5, device=d0)

        d1 = torch.device('npu:1')
        x1 = torch.zeros(5, 5, device=d1)
        self._test_copy_sync_current_stream(x0, x1)

        x2 = torch.zeros(5, 5, device=d0)
        self._test_copy_sync_current_stream(x0, x2)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_cat_autogpu(self):
        x = torch.randn(4, 4).npu(1)
        y = torch.randn(4, 4).npu(1)
        z = torch.cat([x, y], 0)
        self.assertEqual(z.get_device(), x.get_device())

    @unittest.skipIf(torch_npu.npu.device_count() >= 10, "Loading a npu:9 tensor")
    def test_load_nonexistent_device(self):
        # Setup: create a serialized file object with a 'npu:9' restore location
        tensor = torch.randn(2, device='npu')
        buf = io.BytesIO()
        torch.save(tensor, buf)
        # NB: this might not work in the future if serialization changes
        buf = io.BytesIO(buf.getvalue().replace(b'npu:0', b'npu:9'))

        msg = r'Attempting to deserialize object on NPU device 9'
        with self.assertRaisesRegex(RuntimeError, msg):
            _ = torch.load(buf)

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_multigpu_serialization_remap(self):
        x = [torch.randn(4, 4).npu(0), torch.randn(4, 4).npu(1)]

        def gpu_remap(storage, location):
            if location == 'npu:1':
                return storage.npu(0)

        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f, map_location=gpu_remap)

        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), 0)

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_multigpu_serialization_remap_dict(self):
        x = [torch.randn(4, 4).npu(0), torch.randn(4, 4).npu(1)]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f, map_location={'npu:1': 'npu:0'})
        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), 0)

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_multigpu_storage_clone(self):
        x = torch.randn(4, 4, device='npu:1').storage()
        y = x.clone()
        self.assertEqual(x.get_device(), y.get_device())
        for t in ['byte', 'char', 'short', 'int', 'long', 'half', 'double']:
            self.assertEqual(getattr(x, t)().get_device(), x.get_device())

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_cuda_set_device(self):
        x = torch.randn(5, 5)
        with torch_npu.npu.device(1):
            self.assertEqual(x.npu().get_device(), 1)
            torch_npu.npu.set_device(0)
            self.assertEqual(x.npu().get_device(), 0)
            with torch_npu.npu.device(1):
                self.assertEqual(x.npu().get_device(), 1)
            self.assertEqual(x.npu().get_device(), 0)
            torch_npu.npu.set_device(1)
        self.assertEqual(x.npu().get_device(), 0)

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_current_stream(self):
        d0 = torch.device('npu:0')
        d1 = torch.device('npu:1')

        s0 = torch_npu.npu.current_stream()
        s1 = torch_npu.npu.current_stream(device=1)
        s2 = torch_npu.npu.current_stream(device=0)

        self.assertEqual(d0, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(s0, s2)

        with torch_npu.npu.device(d1):
            s0 = torch_npu.npu.current_stream()
            s1 = torch_npu.npu.current_stream(1)
            s2 = torch_npu.npu.current_stream(d0)

        self.assertEqual(d1, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(s0, s1)

        with self.assertRaisesRegex(ValueError,
                                    "Expected a npu device, but got: cpu"):
            torch_npu.npu.current_stream(torch.device('cpu'))

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    @skipCUDANonDefaultStreamIf(True)
    def test_default_stream(self):
        d0 = torch.device('npu:0')
        d1 = torch.device('npu:1')

        with torch_npu.npu.device(d0):
            s0 = torch_npu.npu.default_stream()

        with torch_npu.npu.device(d1):
            s1 = torch_npu.npu.default_stream()

        s2 = torch_npu.npu.default_stream(device=0)
        s3 = torch_npu.npu.default_stream(d1)

        self.assertEqual(d0, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(d1, s3.device)
        self.assertEqual(s0, s2)
        self.assertEqual(s1, s3)

        with torch_npu.npu.device(d0):
            self.assertEqual(torch_npu.npu.current_stream(), s0)

        with torch_npu.npu.device(d1):
            self.assertEqual(torch_npu.npu.current_stream(), s1)

        with self.assertRaisesRegex(ValueError,
                                    "Expected a npu device, but got: cpu"):
            torch_npu.npu.default_stream(torch.device('cpu'))

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_stream_event_device(self):
        d0 = torch.device('npu:0')
        d1 = torch.device('npu:1')
        e0 = torch_npu.npu.Event()

        self.assertEqual(None, e0.device)

        with torch_npu.npu.device(d0):
            s0 = torch_npu.npu.current_stream()
            s0.record_event(e0)

        with torch_npu.npu.device(d1):
            s1 = torch_npu.npu.Stream()
            e1 = s1.record_event()

        self.assertEqual(s0.device, torch.device('npu:0'))
        self.assertEqual(e0.device, torch.device('npu:0'))
        self.assertEqual(s1.device, torch.device('npu:1'))
        self.assertEqual(e1.device, torch.device('npu:1'))

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_stream_context(self):
        s0 = torch_npu.npu.current_stream()
        s1 = torch_npu.npu.Stream(device=1)
        s2 = torch_npu.npu.Stream(device=0)

        with torch_npu.npu.device(s1.device):
            prev_stream_on_cuda1 = torch_npu.npu.current_stream()

        self.assertEqual(torch_npu.npu.current_stream(), s0)
        self.assertEqual(0, torch_npu.npu.current_device())
        with torch_npu.npu.stream(s1):
            self.assertEqual(torch_npu.npu.current_stream(), s1)
            self.assertEqual(1, torch_npu.npu.current_device())
            with torch_npu.npu.stream(s2):
                self.assertEqual(torch_npu.npu.current_stream(), s2)
                self.assertEqual(0, torch_npu.npu.current_device())
                with torch_npu.npu.stream(s0):
                    self.assertEqual(torch_npu.npu.current_stream(), s0)
                    self.assertEqual(0, torch_npu.npu.current_device())
                self.assertEqual(torch_npu.npu.current_stream(), s2)
                self.assertEqual(0, torch_npu.npu.current_device())
            self.assertEqual(torch_npu.npu.current_stream(), s1)
            self.assertEqual(1, torch_npu.npu.current_device())

        with torch_npu.npu.device(s1.device):
            self.assertEqual(prev_stream_on_cuda1, torch_npu.npu.current_stream())

        self.assertEqual(torch_npu.npu.current_stream(), s0)
        self.assertEqual(0, torch_npu.npu.current_device())

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_streams_multi_gpu(self):
        default_stream = torch_npu.npu.current_stream()
        self.assertEqual(default_stream.device, torch.device('npu:0'))
        stream = torch_npu.npu.Stream(device=1)
        self.assertEqual(stream.device, torch.device('npu:1'))
        with torch_npu.npu.device(1):
            self.assertEqual(
                torch_npu.npu.current_stream().device, torch.device('npu:1'))
            self.assertNotEqual(torch_npu.npu.current_stream(), default_stream)

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_streams_multi_gpu_query(self):
        d0 = torch.device('npu:0')
        d1 = torch.device('npu:1')
        torch_npu.npu.synchronize(d0)
        torch_npu.npu.synchronize(d1)

        with torch_npu.npu.device(d0):
            s0 = torch_npu.npu.current_stream()

        with torch_npu.npu.device(d1):
            s1 = torch_npu.npu.current_stream()
            torch_npu.npu._sleep(TestNpuMultiNpu.FIFTY_MIL_CYCLES)

        self.assertTrue(s0.query())
        self.assertFalse(s1.query())

        with torch_npu.npu.device(d0):
            self.assertTrue(s0.query())
            self.assertFalse(s1.query())

        with torch_npu.npu.device(d1):
            self.assertTrue(s0.query())
            self.assertFalse(s1.query())

        # deliberately using a different device
        with torch_npu.npu.device(d0):
            s1.synchronize()

        self.assertTrue(s0.query())
        self.assertTrue(s1.query())

        with torch_npu.npu.device(d0):
            self.assertTrue(s0.query())
            self.assertTrue(s1.query())

        with torch_npu.npu.device(d1):
            self.assertTrue(s0.query())
            self.assertTrue(s1.query())

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_streams_multi_gpu_eq(self):
        d0 = torch.device('npu:0')
        d1 = torch.device('npu:1')

        with torch_npu.npu.device(d0):
            s0 = torch_npu.npu.current_stream()
            s1 = torch_npu.npu.current_stream()

        with torch_npu.npu.device(d1):
            s2 = torch_npu.npu.current_stream()
            s3 = torch_npu.npu.current_stream()

        self.assertTrue(s0 == s0)
        self.assertTrue(s0 == s1)
        self.assertTrue(s2 == s2)
        self.assertTrue(s2 == s3)
        self.assertFalse(s0 == s2)
        self.assertFalse(s1 == s3)

        self.assertEqual(s0.device, s1.device)
        self.assertEqual(s0.cuda_stream, s1.cuda_stream)
        self.assertEqual(s2.device, s3.device)
        self.assertEqual(s2.cuda_stream, s3.cuda_stream)
        self.assertNotEqual(s0.device, s3.device)

        self.assertEqual(hash(s0), hash(s1))
        self.assertEqual(hash(s2), hash(s3))
        self.assertNotEqual(hash(s0), hash(s3))

    @unittest.skipIf(not TEST_MULTINPU, "multi-NPU not supported")
    def test_streams_priority(self):
        low, high = torch_npu.npu.Stream.priority_range()
        s0 = torch_npu.npu.Stream(device=0, priority=low)

        self.assertEqual(low, s0.priority)
        self.assertEqual(torch.device('npu:0'), s0.device)

        s1 = torch_npu.npu.Stream(device=1, priority=high)

        self.assertEqual(high, s1.priority)
        self.assertEqual(torch.device('npu:1'), s1.device)

    @unittest.skipIf(not TEST_MULTINPU, "multi-NPU not supported")
    def test_tensor_device(self):
        self.assertEqual(torch_npu.npu.FloatTensor(1).get_device(), 0)
        self.assertEqual(torch_npu.npu.FloatTensor(1, device=1).get_device(), 1)
        with torch_npu.npu.device(1):
            self.assertEqual(torch_npu.npu.FloatTensor(1).get_device(), 1)
            self.assertEqual(torch_npu.npu.FloatTensor(1, device=0).get_device(), 0)
            self.assertEqual(torch_npu.npu.FloatTensor(1, device=None).get_device(), 1)

    @staticmethod
    def _stream_synchronize(self, spin_time_cycles):
        s = torch_npu.npu.current_stream()
        e_tik = torch_npu.npu.Event(enable_timing=True)
        e_tok = torch_npu.npu.Event(enable_timing=True)

        e_tik.record(s)
        torch_npu.npu._sleep(spin_time_cycles)
        e_tok.record(s)
        s.synchronize()

        self.assertTrue(s.query())

        # not necessary to check e_tik and e_tok, as elapsed_time would throw
        # exception if otherwise.
        return e_tik.elapsed_time(e_tok)

    @staticmethod
    def _event_synchronize(self, spin_time_cycles):
        s = torch_npu.npu.current_stream()
        e_tik = torch_npu.npu.Event(enable_timing=True)
        e_tok = torch_npu.npu.Event(enable_timing=True)

        e_tik.record(s)
        torch_npu.npu._sleep(spin_time_cycles)
        s.record_event(e_tok)
        e_tok.synchronize()

        self.assertTrue(s.query())

        # not necessary to check e_tik and e_tok, as elapsed_time would throw
        # exception if otherwise.
        return e_tik.elapsed_time(e_tok)

    @staticmethod
    def _event_wait(self, spin_time_cycles):
        s0 = torch_npu.npu.current_stream()
        s1 = torch_npu.npu.Stream()
        e_tik = torch_npu.npu.Event(blocking=True, enable_timing=True)
        e_tok = torch_npu.npu.Event(blocking=True, enable_timing=True)

        e_tik.record(s0)
        torch_npu.npu._sleep(spin_time_cycles - 10)
        e_sync = torch_npu.npu.Event(blocking=True)
        e_sync.record()
        e_sync.wait(s1)
        with torch_npu.npu.stream(s1):
            torch_npu.npu._sleep(10)
        s1.synchronize()
        e_tok.record()
        e_tok.synchronize()

        self.assertTrue(s0.query())
        self.assertTrue(s1.query())
        self.assertTrue(e_sync.query())

        # not necessary to check e_tik and e_tok, as elapsed_time would throw
        # exception if otherwise.
        return e_tik.elapsed_time(e_tok)

    @staticmethod
    def _test_stream_event_nogil(self, sync_func, p2c, c2p):
        with torch_npu.npu.device('npu:1'):
            c2p.put(0)
            p2c.get()
            c2p.put(sync_func(self, TestNpuMultiNpu.FIFTY_MIL_CYCLES))

    @skipIfRocm
    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_stream_event_nogil(self):
        for sync_func in [TestNpuMultiNpu._stream_synchronize,
                          TestNpuMultiNpu._event_synchronize,
                          TestNpuMultiNpu._event_wait]:
            p2c = queue.Queue()
            c2p = queue.Queue()
            e_tik = torch_npu.npu.Event(enable_timing=True)
            e_tok = torch_npu.npu.Event(enable_timing=True)

            t = threading.Thread(
                target=TestNpuMultiNpu._test_stream_event_nogil,
                args=(self, sync_func, p2c, c2p))
            t.daemon = True
            t.start()

            c2p.get()
            with torch_npu.npu.device('npu:0'):
                e_tik.record()
                p2c.put(0)
                parent_time = sync_func(self, TestNpuMultiNpu.FIFTY_MIL_CYCLES)
                child_time = c2p.get()
                e_tok.record()
                e_tok.synchronize()
                total_time = e_tik.elapsed_time(e_tok)

            # Without GIL, synchronizations in parent and child threads can
            # overlap. The total execution time should be a little bit longer
            # than spinning fifty million cycles and much shorter than twice of
            # that. However, testing absolute execution time is not reliable as
            # it may vary on different hardware in different environments.
            # Therefore, this test uses relative comparisons, checking if the
            # sum of parent and child threads execution time is greater than the
            # real execution time by least 40%.
            self.assertGreater(parent_time + child_time, total_time * 1.4)

    # This test is flaky for ROCm, see issue #62602
    @skipIfRocm
    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_events_wait(self):
        d0 = torch.device('npu:0')
        d1 = torch.device('npu:1')
        torch_npu.npu.synchronize(d0)
        torch_npu.npu.synchronize(d1)

        with torch_npu.npu.device(d0):
            s0 = torch_npu.npu.current_stream()
            torch_npu.npu._sleep(TestNpuMultiNpu.FIFTY_MIL_CYCLES)
            e0 = torch_npu.npu.Event()
            s0.record_event(e0)

        with torch_npu.npu.device(d1):
            s1 = torch_npu.npu.current_stream()

        self.assertFalse(s0.query())
        self.assertTrue(s1.query())

        s1.wait_event(e0)
        s1.synchronize()

        self.assertTrue(e0.query())
        self.assertTrue(s0.query())
        self.assertTrue(s1.query())

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_events_multi_gpu_query(self):
        d0 = torch.device('npu:0')
        d1 = torch.device('npu:1')

        with torch_npu.npu.device(d0):
            s0 = torch_npu.npu.current_stream()
            e0 = s0.record_event()
            s0.synchronize()

        with torch_npu.npu.device(d1):
            s1 = torch_npu.npu.current_stream()
            torch_npu.npu._sleep(TestNpuMultiNpu.FIFTY_MIL_CYCLES)
            e1 = s1.record_event()

        self.assertTrue(e0.query())
        self.assertFalse(e1.query())

        with torch_npu.npu.device(d0):
            self.assertTrue(e0.query())
            self.assertFalse(e1.query())

        with torch_npu.npu.device(d1):
            self.assertTrue(e0.query())
            self.assertFalse(e1.query())

        # deliberately using a different device
        with torch_npu.npu.device(d0):
            e1.synchronize()

        self.assertTrue(e0.query())
        self.assertTrue(e1.query())

        with torch_npu.npu.device(d0):
            self.assertTrue(e0.query())
            self.assertTrue(e1.query())

        with torch_npu.npu.device(d1):
            self.assertTrue(e0.query())
            self.assertTrue(e1.query())

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    @skipIfRocm
    def test_events_multi_gpu_elapsed_time(self):
        d0 = torch.device('npu:0')
        d1 = torch.device('npu:1')

        with torch_npu.npu.device(d0):
            s0 = torch_npu.npu.current_stream()
            e0 = torch_npu.npu.Event(enable_timing=True)
            torch_npu.npu._sleep(10)
            s0.record_event(e0)

        with torch_npu.npu.device(d1):
            s1 = torch_npu.npu.current_stream()
            e1 = torch_npu.npu.Event(enable_timing=True)
            torch_npu.npu._sleep(TestNpuMultiNpu.FIFTY_MIL_CYCLES)
            s1.record_event(e1)

        e0.synchronize()
        e1.synchronize()
        with torch_npu.npu.device(d0):
            with self.assertRaises(RuntimeError):
                self.assertGreater(e0.elapsed_time(e1), 0)

        with torch_npu.npu.device(d1):
            with self.assertRaises(RuntimeError):
                self.assertGreater(e0.elapsed_time(e1), 0)

        with torch_npu.npu.device(d0):
            s0 = torch_npu.npu.current_stream()
            e2 = torch_npu.npu.Event(enable_timing=True)
            torch_npu.npu._sleep(TestNpuMultiNpu.FIFTY_MIL_CYCLES)
            s0.record_event(e2)
            s0.synchronize()

        self.assertGreater(e0.elapsed_time(e2), 0)

        # deliberately calling from a different device
        with torch_npu.npu.device(d1):
            self.assertGreater(e0.elapsed_time(e2), 0)

    @contextlib.contextmanager
    def _get_external_stream(self, device):
        cudart = torch_npu.npu.cudart()
        stream = ctypes.c_ulonglong(0)
        stream_p = ctypes.POINTER(ctypes.c_void_p)(stream)
        stream_p_int = ctypes.cast(stream_p, ctypes.c_void_p).value
        with device:
            try:
                out = cudart.cudaStreamCreate(stream_p_int)
                self.assertEqual(out, 0)
                self.assertNotEqual(stream.value, 0)
                yield stream.value
            finally:
                out = cudart.cudaStreamDestroy(stream.value)
                self.assertEqual(out, 0)

    def test_external_streams(self):
        device = torch_npu.npu.device(0)
        with self._get_external_stream(device) as stream_v:
            ext_stream = torch_npu.npu.ExternalStream(stream_v)
            self.assertEqual(stream_v, ext_stream.cuda_stream)
            self.assertEqual(ext_stream.device.index, device.idx)

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_external_streams_multi_device(self):
        device = torch_npu.npu.device(1)
        with self._get_external_stream(device) as stream_v:
            ext_stream = torch_npu.npu.ExternalStream(
                stream_v, device=device)
            self.assertEqual(stream_v, ext_stream.cuda_stream)
            self.assertEqual(ext_stream.device.index, device.idx)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_caching_pinned_memory_multi_gpu(self):
        # checks that the events preventing pinned memory from being re-used
        # too early are recorded on the correct NPU
        cycles_per_ms = get_cycles_per_ms()

        t = torch.FloatTensor([1]).pin_memory()
        ptr = t.data_ptr()
        gpu_tensor0 = torch_npu.npu.FloatTensor([0], device=0)
        gpu_tensor1 = torch_npu.npu.FloatTensor([0], device=1)

        with torch_npu.npu.device(1):
            torch_npu.npu._sleep(int(1000 * cycles_per_ms))  # delay the copy by 1s
            gpu_tensor1.copy_(t, non_blocking=True)

        del t
        t = torch.FloatTensor([2]).pin_memory()
        self.assertNotEqual(t.data_ptr(), ptr, msg='allocation re-used too soon')

        with torch_npu.npu.device(0):
            gpu_tensor0.copy_(t, non_blocking=True)

        self.assertEqual(gpu_tensor1[0], 1)
        self.assertEqual(gpu_tensor0[0], 2)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_get_set_rng_state_all(self):
        states = torch_npu.npu.get_rng_state_all()
        before0 = torch_npu.npu.FloatTensor(100, device=0).normal_()
        before1 = torch_npu.npu.FloatTensor(100, device=1).normal_()
        torch_npu.npu.set_rng_state_all(states)
        after0 = torch_npu.npu.FloatTensor(100, device=0).normal_()
        after1 = torch_npu.npu.FloatTensor(100, device=1).normal_()
        self.assertEqual(before0, after0, atol=0, rtol=0)
        self.assertEqual(before1, after1, atol=0, rtol=0)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_rng_state_offset(self):
        before = torch_npu.npu.get_rng_state()
        torch_npu.npu._set_rng_state_offset(100)
        offset = torch_npu.npu._get_rng_state_offset()
        torch_npu.npu.set_rng_state(before)
        self.assertEqual(offset, 100)

    # Verifies that mem_get_info works, including when called for a different device
    def test_mem_get_info(self):
        def _test(idx):
            before_free_bytes, before_available_bytes = torch_npu.npu.mem_get_info(idx)
            # increasing to 8MB to force acquiring a new block and overcome blocksize differences across platforms
            t = torch.randn(1024 * 1024 * 8, device='npu:' + str(idx))
            if IS_JETSON:
                # w/o syncing, mem_get_info will run before memory allocated has actually increased.
                # This race condition causes consistent failure
                torch_npu.npu.synchronize()
            after_free_bytes, after_available_bytes = torch_npu.npu.mem_get_info(idx)

            self.assertLess(after_free_bytes, before_free_bytes)
            self.assertEqual(before_available_bytes, after_available_bytes)

        _test(0)
        if TEST_MULTINPU:
            _test(1)

    # Test that wrap_with_cuda_memory_check successfully detects leak
    def test_cuda_memory_leak_detection(self):
        ln = []

        @self.wrap_with_cuda_memory_check
        def no_leak():
            pass

        @self.wrap_with_cuda_memory_check
        def leak_gpu0():
            # increasing to 8MB to force acquiring a new block and overcome blocksize differences across platforms
            ln.append(torch.randn(1024 * 1024 * 8, device=torch.device("npu:0")))

        no_leak()
        regex = r"NPU driver API confirmed .+ on device 0.+"
        if IS_JETSON:
            try:
                leak_gpu0()
            except RuntimeError as e:
                import re
                assert re.match(regex, str(e)), str(e) + "\n does not match: \n" + regex
        else:
            # assertRaisesRegex does not pass with Python for Jetson,
            # even though the RuntimeError matches regex using re.match
            with self.assertRaisesRegex(RuntimeError, regex):
                leak_gpu0()

        if TEST_MULTINPU:
            @self.wrap_with_cuda_memory_check
            def leak_gpu1():
                # increasing to 8MB to force acquiring a new block and overcome blocksize differences across platforms
                ln.append(torch.randn(1024 * 1024 * 8, device=torch.device("npu:1")))

            with self.assertRaisesRegex(RuntimeError, r"NPU driver API confirmed .+ on device 1.+"):
                leak_gpu1()

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_streaming_backwards_device_transfer(self):
        # This function must run with non-default current streams on all devices, otherwise it's meaningless.
        # The intention is to test that to()'s backward (CopyBackward) interacts properly with the
        # synchronization logic in torch/csrc/autograd/input_buffer.cpp.
        dev0 = torch.device("npu:0")
        dev1 = torch.device("npu:1")

        # Unfortunately I need to make the tensors largeish.
        # Bigger tensors = longer D2D transfers = more likely to expose races.
        size = 2**26

        a = torch.full((size,), 1, device=dev1, dtype=torch.float64, requires_grad=True)
        b = torch.full((size,), 1, device=dev1, dtype=torch.float64, requires_grad=True)

        # Here to_backward_recipient = a*b is used only once, so MulBackward's InputBuffer slot only expects 1 input.
        # This tests the situation where we don't call InputBuffer::accumulate for MulBackward's InputBuffer.
        to_backward_recipient = a * b
        s = to_backward_recipient.to(device="npu:0").sum()
        torch_npu.npu.synchronize(device=dev0)
        torch_npu.npu.synchronize(device=dev1)
        s.backward()
        self.assertTrue(a.grad.sum().item() == size)
        self.assertTrue(b.grad.sum().item() == size)

        # Here to_backward_recipient = a*b is used twice, so MulBackward's InputBuffer slot expects 2 inputs.
        # This tests the situation where we do call InputBuffer::accumulate for MulBackward's InputBuffer.
        a.grad = None
        b.grad = None
        to_backward_recipient = a * b
        # Multiply by 2 here so to's backward creates gradient values that are different from the case above,
        # to mitigate weirdness if the caching allocator happens to reuse memory regions that were populated
        # with 1s by the case above
        s0 = to_backward_recipient.to(device="npu:0").sum() * 2.
        s1 = to_backward_recipient.to(device="npu:0").sum() * 2.
        torch_npu.npu.synchronize(device=dev0)
        torch_npu.npu.synchronize(device=dev1)
        s0.backward(retain_graph=True)
        s1.backward()
        self.assertTrue(a.grad.sum().item() == 4 * size)
        self.assertTrue(b.grad.sum().item() == 4 * size)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    @unittest.skipIf(IS_SANDCASTLE or IS_REMOTE_GPU, "Does not work on Sandcastle")
    def test_cuda_init_race(self):
        import subprocess
        subprocess.check_call([sys.executable, '-c', """\
import torch
import torch_npu
import threading

def worker(rank):
    torch.tensor([1.]).npu(rank)

t1 = threading.Thread(target=worker, args=(0,))
t2 = threading.Thread(target=worker, args=(1,))
t1.start()
t2.start()
"""])

    def test_grad_scaling_unscale(self, dtype=torch.float):
        inv_scale = torch.full((1,), 0.25, dtype=torch.float, device="npu:0")
        found_inf = torch.full((1,), 0.0, dtype=torch.float, device="npu:0")

        size = 10
        g = torch.full((size, size), 4.0, dtype=dtype, device="npu:0")
        ginf = g.clone()
        ginf[2, 2] = float('inf')
        gnan = g.clone()
        gnan[2, 2] = float('nan')

        # Tries selected combinations of
        #  - contiguous grads
        #  - g.clone().t() which is not contiguous but still non overlapping and dense
        #  - variants of g.clone()[:, :5] which are not non overlapping and dense
        # Non overlapping and dense grads route into a multi tensor apply kernel,
        # others use a fallback per-tensor kernel, so we should try both.
        cases = (
            ([g.clone(), g.clone()], False),
            ([g.clone(), g.clone().t()], False),
            ([g.clone(), g.clone()[:, :5]], False),
            ([g.clone()[:, :5], g.clone()[:, :5]], False),
            ([g.clone(), ginf.clone()], True),
            ([g.clone(), gnan.clone()], True),
            ([g.clone(), ginf.clone()[:, :5]], True),
            ([g.clone(), gnan.clone()[:, :5]], True),
            ([ginf.clone(), g.clone()[:, :5]], True),
            ([ginf.clone()[:, :5], g.clone()[:, :5]], True),
        )

        for grads, has_inf in cases:
            found_inf.zero_()
            torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
            if has_inf:
                self.assertEqual(found_inf, 1.0)
            else:
                self.assertEqual(found_inf, 0.0)
                for grad in grads:
                    self.assertEqual(grad, torch.ones_like(grad), rtol=1e-5, atol=1e-7)

        # When passing lists with mismatched dtypes to a raw
        # _amp_foreach_non_finite_check_and_unscale_ call,
        # it's expected to fall back to single-tensor TensorIterator kernel.
        grads = [g.clone(), g.to(dtype=torch.float16)]
        torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
        for grad in grads:
            self.assertEqual(grad, torch.ones_like(grad), rtol=1e-5, atol=1e-7)

        # Passing lists with mismatched devices to a raw
        # _amp_foreach_non_finite_check_and_unscale_ call should raise errors.
        if TEST_MULTINPU:
            with self.assertRaisesRegex(RuntimeError, r"Expected all tensors to be on the same device"):
                torch._amp_foreach_non_finite_check_and_unscale_([g.clone(), g.to(device="cuda:1")],
                                                                 found_inf,
                                                                 inv_scale)

        # Creates a list of grads with mismatched dtypes and devices, to ensure
        # scaler._unscale_grads_ organizes grads by dtype and device before calling
        # _amp_foreach_non_finite_check_and_unscale_ on each set.
        # If inject_inf >= 0, writes an inf into one grad for _unscale_grads_ to find.
        def perfect_storm_grads(inject_inf):
            grads = [g.clone(), g.clone()[:, :5], g.to(dtype=torch.float16), g.to(dtype=torch.float16)]
            if TEST_MULTINPU:
                grads += [g.to(device="npu:1"),
                          g.to(device="npu:1")[:, :5],
                          g.to(device="npu:1", dtype=torch.float16),
                          g.to(device="npu:1", dtype=torch.float16)]
            if inject_inf >= 0:
                grads[inject_inf][2, 2] = float('inf')
            return grads

        scaler = torch_npu.npu.amp.GradScaler()
        dummy_params = [torch.empty_like(g) for g in perfect_storm_grads(-1)]
        dummy_opt = torch.optim.SGD(dummy_params, lr=1.)

        # Ensures the inf/nan checking can find an inf injected onto any grad in the perfect storm.
        for inject_inf in range(-1, len(dummy_params)):
            found_inf = torch.full((1,), 0.0, dtype=torch.float, device="npu:0")
            grads = perfect_storm_grads(inject_inf)
            for i, p in enumerate(dummy_params):
                p.grad = grads[i]
            found_inf_per_device = scaler._unscale_grads_(dummy_opt, inv_scale, found_inf, True)
            if inject_inf < 0:
                # No inf was injected, ensures unscaling worked normally.
                self.assertTrue(sum(v.item() for v in found_inf_per_device.values()) == 0)
                for grad in grads:
                    self.assertEqual(grad, torch.ones_like(grad), rtol=1e-5, atol=1e-7)
            else:
                # inf was injected, ensures inf was found.
                self.assertTrue(sum(v.item() for v in found_inf_per_device.values()) == 1)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_grad_scaling_device_as_key(self):
        # Ensure that different instances of "device" objects that point to the same device
        # are treated as identical keys by dicts.  GradScaler relies on this behavior, and may
        # error otherwise in a way that's difficult to detect (a silent performance hit).
        d = {}
        t = torch.empty((1,), device="npu:0")
        dev0a = torch.device("npu:0")
        dev0b = torch.device("npu:0")
        dev1a = torch.device("npu:1")
        dev1b = torch.device("npu:1")

        self.assertTrue(hash(dev0a) == hash(dev0b))
        self.assertTrue(hash(dev1a) == hash(dev1b))

        d[dev0a] = "0a"
        d[dev0b] = "0b"
        self.assertTrue(len(d) == 1)
        self.assertTrue(d[dev0a] == "0b")
        d[t.device] = "t"
        self.assertTrue(len(d) == 1)
        self.assertTrue(d[dev0a] == "t")

        d[dev1a] = "1a"
        d[dev1b] = "1b"
        self.assertTrue(len(d) == 2)
        self.assertTrue(d[dev1a] == "1b")

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_grad_scaling_scale(self):
        scaler = torch_npu.npu.amp.GradScaler(init_scale=2.)
        t0 = torch.full((1,), 4.0, dtype=torch.float32, device="npu:0")
        t1 = torch.full((1,), 4.0, dtype=torch.float32, device="npu:1")
        # Create some nested iterables of tensors on different devices.
        outputs = (t1.clone(), (t0.clone(), t1.clone()), [t0.clone(), (t1.clone(), t0.clone())])
        outputs = scaler.scale(outputs)
        self.assertTrue(outputs[0] == 8.0 and outputs[1][0] == 8.0 and outputs[1][1] == 8.0 and
                        outputs[2][0] == 8.0 and outputs[2][1][0] == 8.0 and outputs[2][1][1] == 8.0)
        self.assertTrue(scaler._scale.device == t1.device)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_grad_scaling_multigpu(self):
        # Same as above, but runs some of the models on device 1.
        # GradScaler should transparently handle losses and gradients on multiple devices.
        # This test could be combined with the test above, but I think it makes sense to treat
        # multi-NPU operations separately.
        dev0 = torch.device("npu:0")
        dev1 = torch.device("npu:1")

        for enabled in True, False:
            mod_control0, mod_scaling0, opt_control0, opt_scaling0, data, loss_fn, skip_iter = \
                _create_scaling_case()
            mod_control1, mod_scaling1, opt_control1, opt_scaling1 = \
                _create_scaling_models_optimizers(device=dev1)

            scaler = torch_npu.npu.amp.GradScaler(init_scale=128., growth_factor=2.0,
                                                  enabled=enabled, growth_interval=1)

            def run(model0, model1, optimizer0, optimizer1, try_scaling_api):
                for i, (input1, target) in enumerate(data):
                    optimizer0.zero_grad()
                    optimizer1.zero_grad()
                    output0 = model0(input1)
                    output1 = model1(input1.to(dev1))
                    loss0 = loss_fn(0.3 * output0 + 0.7 * output1.to(dev0), target)
                    loss1 = loss_fn(0.6 * output0.to(dev1) - 0.4 * output1, target.to(dev1))

                    if try_scaling_api:
                        scaler.scale(loss0).backward(retain_graph=True)
                        scaler.scale(loss1).backward()
                        if i == skip_iter and scaler.is_enabled():
                            model1[1].weight.grad.data.fill_(float('inf'))

                        # As an additional stress test, separately unscale for one of the optimizers.
                        scaler.unscale_(optimizer0)

                        scaler.step(optimizer0)
                        scaler.step(optimizer1)

                        # Make sure the found_infs were collected properly across optimizers and devices.
                        if scaler.is_enabled():
                            self.assertTrue(len(scaler._found_inf_per_device(optimizer0)) == 1)
                            self.assertTrue(len(scaler._found_inf_per_device(optimizer1)) == 1)
                            self.assertTrue(scaler._found_inf_per_device(optimizer0)[dev0].item() == 0.)
                            self.assertTrue(scaler._found_inf_per_device(optimizer1)[dev1].item() ==
                                            float(i == skip_iter))

                        scaler.update()
                    else:
                        loss0.backward(retain_graph=True)
                        loss1.backward()
                        optimizer0.step()
                        if (not scaler.is_enabled()) or (i != skip_iter):
                            optimizer1.step()

            run(mod_control0, mod_control1, opt_control0, opt_control1, False)
            run(mod_scaling0, mod_scaling1, opt_scaling0, opt_scaling1, True)

            # The loss scale should have been multiplied by the growth factor 3 times and the backoff factor once.
            self.assertTrue(scaler.get_scale() == (128. * scaler.get_growth_factor()**3 *
                                                   scaler.get_backoff_factor()**1) if enabled else 1.0)

            # Copy mod_control1 and mod_scaling1 back the device 0 for comparison
            mod_control1.to(dev0)
            mod_scaling1.to(dev0)

            for c, s in zip(chain(mod_control0.parameters(), mod_control1.parameters()),
                            chain(mod_scaling0.parameters(), mod_scaling1.parameters())):
                self.assertEqual(c, s, rtol=1e-5, atol=1e-7)

    @unittest.skipIf(not TEST_MULTINPU, "Test needs multiple NPUs")
    def test_cuda_device_memory_allocated(self):
        from torch.cuda import memory_allocated
        device_count = torch_npu.npu.device_count()
        current_alloc = [memory_allocated(idx) for idx in range(device_count)]
        x = torch.ones(10, device="npu:0")
        self.assertGreater(memory_allocated(0), current_alloc[0])
        self.assertTrue(all(memory_allocated(torch_npu.npu.device(idx)) ==
                        current_alloc[idx] for idx in range(1, device_count)))


class TestNpuComm(TestCase):
    def _test_broadcast(self, input1):
        if not TEST_MULTINPU:
            raise unittest.SkipTest("only one NPU detected")
        # test regular
        results = comm.broadcast(input1, (0, 1))
        for i, t in enumerate(results):
            self.assertEqual(t.get_device(), i)
            self.assertEqual(t, input1)
            if input1.is_npu and input1.get_device() == i:  # test not copying on same device
                self.assertEqual(t.data_ptr(), input1.data_ptr())
        # test out=
        for inplace in [True, False]:
            if inplace:
                outputs = [torch.empty_like(input1, device=0), torch.empty_like(input1, device=1)]
            else:
                outputs = [input1.npu(0), torch.empty_like(input1, device=1)]
            results = comm.broadcast(input1, out=outputs)
            for r, s in zip(results, outputs):
                self.assertIs(r, s)
            for i, t in enumerate(results):
                self.assertEqual(t.get_device(), i)
                self.assertEqual(t, input1)
        # test error msg
        with self.assertRaisesRegex(RuntimeError, r"Exactly one of 'devices' and 'out'"):
            comm.broadcast(input1, (0, 1), out=outputs)
        with self.assertRaisesRegex(RuntimeError,
                                    r"Expected all output tensors to be NPU tensors, but output tensor at index 1"):
            comm.broadcast(input1, out=[input1.npu(0), input1.cpu()])
        with self.assertRaisesRegex(RuntimeError,
                                    r"Expected all output tensors to have same shape as the source .+ at index 1"):
            comm.broadcast(input1, out=[input1.npu(0), input1.npu(1).unsqueeze(0)])

    def test_broadcast_cpu(self):
        self._test_broadcast(torch.randn(5, 5))

    def test_broadcast_gpu(self):
        self._test_broadcast(torch.randn(5, 5).npu())

    def _test_broadcast_coalesced(self, tensors, buffer_size):
        b_tensors = [comm.broadcast(t, (0, 1)) for t in tensors]
        for (_, bt), t in zip(b_tensors, tensors):
            self.assertEqual(bt.get_device(), 1)
            self.assertEqual(bt, t)
            self.assertIsInstance(bt, type(t))

        bc_tensors = comm.broadcast_coalesced(tensors, (0, 1), buffer_size=buffer_size)
        bc_tensors_t = list(zip(*bc_tensors))
        self.assertEqual(b_tensors, bc_tensors_t)
        for (_, bt), (_, bct) in zip(b_tensors, bc_tensors_t):
            self.assertEqual(bt.get_device(), bct.get_device())
            self.assertIsInstance(bct, type(bt))

        # check that tensors on device[0] are returned as-is
        for out_tensors in (b_tensors, bc_tensors_t):
            for inp_t, (out_t, _) in zip(tensors, out_tensors):
                self.assertIs(inp_t, out_t)

        # check that the tensors not on device[0] have different version counters
        # NOTE [ Version Counter in comm.*_coalesced ]
        versions = [t._version for _, t in bc_tensors_t]
        for old_version, (_, t) in zip(versions, bc_tensors_t):
            self.assertEqual(t._version, old_version)
            t.zero_()
            self.assertEqual(t._version, old_version + 1)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    # Note: fails sometimes on the CI, passes on dual gfx906
    def test_broadcast_coalesced(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            self.genSparseTensor((2, 3), 2, 1, False, 'npu', torch.float64)[0],
            torch.randn(numel).long().npu(),
            torch.randn(numel).npu(),
            self.genSparseTensor((2, 3), 2, 10, False, 'npu', torch.float64)[0],
            self.genSparseTensor((2, 3), 2, 5, False, 'npu', torch.float64)[0],
            self.genSparseTensor((3, 3), 2, 7, False, 'npu', torch.int64)[0],
            self.genSparseTensor((2, 3), 2, 2, False, 'npu', torch.float32)[0],
            torch.randn(numel).long().npu(),
            torch.randn(numel).long().npu(),
            self.genSparseTensor((2, 7), 2, 3, False, 'npu', torch.int64)[0],
            torch.randn(numel * 2).int().npu(),  # int is 2x shorter
            torch.randn(numel).npu(),
        ]
        self._test_broadcast_coalesced(tensors, num_bytes * 5 // 2)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_broadcast_coalesced_dense_only(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            torch.randn(numel).long().npu(),
            torch.randn(numel).npu(),
            torch.randn(numel).long().npu(),
            torch.randn(numel).long().npu(),
            torch.randn(numel * 2).int().npu(),  # int is 2x shorter
            torch.randn(numel).npu(),
        ]
        self._test_broadcast_coalesced(tensors, num_bytes * 5 // 2)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_broadcast_coalesced_empty_tensors(self):
        tensors = [
            torch.tensor([]).byte().npu(),
            torch.randn(5).npu(),
            torch.randn(5).double().npu()
        ]
        self._test_broadcast_coalesced(tensors, 256)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_reduce_add(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        x_npu = x.npu(0)
        y_npu = y.npu(1)
        result = comm.reduce_add((x_npu, y_npu))
        self.assertEqual(result.get_device(), 0)
        self.assertEqual(result.cpu(), x + y)

    def _test_reduce_add_coalesced(self, tensors, buffer_size):
        dup_tensors = [tensors, [t.npu(1) for t in tensors]]

        r_tensors = [comm.reduce_add(t) for t in zip(*dup_tensors)]
        for r, t in zip(r_tensors, tensors):
            self.assertEqualTypeString(r, t)
            self.assertEqual(r.coalesce() if r.is_sparse else r, t * 2)

        rc_tensors = comm.reduce_add_coalesced(dup_tensors, buffer_size=buffer_size)
        self.assertEqual(r_tensors, rc_tensors)
        for r, rc in zip(r_tensors, rc_tensors):
            self.assertEqualTypeString(rc, r)

        # Since we have both npu:0 and npu:1 inputs, the outputs must be new.
        # We can check that they have different version counters.
        # NOTE [ Version Counter in comm.*_coalesced ]
        versions = [t._version for t in rc_tensors]
        for old_version, t in zip(versions, rc_tensors):
            self.assertEqual(t._version, old_version)
            t.zero_()
            self.assertEqual(t._version, old_version + 1)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_reduce_add_coalesced(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            self.genSparseTensor((2, 3), 2, 1, False, 'npu', torch.float64)[0],
            torch.randn(numel).long().npu(),
            torch.randn(numel).npu(),
            self.genSparseTensor((2, 3), 2, 10, False, 'npu', torch.float64)[0],
            self.genSparseTensor((2, 3), 2, 5, False, 'npu', torch.float64)[0],
            self.genSparseTensor((3, 3), 2, 7, False, 'npu', torch.int64)[0],
            self.genSparseTensor((2, 3), 2, 2, False, 'npu', torch.float32)[0],
            torch.randn(numel).long().npu(),
            torch.randn(numel).long().npu(),
            self.genSparseTensor((2, 7), 2, 3, False, 'npu', torch.int64)[0],
            torch.randn(numel * 2).int().npu(),  # int is 2x shorter
            torch.randn(numel).npu(),
        ]
        self._test_reduce_add_coalesced(tensors, num_bytes * 5 // 2)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_reduce_add_coalesced_dense_only(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            torch.randn(numel).long().npu(),
            torch.randn(numel).npu(),
            torch.randn(numel).long().npu(),
            torch.randn(numel).long().npu(),
            torch.randn(numel * 2).int().npu(),  # int is 2x shorter
            torch.randn(numel).npu(),
        ]
        self._test_reduce_add_coalesced(tensors, num_bytes * 5 // 2)

    def _test_scatter(self, input1, chunk_sizes=None, dim=0):
        if not TEST_MULTINPU:
            raise unittest.SkipTest("only one NPU detected")
        if chunk_sizes is None:
            ref_chunk_sizes = tuple(repeat(input1.size(dim) // 2, 2))
        else:
            ref_chunk_sizes = chunk_sizes

        # test regular
        result = comm.scatter(input1, (0, 1), chunk_sizes, dim)
        self.assertEqual(len(result), 2)
        chunk_start = 0
        for i, r in enumerate(result):
            chunk_end = chunk_start + ref_chunk_sizes[i]
            index = [slice(None, None) for _ in range(input1.dim())]
            index[dim] = slice(chunk_start, chunk_end)
            self.assertEqual(r, input1[tuple(index)], atol=0, rtol=0)
            chunk_start = chunk_end
            if r.device == input1.device:
                self.assertEqual(r.data_ptr(), input1.data_ptr())  # for target @ same device, a view should be returned

        # test out
        out = [torch.empty_like(t) for t in result]
        result = comm.scatter(input1, dim=dim, out=out)
        self.assertEqual(len(result), 2)
        chunk_start = 0
        for i, r in enumerate(result):
            self.assertIs(r, out[i])
            chunk_end = chunk_start + ref_chunk_sizes[i]
            index = [slice(None, None) for _ in range(input1.dim())]
            index[dim] = slice(chunk_start, chunk_end)
            self.assertEqual(r, input1[tuple(index)], atol=0, rtol=0)
            chunk_start = chunk_end

        # test error msg
        if chunk_sizes is not None:
            with self.assertRaisesRegex(RuntimeError, r"Expected devices and chunk_sizes to be of same length"):
                comm.scatter(input1, [0 for _ in range(len(chunk_sizes) + 1)], dim=dim, chunk_sizes=chunk_sizes)
        with self.assertRaisesRegex(RuntimeError, r"'devices' must not be specified"):
            comm.scatter(input1, (0, 1), dim=dim, out=out)
        with self.assertRaisesRegex(RuntimeError, r"Expected at least one device to scatter to"):
            comm.scatter(input1, (), dim=dim)
        with self.assertRaisesRegex(RuntimeError, r"Expected at least one output tensor to scatter to"):
            comm.scatter(input1, dim=dim, out=[])
        with self.assertRaisesRegex(RuntimeError,
                                    r"Expected all output tensors to be NPU tensors, but output tensor at index 0"):
            comm.scatter(input1, dim=dim, out=([out[0].cpu()] + out[1:]))
        with self.assertRaisesRegex(RuntimeError, r"Output tensor at index 0 has incorrect shape"):
            comm.scatter(input1, dim=dim, out=([out[0].unsqueeze(0)] + out[1:]))
        with self.assertRaisesRegex(RuntimeError, r"Total size for output tensors along scatter dim \d+ does not match"):
            index = [slice(None, None) for _ in range(input1.dim())]
            index[dim] = slice(1, None)
            comm.scatter(input1, dim=dim, out=([out[0][tuple(index)]] + out[1:]))

    def test_scatter_cpu(self):
        self._test_scatter(torch.randn(4, 4), dim=0)

    def test_scatter_cpu_dim(self):
        self._test_scatter(torch.randn(4, 4), dim=1)

    def test_scatter_cpu_neg_dim(self):
        self._test_scatter(torch.randn(4, 4), dim=-2)

    def test_scatter_cpu_sizes(self):
        self._test_scatter(torch.randn(6, 4), chunk_sizes=(2, 4))

    def test_scatter_gpu(self):
        self._test_scatter(torch.randn(4, 4).npu(), dim=0)

    def test_scatter_gpu_dim(self):
        self._test_scatter(torch.randn(4, 4).npu(), dim=1)

    def test_scatter_gpu_neg_dim(self):
        self._test_scatter(torch.randn(4, 4).npu(), dim=-2)

    def test_scatter_gpu_sizes(self):
        self._test_scatter(torch.randn(6, 4).npu(), chunk_sizes=(2, 4))

    def _test_gather(self, dim):
        if not TEST_MULTINPU:
            raise unittest.SkipTest("only one NPU detected")
        x = torch.randn(2, 5, device=0)
        y = torch.randn(2, 5, device=1)
        expected_size = list(x.size())
        expected_size[dim] += y.size(dim)
        expected_size = torch.Size(expected_size)

        destinations = [None, torch.device('npu:0'), torch.device('cpu')]
        if torch_npu.npu.device_count() > 2:
            destinations.append(torch.device('npu:2'))
        with torch_npu.npu.device(1):
            for destination in destinations:
                if destination is None:
                    expected_device = torch.device('npu', torch_npu.npu.current_device())
                else:
                    expected_device = destination
                for use_out in [True, False]:
                    if use_out:
                        out = torch.empty(expected_size, device=expected_device)
                        result = comm.gather((x, y), dim, out=out)
                        self.assertIs(out, result)
                    else:
                        result = comm.gather((x, y), dim, destination=destination)

                    self.assertEqual(result.device, expected_device)
                    self.assertEqual(result.size(), expected_size)

                    index = [slice(None, None), slice(None, None)]
                    index[dim] = slice(0, x.size(dim))
                    self.assertEqual(result[tuple(index)], x)
                    index[dim] = slice(x.size(dim), x.size(dim) + y.size(dim))
                    self.assertEqual(result[tuple(index)], y)

        # test error msg
        with self.assertRaisesRegex(RuntimeError, r"'destination' must not be specified"):
            comm.gather((x, y), dim, destination='cpu', out=torch.empty(expected_size, device='cpu'))
        with self.assertRaisesRegex(RuntimeError, r"Expected at least one tensor to gather from"):
            comm.gather(())
        with self.assertRaisesRegex(RuntimeError, r"Expected all input tensors to be NPU tensors, "):
            comm.gather((x.cpu(), y))
        with self.assertRaisesRegex(RuntimeError, r"Expected all input tensors to have the same number of dimensions"):
            comm.gather((x, y.unsqueeze(0)))
        with self.assertRaisesRegex(RuntimeError, r"Input tensor at index 1 has invalid shape"):
            if dim in [0, -2]:
                comm.gather((x, y[:, 1:]), dim=dim)
            elif dim in [1, -1]:
                comm.gather((x, y[1:, :]), dim=dim)

    def test_gather(self):
        self._test_gather(0)

    def test_gather_dim(self):
        self._test_gather(1)

    def test_gather_neg_dim(self):
        self._test_gather(-1)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_memory_format_scatter_gather(self):
        nhwc = torch.randn((10, 3, 32, 32), device='cpu').contiguous(memory_format=torch.channels_last)
        results = torch_npu.npu.comm.scatter(nhwc, (0, 1), None, 0)
        for result in results:
            self.assertFalse(result.is_contiguous())
            self.assertTrue(result.is_contiguous(memory_format=torch.channels_last))

        gathered = torch_npu.npu.comm.gather(results)
        self.assertTrue(gathered.is_contiguous(memory_format=torch.channels_last))

    @unittest.skipIf(not TEST_MULTINPU, "Test needs multiple NPUs")
    def test_scatter_namedtuple(self):
        # tests ability to scatter namedtuples and retrieve a list where each
        # element is of the expected namedtuple type.
        fields = ("a", "b")
        TestNamedTupleInput_0 = collections.namedtuple("NamedTuple", fields)
        num_gpus = torch_npu.npu.device_count()
        a = torch.rand(num_gpus * 2, device=0)
        b = torch.rand(num_gpus * 2, device=0)
        a_tensors_for_gpu = [a[2 * i: 2 * i + 2].to(i) for i in range(num_gpus)]
        b_tensors_for_gpu = [b[2 * i: 2 * i + 2].to(i) for i in range(num_gpus)]

        inp = TestNamedTupleInput_0(a, b)
        target_gpus = [torch.device(i) for i in range(num_gpus)]
        scatter_out = scatter_gather.scatter(inp, target_gpus)

        for i, x in enumerate(scatter_out):
            self.assertTrue(isinstance(x, type(inp)))
            self.assertEqual(x._fields, fields)
            expected_a = a_tensors_for_gpu[i]
            expected_b = b_tensors_for_gpu[i]
            self.assertEqual(expected_a, x.a)
            self.assertEqual(expected_b, x.b)

        class TestNamedTupleInput_1(NamedTuple):
            a: torch.tensor
            b: torch.tensor

        a = torch.rand(num_gpus * 2, device=0)
        b = torch.rand(num_gpus * 2, device=0)
        a_tensors_for_gpu = [a[2 * i: 2 * i + 2].to(i) for i in range(num_gpus)]
        b_tensors_for_gpu = [b[2 * i: 2 * i + 2].to(i) for i in range(num_gpus)]
        inp = TestNamedTupleInput_1(a, b)

        scatter_out = scatter_gather.scatter(inp, target_gpus)
        for i, x in enumerate(scatter_out):
            self.assertTrue(isinstance(x, type(inp)))
            self.assertEqual(x._fields, fields)
            expected_a = a_tensors_for_gpu[i]
            expected_b = b_tensors_for_gpu[i]
            self.assertEqual(expected_a, x.a)
            self.assertEqual(expected_b, x.b)

    @unittest.skipIf(not TEST_MULTINPU, "Test needs multiple NPUs")
    def test_gather_namedtuple(self):
        # tests ability to gather a list of namedtuples and return a namedtuple where each
        # element is of the expected tensor type.
        fields = ['a', 'b']
        TestNamedTupleInput_0 = collections.namedtuple('NamedTuple', fields)

        num_gpus = torch_npu.npu.device_count()
        a = torch.rand(num_gpus * 2, device=0)
        b = torch.rand(num_gpus * 2, device=1)
        out1 = TestNamedTupleInput_0(a, b)

        a = torch.rand(num_gpus * 2, device=1)
        b = torch.rand(num_gpus * 2, device=0)
        out2 = TestNamedTupleInput_0(a, b)

        outputs = [out1, out2]

        out = scatter_gather.gather(outputs, 'cpu')  # test on CPU
        for i, x in enumerate(out):
            self.assertTrue(isinstance(x, type(out2[-1])))  # x must be a tensor
            cat = torch.cat((outputs[0][i].to('cpu'), outputs[1][i].to('cpu')))
            self.assertTrue(torch.equal(x, cat))

        out = scatter_gather.gather(outputs, 0)  # test on NPU
        for i, x in enumerate(out):
            self.assertTrue(isinstance(x, type(out2[-1])))
            cat = torch.cat((outputs[0][i].to(0), outputs[1][i].to(0)))
            self.assertTrue(torch.equal(x, cat))

        class TestNamedTupleInput_1(NamedTuple):
            a: torch.tensor
            b: torch.tensor

        a = torch.rand(num_gpus * 2, device=0)
        b = torch.rand(num_gpus * 2, device=1)
        out1 = TestNamedTupleInput_1(a, b)

        a = torch.rand(num_gpus * 2, device=1)
        b = torch.rand(num_gpus * 2, device=0)
        out2 = TestNamedTupleInput_1(a, b)

        outputs = [out1, out2]

        out = scatter_gather.gather(outputs, 0)  # test on NPU
        for i, x in enumerate(out):
            self.assertTrue(isinstance(x, type(out2[-1])))
            cat = torch.cat((outputs[0][i].to(0), outputs[1][i].to(0)))
            self.assertTrue(torch.equal(x, cat))

        out = scatter_gather.gather(outputs, 'cpu')  # test on CPU
        for i, x in enumerate(out):
            self.assertTrue(isinstance(x, type(out2[-1])))
            cat = torch.cat((outputs[0][i].to('cpu'), outputs[1][i].to('cpu')))
            self.assertTrue(torch.equal(x, cat))


instantiate_parametrized_tests(TestNpuMultiNpu)


if __name__ == '__main__':
    run_tests()
