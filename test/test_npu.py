# Copyright (c) 2021, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from itertools import product
import collections
import gc
import numpy as np

import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import freeze_rng_state


class TestNpu(TestCase):

    FIFTY_MIL_CYCLES = 50000000

    def _check_memory_stat_consistency(self):
        snapshot = torch_npu.npu.memory_snapshot()

        expected_each_device = collections.defaultdict(lambda: collections.defaultdict(int))

        for segment in snapshot:
            expected = expected_each_device[segment["device"]]
            pool_str = segment["segment_type"] + "_pool"

            expected["segment.all.current"] += 1
            expected["segment." + pool_str + ".current"] += 1

            expected["allocated_bytes.all.current"] += segment["allocated_size"]
            expected["allocated_bytes." + pool_str + ".current"] += segment["allocated_size"]

            expected["reserved_bytes.all.current"] += segment["total_size"]
            expected["reserved_bytes." + pool_str + ".current"] += segment["total_size"]

            expected["active_bytes.all.current"] += segment["active_size"]
            expected["active_bytes." + pool_str + ".current"] += segment["active_size"]

            is_split = len(segment["blocks"]) > 1
            for block in segment["blocks"]:
                if block["state"] == "active_allocated":
                    expected["allocation.all.current"] += 1
                    expected["allocation." + pool_str + ".current"] += 1

                if block["state"].startswith("active_"):
                    expected["active.all.current"] += 1
                    expected["active." + pool_str + ".current"] += 1

                if block["state"] == "inactive" and is_split:
                    expected["inactive_split.all.current"] += 1
                    expected["inactive_split." + pool_str + ".current"] += 1
                    expected["inactive_split_bytes.all.current"] += block["size"]
                    expected["inactive_split_bytes." + pool_str + ".current"] += block["size"]

        for device, expected in expected_each_device.items():
            stats = torch_npu.npu.memory_stats(device)
            for k, v in expected.items():
                self.assertEqual(v, stats[k])

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
                return torch.npu.FloatTensor(*size)

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

        def assert_change_by_tensor():
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

        assert_change_by_tensor()

        # test empty_cache and reset_peak
        assert_change(0, empty_cache=True)
        assert_change(0, reset_peak=True)

    def test_memory_stats(self):
        gc.collect()
        torch_npu.npu.empty_cache()
        for _ in self._test_memory_stats_generator(self):
            self._check_memory_stat_consistency()

    def test_memory_allocation(self):
        gc.collect()
        torch_npu.npu.empty_cache()
        mem = None
        size = 1
        prev = 0
        try:
            prev = torch_npu.npu.memory_allocated()
            mem = torch_npu.npu.caching_allocator_alloc(size)
            self.assertGreater(torch_npu.npu.memory_allocated(), prev)
        finally:
            if mem is not None:
                torch_npu.npu.caching_allocator_delete(mem)
                self.assertEqual(torch_npu.npu.memory_allocated(), prev)

    def test_out_of_memory(self):
        tensor = torch.zeros(1024, device='npu')

        with self.assertRaisesRegex(AssertionError, "Tried to allocate 8000000000.00 GiB"):
            with self.assertRaisesRegex(RuntimeError, "Tried to allocate 8000000000.00 GiB"):
                torch.empty(1024 * 1024 * 1024 * 8000000000, dtype=torch.int8, device='npu')

        # ensure out of memory error doesn't disturb subsequent kernel
        tensor.fill_(1)
        self.assertTrue((tensor == 1).all())

    def _test_copy_sync_current_stream(self, x, y):
        x_plus_one = x + 1
        s0 = torch_npu.npu.Stream(device=x.device)
        s1 = torch_npu.npu.Stream(device=y.device)
        s2 = torch_npu.npu.Stream(device=x.device)
        s3 = torch_npu.npu.Stream(device=y.device)

        # same dst stream different src streams
        with torch_npu.npu.stream(s0):
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
            with torch_npu.npu.stream(s0):
                y.copy_(x_plus_one)

        with torch_npu.npu.stream(s3), torch_npu.npu.stream(s0):
            y.copy_(x)

        s0.synchronize()
        # Similarly, both copy() ops are synchronized on s0.
        self.assertEqual(y, x)

    def test_copy_non_blocking(self):
        def _test_copy_non_blocking(a, b):
            event = torch_npu.npu.Event()
            a.copy_(b, non_blocking=True)
            event.record()
            event.synchronize()
            self.assertEqual(a, b)

        # 10MB copies
        x = torch.ones(10000000, dtype=torch.uint8).npu()
        y = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        _test_copy_non_blocking(x, y)

        x = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        y = torch.ones(10000000, dtype=torch.uint8).npu()
        _test_copy_non_blocking(x, y)

    def test_to_non_blocking(self):
        stream = torch_npu.npu.current_stream()

        def _test_to_non_blocking(a, non_blocking, dst):
            torch_npu.npu.synchronize()
            # Pushes an 0.1 second spin to stream so if the copy is non blocking,
            # stream will almost surely be active when we query().
            b = a.to(device=dst, non_blocking=non_blocking)
            stream.synchronize()
            self.assertEqual(a, b)
            self.assertTrue(b.is_pinned() == (non_blocking and dst == "cpu"))

        for dst, try_non_blocking in product(("npu", ), (True, )):
            # Creates source on the opposite device from destination.
            src = torch.randn(1000000,
                              device="npu" if dst == "cpu" else "cpu")
            if dst == "npu":
                src = src.pin_memory()
            _test_to_non_blocking(src, try_non_blocking, dst)

    def test_to_cpu_blocking_by_default(self):
        src = torch.randn(1000000, device="npu")
        torch_npu.npu.synchronize()
        dst = src.to(device="cpu")
        self.assertEqual(src, dst)
        self.assertFalse(dst.is_pinned())

    def test_torch_manual_seed_seeds_npu_devices(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float()  # Not support device RNG (.npu()).
            torch.manual_seed(2)
            self.assertEqual(torch_npu.npu.initial_seed(), 2)
            x.uniform_()
            torch.manual_seed(2)
            y = x.clone().uniform_()
            self.assertEqual(x, y)
            self.assertEqual(torch_npu.npu.initial_seed(), 2)

    def test_manual_seed(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float()  # Not support device RNG (.npu()).
            torch_npu.npu.manual_seed(2)
            torch.manual_seed(2)
            self.assertEqual(torch_npu.npu.initial_seed(), 2)
            x.uniform_()
            a = torch.bernoulli(torch.full_like(x, 0.5))
            torch.manual_seed(2)
            y = x.clone().uniform_()
            b = torch.bernoulli(torch.full_like(x, 0.5))
            self.assertEqual(x, y)
            self.assertEqual(a, b)
            self.assertEqual(torch_npu.npu.initial_seed(), 2)

    def test_get_set_rng_state(self):
        with freeze_rng_state():
            torch.manual_seed(3)
            cpu_state = torch.get_rng_state()
            npu_state = torch_npu.npu.get_rng_state()
            self.assertEqual(int(cpu_state[0]), 3)
            self.assertEqual(cpu_state[0], npu_state[0])
            torch_npu.npu.manual_seed(2)
            cpu_state_new = torch.get_rng_state()
            npu_state = torch_npu.npu.get_rng_state()
            self.assertEqual(cpu_state, cpu_state_new)
            self.assertEqual(int(npu_state[0]), 2)

    def test_get_device_index(self):
        from torch_npu.npu import _get_device_index
        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            _get_device_index('npu0', optional=True)

        with self.assertRaisesRegex(ValueError, "Expected a npu device"):
            cpu_device = torch.device('cpu')
            _get_device_index(cpu_device, optional=True)

    def test_npu_synchronize(self):
        torch_npu.npu.synchronize()
        torch_npu.npu.synchronize('npu')
        torch_npu.npu.synchronize('npu:0')
        torch_npu.npu.synchronize(0)
        torch_npu.npu.synchronize(torch.device('npu:0'))

        with self.assertRaisesRegex(ValueError, "Expected a npu device, but"):
            torch_npu.npu.synchronize(torch.device("cpu"))

        with self.assertRaisesRegex(AssertionError, "Expected a npu device, but"):
            with self.assertRaisesRegex(ValueError, "Expected a npu device, but"):
                torch_npu.npu.synchronize("cpu")

    def test_streams(self):
        default_stream = torch_npu.npu.current_stream()
        user_stream = torch_npu.npu.Stream()
        self.assertEqual(torch_npu.npu.current_stream(), default_stream)
        self.assertNotEqual(default_stream, user_stream)
        self.assertNotEqual(user_stream.npu_stream, 0)
        with torch_npu.npu.stream(user_stream):
            self.assertEqual(torch_npu.npu.current_stream(), user_stream)

    def test_stream_event_repr(self):
        s = torch_npu.npu.current_stream()
        self.assertTrue("torch_npu.npu.Stream" in s.__repr__())
        e = torch_npu.npu.Event()
        self.assertTrue("torch_npu.npu.Event" in e.__repr__())
        s.record_event(e)
        self.assertTrue("torch_npu.npu.Event" in e.__repr__())

    def test_events(self):
        stream = torch_npu.npu.current_stream()
        event = torch_npu.npu.Event(enable_timing=True)
        self.assertTrue(event.query())
        start_event = torch_npu.npu.Event(enable_timing=True)
        stream.record_event(start_event)
        stream.record_event(event)
        event.synchronize()
        self.assertTrue(event.query())
        self.assertGreater(start_event.elapsed_time(event), 0)

    @staticmethod
    def _stream_synchronize(self, spin_time_cycles):
        s = torch_npu.npu.current_stream()
        e_tik = torch_npu.npu.Event(enable_timing=True)
        e_tok = torch_npu.npu.Event(enable_timing=True)

        e_tik.record(s)
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
        e_sync = torch_npu.npu.Event(blocking=True)
        e_sync.record()
        e_sync.wait(s1)
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
            c2p.put(sync_func(self, TestNpu.FIFTY_MIL_CYCLES))

    def test_noncontiguous_pinned_memory(self):
        # See issue #3266
        x = torch.arange(0, 10).view((2, 5))
        self.assertEqual(x.t(), x.t().pin_memory())

    def test_caching_pinned_memory(self):

        # check that allocations are re-used after deletion
        t = torch.FloatTensor([1]).pin_memory()
        ptr = t.data_ptr()
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertEqual(t.data_ptr(), ptr)

        # check that the allocation is not re-used if it's in-use by a copy
        npu_tensor = torch.npu.FloatTensor([0])
        npu_tensor.copy_(t, non_blocking=True)
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertEqual(list(npu_tensor), [1])
    
    def test_function_torch_empty_and_to(self):
        x = torch.empty((2, 3), dtype=torch.float16, device='npu')
        x_int32 = x.to(torch.int32)
        res = x_int32 + 1

    def test_function_npu(self):
        x = torch.empty((2, 3), dtype=torch.float16, device='cpu')
        x_npu = x.npu()
        res = x_npu + 1

    def test_function_torch_empty_with_format(self):
        x = torch.empty_with_format((2, 3), dtype=torch.float32, device='npu')
        res = x + 1
    
    def test_function_torch_empty_like(self):
        x = torch.empty((2, 3), dtype=torch.float32, device='npu')
        x_like = torch.empty_like(x)
        res = x_like + 1

    def test_function_torch_empty_strided(self):
        x = torch.empty_strided((2, 3), (1, 2), dtype=torch.int8, device='npu')
    
    def test_function_tensor_new_empty(self):
        x = torch.ones(()).npu()
        x_new_empty = x.new_empty((2, 3), dtype=torch.float16, device='npu')
        res = x_new_empty + 1

    def test_function_tensor_new_empty_strided(self):
        x = torch.ones(()).npu()
        x_new = x.new_empty_strided([2, 3], [3, 1], dtype=torch.float32, device='npu')
        res = x_new + 1

    def test_function_torch_device(self):
        device1 = torch.device("npu:1")
        device2 = torch.device("npu")
        device3 = torch.device("npu", 2)
        self.assertEqual(str(device1), f"device(type='{torch_npu.npu.npu_device}', index=1)")
        self.assertEqual(str(device2), f"device(type='{torch_npu.npu.npu_device}', index=None)")
        self.assertEqual(str(device3), f"device(type='{torch_npu.npu.npu_device}', index=2)")

    def test_function_tensor_data_npu(self):
        x = torch.ones(())
        x.data = x.data.npu()

    def test_function_tensor_new_full(self):
        x_cpu = torch.tensor((), dtype=torch.float32)
        cpu_out = x_cpu.new_full((2, 3), 3.1)
        
        x = torch.tensor((), dtype=torch.float32).npu()
        npu_output1 = x.new_full((2, 3), 3.1, device=None, requires_grad=False)
        npu_output2 = x.new_full((2, 3), 3.1, device='cpu', requires_grad=False)
        npu_output3 = x.new_full((2, 3), 3.1, device='npu', requires_grad=False)
        self.assertRtolEqual(cpu_out.numpy(), npu_output1.cpu().numpy())
        self.assertRtolEqual(cpu_out.numpy(), npu_output2.cpu().numpy())
        self.assertRtolEqual(cpu_out.numpy(), npu_output3.cpu().numpy())

    def test_function_tensor_new_ones(self):
        x_cpu = torch.tensor((), dtype=torch.float32)
        cpu_out = x_cpu.new_ones((2, 3))

        x = torch.tensor((), dtype=torch.float32).npu()
        npu_output1 = x.new_ones((2, 3), device=None, requires_grad=False)
        npu_output2 = x.new_ones((2, 3), device='cpu', requires_grad=False)
        npu_output3 = x.new_ones((2, 3), device='npu', requires_grad=False)
        self.assertRtolEqual(cpu_out.numpy(), npu_output1.cpu().numpy())
        self.assertRtolEqual(cpu_out.numpy(), npu_output2.cpu().numpy())
        self.assertRtolEqual(cpu_out.numpy(), npu_output3.cpu().numpy())

    def test_function_tensor_new_tensor(self):
        x_cpu = torch.tensor((), dtype=torch.float32)
        x = torch.tensor((), dtype=torch.float32).npu()

        list_input = [[1, 2, 3], [4, 5, 6]]
        cpu_out = x_cpu.new_tensor(list_input)
        npu_output1 = x.new_tensor(list_input, device=None, requires_grad=False)
        npu_output2 = x.new_tensor(list_input, device='cpu', requires_grad=False)
        npu_output3 = x.new_tensor(list_input, device='npu', requires_grad=False)
        self.assertRtolEqual(cpu_out.numpy(), npu_output1.cpu().numpy())
        self.assertRtolEqual(cpu_out.numpy(), npu_output2.cpu().numpy())
        print(cpu_out.numpy().dtype, npu_output3.cpu().numpy().dtype)
        self.assertRtolEqual(cpu_out.numpy(), npu_output3.cpu().numpy())

        np_input = np.array(list_input)
        cpu_out = x_cpu.new_tensor(np_input)
        npu_output1 = x.new_tensor(np_input, device=None, requires_grad=False)
        npu_output2 = x.new_tensor(np_input, device='cpu', requires_grad=False)
        npu_output3 = x.new_tensor(np_input, device='npu', requires_grad=False)
        self.assertRtolEqual(cpu_out.numpy(), npu_output1.cpu().numpy())
        self.assertRtolEqual(cpu_out.numpy(), npu_output2.cpu().numpy())
        self.assertRtolEqual(cpu_out.numpy(), npu_output3.cpu().numpy())

        tensor_input = torch.tensor(list_input)
        cpu_out = x_cpu.new_tensor(tensor_input)
        npu_output1 = x.new_tensor(tensor_input, device=None, requires_grad=False)
        npu_output2 = x.new_tensor(tensor_input, device='cpu', requires_grad=False)
        npu_output3 = x.new_tensor(tensor_input, device='npu', requires_grad=False)
        self.assertRtolEqual(cpu_out.numpy(), npu_output1.cpu().numpy())
        self.assertRtolEqual(cpu_out.numpy(), npu_output2.cpu().numpy())
        self.assertRtolEqual(cpu_out.numpy(), npu_output3.cpu().numpy())

    def test_function_tensor_new_zeros(self):
        x_cpu = torch.tensor((), dtype=torch.float32)
        cpu_out = x_cpu.new_zeros((2, 3))

        x = torch.tensor((), dtype=torch.float32).npu()
        npu_output1 = x.new_zeros((2, 3), device=None, requires_grad=False)
        npu_output2 = x.new_zeros((2, 3), device='cpu', requires_grad=False)
        npu_output3 = x.new_zeros((2, 3), device='npu', requires_grad=False)
        self.assertRtolEqual(cpu_out.numpy(), npu_output1.cpu().numpy())
        self.assertRtolEqual(cpu_out.numpy(), npu_output2.cpu().numpy())
        self.assertRtolEqual(cpu_out.numpy(), npu_output3.cpu().numpy())

if __name__ == '__main__':
    run_tests()
