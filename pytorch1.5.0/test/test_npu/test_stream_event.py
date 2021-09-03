# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
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

import unittest
import time
import torch
import torch.npu
import threading
from contextlib import contextmanager
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfRocm, PY3

TEST_NPU = torch.npu.is_available()
TEST_MULTINPU = TEST_NPU and torch.npu.device_count() >= 2

def skipNPUNonDefaultStreamIf(condition):
    def dec(fn):
        if getattr(fn, '_do_npu_non_default_stream', True):  # if current True
            fn._do_npu_non_default_stream = not condition
        return fn
    return dec


class TestDevice(unittest.TestCase):

    def setUp(self) -> None:
        # before one test
        pass

    def tearDown(self) -> None:
        # after one test
        pass

    @classmethod
    def setUpClass(cls) -> None:
        # before all test
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        # after all test
        pass

    '''
    单元测试
    '''
    def test_get_device_index(self):
        from torch.npu._utils import _get_device_index
        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
            _get_device_index('npu0', optional=True)
        with self.assertRaisesRegex(ValueError, "Expected a npu device"):
            cpu_device = torch.device('cpu')
            _get_device_index(cpu_device, optional=True)

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_npu_set_device(self):
        x = torch.randn(5, 5)
        with torch.npu.device(1):
            self.assertEqual(x.to('npu').get_device(), 1)
            torch.npu.set_device(0)
            self.assertEqual(x.to('npu').get_device(), 0)
            with torch.npu.device(1):
                self.assertEqual(x.to('npu').get_device(), 1)
            self.assertEqual(x.to('npu').get_device(), 0)
            torch.npu.set_device(1)
        self.assertEqual(x.to('npu').get_device(), 0)

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_current_stream(self):
        d0 = torch.device('npu:0')
        d1 = torch.device('npu:1')

        s0 = torch.npu.current_stream()
        s1 = torch.npu.current_stream(device=1)
        s2 = torch.npu.current_stream(device=0)

        self.assertEqual(d0, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(s0, s2)

        with torch.npu.device(d1):
            s0 = torch.npu.current_stream()
            s1 = torch.npu.current_stream(1)
            s2 = torch.npu.current_stream(d0)

        self.assertEqual(d1, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(s0, s1)

        with self.assertRaisesRegex(ValueError, "Expected a npu device, but got: cpu"):
            torch.npu.current_stream(torch.device('cpu'))

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    @skipNPUNonDefaultStreamIf(True)
    def test_default_stream(self):
        d0 = torch.device('npu:0')
        d1 = torch.device('npu:1')

        with torch.npu.device(d0):
            s0 = torch.npu.default_stream()

        with torch.npu.device(d1):
            s1 = torch.npu.default_stream()

        s2 = torch.npu.default_stream(device=0)
        s3 = torch.npu.default_stream(d1)

        self.assertEqual(d0, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(d1, s3.device)
        self.assertEqual(s0, s2)
        self.assertEqual(s1, s3)

        with torch.npu.device(d0):
            self.assertEqual(torch.npu.current_stream(), s0)

        with torch.npu.device(d1):
            self.assertEqual(torch.npu.current_stream(), s1)

        with self.assertRaisesRegex(ValueError, "Expected a npu device, but got: cpu"):
            torch.npu.default_stream(torch.device('cpu'))

    def test_npu_synchronize(self):
        torch.npu.synchronize()
        torch.npu.synchronize('npu')
        torch.npu.synchronize('npu:0')
        torch.npu.synchronize(0)
        torch.npu.synchronize(torch.device('npu:0'))

        if TEST_MULTINPU:
            torch.npu.synchronize('npu:1')
            torch.npu.synchronize(1)
            torch.npu.synchronize(torch.device('npu:1'))

        with self.assertRaisesRegex(ValueError, "Expected a npu device, but"):
            torch.npu.synchronize(torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "Expected a npu device, but"):
            torch.npu.synchronize("cpu")

    def test_stream_event_repr(self):
        s = torch.npu.current_stream()
        self.assertTrue("torch.npu.Stream" in s.__repr__())
        e = torch.npu.Event()
        self.assertTrue("torch.npu.Event" in e.__repr__())
        s.record_event(e)
        self.assertTrue("torch.npu.Event" in e.__repr__())

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_events_wait(self):
        d0 = torch.device('npu:0')
        d1 = torch.device('npu:1')
        torch.npu.synchronize(d0)
        torch.npu.synchronize(d1)

        with torch.npu.device(d0):
            s0 = torch.npu.current_stream()
            time.sleep(2)
            #torch.npu._sleep(TestCuda.FIFTY_MIL_CYCLES)
            e0 = torch.npu.Event()
            s0.record_event(e0)

        with torch.npu.device(d1):
            s1 = torch.npu.current_stream()

        #self.assertFalse(s0.query())
        #self.assertTrue(s1.query())

        s1.wait_event(e0)
        s1.synchronize()

        self.assertTrue(e0.query())
        #self.assertTrue(s0.query())
        #self.assertTrue(s1.query())

    @skipNPUNonDefaultStreamIf(True)
    def test_streams(self):
        default_stream = torch.npu.current_stream()
        user_stream = torch.npu.Stream()
        self.assertEqual(torch.npu.current_stream(), default_stream)
        self.assertNotEqual(default_stream, user_stream)
        self.assertEqual(default_stream.npu_stream, 0)
        self.assertNotEqual(user_stream.npu_stream, 0)
        print("user_stream :", user_stream)
        with torch.npu.stream(user_stream):
            self.assertEqual(torch.npu.current_stream(), user_stream)
        #self.assertTrue(user_stream.query())
        #tensor1 = torch.ByteTensor(5).pin_memory()
        #tensor2 = tensor1.npu(non_blocking=True) + 1
        default_stream.synchronize()
        #self.assertTrue(default_stream.query())

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_stream_event_device(self):
        d0 = torch.device('npu:0')
        d1 = torch.device('npu:1')
        e0 = torch.npu.Event()

        self.assertEqual(None, e0.device)

        with torch.npu.device(d0):
            s0 = torch.npu.current_stream()
            s0.record_event(e0)

        with torch.npu.device(d1):
            s1 = torch.npu.Stream()
            e1 = s1.record_event()

        self.assertEqual(s0.device, torch.device('npu:0'))
        self.assertEqual(e0.device, torch.device('npu:0'))
        self.assertEqual(s1.device, torch.device('npu:1'))
        self.assertEqual(e1.device, torch.device('npu:1'))

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    # Note: fails sometimes on the CI, passes on dual gfx906
    @skipIfRocm
    def test_stream_context(self):
        s0 = torch.npu.current_stream()
        s1 = torch.npu.Stream(device=1)
        s2 = torch.npu.Stream(device=0)

        with torch.npu.device(s1.device):
            prev_stream_on_npu1 = torch.npu.current_stream()

        self.assertEqual(torch.npu.current_stream(), s0)
        self.assertEqual(0, torch.npu.current_device())
        with torch.npu.stream(s1):
            self.assertEqual(torch.npu.current_stream(), s1)
            self.assertEqual(1, torch.npu.current_device())
            with torch.npu.stream(s2):
                self.assertEqual(torch.npu.current_stream(), s2)
                self.assertEqual(0, torch.npu.current_device())
                with torch.npu.stream(s0):
                    self.assertEqual(torch.npu.current_stream(), s0)
                    self.assertEqual(0, torch.npu.current_device())
                self.assertEqual(torch.npu.current_stream(), s2)
                self.assertEqual(0, torch.npu.current_device())
            self.assertEqual(torch.npu.current_stream(), s1)
            self.assertEqual(1, torch.npu.current_device())

    def test_elapsed_time(self):
        start = torch.npu.Event()
        stop = torch.npu.Event()
        torch.npu.Event.record(start)
        stream1 = torch.npu.Stream()
        torch.npu.Event.record(stop)
        time.sleep(2)
        stop.synchronize()
        times = start.elapsed_time(stop)
        print("times =",times)

    def test_wait_stream(self):
        stream = torch.npu.Stream()
        first = True
        print("\ntest_wait_stream()")
        print("torch.npu.current_stream().wait_stream(stream)  begin")
        torch.npu.current_stream().wait_stream(stream)
        print("torch.npu.current_stream().wait_stream(stream)  end")

    '''
    场景测试
    '''
    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_streams_multi_npu(self):
        default_stream = torch.npu.current_stream()
        self.assertEqual(default_stream.device, torch.device('npu:0'))
        stream = torch.npu.Stream(device=1)
        self.assertEqual(stream.device, torch.device('npu:1'))
        with torch.npu.device(1):
            self.assertEqual(torch.npu.current_stream().device, torch.device('npu:1'))
            self.assertNotEqual(torch.npu.current_stream(), default_stream)

    @unittest.skipIf(not TEST_MULTINPU, "detected only one NPU")
    def test_streams_multi_npu_eq(self):
        d0 = torch.device('npu:0')
        d1 = torch.device('npu:1')

        with torch.npu.device(d0):
            s0 = torch.npu.current_stream()
            s1 = torch.npu.current_stream()

        with torch.npu.device(d1):
            s2 = torch.npu.current_stream()
            s3 = torch.npu.current_stream()

        self.assertTrue(s0 == s0)
        self.assertTrue(s0 == s1)
        self.assertTrue(s2 == s2)
        self.assertTrue(s2 == s3)
        self.assertFalse(s0 == s2)
        self.assertFalse(s1 == s3)

        self.assertEqual(s0.device, s1.device)
        self.assertEqual(s0.npu_stream, s1.npu_stream)
        self.assertEqual(s2.device, s3.device)
        self.assertEqual(s2.npu_stream, s3.npu_stream)
        self.assertNotEqual(s0.device, s3.device)

        self.assertEqual(hash(s0), hash(s1))
        self.assertEqual(hash(s2), hash(s3))
        self.assertNotEqual(hash(s0), hash(s3))

    @skipIfRocm
    @unittest.skipIf(not PY3, "Barrier is unavailable before Python3")
    def test_cublas_multiple_threads_same_device(self):
        # Note, these parameters should be very carefully tuned
        # Too small number makes it hard for the racing condition
        # to happen, while too large number sometimes cause hang
        size = 1024
        num_threads = 2
        trials = 3
        test_iters = 100

        weight = torch.ones((size, size), device='cpu').to('npu')
        results = {}
        barrier = threading.Barrier(num_threads)

        def _worker(t):
            my_stream = torch.npu.Stream()
            # Hard sync so we don't need to worry about creating and using tensors
            # across streams or the fact that default streams are thread-local.
            # Those issues are not the target of this test.
            torch.npu.synchronize()
            # Line up threads to increase likelihood of race conditions.
            barrier.wait()
            with torch.npu.stream(my_stream):
                for i in range(test_iters):
                    # If all threads are sharing the same cublas handle,
                    # the following sequence may occur:
                    # thread 0 calls cublasSetStream()
                    # thread 1 calls cublasSetStream()
                    # thread 0 launches its raw gemm, which it thinks is in
                    #          its own stream, but is actually in thread 1's stream.
                    # thread 0 enqueues its div_, which IS is its own stream,
                    #          but actually now races with its gemm.
                    results[t] = torch.mm(results[t], weight)
                    results[t].div_(float(size))
            torch.npu.synchronize()

    def test_copy_non_blocking(self):
        def _test_copy_non_blocking(a, b):
            event = torch.npu.Event()
            a.copy_(b, non_blocking=True)
            event.record()
            self.assertFalse(event.query())
            event.synchronize()
            self.assertEqual(a.tolist(), b.tolist())

        # 10MB copies
        x = torch.ones(10000000, dtype=torch.int32).npu()
        y = torch.zeros(10000000, dtype=torch.int32).pin_memory()
        _test_copy_non_blocking(x, y)

        x = torch.zeros(10000000, dtype=torch.int32).pin_memory()
        y = torch.ones(10000000, dtype=torch.int32).npu()
        _test_copy_non_blocking(x, y)
    

if __name__ == "__main__":
    unittest.main()
