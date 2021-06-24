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


class TestPinMemory(unittest.TestCase):

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

    #@unittest.skipIf(PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property")
    def test_pin_memory(self):
        x = torch.randn(3, 5)
        self.assertFalse(x.is_pinned())
        if not torch.npu.is_available():
            self.assertRaises(RuntimeError, lambda: x.pin_memory())
        else:
            pinned = x.pin_memory()
            print("x：", x )
            print("pinned:", pinned)
            #self.assertTrue(pinned.is_pinned())
            self.assertEqual(pinned.numpy().all(), x.numpy().all())
            self.assertNotEqual(pinned.data_ptr(), x.data_ptr())
            # test that pin_memory on already pinned tensor has no effect
            self.assertIs(pinned.numpy().all(), pinned.pin_memory().numpy().all())
            #self.assertEqual(pinned.data_ptr(), pinned.pin_memory().data_ptr())
    
    def test_noncontiguous_pinned_memory(self):
        # See issue #3266
        x = torch.arange(0, 10).view((2, 5))
        self.assertEqual(x.t().tolist(), x.t().pin_memory().tolist())
        self.assertFalse((x.t().numpy()-x.t().pin_memory().numpy()).all())

    def test_caching_pinned_memory(self):
        #cycles_per_ms = get_cycles_per_ms()

        # check that allocations are re-used after deletion
        t = torch.FloatTensor([1]).pin_memory()
        ptr = t.data_ptr()
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertEqual(t.data_ptr(), ptr, 'allocation not reused')

        # check that the allocation is not re-used if it's in-use by a copy
        npu_tensor = torch.npu.FloatTensor([0])
        #torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the copy
        time.sleep(5)
        npu_tensor.copy_(t, non_blocking=True)
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertNotEqual(t.data_ptr(), ptr, 'allocation re-used too soon')
        self.assertEqual(npu_tensor.tolist(), [1])
        #self.assertEqual(list(npu_tensor), [1])

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    def test_caching_pinned_memory_multi_npu(self):
        # checks that the events preventing pinned memory from being re-used
        # too early are recorded on the correct GPU
        #cycles_per_ms = get_cycles_per_ms()

        t = torch.FloatTensor([1]).pin_memory()
        ptr = t.data_ptr()
        npu_tensor0 = torch.npu.FloatTensor([0], device=0)
        npu_tensor1 = torch.npu.FloatTensor([0], device=1)

        with torch.npu.device(1):
            #torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the copy
            time.sleep(5)
            npu_tensor1.copy_(t, non_blocking=True)

        del t
        t = torch.FloatTensor([2]).pin_memory()
        self.assertNotEqual(t.data_ptr(), ptr, 'allocation re-used too soon')

        with torch.npu.device(0):
            npu_tensor0.copy_(t, non_blocking=True)

        self.assertEqual(npu_tensor1[0].item(), 1)
        self.assertEqual(npu_tensor0[0].item(), 2)


    def test_empty_shared(self):
        t = torch.Tensor()
        t.share_memory_()

if __name__ == "__main__":
    unittest.main()
