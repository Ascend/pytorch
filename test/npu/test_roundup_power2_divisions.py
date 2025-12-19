import os
import gc
import unittest
import numpy as np

import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "roundup_power2_divisions:[8:1,64:4,128:8,256:2,>:16]"
# roundup_power2_divisions: 0:1, 1:1, 2:1, 3:1, 4:0, 5:0, 6:4, 7:8, 8:2, 9:16, 10:16, 11:16, 12:16, 13:16, 14:16, 15:16


class Test_roundup_power2_divisions(TestCase):
    def test_divisions(self):
        def power2_div(size, div_factor):
            pow2 = 1
            while pow2 < size:
                pow2 = pow2 * 2
            if pow2 == size:
                return pow2
            step = pow2 / 2 / div_factor
            ret = pow2 / 2
            while ret < size:
                ret = ret + step
            return ret

        def align_to_512_or_2m(size):
            new_size = ((size + 511) // 512) * 512
            if new_size < 10485760:
                return new_size
            else:
                return ((size + 2097152 - 1) // 2097152) * 2097152

        def do_test(nelems, divisions=0):
            nbytes = nelems
            # do not use torch.rand
            x = torch.from_numpy(np.random.randn(nelems).astype(np.uint8)).to("npu")
            requested_bytes = torch.npu.memory_stats()["requested_bytes.all.current"]
            do_assert(requested_bytes, nbytes)

            max_memory_allocated = torch.npu.max_memory_allocated()
            allocated_bytes = torch.npu.memory_stats()["allocated_bytes.all.current"]
            active_bytes = torch.npu.memory_stats()["active_bytes.all.current"]
            do_assert(max_memory_allocated, allocated_bytes)
            do_assert(max_memory_allocated, active_bytes)

            if divisions > 1:
                do_assert(max_memory_allocated, power2_div(nbytes, divisions))
            else:
                do_assert(max_memory_allocated, align_to_512_or_2m(nbytes))

        def do_assert(x, y):
            self.assertEqual(x, y)

        torch.npu.memory.empty_cache()
        do_test(513)
        do_test(1025)
        do_test(33 * 1024 * 1024) # index:5, division:0
        do_test(65 * 1024 * 1024, 4) # index:6, division:4
        do_test(200 * 1024 * 1024, 8) # index:7, division:8
        do_test(510 * 1024 * 1024, 2) # index:8, division:2
        do_test(513 * 1024 * 1024, 16) # index:9, division:16

    @SupportedDevices(['Ascend910A'])
    def test_add_32(self):
        torch.npu.memory.empty_cache()

        x = torch.from_numpy(np.random.randn(512).astype(np.uint8)).to("npu")
        allocated_bytes = torch.npu.memory_stats()["allocated_bytes.all.current"]
        active_bytes = torch.npu.memory_stats()["active_bytes.all.current"]
        self.assertEqual(allocated_bytes, active_bytes)
        self.assertEqual(allocated_bytes, 1024)


if __name__ == '__main__':
    run_tests()
