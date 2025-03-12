import os
import shutil
import unittest
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestAllocator(TestCase):
    @SupportedDevices(['Ascend910B'])
    def test_huge_memory_alloc_20M(self):
        prev = torch_npu.npu.memory_allocated()
        a = torch.rand(1024 * 1024 * 40, dtype=torch.float32).npu()
        torch.npu.synchronize()
        # 实际申请1G内存
        self.assertEqual(torch_npu.npu.memory_allocated(), prev + ((4 * 40 * 1024 * 1024 + 32) // 512 + 1) * 512)

    @SupportedDevices(['Ascend910B'])
    def test_huge_memory_alloc_512B(self):
        prev = torch_npu.npu.memory_allocated()
        a = torch.rand(8 * 8 * 16, dtype=torch.float32).npu() # 512B
        torch.npu.synchronize()
        # 实际申请1M内存
        self.assertEqual(torch_npu.npu.memory_allocated(), prev + ((8 * 8 * 16 * 4 + 32) // 512 + 1) * 512)

if __name__ == '__main__':
    run_tests()
