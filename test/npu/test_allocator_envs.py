import os
import math
import shutil
import unittest
import torch
import torch_npu

import torch_npu.npu.utils as utils

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestAllocator(TestCase):
    def test_huge_memory_alloc_20M(self):
        prev = torch_npu.npu.memory_allocated()
        a = torch.rand(1024 * 1024 * 40, dtype=torch.float32).npu()
        # 实际申请1G内存
        if (utils.get_soc_version() >= 260):
            self.assertEqual(torch_npu.npu.memory_allocated(), prev + math.ceil((4 * 40 * 1024 * 1024) / 512) * 512)
        else:
            self.assertEqual(torch_npu.npu.memory_allocated(),
                             prev + math.ceil((4 * 40 * 1024 * 1024 + 32) / 512) * 512)

    def test_huge_memory_alloc_512B(self):
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
        prev = torch_npu.npu.memory_allocated()
        a = torch.rand(8 * 8 * 16, dtype=torch.float32).npu()  # 512B
        # 实际申请1M内存
        if (utils.get_soc_version() >= 260):
            self.assertEqual(torch_npu.npu.memory_allocated(), prev + math.ceil((8 * 8 * 16 * 4) / 512) * 512)
        else:
            self.assertEqual(torch_npu.npu.memory_allocated(), prev + math.ceil((8 * 8 * 16 * 4 + 32) / 512) * 512)

    def test_huge_memory_alloc_512B_by_vm(self):
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        prev = torch_npu.npu.memory_allocated()
        a = torch.rand(8 * 8 * 16, dtype=torch.float32).npu()  # 512B
        # 实际申请1M内存
        if (utils.get_soc_version() >= 260):
            self.assertEqual(torch_npu.npu.memory_allocated(), prev + math.ceil((8 * 8 * 16 * 4) / 512) * 512)
        else:
            self.assertEqual(torch_npu.npu.memory_allocated(), prev + math.ceil((8 * 8 * 16 * 4 + 32) / 512) * 512)
        del os.environ["PYTORCH_NPU_ALLOC_CONF"]

if __name__ == '__main__':
    run_tests()
