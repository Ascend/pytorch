import os
import gc

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
import torch_npu

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True,segment_size_mb:128"


class Test_expandable_segments(TestCase):
    def test_empty_virt_addr_cache(self):
        gc.collect()
        torch_npu.npu.empty_cache()
        x = torch.empty((2, 1024, 1024), device="npu", dtype=torch.float32)
        self.assertEqual(torch_npu.npu.memory_reserved(), 128 * 1024 * 1024)


if __name__ == '__main__':
    run_tests()
