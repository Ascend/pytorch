import os
import gc

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"


class Test_expandable_segments(TestCase):
    def test_empty_virt_addr_cache(self):
        gc.collect()
        torch_npu.npu.empty_cache()
        prev = 0

        x = torch.empty((7500, 1024, 1024), device="npu")
        del x
        last_r = torch_npu.npu.memory_reserved()

        torch_npu.npu.empty_virt_addr_cache()
        new_r = torch_npu.npu.memory_reserved()
        self.assertEqual(new_r, prev)
        self.assertEqual(torch_npu.npu.max_memory_reserved(), last_r)

        # test re-alloc after empty virtual address
        try:
            y = torch.empty((7500, 1024, 1024), device="npu")
            self.assertGreater(torch_npu.npu.memory_allocated(), prev)
        finally:
            if y is not None:
                del y
                self.assertEqual(torch_npu.npu.memory_allocated(), prev)
                torch_npu.npu.empty_virt_addr_cache()
                # empty unmapped physical handles with empty_cache()
                torch_npu.npu.empty_cache()
                self.assertEqual(torch_npu.npu.memory_reserved(), prev)

if __name__ == '__main__':
    run_tests()
