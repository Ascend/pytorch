import torch
import torch.nn as nn

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestErrorCode(TestCase):

    def test_set_per_process_memory_fraction(self):
        with self.assertRaisesRegex(TypeError, "ERR00002 PTA invalid type"):
            torch_npu.npu.set_per_process_memory_fraction(1)
    
    def test_div(self):
        x1 = torch.tensor(1).npu()
        x2 = torch.tensor(1).npu()
        with self.assertRaisesRegex(RuntimeError, "ERR01001 OPS invalid parameter"):
            torch.div(x1, x2, rounding_mode="test")
    

if __name__ == "__main__":
    run_tests()
