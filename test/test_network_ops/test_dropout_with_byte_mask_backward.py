import torch
import numpy as np

import torch_npu

from torch_npu.contrib.module.npu_modules import DropoutWithByteMask
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestDropoutWithByteMaskBackward(TestCase):
    def npu_op_exec(self, input1, prob):
        input1.requires_grad = True
        m = DropoutWithByteMask(p=prob).npu()
        output = m(input1)
        output.backward(torch.ones_like(output))

    def test_DropoutWithByteMaskBackward(self):
        torch.manual_seed(5)
        items = [[np.float16, 2, (4, 4)], [np.float16, 0, (32, 384, 1024)]]
        for item in items:
            cpu_input, npu_input = create_common_tensor(item, 0, 1)
            # result is random,only check api can exec success!
            self.npu_op_exec(npu_input, prob=0.2)


if __name__ == "__main__":
    run_tests()
