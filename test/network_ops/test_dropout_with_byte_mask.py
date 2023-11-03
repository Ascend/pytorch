import torch
import numpy as np

import torch_npu

from torch_npu.contrib.module.npu_modules import DropoutWithByteMask
from torch_npu.contrib.function import npu_functional as F
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestDropoutWithByteMask(TestCase):
    def npu_op_exec(self, input1, prob):
        m = DropoutWithByteMask(p=prob).npu()
        out1 = m(input1)
        out2 = F.dropout_with_byte_mask(input1, p=prob)
        out3 = torch_npu.dropout_with_byte_mask(input1, p=prob, train=True)

    def test_DropoutWithByteMask(self):
        torch.manual_seed(5)
        items = [[np.float16, 2, (4, 4)], [np.float16, 0, (32, 384, 1024)]]
        for item in items:
            cpu_input, npu_input = create_common_tensor(item, 0, 1)
            # result is random,only check api can exec success!
            self.npu_op_exec(npu_input, prob=0.2)


if __name__ == "__main__":
    run_tests()
