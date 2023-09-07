import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestIm2col(TestCase):

    def cpu_op_exec(self, input1, ksizes, rates, padding, strides):
        output = torch._C._nn.im2col(input1, ksizes, rates, padding, strides)
        return output

    def npu_op_exec(self, input1, ksizes, rates, padding, strides):
        output = torch._C._nn.im2col(input1, ksizes, rates, padding, strides)
        output = output.cpu()
        return output

    def test_im2col_float16_shape_format(self):
        shape_format = [
            [np.float16, -1, (1, 3, 3, 3)],
            [np.float16, -1, (2, 16, 4, 6)],
            [np.float16, -1, (2, 16, 4)],
            [np.float16, -1, (256, 16, 3)],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1.float(), (1, 1), (1, 1), (0, 0), (1, 1)).half()
            npu_output = self.npu_op_exec(npu_input1, (1, 1), (1, 1), (0, 0), (1, 1))
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
