import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestIsNonzero(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.is_nonzero(input1)
        return output

    def npu_op_exec(self, input1):
        output = torch.is_nonzero(input1)
        return output

    def test_isnonzero_shape_format(self, device="npu"):
        dtype_list = [np.float16, np.float32, np.int32, np.bool_]
        format_list = [0]
        shape_list = [[1], [1, 1, 1], [1, 1, 1, 1]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output == npu_output


if __name__ == "__main__":
    run_tests()
