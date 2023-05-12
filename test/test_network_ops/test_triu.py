import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestTriu(TestCase):

    def cpu_op_exec(self, input1):
        output = torch.triu(input1, 1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.triu(input1, 1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_exec(self, input1):
        output = input1.triu_(1)
        output = output.numpy()
        return output

    def npu_op_inplace_exec(self, input1):
        output = input1.triu_(1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_triu(self):
        dtype_list = [np.float32, np.float16]
        format_list = [0, 3]
        shape_list = [[5, 5]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_triu_inplace(self):
        dtype_list = [np.float32, np.float16]
        format_list = [0, 3]
        shape_list = [[5, 5]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_inplace_exec(cpu_input)
            npu_output = self.npu_op_inplace_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
