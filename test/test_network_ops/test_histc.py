import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestHistc(TestCase):
    def test_histc_float32(self):
        dtype_list = [np.float32]
        format_list = [2]
        shape_list = [[5, 5],[4, 5, 6]]
        bins_list = [5, 10, 100]
        shape_format = [
            [i, j, k, l] for i in dtype_list for j in format_list for k in shape_list for l in bins_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[:-1], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, item[-1])
            npu_output = self.npu_op_exec(npu_input, item[-1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_histc_float16(self):
        dtype_list = [np.float16]
        format_list = [2]
        shape_list = [[5, 5],[4, 5, 6]]
        bins_list = [5, 10, 100]
        shape_format = [
            [i, j, k, l] for i in dtype_list for j in format_list for k in shape_list for l in bins_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[:-1], 0, 100)
            cpu_input = cpu_input.float()
            cpu_output = self.cpu_op_exec(cpu_input, item[-1])
            npu_output = self.npu_op_exec(npu_input, item[-1])
            self.assertRtolEqual(cpu_output, npu_output)

    def cpu_op_exec(self, input1, bins=100):
        output = torch.histc(input1, bins)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, bins=100):
        input1 = input1.to("npu")
        output = torch.histc(input1, bins)
        output = output.to("cpu")
        output = output.numpy()
        return output


if __name__ == "__main__":
    run_tests()
