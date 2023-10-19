import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestNpuNativeFormatPrint(TestCase):
    def cpu_op_exec(self, inputs, split):
        cpu_output = inputs[split]
        cpu_output = cpu_output.numpy()
        return cpu_output

    def npu_op_exec(self, inputs, split):
        npu_output = inputs[split]
        npu_output = npu_output.cpu().numpy()
        return npu_output

    def test_npu_native_format_print(self):
        dtype_list = [np.float32]
        format_list = [3, 29, 30]
        shape_list = [[256], [8, 64], [2, 8, 64], [2, 8, 64, 128]]
        shape_format = [
            [d, i, j] for d in dtype_list for i in format_list for j in shape_list
        ]

        split_dict = {1: [1], 2: [0, 1], 3: [0, 1, 0], 4: [0, 1, 0, 0]}
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0:3], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input, split_dict[cpu_input.dim()])
            npu_output = self.npu_op_exec(npu_input, split_dict[npu_input.dim()])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
