import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAbs(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.abs(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.abs(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_abs_shape_format_fp16(self, device="npu"):
        format_list = [0, 3]
        shape_list = [[5], [5, 10], [1, 3, 2], [52, 15, 15, 20]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -10, 10)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_abs_shape_format_fp32(self, device="npu"):
        format_list = [0, 3]
        shape_list = [[5], [5, 10], [1, 3, 2], [52, 15, 15, 20]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -10, 10)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
