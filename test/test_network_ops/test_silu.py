import torch
import torch.nn.functional as F
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSilu(TestCase):

    def cpu_op_exec(self, input1):
        output = F.silu(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = F.silu(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_inplace(self, input1):
        torch_npu.npu_silu_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def test_silu_shape_format_fp16(self):
        format_list = [0]
        shape_list = [1, (64, 10), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_silu_shape_format_fp32(self):
        format_list = [0, 3, 4, 29]
        shape_list = [1, (32, 32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_silu_inplace_shape_format_fp16(self):
        format_list = [0]
        shape_list = [1, (64, 10), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec_inplace(npu_input)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_silu_inplace_shape_format_fp32(self):
        format_list = [0, 3, 4, 29]
        shape_list = [1, (32, 32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec_inplace(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
