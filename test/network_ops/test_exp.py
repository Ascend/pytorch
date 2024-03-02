import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestExp(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.exp(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.exp(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_tensor_op_exec(self, input1):
        output = input1.exp()
        output = output.numpy()
        return output

    def npu_tensor_op_exec(self, input1):
        output = input1.exp()
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_tensor_inplace_op_exec(self, input1):
        input1.exp_()
        input1 = input1.numpy()
        return input1

    def npu_tensor_inplace_op_exec(self, input1):
        input1.exp_()
        input1 = input1.to("cpu")
        input1 = input1.numpy()
        return input1

    def test_exp_shape_format_fp16(self):
        format_list = [0, 3]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_exp_shape_format_fp32(self):
        format_list = [0, 3]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_tensor_exp_shape_format_fp16(self):
        format_list = [0, 3]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_tensor_op_exec(cpu_input)
            npu_output = self.npu_tensor_op_exec(npu_input)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_tensor_exp_shape_format_fp32(self):
        format_list = [0, 3]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_output = self.cpu_tensor_op_exec(cpu_input)
            npu_output = self.npu_tensor_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_tensor_inplace_exp_shape_format_fp16(self):
        format_list = [0, 3]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_tensor_inplace_op_exec(cpu_input)
            npu_output = self.npu_tensor_inplace_op_exec(npu_input)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_tensor_inplace_exp_shape_format_fp32(self):
        format_list = [0, 3]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_output = self.cpu_tensor_inplace_op_exec(cpu_input)
            npu_output = self.npu_tensor_inplace_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

if __name__ == "__main__":
    run_tests()
