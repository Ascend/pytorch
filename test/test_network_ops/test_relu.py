import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestRelu(TestCase):
    def cpu_op_back_exec(self, input1):
        w = torch.ones_like(input1)
        input1.requires_grad_(True)
        output = torch.relu(input1)
        output.backward(w)
        res = input1.grad
        res = res.numpy()
        return output.detach().numpy(), res

    def npu_op_back_exec(self, input1):
        w = torch.ones_like(input1)
        input1.requires_grad_(True)
        output = torch.relu(input1)
        output.backward(w)
        output = output.to("cpu")
        res = input1.grad.to("cpu")
        res = res.numpy()
        return output.detach().numpy(), res

    def cpu_inp_op_exec(self, input1):
        output = torch.relu_(input1)
        output = output.numpy()
        return output

    def npu_inp_op_exec(self, input1):
        output = torch.relu_(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_relu_shape_format_fp32(self, device='npu'):
        format_list = [-1]
        shape_list = [(1000, 1280), (32, 3, 3), (1024, 464, 7, 7)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output, cpu_res = self.cpu_op_back_exec(cpu_input1)
            npu_output, npu_res = self.npu_op_back_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_res = cpu_res.astype(npu_res.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_res, npu_res)

    def test_relu_shape_format_fp16(self, device='npu'):
        format_list = [-1]
        shape_list = [(1000, 1280), (32, 3, 3), (1024, 464, 7, 7)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output, cpu_res = self.cpu_op_back_exec(cpu_input1)
            npu_output, npu_res = self.npu_op_back_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_res = cpu_res.astype(npu_res.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_res, npu_res)

    def test_relu_shape_format_fp16_inp(self, device='npu'):
        format_list = [-1]
        shape_list = [(1000, 1280), (32, 3, 3), (1024, 464, 7, 7)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_inp_op_exec(cpu_input1)
            npu_output = self.npu_inp_op_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
