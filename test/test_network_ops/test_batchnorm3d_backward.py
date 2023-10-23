import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestBn2d(TestCase):
    def cpu_op_exec(self, input1, dim):
        input1.requires_grad_(True)
        m = torch.nn.BatchNorm3d(dim)
        input_cpu = m(input1)
        input_cpu = input_cpu.detach().numpy()
        w = torch.ones_like(input1)
        tmp = m(input1)
        tmp.backward(w)
        output = input1.grad
        output = output.detach().numpy()
        return output, input_cpu

    def npu_op_exec_new(self, input1, dim):
        w = torch.ones_like(input1)
        w = w.to("npu")
        m = torch.nn.BatchNorm3d(dim)
        m = m.to("npu")
        input_npu = m(input1)
        input_npu = input_npu.to("cpu")
        input_npu = input_npu.detach().numpy()
        input1.requires_grad_(True)
        tmp = m(input1)
        tmp.backward(w)
        output = input1.grad.to("cpu")
        output = output.detach().numpy()
        return output, input_npu

    def test_batchnorm3d_shape_format_fp16(self):
        format_list = [30]
        shape_list = [[256, 164, 7, 7, 7], [148, 16, 28, 28, 28]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output, cpu_input = self.cpu_op_exec(cpu_input1, item[2][1])
            npu_output, npu_input = self.npu_op_exec_new(npu_input1, item[2][1])
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            cpu_input = cpu_input.astype(npu_input.dtype)
            self.assertRtolEqual(cpu_input, npu_input)

    def test_batchnorm3d_shape_format_fp32(self):
        format_list = [30]
        shape_list = [(256, 32, 7, 7, 7)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output, cpu_input = self.cpu_op_exec(cpu_input1, item[2][1])
            npu_output, npu_input = self.npu_op_exec_new(npu_input1, item[2][1])
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            cpu_input = cpu_input.astype(npu_input.dtype)
            self.assertRtolEqual(cpu_input, npu_input)


if __name__ == "__main__":
    run_tests()
