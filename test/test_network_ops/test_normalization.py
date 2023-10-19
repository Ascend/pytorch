import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestNormalization(TestCase):
    def op_exec(self, npu_flag, input1, dim):
        m = torch.nn.BatchNorm2d(dim)
        if npu_flag:
            m = m.to("npu")
        input_new = m(input1)
        if npu_flag:
            input_new = input_new.to("cpu")
        input_new = input_new.detach().numpy()
        input1.requires_grad_(True)
        w = torch.ones_like(input1)
        if npu_flag:
            w = w.to("npu")
        tmp = m(input1)
        tmp.backward(w)
        output = input1.grad
        if npu_flag:
            output = output.to("cpu")
        output = output.detach().numpy()
        return output, input_new

    def test_batchnorm_shape_format_fp16(self, device='npu'):
        format_list = [0]
        shape_list = [[256, 672, 7, 7], [1024, 58, 28, 28]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output, cpu_input = self.op_exec(0, cpu_input1, item[2][1])
            npu_output, npu_input = self.op_exec(1, npu_input1, item[2][1])
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            cpu_input = cpu_input.astype(npu_input.dtype)
            self.assertRtolEqual(cpu_input, npu_input)

    def test_batchnorm_shape_format_fp32(self, device='npu'):
        format_list = [0]
        shape_list = [(256, 32, 112, 112)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output, cpu_input = self.op_exec(0, cpu_input1, item[2][1])
            npu_output, npu_input = self.op_exec(1, npu_input1, item[2][1])
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            cpu_input = cpu_input.astype(npu_input.dtype)
            self.assertRtolEqual(cpu_input, npu_input)


if __name__ == "__main__":
    run_tests()
