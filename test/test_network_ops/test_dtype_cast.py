import sys

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestDtypeCast(TestCase):
    def cpu_op_exec(self, input1, dst_dtype):
        input1.requires_grad = True
        output = input1.to(dst_dtype)
        output.backward(torch.ones_like(output))
        input1_grad = input1.grad
        return output.detach(), input1_grad

    def npu_op_exec(self, input1, dst_dtype):
        input1.requires_grad = True
        output = torch_npu.npu_dtype_cast(input1, dst_dtype)
        output.backward(torch.ones_like(output))
        input1_grad = input1.grad
        return output.cpu().detach(), input1_grad.cpu()

    def test_dtype_base(self):
        a = torch.rand(2).npu()
        a.requires_grad = True
        b = torch_npu.npu_dtype_cast(a, torch.half)
        if b.requires_grad is not True:
            raise RuntimeError("the output.requires_grad of npu_dtype_cast should be same with input, but not so.")

    def test_dtype_cast_shape_format(self):
        shape_format = [
            [np.float32, 0, 1],
            [np.float32, 0, (64, 10)],
            [np.float32, 4, (32, 1, 3, 3)],
            [np.float32, 29, (10, 128)]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_output, cpu_input_grad = self.cpu_op_exec(cpu_input, torch.half)
            npu_output, npu_input_grad = self.npu_op_exec(npu_input, torch.half)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)

    def test_dtype_cast_shape_format_fp16(self):
        shape_format = [
            [np.float16, 0, 1],
            [np.float16, 0, (64, 10)],
            [np.float16, 4, (32, 1, 3, 3)],
            [np.float16, 29, (10, 128)]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_output, cpu_input_grad = self.cpu_op_exec(cpu_input, torch.float)
            npu_output, npu_input_grad = self.npu_op_exec(npu_input, torch.float)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)


if __name__ == "__main__":
    run_tests()
