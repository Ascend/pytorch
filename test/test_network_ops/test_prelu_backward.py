import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestPreluBackward(TestCase):
    @staticmethod
    def cpu_op_back_exec_ext(input1):
        is_float16 = input1.dtype == torch.float16
        if is_float16:
            input1 = input1.to(torch.float32)
        num_parameters = input1.shape[1] if input1.dim() > 1 else 1
        input1.requires_grad = True
        prelu = torch.nn.PReLU(num_parameters)
        weight = torch.ones([num_parameters], dtype=input1.dtype) * 0.25
        prelu.weight.data = weight.data
        output = prelu(input1)
        loss = output.sum()
        loss.backward(torch.ones_like(loss))
        input_grad = input1.grad.numpy()
        if is_float16:
            return input_grad.astype(np.float16)
        return input_grad

    @staticmethod
    def npu_op_back_exec_ext(input1):
        num_parameters = input1.shape[1] if input1.dim() > 1 else 1
        input1.requires_grad = True
        prelu = torch.nn.PReLU(num_parameters)
        weight = torch.ones([num_parameters], dtype=input1.dtype) * 0.25
        prelu.weight.data = weight.data.npu()
        output = prelu(input1)
        loss = output.sum()
        loss.backward(torch.ones_like(loss))
        input_grad = input1.grad.detach().cpu().numpy()
        return input_grad

    def test_PreluBackward_shape_format_fp32(self):
        shape_format = [
            [np.float32, 0, (17, 12, 38, 15)],
            [np.float32, 0, (1, 12, 38, 5)],
            [np.float32, 0, (124, 12, 38, 25)],
            [np.float32, 0, (4, 12, 38, 5)],
            [np.float32, 0, (10, 12, 38, 45)],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -2, 2)
            cpu_output = self.cpu_op_back_exec_ext(cpu_input)
            npu_output = self.npu_op_back_exec_ext(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_PreluBackward_shape_format_fp16(self):
        shape_format = [
            [np.float16, 0, (3, 5, 4)],
            [np.float16, 0, (32, 1, 1)],
            [np.float16, 0, (3, 224, 224)],
            [np.float16, 0, (5, 32, 112)],
            [np.float16, 0, (2, 672, 7)],
            [np.float16, 0, (6, 288, 14)],
            [np.float16, 0, (4, 58, 28)],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -2, 2)
            cpu_output = self.cpu_op_back_exec_ext(cpu_input)
            npu_output = self.npu_op_back_exec_ext(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
