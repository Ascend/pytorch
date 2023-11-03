import torch
import torch.nn as nn
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestConvTranspose2dBackward(TestCase):

    def cpu_op_exec(self, input1, weight, bias):
        input1.requires_grad = True
        weight.requires_grad = True
        bias.requires_grad = True

        res_forward = nn.functional.conv_transpose2d(input1, weight, padding=1, bias=bias)
        grads = torch.ones_like(res_forward)
        res_forward.backward(grads, retain_graph=True)
        input_grad = input1.grad
        weight_grad = weight.grad
        return res_forward, input_grad, weight_grad

    def npu_op_exec(self, input1, weight, bias):
        input1.requires_grad = True
        weight.requires_grad = True
        bias.requires_grad = True

        res_forward = nn.functional.conv_transpose2d(input1, weight, padding=1, bias=bias)
        grads = torch.ones_like(res_forward).npu()
        res_forward.backward(grads, retain_graph=True)
        res_forward = res_forward.cpu()
        input_grad = input1.grad.cpu()
        weight_grad = weight.grad.cpu()
        return res_forward, input_grad, weight_grad

    def conv_transpose2d_backward_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 2)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -2, 2)
            cpu_bias, npu_bias = create_common_tensor((item[0][0], 0, 4), -100, 100)
            if item[0][0] == np.float16:
                cpu_input1 = cpu_input1.float()
                cpu_input2 = cpu_input2.float()
                cpu_bias = cpu_bias.float()
            else:
                # Synchronous accuracy loss: The operator converts fp32 to fp16 for calculation.
                cpu_input1 = cpu_input1.half().float()
                cpu_input2 = cpu_input2.half().float()
                cpu_bias = cpu_bias.half().float()
            cpu_output, cpu_input_grad, cpu_weight_grad = self.cpu_op_exec(cpu_input1, cpu_input2, bias=cpu_bias)
            npu_output, npu_input_grad, npu_weight_grad = self.npu_op_exec(npu_input1, npu_input2, bias=npu_bias)
            if item[0][0] == np.float16:
                cpu_output = cpu_output.half()
                cpu_input_grad = cpu_input_grad.half()
                cpu_weight_grad = cpu_weight_grad.half()

            self.assertRtolEqual(cpu_output.detach(), npu_output.detach(), prec=1e-3)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad, prec=1e-3)
            self.assertRtolEqual(cpu_weight_grad, npu_weight_grad, prec=1e-3)

    def test_conv_transpose2d_backward_shape_format(self):
        shape_format = [
            [[np.float16, 0, [1, 4, 5, 5]], [np.float16, 0, [4, 4, 3, 3]]],
            [[np.float32, 0, [1, 4, 5, 5]], [np.float32, 0, [4, 4, 3, 3]]]
        ]
        self.conv_transpose2d_backward_result(shape_format)

    def test_conv_transpose2d_backward_allow_hf32(self):
        torch.npu.conv.allow_hf32 = True
        shape_format = [
            [[np.float16, 0, [1, 4, 5, 5]], [np.float16, 0, [4, 4, 3, 3]]]
        ]
        self.conv_transpose2d_backward_result(shape_format)
        torch.npu.conv.allow_hf32 = False


if __name__ == "__main__":
    np.random.seed(1234)
    run_tests()
