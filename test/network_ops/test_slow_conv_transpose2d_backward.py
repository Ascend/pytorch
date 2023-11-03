import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSlowConvTranspose2dBackward(TestCase):
    weight_grad = []
    input_grad = []
    bias_grad = []

    def get_weight_grad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def get_input_grad(self, grad):
        self.input_grad.append(grad.to("cpu"))

    def get_bias_grad(self, grad):
        self.bias_grad.append(grad.to("cpu"))

    def cpu_op_exec(self, input1, weight, bias, kernel_size):
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.get_input_grad(grad))
        weight.requires_grad = True
        weight.register_hook(lambda grad: self.get_weight_grad(grad))
        bias.requires_grad = True
        bias.register_hook(lambda grad: self.get_bias_grad(grad))

        res_forward = torch._C._nn.slow_conv_transpose2d(input1, weight, kernel_size=kernel_size, bias=bias)
        grads = torch.ones_like(res_forward).float()
        res_forward.backward(grads)
        res_forward = res_forward.detach().numpy()
        return res_forward

    def npu_op_exec(self, input1, weight, bias, kernel_size):
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.get_input_grad(grad))
        weight.requires_grad = True
        weight.register_hook(lambda grad: self.get_weight_grad(grad))
        bias.requires_grad = True
        bias.register_hook(lambda grad: self.get_bias_grad(grad))
        res_forward = torch._C._nn.slow_conv_transpose2d(input1, weight, kernel_size=kernel_size, bias=bias)
        grads = torch.ones_like(res_forward).float()
        grads = grads.to("npu")
        res_forward.backward(grads)
        res_forward = res_forward.to("cpu")
        res_forward = res_forward.detach().numpy()
        return res_forward

    def test_slow_conv_transpose2d_backward_shape_format_fp16(self):
        shape_format = [
            [[np.float16, 0, (1, 12, 20, 20)], [np.float16, 0, (12, 12, 3, 3)], [np.float16, 0, 12]],
            [[np.float16, 0, (1, 4, 5, 5)], [np.float16, 0, (4, 4, 3, 3)], [np.float16, 0, 4]]
        ]
        for item in shape_format:
            # Note:when bias is True, the value range is both pos and neg which will be not enough precision!
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 10)
            cpu_weight, npu_weight = create_common_tensor(item[1], 0, 10)
            cpu_bias, npu_bias = create_common_tensor(item[2], 0, 10)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_weight = cpu_weight.to(torch.float32)
            cpu_bias = cpu_bias.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_weight, cpu_bias, item[1][-1][-2:])
            npu_output = self.npu_op_exec(npu_input1, npu_weight, npu_bias, item[1][-1][-2:])

            cpu_output = cpu_output.astype(np.float16)
            npu_output = npu_output.astype(np.float16)
            self.input_grad[0] = self.input_grad[0].to(torch.float16)
            self.input_grad[1] = self.input_grad[1].to(torch.float16)
            self.weight_grad[0] = self.weight_grad[0].to(torch.float16)
            self.weight_grad[1] = self.weight_grad[1].to(torch.float16)
            self.bias_grad[0] = self.bias_grad[0].to(torch.float16)
            self.bias_grad[1] = self.bias_grad[1].to(torch.float16)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(self.input_grad[0], self.input_grad[1])
            self.assertRtolEqual(self.weight_grad[0], self.weight_grad[1])
            self.assertRtolEqual(self.bias_grad[0], self.bias_grad[1])

    def test_slow_conv_transpose2d_backward_shape_format_fp32(self):
        shape_format = [
            [[np.float32, 0, (1, 4, 5, 5)], [np.float32, 0, (4, 4, 3, 3)], [np.float32, 0, 4]],
            [[np.float32, 0, (1, 12, 20, 20)], [np.float32, 0, (12, 12, 3, 3)], [np.float32, 0, 12]],
        ]
        for item in shape_format:
            # Note:when bias is True, the value range is both pos and neg which will be not enough precision!
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 10)
            cpu_weight, npu_weight = create_common_tensor(item[1], 0, 10)
            cpu_bias, npu_bias = create_common_tensor(item[2], 0, 10)

            cpu_output = self.cpu_op_exec(cpu_input1, cpu_weight, cpu_bias, item[1][-1][-2:])
            npu_output = self.npu_op_exec(npu_input1, npu_weight, npu_bias, item[1][-1][-2:])

            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-1)
            self.assertRtolEqual(self.input_grad[0], self.input_grad[1], prec=1.e-1)
            self.assertRtolEqual(self.weight_grad[0], self.weight_grad[1], prec=1.e-1)
            self.assertRtolEqual(self.bias_grad[0], self.bias_grad[1], prec=1.e-1)


if __name__ == "__main__":
    run_tests()
