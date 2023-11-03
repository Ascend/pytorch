import math
import torch
import numpy as np
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestUpsampleNearest1DBackward(TestCase):
    def cpu_op_exec(self, input1, grads, size):
        input1.requires_grad_(True)
        output = F.interpolate(input1, size, mode="nearest")
        output.backward(grads)
        gradcpu = input1.grad
        return output.detach().numpy(), gradcpu.detach().numpy()

    def cpu_op_scale_exec(self, input1, grads, scale):
        input1.requires_grad_(True)
        output = F.interpolate(input1, scale_factor=scale, mode="nearest")
        output.backward(grads)
        gradcpu = input1.grad
        return output.detach().numpy(), gradcpu.detach().numpy()

    def npu_op_exec(self, input1, grads, size):
        input1.requires_grad_(True)
        output = F.interpolate(input1, size, mode="nearest")
        output.backward(grads)
        gradnpu = input1.grad
        gradnpu = gradnpu.to("cpu")
        output = output.to("cpu")
        return output.detach().numpy(), gradnpu.detach().numpy()

    def npu_op_scale_exec(self, input1, grads, scale):
        input1.requires_grad_(True)
        output = F.interpolate(input1, scale_factor=scale, mode="nearest")
        output.backward(grads)
        gradnpu = input1.grad
        gradnpu = gradnpu.to("cpu")
        output = output.to("cpu")
        return output.detach().numpy(), gradnpu.detach().numpy()

    def test_upsample_nearest1d_backward_shape_format(self):
        test_cases1 = [
            [[np.float32, 3, (2, 2, 3)], [1, ]],
            [[np.float32, 0, (2, 1, 1)], [4, ]],
            [[np.float32, 0, (4, 1, 2)], [4, ]],
            [[np.float32, 0, (1, 1, 1)], [1, ]]
        ]
        for item in test_cases1:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            size = list(item[0][2])
            size[2] = item[1][0]

            grad_item = []
            grad_item.append(item[0][0])
            grad_item.append(item[0][1])
            grad_item.append(size)
            cpu_grads, npu_grads = create_common_tensor(grad_item, 0, 100)

            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            if cpu_grads.dtype == torch.float16:
                cpu_grads = cpu_grads.to(torch.float32)

            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input, cpu_grads, item[1])
            npu_output, npu_grad = self.npu_op_exec(npu_input, npu_grads, item[1])

            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_grad = cpu_grad.astype(npu_grad.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)

    def test_upsample_nearest1d_backward_shape_format_scale(self):
        test_cases2 = [
            [[np.float32, 3, (2, 2, 3)], 0.4],
            [[np.float32, 0, (2, 1, 1)], 4],
            [[np.float32, 0, (4, 1, 2)], 2],
            [[np.float32, 0, (1, 1, 1)], 1]
        ]
        for item in test_cases2:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)

            size = list(item[0][2])
            size[2] = item[1] * item[0][2][2]
            size[2] = math.floor(size[2])

            grad_item = []
            grad_item.append(item[0][0])
            grad_item.append(item[0][1])
            grad_item.append(size)
            cpu_grads, npu_grads = create_common_tensor(grad_item, 0, 100)

            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            if cpu_grads.dtype == torch.float16:
                cpu_grads = cpu_grads.to(torch.float32)

            cpu_output, cpu_grad = self.cpu_op_scale_exec(cpu_input, cpu_grads, item[1])
            npu_output, npu_grad = self.npu_op_scale_exec(npu_input, npu_grads, item[1])

            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_grad = cpu_grad.astype(npu_grad.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()
