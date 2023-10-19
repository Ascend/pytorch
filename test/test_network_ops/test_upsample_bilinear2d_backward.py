import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestUpsampleBilinear2dBackward(TestCase):
    def cpu_op_exec(self, inputs, shapes):
        inputs.requires_grad_(True)
        output = torch._C._nn.upsample_bilinear2d(inputs, shapes, True, 0, 0)
        output.backward(torch.ones_like(output))
        gradcpu = inputs.grad
        return output.detach().numpy(), gradcpu.detach().numpy()

    def npu_op_exec(self, inputs, shapes):
        inputs.requires_grad_(True)
        output = torch._C._nn.upsample_bilinear2d(inputs, shapes, True, 0, 0)
        inputback = torch.ones_like(output)
        output.backward(inputback)
        out = output.to("cpu")
        grad = inputs.grad
        grad = grad.to("cpu")
        return out.detach().numpy(), grad.detach().numpy()

    def cpu_op_scale_exec(self, inputs, scale_factor):
        inputs.requires_grad_(True)
        output = torch.nn.functional.interpolate(inputs, scale_factor=scale_factor, mode="bilinear")
        output.backward(torch.ones_like(output))
        gradcpu = inputs.grad
        return output.detach().numpy(), gradcpu.detach().numpy()

    def npu_op_scale_exec(self, inputs, scale_factor):
        inputs.requires_grad_(True)
        output = torch.nn.functional.interpolate(inputs, scale_factor=scale_factor, mode="bilinear")
        inputback = torch.ones_like(output)
        output.backward(inputback)
        out = output.to("cpu")
        grad = inputs.grad
        grad = grad.to("cpu")
        return out.detach().numpy(), grad.detach().numpy()

    def test_UpsampleBilinear2d_common_shape_format(self):
        shape_format = [
            [[np.float32, -1, (4, 3, 1, 5)], (2, 2)],
            [[np.float32, -1, (2, 3, 2, 1)], (3, 3)],
            [[np.float32, -1, (1, 4, 2, 2)], (4, 4)],
            [[np.float16, -1, (4, 10, 16, 14)], (5, 5)],
            [[np.float16, -1, (8, 8, 8, 8)], (1, 2)],
            [[np.float16, -1, (10, 4, 3, 2)], (2, 4)]
        ]
        for item in shape_format:
            cpu_inputs, npu_inputs = create_common_tensor(item[0], 1, 100)
            if cpu_inputs.dtype == torch.float16:
                cpu_inputs = cpu_inputs.to(torch.float32)
            cpu_output, cpu_grad = self.cpu_op_exec(cpu_inputs, item[1])
            npu_output, npu_grad = self.npu_op_exec(npu_inputs, item[1])
            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_grad = cpu_grad.astype(npu_grad.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)

    def test_UpsampleBilinear2d_common_scale_shape_format(self):
        shape_format = [
            [[np.float32, -1, (4, 3, 1, 5)], (2, 2)],
            [[np.float32, -1, (2, 3, 2, 1)], (3, 3)],
            [[np.float32, -1, (1, 4, 2, 2)], (4, 4)],
            [[np.float16, -1, (4, 10, 16, 14)], (5, 5)],
            [[np.float16, -1, (8, 8, 8, 8)], (1, 2)],
            [[np.float16, -1, (10, 4, 3, 2)], (2, 4)]
        ]
        for item in shape_format:
            cpu_inputs, npu_inputs = create_common_tensor(item[0], 1, 100)
            if cpu_inputs.dtype == torch.float16:
                cpu_inputs = cpu_inputs.to(torch.float32)
            cpu_output, cpu_grad = self.cpu_op_scale_exec(cpu_inputs, item[1])
            npu_output, npu_grad = self.npu_op_scale_exec(npu_inputs, item[1])
            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_grad = cpu_grad.astype(npu_grad.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()
