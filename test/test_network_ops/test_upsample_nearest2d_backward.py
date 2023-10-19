import torch
import numpy as np
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestUpsampleNearest2DBackward(TestCase):
    def cpu_op_exec(self, input1, size):
        input1.requires_grad_(True)
        output = F.interpolate(input1, size, mode="nearest")
        output.backward(torch.ones_like(output))
        return output.detach().numpy(), input1.grad.numpy()

    def npu_op_exec(self, input1, size):
        input1.requires_grad_(True)
        output = F.interpolate(input1, size, mode="nearest")
        inputback = torch.ones_like(output)
        output.backward(inputback)
        out = output.to("cpu")
        grad = input1.grad
        grad = grad.to("cpu")
        return out.detach().numpy(), grad.detach().numpy()

    def test_upsample_bilinear2d_shape_format(self):
        shape_format = [
            [[np.float32, 0, (2, 3, 4, 4)], [2, 2]],
            [[np.float16, 0, (2, 3, 4, 4)], [2, 2]],
            [[np.float32, 0, (5, 3, 6, 4)], [10, 10]],
            [[np.float16, 0, (5, 3, 6, 4)], [10, 10]],
            [[np.float32, 0, (2, 3, 2, 4)], [10, 10]],
            [[np.float16, -1, (2, 3, 2, 3)], [10, 10]]
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input, item[1])
            npu_output, npu_grad = self.npu_op_exec(npu_input, item[1])

            cpu_grad = cpu_grad.astype(npu_grad.dtype)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()
