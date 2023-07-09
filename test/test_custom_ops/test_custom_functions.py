import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestCustomFunctions(TestCase):
    def benchmark_backward_exec(self, x, loss):
        x.requires_grad_(True)
        y = x.to(torch.float16)
        y.copy_(x)
        y.backward(loss, retain_graph=True)
        return x.grad.numpy()

    def custom_function_backward_exec(self, x, loss):
        x.requires_grad_(True)
        y = x.to(torch.float16)
        y.copy_(x)
        y.backward(loss, retain_graph=True)
        dx = x.grad.cpu()
        return dx.numpy()

    def test_backward_exec(self):
        shape_format = [
            [np.float32, 0, (5, 3, 6, 4)],
        ]

        for item in shape_format:
            cpu_x, npu_x = create_common_tensor(item, 0, 100)
            cpu_loss, npu_loss = create_common_tensor(item, 0, 100)
            custom_output = self.benchmark_backward_exec(cpu_x, cpu_loss)
            npu_output = self.custom_function_backward_exec(npu_x, npu_loss)
            self.assertRtolEqual(custom_output, npu_output)


if __name__ == "__main__":
    run_tests()
