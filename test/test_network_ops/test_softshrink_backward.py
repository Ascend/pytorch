import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

cpu_input_grad = None
npu_input_grad = None


def cpu_input_grad_hook(grad):
    global cpu_input_grad
    cpu_input_grad = grad.numpy()


def npu_input_grad_hook(grad):
    global npu_input_grad
    npu_input_grad = grad.cpu().numpy()


class TestSoftShrinkBackward(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input_grad = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input_grad_data = torch.from_numpy(input_grad)
        npu_input_x = torch.from_numpy(input_x)
        return npu_input_grad_data, npu_input_x

    def cpu_op_exec(self, input_x, input_grad, lambd):
        input_x.requires_grad_(True)
        input_x.register_hook(cpu_input_grad_hook)
        m = torch.nn.Softshrink(lambd=lambd)
        output = m(input_x)
        output.backward(input_grad)

    def npu_op_exec(self, input_x, input_grad, lambd):
        input_x = input_x.to("npu")
        input_grad = input_grad.to("npu")
        input_x.requires_grad_(True)
        input_x.register_hook(npu_input_grad_hook)
        m = torch.nn.Softshrink(lambd=lambd).npu()
        output = m(input_x)
        output.backward(input_grad)

    def test_softshrink_3_3_float32(self):
        input_grad, input_x = self.generate_data(-1, 1, (3, 3), np.float32)
        self.cpu_op_exec(input_x, input_grad, 0.5)
        self.npu_op_exec(input_x, input_grad, 0.5)
        self.assertRtolEqual(cpu_input_grad, npu_input_grad)

    def test_softshrink_100_100_float32(self):
        input_grad, input_x = self.generate_data(-1, 1, (100, 100), np.float32)
        self.cpu_op_exec(input_x, input_grad, 0.5)
        self.npu_op_exec(input_x, input_grad, 0.5)
        self.assertRtolEqual(cpu_input_grad, npu_input_grad)

    def test_softshrink_10_10_10_10_float32(self):
        input_grad, input_x = self.generate_data(-1, 1, (10, 10, 10, 10), np.float32)
        self.cpu_op_exec(input_x, input_grad, 0.5)
        self.npu_op_exec(input_x, input_grad, 0.5)
        self.assertRtolEqual(cpu_input_grad, npu_input_grad)


if __name__ == "__main__":
    run_tests()
