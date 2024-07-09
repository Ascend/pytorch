import unittest
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestRotaryMul(TestCase):
    def rotary_mul(self, x, r1, r2):
        x1, x2 = torch.chunk(x, 2, -1)
        x_new = torch.cat((-x2, x1), dim=-1)
        output = r1 * x + r2 * x_new
        return output

    def gen_data(self, shape, dtype):
        cpu_input = torch.rand(shape, dtype=dtype)
        npu_input = cpu_input.npu()
        return cpu_input, npu_input

    def cpu_to_exec(self, x, r1, r2):
        x.requires_grad = True
        r1.requires_grad = True
        r2.requires_grad = True
        out = self.rotary_mul(x, r1, r2)
        out.backward(torch.ones_like(out))
        x_grad = x.grad.numpy()
        r1_grad = r1.grad.numpy()
        r2_grad = r2.grad.numpy()
        return x_grad, r1_grad, r2_grad

    def npu_to_exec(self, x, r1, r2):
        x.requires_grad = True
        r1.requires_grad = True
        r2.requires_grad = True
        out = torch_npu.npu_rotary_mul(x, r1, r2)
        out.backward(torch.ones_like(out))
        x_grad = x.grad.detach().cpu().numpy()
        r1_grad = r1.grad.detach().cpu().numpy()
        r2_grad = r2.grad.detach().cpu().numpy()
        return x_grad, r1_grad, r2_grad

    @SupportedDevices(['Ascend910B'])
    def test_rotary_mul_backward(self):
        dtype_list = [torch.float32]
        shape_list = [
            [[2, 8192, 5, 128], [1, 8192, 1, 128], [1, 8192, 1, 128]],
            [[8192, 2, 5, 128], [8192, 1, 1, 128], [8192, 1, 1, 128]],
            [[2048, 4, 32, 64], [2048, 4, 1, 64], [2048, 4, 1, 64]],
        ]
        items = [
            [shape, dtype]
            for shape in shape_list
            for dtype in dtype_list
        ]
        for shape, dtype in items:
            cpu_x, npu_x = self.gen_data(shape[0], dtype)
            cpu_r1, npu_r1 = self.gen_data(shape[1], dtype)
            cpu_r2, npu_r2 = self.gen_data(shape[2], dtype)
            cpu_grad1, cpu_grad2, cpu_grad3 = self.cpu_to_exec(cpu_x, cpu_r1, cpu_r2)
            npu_grad1, npu_grad2, npu_grad3 = self.npu_to_exec(npu_x, npu_r1, npu_r2)
            self.assertRtolEqual(cpu_grad1, npu_grad1)
            self.assertRtolEqual(cpu_grad2, npu_grad2)
            self.assertRtolEqual(cpu_grad3, npu_grad3)


if __name__ == '__main__':
    run_tests()
