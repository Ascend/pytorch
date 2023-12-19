import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestGridSampler2dBackward(TestCase):
    def get_attrs(self):
        attrs = [
            [0, True],
            [1, True],
            [0, False],
            [1, False]
        ]
        return attrs

    def cpu_op_exec(self, input1, sample, pad_mode, align):
        input1.requires_grad = True
        sample.requires_grad = True
        out = torch.grid_sampler_2d(input1, sample, 0, pad_mode, align)
        out.backward(torch.ones_like(out))
        dx = input1.grad.numpy()
        dgrid = sample.grad.numpy()
        return dx, dgrid

    def npu_op_exec(self, input1, sample, pad_mode, align):
        input1.requires_grad = True
        sample.requires_grad = True
        out = torch.grid_sampler_2d(input1, sample, 0, pad_mode, align)
        out.backward(torch.ones_like(out))
        dx = input1.grad
        dgrid = sample.grad
        dx = dx.to("cpu").numpy()
        dgrid = dgrid.to("cpu").numpy()
        return dx, dgrid

    def test_grid_sampler_2d_backward_fp32(self, device="npu"):
        shape_list = [[100, 1, 28, 28], [100, 64, 32, 28]]
        shape_format = [
            [np.float32, -1, j] for j in shape_list
        ]
        sample_format = [np.float32, -1, [100, 1, 1, 2]]
        attrs = self.get_attrs()
        for item in shape_format:
            for attr in attrs:
                cpu_input, npu_input = create_common_tensor(item, 0, 100)
                cpu_sample, npu_sample = create_common_tensor(sample_format, -1, 1)
                cpu_output_dx, cpu_output_dgrid = self.cpu_op_exec(cpu_input, cpu_sample, *attr)
                npu_output_dx, npu_output_dgrid = self.npu_op_exec(npu_input, npu_sample, *attr)
                self.assertRtolEqual(cpu_output_dx, npu_output_dx)
                self.assertRtolEqual(cpu_output_dgrid, npu_output_dgrid)

    def test_grid_sampler_2d_backward_fp16(self, device="npu"):
        def cpu_op_fp16_exec(input1, sample, pad_mode, align):
            input1 = input1.to(torch.float32)
            sample = sample.to(torch.float32)
            input1.requires_grad = True
            sample.requires_grad = True
            out = torch.grid_sampler(input1, sample, 0, pad_mode, align)
            out.backward(torch.ones_like(out))
            dx = input1.grad
            dgrid = sample.grad
            dx = dx.numpy().astype(np.float16)
            dgrid = dgrid.numpy().astype(np.float16)
            return dx, dgrid

        shape_list = [[100, 1, 28, 28], [100, 64, 32, 28]]
        shape_format = [
            [np.float16, -1, j] for j in shape_list
        ]
        sample_format = [np.float16, -1, [100, 1, 1, 2]]
        attrs = self.get_attrs()
        for item in shape_format:
            for attr in attrs:
                cpu_input, npu_input = create_common_tensor(item, 0, 100)
                cpu_sample, npu_sample = create_common_tensor(sample_format, -1, 1)
                cpu_output_dx, cpu_output_dgrid = cpu_op_fp16_exec(cpu_input, cpu_sample, *attr)
                npu_output_dx, npu_output_dgrid = self.npu_op_exec(npu_input, npu_sample, *attr)
                self.assertRtolEqual(cpu_output_dx, npu_output_dx)
                self.assertRtolEqual(cpu_output_dgrid, npu_output_dgrid)


if __name__ == "__main__":
    np.random.seed(1234)
    run_tests()
