import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestCummaxminBackward(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input = torch.from_numpy(input_x)
        return npu_input

    def generate_dimname_data(self, min_d, max_d, shape, dtype):
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input = torch.from_numpy(input_x)
        npu_input.names = ['N', 'C', 'H', 'W']
        return npu_input

    def cpu_cummax_op_exec(self, input_x, dim):
        input_x.requires_grad = True
        output, indices = torch.cummax(input_x, dim)
        output.backward(torch.ones_like(output))
        output_grad = input_x.grad
        output_grad = output_grad.detach().numpy()
        output = output.detach().numpy()
        return output, output_grad

    def npu_cumax_op_exec(self, input_x, dim):
        input_x.requires_grad = True
        output, indices = torch.cummax(input_x, dim)
        output.backward(torch.ones_like(output))
        output = output.to("cpu")
        output_grad = input_x.grad
        output_grad = output_grad.to("cpu")
        output_grad = output_grad.detach().numpy()
        output = output.detach().numpy()
        return output, output_grad

    def cpu_cummin_op_exec(self, input_x, dim):
        input_x.requires_grad = True
        output, indices = torch.cummin(input_x, dim)
        output.backward(torch.ones_like(output))
        output_grad = input_x.grad
        output_grad = output_grad.detach().numpy()
        output = output.detach().numpy()
        return output, output_grad

    def npu_cummin_op_exec(self, input_x, dim):
        input_x.requires_grad = True
        output, indices = torch.cummin(input_x, dim)
        output.backward(torch.ones_like(output))
        output = output.to("cpu")
        output_grad = input_x.grad
        output_grad = output_grad.to("cpu")
        output_grad = output_grad.detach().numpy()
        output = output.detach().numpy()
        return output, output_grad

    def test_cummax_backward_dim2_0_float32(self):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        input_x2 = input_x1.clone().npu()
        cpu_output, cpuoutput_grad = self.cpu_cummax_op_exec(input_x1, 1)
        npu_output, npuoutput_grad = self.npu_cumax_op_exec(input_x2, 1)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpuoutput_grad, npuoutput_grad)

    def test_cummax_backward_dim3_0_float32(self):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3), np.float32)
        input_x2 = input_x1.clone().npu()
        cpu_output, cpuoutput_grad = self.cpu_cummax_op_exec(input_x1, 0)
        npu_output, npuoutput_grad = self.npu_cumax_op_exec(input_x2, 0)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpuoutput_grad, npuoutput_grad)

    def test_cummax_backward_dim6_4_float32(self):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3, 3, 3), np.float32)
        input_x2 = input_x1.clone().npu()
        cpu_output, cpuoutput_grad = self.cpu_cummax_op_exec(input_x1, 4)
        npu_output, npuoutput_grad = self.npu_cumax_op_exec(input_x2, 4)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpuoutput_grad, npuoutput_grad)

    def test_cummax_backward_dim6_5_float32(self):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3, 3, 3), np.float32)
        input_x2 = input_x1.clone().npu()
        cpu_output, cpuoutput_grad = self.cpu_cummax_op_exec(input_x1, 5)
        npu_output, npuoutput_grad = self.npu_cumax_op_exec(input_x2, 5)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpuoutput_grad, npuoutput_grad)

    def test_cummax_backward_10dim6_2_float32(self):
        input_x1 = self.generate_data(-1, 1, (10, 10, 10, 10, 10, 10), np.float32)
        input_x2 = input_x1.clone().npu()
        cpu_output, cpuoutput_grad = self.cpu_cummax_op_exec(input_x1, 2)
        npu_output, npuoutput_grad = self.npu_cumax_op_exec(input_x2, 2)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpuoutput_grad, npuoutput_grad)

    def test_cummax_backward_dim4_H_float32_dimname(self):
        input_x1 = self.generate_dimname_data(-1, 1, (3, 3, 3, 3), np.float32)
        input_x2 = input_x1.clone().npu()
        cpu_output, cpuoutput_grad = self.cpu_cummax_op_exec(input_x1, 'H')
        npu_output, npuoutput_grad = self.npu_cumax_op_exec(input_x2, 'H')
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpuoutput_grad, npuoutput_grad)

    def test_cummin_backward_dim2_0_float32(self):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        input_x2 = input_x1.clone().npu()
        cpu_output, cpu_argmin = self.cpu_cummin_op_exec(input_x1, 1)
        npu_output, npu_argmin = self.npu_cummin_op_exec(input_x2, 1)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_argmin, npu_argmin)

    def test_cummin_backward_dim6_4_float32(self):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3, 3, 3), np.float32)
        input_x2 = input_x1.clone().npu()
        cpu_output, cpu_argmin = self.cpu_cummin_op_exec(input_x1, 4)
        npu_output, npu_argmin = self.npu_cummin_op_exec(input_x2, 4)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_argmin, npu_argmin)

    def test_cummin_backward_dim6_5_float32(self):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3, 3, 3), np.float32)
        input_x2 = input_x1.clone().npu()
        cpu_output, cpu_argmin = self.cpu_cummin_op_exec(input_x1, 5)
        npu_output, npu_argmin = self.npu_cummin_op_exec(input_x2, 5)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_argmin, npu_argmin)

    def test_cummin_backward_10dim6_2_float32(self):
        input_x1 = self.generate_data(-1, 1, (10, 10, 10, 10, 10, 10), np.float32)
        input_x2 = input_x1.clone().npu()
        cpu_output, cpu_argmin = self.cpu_cummin_op_exec(input_x1, 2)
        npu_output, npu_argmin = self.npu_cummin_op_exec(input_x2, 2)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_argmin, npu_argmin)

    def test_cummin_backward_dim4_H_float32_dimname(self):
        input_x1 = self.generate_dimname_data(-1, 1, (3, 3, 3, 3), np.float32)
        input_x2 = input_x1.clone().npu()
        cpu_output, cpu_argmin = self.cpu_cummin_op_exec(input_x1, 'H')
        npu_output, npu_argmin = self.npu_cummin_op_exec(input_x2, 'H')
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_argmin, npu_argmin)


if __name__ == "__main__":
    run_tests()
