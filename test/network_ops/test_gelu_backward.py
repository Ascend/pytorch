import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestGeluBackward(TestCase):
    def generate_single_data(self, min_val, max_val, shape, dtype):
        input1 = np.random.uniform(min_val, max_val, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input1):
        input1.requires_grad_(True)
        output = torch.nn.functional.gelu(input1)
        z = output.sum()
        z.backward()
        res = input1.grad
        return res.detach().numpy()

    def npu_op_exec(self, input1):
        input1 = input1.to("npu")
        input1.requires_grad = True
        output = torch.nn.functional.gelu(input1)
        z = output.sum()
        z.backward()
        res = input1.grad.to("cpu")
        return res.detach().numpy()

    def test_gelu_backward_float32_1(self, device="npu"):
        input1 = self.generate_single_data(0, 100, (4, 3, 1, 1), np.float32)
        cpu_input1 = copy.deepcopy(input1)
        cpu_output = self.cpu_op_exec(cpu_input1)
        npu_output = self.npu_op_exec(input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_gelu_backward_float32_2(self, device="npu"):
        input1 = self.generate_single_data(0, 100, (15, 3, 1), np.float32)
        cpu_input1 = copy.deepcopy(input1)
        cpu_output = self.cpu_op_exec(cpu_input1)
        npu_output = self.npu_op_exec(input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_gelu_backward_float32_3(self, device="npu"):
        input1 = self.generate_single_data(0, 100, (4, 4), np.float32)
        cpu_input1 = copy.deepcopy(input1)
        cpu_output = self.cpu_op_exec(cpu_input1)
        npu_output = self.npu_op_exec(input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_gelu_backward_float16(self, device="npu"):
        input1 = self.generate_single_data(0, 100, (5, 10, 100), np.float16)
        cpu_input1 = input1.to(torch.float32)
        cpu_output = self.cpu_op_exec(cpu_input1)
        cpu_output = cpu_output.astype(np.float16)
        npu_output = self.npu_op_exec(input1)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    np.random.seed(20)
    run_tests()
