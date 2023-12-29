import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestGelu(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input1):
        output = torch.nn.functional.gelu(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        input1_npu = input1.to('npu')
        output = torch.nn.functional.gelu(input1_npu)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_fp16(self, input1):
        input1 = input1.to(torch.float32)
        output = torch.nn.functional.gelu(input1)
        output = output.numpy()
        output = output.astype(np.float16)
        return output

    def npu_op_exec_fp16(self, input1):
        input1 = input1.to(torch.float32).to('npu')
        output = torch.nn.functional.gelu(input1)
        output = output.to("cpu")
        output = output.numpy().astype(np.float16)
        return output

    def test_gelu_float32_1(self, device="npu"):
        input1 = self.generate_data(0, 100, (4, 3), np.float32)
        cpu_input1 = copy.deepcopy(input1)
        cpu_output = self.cpu_op_exec(cpu_input1)
        npu_output = self.npu_op_exec(input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_gelu_float32_2(self, device="npu"):
        input1 = self.generate_data(0, 1000, (4, 3), np.float32)
        cpu_input1 = copy.deepcopy(input1)
        cpu_output = self.cpu_op_exec(cpu_input1)
        npu_output = self.npu_op_exec(input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_gelu_float16_1(self, device="npu"):
        npu_input1 = self.generate_data(0, 100, (5, 3), np.float16)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_output = self.cpu_op_exec_fp16(cpu_input1)
        npu_output = self.npu_op_exec_fp16(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_gelu_float16_2(self, device="npu"):
        npu_input1 = self.generate_data(0, 1000, (5, 3), np.float16)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_output = self.cpu_op_exec_fp16(cpu_input1)
        npu_output = self.npu_op_exec_fp16(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_gelu_float16_3(self, device="npu"):
        npu_input1 = self.generate_data(0, 1000, (3, 3), np.float16)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_output = self.cpu_op_exec_fp16(cpu_input1)
        npu_output = self.npu_op_exec_fp16(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    np.random.seed(1234)
    run_tests()
