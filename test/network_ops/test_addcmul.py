import sys

import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAddCMul(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input3 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)

        return npu_input1, npu_input2, npu_input3

    def generate_output_data(self, min_d, max_d, shape, dtype):
        output_y = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_output_y = torch.from_numpy(output_y)
        return npu_output_y

    def cpu_op_exec(self, input1, input2, input3, scalar):
        output = torch.addcmul(input1, input2, input3, value=scalar)
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, input2, input3, scalar, output_y):
        output = output_y
        torch.addcmul(input1, input2, input3, value=scalar, out=output_y)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, input3, scalar):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        output = torch.addcmul(input1, input2, input3, value=scalar)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, input3, scalar, output_y):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        output = output_y.to("npu")
        torch.addcmul(input1, input2, input3, value=scalar, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_addcmul_3_3_float32(self, device="npu"):
        input1, input2, input3 = self.generate_data(0, 100, (3, 3), np.float32)
        cpu_output = self.cpu_op_exec(input1, input2, input3, 0.5)
        npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcmul_10_10_float32(self, device="npu"):
        input1, input2, input3 = self.generate_data(0, 100, (10, 10), np.float32)
        cpu_output = self.cpu_op_exec(input1, input2, input3, 0.5)
        npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcmul_3_3_float16(self, device="npu"):
        input1, input2, input3 = self.generate_data(0, 100, (3, 3), np.float16)
        input1_cpu = input1.float()
        input2_cpu = input2.float()
        input3_cpu = input3.float()
        cpu_output = self.cpu_op_exec(input1_cpu, input2_cpu, input3_cpu, 0.5).astype(np.float16)
        npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcmul_10_10_float16(self, device="npu"):
        input1, input2, input3 = self.generate_data(0, 100, (10, 10), np.float16)
        input1_cpu = input1.float()
        input2_cpu = input2.float()
        input3_cpu = input3.float()
        cpu_output = self.cpu_op_exec(input1_cpu, input2_cpu, input3_cpu, 0.5).astype(np.float16)
        npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcmul_10_23_float32(self, device="npu"):
        input1, input2, input3 = self.generate_data(0, 100, (10, 23), np.float32)
        cpu_output = self.cpu_op_exec(input1, input2, input3, 0.5)
        npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_tensor_addcmul_3_3_float32(self, device="npu"):
        input1, input2, input3 = self.generate_data(0, 100, (3, 3), np.float32)
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        input3_npu = input3.npu()
        input1.addcmul_(input2, input3, value=0.5)
        input1_npu.addcmul_(input2_npu, input3_npu, value=0.5)
        self.assertRtolEqual(input1, input1_npu)

    def test_tensor_addcmul_10_10_float32(self, device="npu"):
        input1, input2, input3 = self.generate_data(0, 100, (10, 10), np.float32)
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        input3_npu = input3.npu()
        input1.addcmul_(input2, input3, value=0.5)
        input1_npu.addcmul_(input2_npu, input3_npu, value=0.5)
        self.assertRtolEqual(input1, input1_npu)

    def test_tensor_addcmul_3_3_float16(self, device="npu"):
        input1, input2, input3 = self.generate_data(0, 100, (3, 3), np.float16)
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        input3_npu = input3.npu()
        input1.addcmul_(input2, input3, value=0.5)
        input1_npu.addcmul_(input2_npu, input3_npu, value=0.5)
        self.assertRtolEqual(input1, input1_npu)

    def test_tensor_addcmul_10_10_float16(self, device="npu"):
        input1, input2, input3 = self.generate_data(0, 100, (10, 10), np.float16)
        input1_npu = input1.npu()
        input2_npu = input2.npu()
        input3_npu = input3.npu()
        input1.addcmul_(input2, input3, value=0.5)
        input1_npu.addcmul_(input2_npu, input3_npu, value=0.5)
        self.assertRtolEqual(input1, input1_npu)

if __name__ == "__main__":
    run_tests()
