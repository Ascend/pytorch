import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestDot(TestCase):
    def generate_data(self, min1, max1, shape, dtype):
        input1 = np.random.uniform(min1, max1, shape).astype(dtype)
        input2 = np.random.uniform(min1, max1, shape).astype(dtype)

        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)

        return npu_input1, npu_input2

    def generate_three_data(self, min1, max1, shape, dtype):
        input1 = np.random.uniform(min1, max1, shape).astype(dtype)
        input2 = np.random.uniform(min1, max1, shape).astype(dtype)
        input3 = np.random.uniform(min1, max1, shape).astype(dtype)

        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)

        return npu_input1, npu_input2, npu_input3

    def cpu_op_exec(self, input1, input2):
        output = torch.dot(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.dot(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, input3):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = input3.to("npu")
        torch.dot(input1, input2, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_dot_float32(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 10, (3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_dot_float32_out(self, device="npu"):
        npu_input1, npu_input2, npu_input3 = self.generate_three_data(0, 10, (3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_dot_float16(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 10, (3), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1.float(), npu_input2.float()).astype(np.float16)
        npu_output = self.npu_op_exec(npu_input1.float(), npu_input2.float()).astype(np.float16)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_dot_float16_out(self, device="npu"):
        npu_input1, npu_input2, npu_input3 = self.generate_three_data(0, 10, (3), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1.float(), npu_input2.float()).astype(np.float16)
        npu_output = self.npu_op_exec_out(npu_input1.float(), npu_input2.float(), npu_input3.float()).astype(np.float16)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_big_scale_float32(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 10, (10240), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
