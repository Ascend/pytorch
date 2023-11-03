import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestSinh(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input = torch.from_numpy(input_x)
        return npu_input

    def cpu_op_exec(self, input_x):
        output = torch.sinh(input_x)
        output = output.numpy()
        return output.astype(np.float32)

    def npu_op_exec(self, input_x):
        input1 = input_x.to("npu")
        output = torch.sinh(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input_x, output_y):
        input_x = input_x.to("npu")
        output_y = output_y.to("npu")
        output_y = torch.sinh(input_x, out=output_y)
        output_y = output_y.to("cpu")
        output_y = output_y.numpy()
        return output_y

    def npu_op_exec_inplace(self, input_x):
        input_x = input_x.to("npu")
        input_x.sinh_()
        output = input_x.to("cpu")
        output = output.numpy()
        return output

    def test_sinh_3_3(self):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1)
        npu_output1 = self.npu_op_exec(input_x1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_sinh_3_3_3_3(self):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1)
        npu_output1 = self.npu_op_exec(input_x1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_sinh_3_3_float16(self):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float16)
        input_x1_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec(input_x1_cpu).astype(np.float16)
        npu_output1 = self.npu_op_exec(input_x1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_sinh_3_3_3_3_float16(self):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3), np.float16)
        input_x1_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec(input_x1_cpu).astype(np.float16)
        npu_output1 = self.npu_op_exec(input_x1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_sinh_3_3_out(self):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        output_y = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1)
        npu_output1 = self.npu_op_exec_out(input_x1, output_y)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_sinh_3_3_3_3_out(self):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3), np.float32)
        output_y = self.generate_data(-1, 1, (3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1)
        npu_output1 = self.npu_op_exec_out(input_x1, output_y)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_sinh_3_3_inplace(self):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1)
        npu_output1 = self.npu_op_exec_inplace(input_x1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_sinh_3_3_3_3_inplace(self):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1)
        npu_output1 = self.npu_op_exec_inplace(input_x1)
        self.assertRtolEqual(cpu_output1, npu_output1)


if __name__ == "__main__":
    run_tests()
