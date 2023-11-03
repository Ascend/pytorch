import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestHardShrink(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input = torch.from_numpy(input_x)
        return npu_input

    def cpu_op_exec(self, input_x, lambd):
        output = torch.nn.functional.hardshrink(input_x, lambd=lambd)
        output = output.numpy()
        return output.astype(np.float32)

    def npu_op_exec(self, input_x, lambd):
        input1 = input_x.to("npu")
        output = torch.nn.functional.hardshrink(input1, lambd=lambd)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_hardshrink_3_3_float32(self):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 0.5)
        npu_output1 = self.npu_op_exec(input_x1, 0.5)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_hardshrink_100_100_float32(self):
        input_x1 = self.generate_data(-1, 1, (100, 100), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 0.5)
        npu_output1 = self.npu_op_exec(input_x1, 0.5)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_hardshrink_3_3_float16(self):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float16)
        input_x1_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec(input_x1_cpu, 0.5).astype(np.float16)
        npu_output1 = self.npu_op_exec(input_x1, 0.5)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_hardshrink_100_100_float16(self):
        input_x1 = self.generate_data(-1, 1, (100, 100), np.float16)
        input_x1_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec(input_x1_cpu, 0.5).astype(np.float16)
        npu_output1 = self.npu_op_exec(input_x1, 0.5)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_hardshrink_10_10_10_10_float32(self):
        input_x1 = self.generate_data(-1, 1, (10, 10, 10, 10), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 0.5)
        npu_output1 = self.npu_op_exec(input_x1, 0.5)
        self.assertRtolEqual(cpu_output1, npu_output1)


if __name__ == "__main__":
    run_tests()
