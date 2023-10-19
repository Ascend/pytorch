import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestFrac(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        return npu_input1, npu_input2

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def generate_three_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input3 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)
        return npu_input1, npu_input2, npu_input3

    def generate_scalar(self, min_d, max_d):
        scalar = np.random.uniform(min_d, max_d)
        return scalar

    def generate_int_scalar(self, min_d, max_d):
        scalar = np.random.randint(min_d, max_d)
        return scalar

    def cpu_op_exec(self, input1):
        output = torch.frac(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.frac(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_(self, input1):
        torch.frac_(input1)
        output = input1.numpy()
        return output

    def npu_op_exec_(self, input1):
        torch.frac_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, out):
        torch.frac(input1, out=out)
        output = out.numpy()
        return output

    def npu_op_exec_out(self, input1, out):
        out = out.to("npu")
        torch.frac(input1, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def test_frac_common_shape_format(self):
        shape_format = [
            [np.float32, -1, (4, 3)],
            [np.float32, -1, (4, 3, 1)],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_frac1_common_shape_format(self):
        shape_format = [
            [np.float32, -1, (4, 3)],
            [np.float32, -1, (4, 3, 1)],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec_(cpu_input1)
            npu_output = self.npu_op_exec_(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_frac_out_common_shape_format(self):
        shape_format = [
            [np.float32, -1, (4, 3)],
            [np.float32, -1, (4, 3, 1)],
        ]
        out = self.generate_single_data(0, 100, (5, 3), np.float32)
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec_out(cpu_input1, out)
            npu_output = self.npu_op_exec_out(npu_input1, out)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
