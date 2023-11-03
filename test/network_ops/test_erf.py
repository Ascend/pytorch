# coding: utf-8

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestErf(TestCase):

    def cpu_op_exec(self, input1):
        output = torch.erf(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.erf(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_(self, input1):
        torch.erf_(input1)
        output = input1.numpy()
        return output

    def npu_op_exec_(self, input1):
        torch.erf_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, cpu_out):
        torch.erf(input1, out=cpu_out)
        output = cpu_out.numpy()
        return output

    def npu_op_exec_out(self, input1, npu_out):
        torch.erf(input1, out=npu_out)
        output = npu_out.to("cpu")
        output = output.numpy()
        return output

    def test_erf_float32_common_shape_format(self, device="npu"):
        shape_format = [
            [np.float32, 0, (4, 3)],
            [np.float32, -1, (2, 4, 3)],
            [np.float32, 3, (20, 13)],
            [np.float32, 4, (20, 13)],
            [np.float32, 29, (20, 13)]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_erf_float321_common_shape_format(self, device="npu"):
        shape_format = [
            [np.float32, 0, (4, 3)],
            [np.float32, -1, (2, 4, 3)],
            [np.float32, 3, (20, 13)],
            [np.float32, 4, (20, 13)],
            [np.float32, 29, (20, 13)]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec_(cpu_input1)
            npu_output = self.npu_op_exec_(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_erf_out_float32_common_shape_format(self, device="npu"):
        shape_format = [
            [np.float32, 0, (4, 3)],
            [np.float32, -1, (2, 4, 3)],
            [np.float32, 3, (20, 13)],
            [np.float32, 4, (20, 13)],
            [np.float32, 29, (20, 13)]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_out, npu_out = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec_out(cpu_input1, cpu_out)
            npu_output = self.npu_op_exec_out(npu_input1, npu_out)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
