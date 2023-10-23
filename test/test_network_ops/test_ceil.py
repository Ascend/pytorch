import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestCeil(TestCase):
    def test_ceil(self):
        cpu_input = torch.randn(10, 10)
        npu_input = cpu_input.to("npu")
        cpu_output = torch.ceil_(cpu_input)
        npu_output = torch.ceil_(npu_input)
        npu_output = npu_output.to("cpu")

        self.assertRtolEqual(cpu_output, npu_output)

    def cpu_op_exec(self, input1):
        output = torch.ceil(input1)
        return output

    def npu_op_exec(self, input1):
        output = torch.ceil(input1)
        output = output.to("cpu")
        return output

    def test_ceil_shape_format(self, device="npu"):
        shape_format = [
            [np.float32, 0, 10],
            [np.float32, 0, (64, 10)],
            [np.float32, 3, (256, 2048, 7, 7)],
            [np.float32, 4, (32, 1, 3, 3)],
            [np.float32, 29, (10, 128)],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
