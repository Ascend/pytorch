import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestIsfinite(TestCase):
    def test_isfinite(self, device="npu"):
        x = torch.Tensor([1, 2, -10]).to("npu")
        self.assertRtolEqual(torch.isfinite(x).to("cpu"), torch.BoolTensor([True, True, True]))

    def cpu_op_exec(self, input1):
        output = torch.isfinite(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.isfinite(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_isfinite_shape_format(self, device="npu"):
        shape_format = [
            [np.int16, 0, (1, 2, 2, 5)],
            [np.int32, 0, (1, 4, 3)],
            [np.int64, 0, (2, 3)],
            [np.float32, 0, (8, 4, 3, 9)],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
