import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestNonzero(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.nonzero(input1)
        output = output.numpy().astype(np.int32)
        return output

    def npu_op_exec(self, input1):
        output = torch.nonzero(input1)
        output = output.to("cpu")
        output = output.numpy().astype(np.int32)
        return output

    def test_zero_input(self):
        cpu_input = torch.zeros(())
        npu_input = cpu_input.npu()
        cpu_output = self.cpu_op_exec(cpu_input)
        npu_output = self.npu_op_exec(npu_input)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_nonzero_shape_format(self, device="npu"):
        dtype_list = [np.float32, np.float16, np.int32, np.int64]
        format_list = [0]
        shape_list = [[256, 10], [256, 256, 100], [5, 256, 256, 100]]

        shape_format = [
            [[i, j, k]] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
