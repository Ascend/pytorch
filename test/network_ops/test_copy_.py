import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestCopy(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = input1.copy_(input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = input1.copy_(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_copy__(self):
        format_list = [0]
        shape_list = [(4, 1), (4, 3, 1)]
        dtype_list = [np.float32, np.int32, np.float16]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_copy_broadcast(self):
        x = torch.randn(10, 5)
        y = torch.randn(5).npu()
        x.copy_(y)
        self.assertEqual(x[3], y)

        x = torch.randn(10, 5).npu()
        y = torch.randn(5)
        x.copy_(y)
        self.assertEqual(x[3], y)


if __name__ == "__main__":
    run_tests()
