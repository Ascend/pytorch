import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestClone(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.clone(input1)
        return output

    def npu_op_exec(self, input1):
        output = torch.clone(input1)
        return output

    def cpu_tensor_op_exec(self, input1):
        output = input1.clone()
        return output

    def npu_tensor_op_exec(self, input1):
        output = input1.clone()
        return output

    def test_clone_op_exec(self):
        shape_format = [
            [[np.float32, -1, (1, 2, 3, 4)]],
            [[np.float32, -1, (2, 3, 4)]],
            [[np.float16, -1, (1, 2, 3, 4)]],
            [[np.float16, -1, (2, 3, 4)]],
            [[np.int32, -1, (1, 2, 3, 4)]],
            [[np.int32, -1, (2, 3, 4)]],
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            couput = self.cpu_op_exec(cpu_input)
            npuput = self.npu_op_exec(npu_input)

            self.assertRtolEqual(couput, npuput)

    def test_tensor_clone_op_exec(self):
        shape_format = [
            [[np.float32, -1, (1, 2, 3, 4)]],
            [[np.float32, -1, (2, 3, 4)]],
            [[np.float16, -1, (1, 2, 3, 4)]],
            [[np.float16, -1, (2, 3, 4)]],
            [[np.int32, -1, (1, 2, 3, 4)]],
            [[np.int32, -1, (2, 3, 4)]],
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            couput = self.cpu_tensor_op_exec(cpu_input)
            npuput = self.npu_tensor_op_exec(npu_input)

            self.assertRtolEqual(couput, npuput)

if __name__ == "__main__":
    run_tests()
