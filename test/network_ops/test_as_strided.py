import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAsStrided(TestCase):
    def cpu_op_exec(self, input1, size, stride, storage_offset):
        output = torch.as_strided(input1, size, stride, storage_offset)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, size, stride, storage_offset):
        output = torch.as_strided(input1, size, stride, storage_offset)
        output = output.cpu().numpy()
        return output

    def test_as_strided(self):
        shape_format = [
            [[np.float32, 0, [3, 3]], (2, 2), (1, 2), 0],
            [[np.float16, 0, [13, 23]], (10, 15), (1, 2), 1],
            [[np.int32, 0, [5, 5]], (3, 3), (1, 2), 1],
            [[np.float32, 2, [32, 8, 2]], (8, 6, 2), (5, 4, 1), 1],
            [[np.int32, 2, [8, 16]], (6, 3), (8, 2), 0],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2], item[3])
            npu_output = self.npu_op_exec(npu_input1, item[1], item[2], item[3])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
