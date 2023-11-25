import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestRepeat(TestCase):
    def cpu_op_exec(self, input1, size):
        output = input1.repeat(size)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, size):
        output = input1.repeat(size)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_repeat_common_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, -1, (1280, 4)], [2, 3]],
            [[np.float32, 0, (1, 6, 4)], [2, 4, 8]],
            [[np.float32, 0, (2, 4, 5)], [2, 6, 10]],
            [[np.int32, 0, (2, 2, 1280, 4)], [2, 2, 3, 5]],
            [[np.int32, 0, (2, 1280, 4)], [3, 2, 6]],
            [[np.int32, 0, (1, 6, 4)], [1, 2, 4, 8]],
            [[np.int32, 0, (2, 4, 5)], [2, 5, 10]],
            [[np.int64, 0, (2, 1280, 4)], [3, 2, 6]],
            [[np.int64, 0, (1, 6, 4)], [1, 2, 4, 8]],
            [[np.int64, 0, (2, 4, 5)], [2, 5, 10]],
            [[np.float16, 0, (1280, 4)], [2, 3]],
            [[np.float16, 0, (1024, 4)], [2, 3, 4]],
            [[np.float16, 0, (1, 6, 4)], [2, 4, 8]],
            [[np.float16, 0, (2, 4, 5)], [2, 6, 10]],
            [[np.bool_, 0, (1024, 4)], [2, 3, 4]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
