import torch
import numpy as np
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestUpsamleNearest2D(TestCase):
    def cpu_op_exec(self, input1, size):
        output = torch.nn.functional.interpolate(input1, size, mode="nearest")
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, size):
        output = torch.nn.functional.interpolate(input1, size, mode="nearest")
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_upsample_nearest2d_shape_format(self):
        shape_format = [
            [[np.float32, 0, (5, 3, 6, 4)], [10, 10]],
            [[np.float16, 0, (5, 3, 6, 4)], [10, 10]],
            [[np.float32, 0, (2, 3, 2, 4)], [10, 10]],
            [[np.float16, -1, (2, 3, 2, 3)], [10, 10]]
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 50)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            size = item[1]
            cpu_output = self.cpu_op_exec(cpu_input, size)
            npu_output = self.npu_op_exec(npu_input, size)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
