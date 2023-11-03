import torch
import numpy as np
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestUpsampleNearest1DBackward(TestCase):

    def cpu_op_exec(self, input1, size):
        output = F.interpolate(input1, size, mode="nearest")
        return output.detach().numpy()

    def cpu_op_scale_exec(self, input1, scale):
        output = F.interpolate(input1, scale_factor=scale, mode="nearest")
        return output.detach().numpy()

    def npu_op_exec(self, input1, size):
        output = F.interpolate(input1, size, mode="nearest")
        output = output.cpu()
        return output.detach().numpy()

    def npu_op_scale_exec(self, input1, scale):
        output = F.interpolate(input1, scale_factor=scale, mode="nearest")
        output = output.cpu()
        return output.detach().numpy()

    def test_upsample_nearest1d_backward_shape_format(self):
        test_cases = [
            [[np.float32, 3, (2, 2, 3)], [1]],
            [[np.float32, 0, (2, 1, 1)], [4]],
            [[np.float32, 0, (20, 12, 6)], [5]],
            [[np.float16, 0, (10, 256, 256)], [2]],
            [[np.float16, 0, (20, 12, 6)], [4]],
            [[np.float64, 0, (10, 256, 256)], [2]],
            [[np.float64, 0, (20, 12, 6)], [4]],
            [[np.uint8, 0, (20, 12, 6)], [5]],
            [[np.uint8, 0, (20, 12, 6)], [4]]
        ]
        for item in test_cases:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            size = list(item[0][2])
            size[2] = item[1][0]

            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_input, item[1])
            npu_output = self.npu_op_exec(npu_input, item[1])
            cpu_output = cpu_output.astype(npu_output.dtype)

            self.assertRtolEqual(cpu_output, npu_output)

    def test_upsample_nearest1d_backward_shape_format_scale(self):
        test_cases = [
            [[np.float32, 3, (2, 2, 3)], 0.4],
            [[np.float32, 0, (2, 1, 1)], 4],
            [[np.float32, 0, (4, 1, 2)], 2],
            [[np.float32, 0, (1, 1, 1)], 1],
            [[np.float16, 3, (2, 2, 3)], 0.4],
            [[np.float16, 0, (2, 1, 1)], 4],
            [[np.float64, 0, (4, 1, 2)], 2],
            [[np.float64, 0, (10, 256, 256)], 5],
            [[np.uint8, 0, (4, 1, 2)], 2],
            [[np.uint8, 0, (20, 10, 10)], 4]
        ]
        for item in test_cases:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)

            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            cpu_output = self.cpu_op_scale_exec(cpu_input, item[1])
            npu_output = self.npu_op_scale_exec(npu_input, item[1])
            cpu_output = cpu_output.astype(npu_output.dtype)

            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
