import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestUpsampleBilinear2d(TestCase):

    def cpu_op_exec(self, inputs, shapes):
        output = torch._C._nn.upsample_bilinear2d(inputs, shapes, True, 0, 0)
        output = output.numpy()
        return output

    def npu_op_exec(self, inputs, shapes):
        output = torch._C._nn.upsample_bilinear2d(inputs, shapes, True, 0, 0)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_scale_exec(self, inputs, scale_factor):
        output = torch.nn.functional.interpolate(inputs, scale_factor=scale_factor, mode="bilinear")
        output = output.numpy()
        return output

    def npu_op_scale_exec(self, inputs, scale_factor):
        output = torch.nn.functional.interpolate(inputs, scale_factor=scale_factor, mode="bilinear")
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_UpsampleBilinear2d_common_shape_format(self):
        shape_format = [
            [[np.float32, -1, (1, 1, 3000, 3000)], (2500, 2500)],
            [[np.float32, -1, (4, 3, 1, 5)], (2, 2)],
            [[np.float32, -1, (2, 3, 2, 1)], (3, 3)],
            [[np.float32, -1, (1, 4, 2, 2)], (4, 4)],
            [[np.float16, -1, (4, 10, 16, 14)], (5, 5)],
            [[np.float16, -1, (8, 8, 8, 8)], (1, 2)],
            [[np.float16, -1, (10, 4, 3, 2)], (2, 4)]
        ]
        for item in shape_format:
            cpu_inputs, npu_inputs = create_common_tensor(item[0], 1, 100)
            if cpu_inputs.dtype == torch.float16:
                cpu_inputs = cpu_inputs.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_inputs, item[1])
            npu_output = self.npu_op_exec(npu_inputs, item[1])
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_UpsampleBilinear2d_scale_common_shape_format(self):
        shape_format = [
            [[np.float32, -1, (1, 1, 3000, 3000)], [2, 2]],
            [[np.float32, 0, (4, 3, 1, 5)], [2, 2]],
            [[np.float32, -1, (2, 3, 2, 1)], [3, 3]],
            [[np.float32, 3, (1, 4, 2, 2)], [4, 4]],
            [[np.float16, 4, (4, 10, 16, 14)], [5, 5]],
            [[np.float16, 0, (8, 8, 8, 8)], [1, 2]],
            [[np.float16, 4, (10, 4, 3, 2)], [2, 4]],
            [[np.float16, 2, (10, 4, 3, 2)], [2, 2]]
        ]
        for item in shape_format:
            cpu_inputs, npu_inputs = create_common_tensor(item[0], 1, 100)
            if cpu_inputs.dtype == torch.float16:
                cpu_inputs = cpu_inputs.to(torch.float32)
            cpu_output = self.cpu_op_scale_exec(cpu_inputs, item[1])
            npu_output = self.npu_op_scale_exec(npu_inputs, item[1])
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
