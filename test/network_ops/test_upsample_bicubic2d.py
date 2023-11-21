import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestUpsampleBicubic2d(TestCase):

    def cpu_op_exec(self, input1, output_size, align_corners, scale_h, scale_w):
        output = torch._C._nn.upsample_bicubic2d(input1, output_size, align_corners, scale_h, scale_w)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, output_size, align_corners, scale_h, scale_w):
        output = torch._C._nn.upsample_bicubic2d(input1, output_size, align_corners, scale_h, scale_w)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @unittest.skip("skip test_upsample_bicubic2d_common_shape_format now")
    def test_upsample_bicubic2d_common_shape_format(self):
        shape_format = [
            [[np.float32, -1, (1, 1, 1, 1)], (1, 1), True, 0, 0, 0, 255],
            [[np.float32, -1, (2, 65535, 2, 2)], (2, 2), True, 0, 0, 0, 255],
            [[np.float32, -1, (65535, 2, 4, 8)], (4, 8), True, 0, 0, 0, 255],
            [[np.float32, -1, (2, 4, 65535, 2)], (65535, 2), True, 0, 0, 0, 255],
            [[np.float32, -1, (1, 31, 149, 2)], (149, 2), True, 0, 0, 0, 255],
            [[np.float32, -1, (10, 10, 786432, 8)], (786432, 8), True, 0, 0, 0, 255],
            [[np.float32, -1, (1, 1, 1, 1)], (2, 2), True, 0, 0, 0, 255],
            [[np.float32, -1, (1, 1, 2, 2)], (4, 4), True, 0, 0, 0, 255],
            [[np.float32, -1, (2, 2, 1, 1)], (2, 2), True, 0, 0, 0, 255],
            [[np.float32, -1, (2, 2, 2, 2)], (10, 10), True, 0, 0, 0, 255],
            [[np.float32, -1, (1, 1, 1, 1)], (2, 2), False, 0.5, 0.5, 0, 255],
            [[np.float32, -1, (1, 1, 2, 2)], (4, 4), False, 0.5, 0.5, 0, 255],
            [[np.float32, -1, (2, 2, 1, 1)], (2, 2), False, 0.5, 0.5, 0, 255],
            [[np.float32, -1, (2, 2, 2, 2)], (10, 10), False, 0.5, 0.5, 0, 255],

        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[5], item[6])
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2], item[3], item[4])
            npu_output = self.npu_op_exec(npu_input1, item[1], item[2], item[3], item[4])
            self.assertRtolEqual(cpu_output, npu_output)

    @unittest.skip("skip test_upsample_bicubic2d_float16_shape_format now")
    def test_upsample_bicubic2d_float16_shape_format(self):
        def cpu_op_exec_fp16(input1, output_size, align_corners, scale_h, scale_w):
            input1 = input1.to(torch.float32)
            output = torch._C._nn.upsample_bicubic2d(input1, output_size, align_corners, scale_h, scale_w)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, -1, (1, 1, 1, 1)], (1, 1), True, 0, 0, 0, 255],
            [[np.float16, -1, (2, 65535, 2, 2)], (2, 2), True, 0, 0, 0, 255],
            [[np.float16, -1, (65535, 2, 4, 8)], (4, 8), True, 0, 0, 0, 255],
            [[np.float16, -1, (2, 4, 65535, 2)], (65535, 2), True, 0, 0, 0, 255],
            [[np.float16, -1, (1, 31, 149, 2)], (149, 2), True, 0, 0, 0, 255],
            [[np.float16, -1, (10, 10, 786432, 8)], (786432, 8), True, 0, 0, 0, 255],
            [[np.float16, -1, (1, 1, 1, 1)], (2, 2), True, 0, 0, 0, 255],
            [[np.float16, -1, (1, 1, 2, 2)], (4, 4), True, 0, 0, 0, 255],
            [[np.float16, -1, (2, 2, 1, 1)], (2, 2), True, 0, 0, 0, 255],
            [[np.float16, -1, (2, 2, 2, 2)], (10, 10), True, 0, 0, 0, 255],
            [[np.float16, -1, (1, 31, 149, 2)], (2, 149), True, 0, 0, 0, 255],
            [[np.float16, -1, (32, 32, 32, 32)], (64, 64), True, 0, 0, 0, 255],
            [[np.float16, -1, (32, 32, 32, 32)], (64, 64), True, 0, 0, 0, 6550.0],
            [[np.float16, -1, (1, 1, 1, 1)], (2, 2), False, 0.5, 0.5, 0, 255],
            [[np.float16, -1, (1, 1, 2, 2)], (4, 4), False, 0.5, 0.5, 0, 255],
            [[np.float16, -1, (2, 2, 1, 1)], (2, 2), False, 0.5, 0.5, 0, 255],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[5], item[6])
            cpu_output = cpu_op_exec_fp16(cpu_input1, item[1], item[2], item[3], item[4])
            npu_output = self.npu_op_exec(npu_input1, item[1], item[2], item[3], item[4])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
