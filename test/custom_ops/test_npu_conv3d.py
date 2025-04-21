import numpy as np
import torch

import torch_npu

from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuConv3d(TestCase):

    def custom_op_exec(self, input1, weight, bias, stride, padding, dilation, groups):
        output = torch.nn.functional.conv3d(input1, weight, bias, stride, padding, dilation, groups)
        output = output.cpu().numpy()
        return output

    def npu_op_exec(self, input1, weight, bias, stride, padding, dilation, groups):
        output = torch_npu.npu_conv3d(input1, weight, bias, stride, padding, dilation, groups)
        output = output.cpu().numpy()
        return output

    def test_npu_conv3d_fp16(self):
        shape_format = [
            # input, weigth, bias, stride, padding, dilation, groups
            [[np.float16, 30, [1, 128, 4, 14, 14]], [np.float16, 30, [1, 128, 3, 3, 3]], None, [1, 1, 1], [1, 1, 1],
             [1, 1, 1], 1],
            [[np.float16, 30, [1, 64, 4, 14, 14]], [np.float16, 30, [1, 64, 3, 3, 3]], None, [1, 1, 1], [2, 2, 2],
             [1, 1, 1], 1],
            [[np.float16, 30, [20, 16, 50, 10, 20]], [np.float16, 30, [33, 16, 3, 3, 3]], [np.float16, 2, [33]],
             [1, 1, 1], [2, 2, 2], [1, 1, 1], 1],
        ]
        for item in shape_format:
            _, input_npu = create_common_tensor(item[0], -1, 1)
            _, weight_npu = create_common_tensor(item[1], -1, 1)
            _, bias_npu = create_common_tensor(item[2], -1, 1) if item[2] else None, None
            custom_output = self.custom_op_exec(input_npu,
                                                weight_npu,
                                                bias_npu,
                                                stride=item[3],
                                                padding=item[4],
                                                dilation=item[5],
                                                groups=item[6])
            npu_output = self.npu_op_exec(input_npu,
                                          weight_npu,
                                          bias_npu,
                                          stride=item[3],
                                          padding=item[4],
                                          dilation=item[5],
                                          groups=item[6])
            self.assertRtolEqual(custom_output, npu_output)

    def test_npu_conv3d_fp32(self):
        torch.npu.config.allow_internal_format = True
        torch.npu.set_compile_mode(jit_compile=True)
        shape_format = [
            # input, weigth, bias, stride, padding, dilation, groups
            [[np.float32, 30, [1, 128, 4, 14, 14]], [np.float32, 30, [1, 128, 3, 3, 3]], None, [1, 1, 1], [1, 1, 1],
             [1, 1, 1], 1],
            [[np.float32, 30, [1, 64, 4, 14, 14]], [np.float32, 30, [1, 64, 3, 3, 3]], None, [1, 1, 1], [2, 2, 2],
             [1, 1, 1], 1],
            [[np.float32, 30, [20, 16, 50, 10, 20]], [np.float32, 30, [33, 16, 3, 3, 3]], [np.float32, 2, [33]],
             [1, 1, 1], [2, 2, 2], [1, 1, 1], 1],
        ]
        for item in shape_format:
            _, input_npu = create_common_tensor(item[0], -1, 1)
            _, weight_npu = create_common_tensor(item[1], -1, 1)
            _, bias_npu = create_common_tensor(item[2], -1, 1) if item[2] else None, None
            custom_output = self.custom_op_exec(input_npu,
                                                weight_npu,
                                                bias_npu,
                                                stride=item[3],
                                                padding=item[4],
                                                dilation=item[5],
                                                groups=item[6])
            npu_output = self.npu_op_exec(input_npu,
                                          weight_npu,
                                          bias_npu,
                                          stride=item[3],
                                          padding=item[4],
                                          dilation=item[5],
                                          groups=item[6])
            self.assertRtolEqual(custom_output, npu_output)


if __name__ == "__main__":
    run_tests()
