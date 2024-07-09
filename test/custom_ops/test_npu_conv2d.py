import numpy as np
import torch

import torch_npu

from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuConv2d(TestCase):

    def custom_op_exec(self, input1, weight, bias, stride, padding, dilation, groups):
        output = torch.nn.functional.conv2d(input1, weight, bias, stride, padding, dilation, groups)
        output = output.cpu().numpy()
        return output

    def npu_op_exec(self, input1, weight, bias, stride, padding, dilation, groups):
        output = torch_npu.npu_conv2d(input1, weight, bias, stride, padding, dilation, groups)
        output = output.cpu().numpy()
        return output

    def test_npu_conv2d_fp16(self):
        shape_format = [
            # input, weigth, bias, stride, padding, dilation, groups
            [[np.float16, 0, [16, 128, 112, 112]], [np.float16, 0, [256, 128, 3, 3]], [np.float16, 2, [256]], [1, 1],
             [1, 1], [1, 1], 1],
            [[np.float16, 0, [1024, 232, 7, 7]], [np.float16, 0, [232, 232, 1, 1]], [np.float16, 2, [232]], [1, 2],
             [1, 2], [1, 2], 1],
            [[np.float16, 0, [1024, 116, 14, 14]], [np.float16, 0, [116, 116, 1, 1]], None, [1, 1], [1, 1], [1, 1], 1],
            [[np.float16, 0, [4, 8, 300, 40]], [np.float16, 0, [16, 8, 3, 3]], None, [1, 2], [1, 2], [1, 2], 1],
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

    def test_npu_conv2d_fp32(self):
        shape_format = [
            # input, weigth, bias, stride, padding, dilation, groups
            [[np.float32, 0, [16, 128, 112, 112]], [np.float32, 0, [256, 128, 3, 3]], [np.float32, 2, [256]], [1, 1],
             [1, 1], [1, 1], 1],
            [[np.float32, 0, [1024, 232, 7, 7]], [np.float32, 0, [232, 232, 1, 1]], [np.float32, 2, [232]], [1, 2],
             [1, 2], [1, 2], 1],
            [[np.float32, 0, [1024, 116, 14, 14]], [np.float32, 0, [116, 116, 1, 1]], None, [1, 1], [1, 1], [1, 1], 1],
            [[np.float32, 0, [4, 8, 300, 40]], [np.float32, 0, [16, 8, 3, 3]], None, [1, 2], [1, 2], [1, 2], 1],
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
