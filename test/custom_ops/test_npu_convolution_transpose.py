#

import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

DEVICE_NAME = torch_npu.npu.get_device_name(0)


class TestConvolutionTranspose(TestCase):

    def supported_op_exec(self, input1, weight, bias, padding, output_padding, stride, dilation, groups):
        dim = input1.dim()
        if dim == 4:
            output = torch.nn.functional.conv_transpose2d(input1, weight, bias, stride, padding,
                                                          output_padding, groups, dilation)
        elif dim == 5:
            output = torch.nn.functional.conv_transpose3d(input1, weight, bias, stride, padding,
                                                          output_padding, groups, dilation)
        return output.cpu().detach()

    def custom_op_exec(self, input1, weight, bias, padding, output_padding, stride, dilation, groups):
        output = torch_npu.npu_convolution_transpose(input1, weight, bias, padding, output_padding,
                                                     stride, dilation, groups)
        return output.cpu().detach()

    def test_npu_convolution_transpose(self):
        items = [[[np.float32, 0, [1, 3, 3, 3]], [np.float32, 0, [3, 2, 3, 3]], [np.float32, 2, [2]],
                  [1, 1], [0, 0], [0, 0], [1, 1], 1],
                 [[np.float16, 2, [20, 16, 50, 10, 20]], [np.float16, 2, [16, 33, 3, 3, 3]], None,
                  [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], 1]]
        if "Ascend910A" in DEVICE_NAME or "Ascend910P" in DEVICE_NAME:
            items0 = [[np.float32, 2, [20, 16, 50, 10, 20]], [np.float32, 2, [16, 33, 3, 3, 3]], None,
                      [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], 1]
            items.append(items0)
        for item in items:
            _, npu_input = create_common_tensor(item[0], 0, 0.001)
            _, weight = create_common_tensor(item[1], 0, 0.001)
            _, bias = create_common_tensor(item[2], 1, 200) if item[2] else _, None
            padding = item[3]
            output_padding = item[4]
            stride = item[5]
            dilation = item[6]
            groups = item[7]

        supported_output = self.supported_op_exec(npu_input, weight, bias, padding, output_padding,
                                                  stride, dilation, groups)
        custom_output = self.custom_op_exec(npu_input, weight, bias, padding, output_padding,
                                            stride, dilation, groups)
        self.assertRtolEqual(supported_output, custom_output)


if __name__ == "__main__":
    run_tests()
