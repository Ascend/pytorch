#

import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestConvolution(TestCase):

    def supported_op_exec(self, input1, weight, bias, stride, padding, dilation, groups):
        dim = input1.dim()
        if dim == 4:
            output = torch.nn.functional.conv2d(input1, weight, bias, stride, padding, dilation, groups)
        if dim == 5:
            is_dilated = False
            for d in dilation:
                is_dilated |= (d != 1)
            if groups == 1 and not is_dilated:
                kernel_size = weight.size()[2]
                output = torch._C._nn.slow_conv3d(input1, weight, kernel_size, bias, stride, padding)
            else:
                output = torch.nn.functional.conv3d(input1, weight, bias, stride, padding, dilation, groups)
        return output.cpu().detach()

    def custom_op_exec(self, input1, weight, bias, stride, padding, dilation, groups):
        output = torch_npu.npu_convolution(input1, weight, bias, stride, padding, dilation, groups)
        return output.cpu().detach()

    def test_npu_convolution(self, device="npu"):
        items = [[[np.float32, 0, [16, 128, 112, 112]], [np.float32, 0, [256, 128, 3, 3]], [np.float32, 2, [256]],
                  [1, 1], [1, 1], [1, 1], 1],
                 [[np.float16, 30, [1, 128, 4, 14, 14]], [np.float16, 30, [1, 128, 3, 3, 3]], None,
                  [1, 1, 1], [1, 1, 1], [1, 1, 1], 1]]
        for item in items:
            _, npu_input = create_common_tensor(item[0], -1, 1)
            _, weight = create_common_tensor(item[1], -1, 1)
            _, bias = create_common_tensor(item[2], -1, 1) if item[2] else _, None
            stride = item[3]
            padding = item[4]
            dilation = item[5]
            groups = item[6]

        supported_output = self.supported_op_exec(npu_input, weight, bias, stride, padding, dilation, groups)
        custom_output = self.custom_op_exec(npu_input, weight, bias, stride, padding, dilation, groups)
        self.assertRtolEqual(supported_output, custom_output)


if __name__ == "__main__":
    run_tests()
