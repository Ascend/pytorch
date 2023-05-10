# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

import torch_npu

from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuConv2d(TestCase):

    def custom_op_exec(self, input1, weight, bias, padding, output_padding, stride, dilation, groups):
        output = torch.nn.functional.conv_transpose2d(input1, weight, bias, stride, padding, output_padding, groups,
                                                      dilation)
        output = output.cpu().numpy()
        return output

    def npu_op_exec(self, input1, weight, bias, padding, output_padding, stride, dilation, groups):
        output = torch_npu.npu_conv_transpose2d(input1, weight, bias, padding, output_padding, stride, dilation, groups)
        output = output.cpu().numpy()
        return output

    def test_npu_conv_transpose2d(self):
        shape_format = [
            # input, weigth, bias, stride, padding, output_padding, dilation, groups
            [[np.float16, 0, [1, 3, 3, 3]], [np.float16, 0, [3, 2, 3, 3]], [np.float16, 2, [2]], [1, 1], [0, 0], [0, 0],
             [1, 1], 1],
            [[np.float32, 0, [1, 3, 3, 3]], [np.float32, 0, [3, 2, 3, 3]], [np.float32, 2, [2]], [1, 1], [0, 0], [0, 0],
             [1, 1], 1],
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
                                                output_padding=item[5],
                                                dilation=item[6],
                                                groups=item[7])
            npu_output = self.npu_op_exec(input_npu,
                                          weight_npu,
                                          bias_npu,
                                          stride=item[3],
                                          padding=item[4],
                                          output_padding=item[5],
                                          dilation=item[6],
                                          groups=item[7])
            self.assertRtolEqual(custom_output, npu_output)


if __name__ == "__main__":
    run_tests()
