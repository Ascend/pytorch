# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

import torch
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestQuantizedMaxPool2d(TestCase):
    def cpu_op_exec(self, input, kernel_size, stride, padding, dilation, ceil_mode):
        output = nn.quantized.functional.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
        return output.numpy()

    def npu_op_exec(self, input, ksize, stride, padding, dilation, ceil_mode):
        output = nn.quantized.functional.max_pool2d(input, ksize, stride, padding, dilation, ceil_mode)
        return output.cpu().numpy()

    def test_quantized_max_pool2d_shape_format_fp16(self, device):
        format_list = [0]
        shape_list = [(32, 16, 16, 16),
                      (16, 1024, 256, 20),
                      (1024, 464, 11, 9),
                      (1, 2048, 15, 15)]
        ksize_list = [(2, 2), (3, 3)]
        stride_list = [(1, 1), (2, 2)]
        padding_list = [(0, 0), (1, 1)]
        dilation_list = [1]
        ceil_mode_list = [False, True]
        shape_format = [
            [np.float16, i, j, k, m, n, o, p] for i in format_list for j in shape_list for k in ksize_list for m in stride_list for n in padding_list for o in dilation_list for p in ceil_mode_list
        ]
        # TODO(Ascend): tbe operator has problem in precision and (x, 1) case and so on.
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            #npu_input = cpu_input
            cpu_input = cpu_input.to(torch.float32)
            qu = torch.nn.quantized.Quantize(1.0, 50, torch.qint8)
            cpu_input = qu(cpu_input)
            npu_input = qu(npu_input)
            #npu_input.to("npu")
            cpu_output = self.cpu_op_exec(cpu_input, item[3], item[4], item[5], item[6], ceil_mode=item[7])
            print(item)
            print(cpu_output.shape)
            npu_output = self.npu_op_exec(npu_input, item[3], item[4], item[5], item[6], ceil_mode=item[7])
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestQuantizedMaxPool2d, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:1")
    run_tests()
