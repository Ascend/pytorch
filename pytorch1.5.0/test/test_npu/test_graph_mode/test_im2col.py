# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
import torch
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from graph_utils import graph_mode

class TestIm2col(TestCase):

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input1, ksizes, rates, padding, strides):
        output = torch._C._nn.im2col(input1, ksizes, rates, padding, strides)
        output = output.numpy()
        return output

    def npu_op_exec_tensor_need_to_npu(self, input1, ksizes, rates, padding, strides):
        output = torch._C._nn.im2col(input1, ksizes, rates, padding, strides)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @graph_mode
    def test_im2col_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input1, ksizes, rates, padding, strides):
            input1 = input1.to(torch.float32)
            output = torch._C._nn.im2col(input1, ksizes, rates, padding, strides)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [np.float16, -1, (1, 3, 3, 3)],
            [np.float16, -1, (2, 16, 4, 6)],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = cpu_op_exec_fp16(cpu_input1, (1,1), (1,1),(0,0),(1,1))
            npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, (1,1), (1,1),(0,0),(1,1))
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestIm2col, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()