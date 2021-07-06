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

import sys
import pdb

import torch
import numpy as np

from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestGroupNormExt(TestCase):
    def cpu_output_exec(self, data_format, input_x, scale, offset,
        shape, shape_param, num_groups, epsilon = 1e-5):

        input_x = input_x.numpy()
        if data_format == "NCHW":
            shape_r = [shape[0],
                    num_groups,
                    shape[1] // num_groups,
                    shape[2],
                    shape[3]]
            shape_param_r = \
                    [1, num_groups, shape_param[0] // num_groups, 1, 1]
        elif data_format == "NHWC":
            shape_r = [shape[0],
                    shape[1],
                    shape[2],
                    num_groups,
                    shape[3] // num_groups]
            shape_param_r = \
                    [1, 1, 1, num_groups, shape_param[0] // num_groups]

        input_x_r = np.reshape(input_x, shape_r)
        scale_r = np.reshape(scale, shape_param_r)
        offset_r = np.reshape(offset, shape_param_r)

        if data_format == "NCHW":
            reduce_axis = (2, 3, 4)
        else:
            reduce_axis = (1, 2, 4)

        reduce_elts = 1.0
        for i in reduce_axis:
            reduce_elts *= shape_r[i]

        mean_muls = input_x_r / reduce_elts
        mean = np.sum(mean_muls, axis = reduce_axis, keepdims = True)

        x_mean_sub = input_x_r - mean
        variance_mul = x_mean_sub * x_mean_sub
        variance_muls = variance_mul / reduce_elts
        variance = np.sum(variance_muls, axis = reduce_axis, keepdims = True)

        normalize_add = variance + epsilon
        normalize_sqrt = np.sqrt(normalize_add)
        normalize_mul = x_mean_sub / normalize_sqrt

        scale_mul = scale_r * normalize_mul
        output = scale_mul + offset_r
        output_y = np.reshape(output, shape).numpy()
        mean_y = np.reshape(mean, -1)
        variance_y = np.reshape(variance, -1)

        return output_y

    def npu_output_exec(self, input_x, scale, offset, num_groups):
        npu_input_x = input_x.to("npu")
        npu_scale   = scale.to("npu")
        npu_offset  = offset.to("npu")

        output = torch.group_norm(
                npu_input_x, num_groups=num_groups, weight=npu_scale, 
                bias=npu_offset)

        return output

    def test_group_norm_case1(self, device):
        shape_format = [
                [[np.float32, 0, (2, 6, 1, 1)], [np.float32, -1, (6,)], 2],
                [[np.float32, 0, (8, 6, 4, 4)], [np.float32, -1, (6,)], 2],
                [[np.float32, 0, (8, 6, 4, 4)], [np.float32, -1, (6,)], 3],
                ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -2, 2)
            cpu_scale, npu_scale = create_common_tensor(item[1], -2, 2)
            cpu_offset, npu_offset = create_common_tensor(item[1], -2, 2)

            cpu_output  = self.cpu_output_exec(
                    'NCHW', cpu_input, cpu_scale, cpu_offset, item[0][2],
                    item[1][2], item[2])
            npu_output = self.npu_output_exec(
                    npu_input, npu_scale, npu_offset, item[2])

            self.assertRtolEqual(cpu_output, npu_output.to('cpu').numpy())

    def test_group_norm_case2(self, device):
        shape_format = [
                [[np.float32, 0, (2, 6, 1, 1)], [np.float32, -1, (6,)], 2, -2e5, 2e5],
                [[np.float32, 0, (8, 6, 4, 4)], [np.float32, -1, (6,)], 2, -2e-38, 2e-38],
                [[np.float32, 0, (8, 6, 4, 4)], [np.float32, -1, (6,)], 6, -2e5, 2e5],
                [[np.float32, 0, (8, 6, 4, 4)], [np.float32, -1, (6,)], 6, -2e-38, 2e-38],
                ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], item[3], item[4])
            cpu_scale, npu_scale = create_common_tensor(item[1], item[3], item[4])
            cpu_offset, npu_offset = create_common_tensor(item[1], item[3], item[4])

            cpu_output  = self.cpu_output_exec(
                    'NCHW', cpu_input, cpu_scale, cpu_offset, item[0][2],
                    item[1][2], item[2])
            npu_output = self.npu_output_exec(
                    npu_input, npu_scale, npu_offset, item[2])

            self.assertRtolEqual(cpu_output, npu_output.to('cpu').numpy())

instantiate_device_type_tests(TestGroupNormExt, globals(), except_for='cpu')
if __name__ == '__main__':
    run_tests()
