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


class TestSplit(TestCase):
    def cpu_op_exec(self, input1, sections, dim):
        output = torch.split(input1, sections, dim)
        output = list(output)
        for i in range(len(output)):
            output[i] = output[i].numpy()
        return output

    def npu_op_exec(self, input1, sections, dim):
        output = torch.split(input1, sections, dim)
        output = list(output)
        for i in range(len(output)):
            output[i] = output[i].to("cpu").numpy()
        return output

    def split_result(self, shape_format):
        for item in shape_format:
            dim = np.random.randint(0, len(item[2]))
            size1 = int(item[2][dim] / 2)
            size2 = int(item[2][dim] - size1)
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)

            cpu_output = self.cpu_op_exec(cpu_input1, [size1, size2], dim)
            npu_output = self.npu_op_exec(npu_input1, [size1, size2], dim)

            for i in range(len(cpu_output)):
                self.assertRtolEqual(cpu_output[i], npu_output[i])

    def test_split_shape_format_fp16_1d(self, device):
        format_list = [0, 3]
        shape_format = [[np.float16, i, [18]] for i in format_list]
        self.split_result(shape_format)

    def test_split_shape_format_fp16_2d(self, device):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [5, 256]] for i in format_list]
        self.split_result(shape_format)

    def test_split_shape_format_fp16_3d(self, device):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [32, 3, 3]] for i in format_list]
        self.split_result(shape_format)

    def test_split_shape_format_fp16_4d(self, device):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [64, 112, 7, 7]] for i in format_list]
        self.split_result(shape_format)

    def test_split_shape_format_fp32_1d(self, device):
        format_list = [0, 3]
        shape_format = [[np.float32, i, [18]] for i in format_list]
        self.split_result(shape_format)

    def test_split_shape_format_fp32_2d(self, device):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [5, 256]] for i in format_list]
        self.split_result(shape_format)

    def test_split_shape_format_fp32_3d(self, device):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [32, 3, 3]] for i in format_list]
        self.split_result(shape_format)

    def test_split_shape_format_fp32_4d(self, device):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [64, 112, 7, 7]] for i in format_list]
        self.split_result(shape_format)


instantiate_device_type_tests(TestSplit, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
