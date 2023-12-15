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

import numpy as np
import torch
import torch.nn as nn

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestDropOutDoMask(TestCase):

    def get_npu_tensor(self, item, minValue, maxValue):
        dtype = item[0]
        npu_format = item[1]
        shape = item[2]
        input1 = np.random.uniform(minValue, maxValue, shape).astype(dtype)
        npu_input = torch.from_numpy(input1).npu()
        if npu_format != -1:
            npu_input = torch_npu.npu_format_cast(npu_input, npu_format)
        return npu_input

    def cpu_op_exec(self, input1):
        out = torch.nn.Dropout(0.5)(input1)
        out = out.numpy()
        return out

    def npu_op_exec(self, input1):
        out = torch.nn.Dropout(0.5)(input1)
        out = out.to("cpu")
        out = out.numpy()
        return out

    def npu_set_dropout_seed_op_exec(self, input1, p, seed):
        torch.manual_seed(seed)
        m = nn.Dropout(p).npu()
        out = m(input1)
        out = out.to("cpu")
        out = out.numpy()
        return out

    def dropout_set_dropout_seed_list_exec(self, list1, p, seed):
        for item in list1:
            npu_input1 = self.get_npu_tensor(item, 0, 100)
            npu_expect_output = self.npu_set_dropout_seed_op_exec(npu_input1, p, seed)
            npu_output = self.npu_set_dropout_seed_op_exec(npu_input1, p, seed)
            self.assertRtolEqual(npu_expect_output, npu_output)

    def dropout_list_exec(self, list1):
        epsilon = 1e-3
        for item in list1:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            # 该算子随机结果的比较方式
            for a, b in zip(cpu_output.flatten(), npu_output.flatten()):
                if abs(a) > 0 and abs(b) > 0 and abs(a - b) > epsilon:
                    print(f'input = {item}, ERROR!')
                    break
            else:
                print(f'input = {item}, Successfully!')

    def test_op_shape_format_fp16(self, device="npu"):
        format_list = [-1]
        shape_list = [1, (256, 1280), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        self.dropout_list_exec(shape_format)

    def test_op_shape_format_fp32(self, device="npu"):
        format_list = [-1]
        shape_list = [1, (256, 1280), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        self.dropout_list_exec(shape_format)

    def test_set_dropout_seed_fp16(self):
        p = 0.5
        seed = 666
        format_list = [-1]
        shape_list = [1, (256, 1280), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float16, i, j]
            for i in format_list
            for j in shape_list
        ]
        self.dropout_set_dropout_seed_list_exec(shape_format, p, seed)

    def test_set_dropout_seed_fp32(self):
        p = 0.5
        seed = 666
        format_list = [-1]
        shape_list = [1, (256, 1280), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float32, i, j]
            for i in format_list
            for j in shape_list
        ]
        self.dropout_set_dropout_seed_list_exec(shape_format, p, seed)


if __name__ == "__main__":
    run_tests()
