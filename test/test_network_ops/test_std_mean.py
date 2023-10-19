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

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestStdMean(TestCase):
    def cpu_op_mean_exec(self, input1, unbiased=True):
        output = torch.std_mean(input1, unbiased=unbiased)
        result = []
        result.append(output[0].numpy())
        result.append(output[1].numpy())
        return result

    def npu_op_mean_exec(self, input1, unbiased=True):
        output = torch.std_mean(input1, unbiased=unbiased)
        result = []
        result.append(output[0].to("cpu").numpy())
        result.append(output[1].to("cpu").numpy())
        return result

    def cpu_op_dim_mean_exec(self, input1, dim, unbiased=True, keepdim=False):
        output = torch.std_mean(input1, dim, unbiased=unbiased, keepdim=keepdim)
        result = []
        result.append(output[0].numpy())
        result.append(output[1].numpy())
        return result

    def npu_op_dim_mean_exec(self, input1, dim, unbiased=True, keepdim=False):
        output = torch.std_mean(input1, dim, unbiased=unbiased, keepdim=keepdim)
        result = []
        result.append(output[0].to("cpu").numpy())
        result.append(output[1].to("cpu").numpy())
        return result

    def test_std_mean_shape_format_fp16(self):
        format_list = [0, 3, 4]
        shape_list = [[1024], [32, 1024], [32, 8, 1024]]
        unbiased_list = [True, False]
        shape_format = [
            [np.float16, i, j, k] for i in format_list for j in shape_list for k in unbiased_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output1 = self.cpu_op_mean_exec(cpu_input1, item[3])
            cpu_output1[0] = cpu_output1[0].astype(np.float16)
            cpu_output1[1] = cpu_output1[1].astype(np.float16)
            npu_output1 = self.npu_op_mean_exec(npu_input1, item[3])
            self.assertRtolEqual(cpu_output1[0], npu_output1[0])
            self.assertRtolEqual(cpu_output1[1], npu_output1[1])

    def test_std_mean_shape_format_fp32(self):
        format_list = [0, 3, 4]
        shape_list = [[1024], [32, 1024], [32, 8, 1024]]
        unbiased_list = [True, False]
        shape_format = [
            [np.float32, i, j, k] for i in format_list for j in shape_list for k in unbiased_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output1 = self.cpu_op_mean_exec(cpu_input1, item[3])
            npu_output1 = self.npu_op_mean_exec(npu_input1, item[3])
            self.assertRtolEqual(cpu_output1[0], npu_output1[0])
            self.assertRtolEqual(cpu_output1[1], npu_output1[1])

    def test_std_dim_mean_shape_format_fp16(self):
        format_list = [0, 3, 4]
        shape_list = [[1024], [32, 1024], [32, 8, 1024]]
        dim_list = [0]
        unbiased_list = [True, False]
        keepdim_list = [True, False]
        shape_format = [
            [np.float16, i, j, k, l, m] for i in format_list for j in shape_list
            for k in dim_list for l in unbiased_list for m in keepdim_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output1 = self.npu_op_dim_mean_exec(cpu_input1, item[3], item[4], item[5])
            cpu_output1[0] = cpu_output1[0].astype(np.float16)
            cpu_output1[1] = cpu_output1[1].astype(np.float16)
            npu_output1 = self.npu_op_dim_mean_exec(npu_input1, item[3], item[4], item[5])
            self.assertRtolEqual(cpu_output1[0], npu_output1[0])
            self.assertRtolEqual(cpu_output1[1], npu_output1[1])

    def test_std_dim_mean_shape_format_fp32(self):
        format_list = [0, 3, 4]
        shape_list = [[1024], [32, 1024], [32, 8, 1024]]
        dim_list = [0]
        unbiased_list = [True, False]
        keepdim_list = [True, False]
        shape_format = [
            [np.float32, i, j, k, l, m] for i in format_list for j in shape_list
            for k in dim_list for l in unbiased_list for m in keepdim_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output1 = self.npu_op_dim_mean_exec(cpu_input1, item[3], item[4], item[5])
            npu_output1 = self.npu_op_dim_mean_exec(npu_input1, item[3], item[4], item[5])
            self.assertRtolEqual(cpu_output1[0], npu_output1[0])
            self.assertRtolEqual(cpu_output1[1], npu_output1[1])

    def test_std_dim_mean_name_fp32(self):
        shape = (1024, 8, 32)
        cpu_input = torch.rand(shape, dtype=torch.float32, names=('N', 'C', 'H'))
        npu_input = cpu_input.npu()
        dim = np.random.choice(['N', 'C', 'H'])
        cpu_output = torch.std_mean(cpu_input, dim=dim)
        npu_output = torch.std_mean(npu_input, dim=dim)
        self.assertRtolEqual(cpu_output[0].numpy(), npu_output[0].cpu().numpy())
        self.assertRtolEqual(cpu_output[1].numpy(), npu_output[1].cpu().numpy())

    def test_std_dim_mean_name_fp16(self):
        shape = (1024, 8, 32)
        cpu_input = torch.rand(shape, dtype=torch.float32)
        npu_input = cpu_input.to(torch.float16).npu()
        cpu_input.names = ['N', 'C', 'H']
        npu_input.names = ['N', 'C', 'H']
        dim = np.random.choice(['N', 'C', 'H'])
        cpu_output = torch.std_mean(cpu_input, dim=dim)
        npu_output = torch.std_mean(npu_input, dim=dim)
        self.assertRtolEqual(cpu_output[0].to(torch.float16).numpy(), npu_output[0].cpu().numpy())
        self.assertRtolEqual(cpu_output[1].to(torch.float16).numpy(), npu_output[1].cpu().numpy())


if __name__ == "__main__":
    run_tests()
