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
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestMarginRankingLoss(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input3 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)

        return npu_input1, npu_input2, npu_input3

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)

        return npu_input1

    def generate_scalar(self, min_d, max_d):
        scalar = np.random.uniform(min_d, max_d)
        return scalar

    def cpu_op_exec(self, input1, input2, target, margin, reduction):
        input1 = input1.float()
        input2 = input2.float()
        target = target.float()
        output = torch.nn.functional.margin_ranking_loss(input1,
                                                         input2,
                                                         target,
                                                         margin,
                                                         reduction=reduction)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, target, margin, reduction):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        target = target.to("npu")

        output = torch.nn.functional.margin_ranking_loss(input1,
                                                         input2,
                                                         target,
                                                         margin,
                                                         reduction=reduction)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_float16_none(self, device):
        npu_input1, npu_input2, npu_target = self.generate_data(
            0, 100, (5, 3), np.float16)
        npu_margin = self.generate_scalar(-100.0, 100.0)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, npu_target,
                                      npu_margin, 'none')
        npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_target,
                                      npu_margin, 'none')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_float32_none(self, device):
        npu_input1, npu_input2, npu_target = self.generate_data(
            0, 100, (5, 3), np.float32)
        npu_margin = self.generate_scalar(-100.0, 100.0)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, npu_target,
                                      npu_margin, 'none')
        npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_target,
                                      npu_margin, 'none')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_float16_mean(self, device):
        npu_input1, npu_input2, npu_target = self.generate_data(
            0, 100, (5, 3), np.float16)
        npu_margin = self.generate_scalar(-100.0, 100.0)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, npu_target,
                                      npu_margin, 'mean')
        npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_target,
                                      npu_margin, 'mean')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_float32_mean(self, device):
        npu_input1, npu_input2, npu_target = self.generate_data(
            0, 100, (5, 3), np.float32)
        npu_margin = self.generate_scalar(-100.0, 100.0)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, npu_target,
                                      npu_margin, 'mean')
        npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_target,
                                      npu_margin, 'mean')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_float16_sum(self, device):
        npu_input1, npu_input2, npu_target = self.generate_data(
            0, 100, (5, 3), np.float16)
        npu_margin = self.generate_scalar(-100.0, 100.0)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, npu_target,
                                      npu_margin, 'sum')
        npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_target,
                                      npu_margin, 'sum')
        self.assertRtolEqual(cpu_output, npu_output)

    def test_float32_sum(self, device):
        npu_input1, npu_input2, npu_target = self.generate_data(
            0, 100, (5, 3), np.float32)
        npu_margin = self.generate_scalar(-100.0, 100.0)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, npu_target,
                                      npu_margin, 'sum')
        npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_target,
                                      npu_margin, 'sum')
        self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestMarginRankingLoss,
                              globals(),
                              except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:0")
    run_tests()
