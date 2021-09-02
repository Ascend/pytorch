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

class TestSoftMarginLoss(TestCase):
    def generate_data(self,min_d, max_d, shape1, shape2, dtype):
        input1 = np.random.uniform(min_d, max_d, shape1).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        if dtype == np.float16:
            stype = torch.float16
        if dtype == np.float32:
            stype = torch.float32
        npu_input2 = torch.ones(size=shape2, dtype=stype)
        return npu_input1, npu_input2

    def cpu_op_exec_default(self,input1, input2):
        stype=input1.dtype
        if stype==torch.float16:
            input1=input1.float()
            input2=input2.float()
        loss = torch.nn.SoftMarginLoss()
        output=loss(input1, input2)
        if stype==torch.float16:
            output=output.half()
        output = output.numpy()
        return output

    def npu_op_exec_default(self,input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        loss = torch.nn.SoftMarginLoss()
        output = loss(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec(self,input1, input2, reduct):
        stype=input1.dtype
        if stype==torch.float16:
            input1=input1.float()
            input2=input2.float()
        loss = torch.nn.SoftMarginLoss(reduction=reduct)
        output = loss(input1, input2)
        if stype==torch.float16:
            output=output.half()
        output = output.numpy()
        return output

    def npu_op_exec(self,input1, input2, reduct):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        loss = torch.nn.SoftMarginLoss(reduction=reduct)
        output = loss(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output
 
    def test_soft_margin_loss_float16(self, device):
        npu_input1, npu_input2 =self.generate_data(-2, 2, (5, 13, 2, 7, 18, 83, 5, 22), (5, 13, 2, 7, 18, 83, 5, 22), np.float16)
        cpu_output = self.cpu_op_exec_default(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_default(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_float16_mean(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (3, 19, 19, 3, 11, 11, 2), (3, 1, 19, 3, 11, 11, 1), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, "mean")
        npu_output = self.npu_op_exec(npu_input1, npu_input2, "mean")
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_float16_none(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (5, 13, 2, 7, 18, 83, 5, 22), (5, 13, 2, 1, 18, 83, 1, 22), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, "none")
        npu_output = self.npu_op_exec(npu_input1, npu_input2, "none")
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_float16_sum(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (1, 8, 2, 2, 5, 8, 2, 8), 
                                                   (1, 8, 2, 2, 1, 1, 1, 1), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, "sum")
        npu_output = self.npu_op_exec(npu_input1, npu_input2, "sum")
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_float32(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (416, 192, 272), (416, 1, 272), np.float32)
        cpu_output = self.cpu_op_exec_default(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_default(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_float32_mean(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (416, 192, 272), (416, 192, 272), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, "mean")
        npu_output = self.npu_op_exec(npu_input1, npu_input2, "mean")
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_float32_none(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (25, 25, 25), (25, 1, 25), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, "none")
        npu_output = self.npu_op_exec(npu_input1, npu_input2, "none")
        self.assertRtolEqual(cpu_output, npu_output)

    def test_soft_margin_loss_float32_sum(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (148, 110, 148), (148, 1, 148), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, "sum")
        npu_output = self.npu_op_exec(npu_input1, npu_input2, "sum")
        self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestSoftMarginLoss, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
