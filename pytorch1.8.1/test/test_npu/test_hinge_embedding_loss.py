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
import torch.nn as nn
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestHingeEmbeddingLoss(TestCase):
    def generate_data(self, min_val, max_val, shape, dtype):
        x = np.random.uniform(min_val, max_val, shape).astype(dtype)
        x = torch.from_numpy(x)
        return x

    def op_exec_cpu(self, input1, target, margin, reduction):
        cpu_output = torch.hinge_embedding_loss(input1, target, margin, reduction)
        cpu_output = cpu_output.numpy()
        return cpu_output

    def op_exec_npu(self, input1, target, margin, reduction):
        input1 = input1.to("npu")
        target = target.to("npu")
        npu_output = torch.hinge_embedding_loss(input1, target, margin, reduction)
        npu_output = npu_output.to("cpu")
        npu_output = npu_output.numpy()
        return npu_output

    def test_hinge_embedding_loss_float32_mean(self, device):
        input1 = self.generate_data(0, 2, (5, 3), np.float32)
        target = self.generate_data(0, 2, (5, 3), np.int32)
        target[target < 1] = -1
        cpu_input1 = copy.deepcopy(input1)
        cpu_target = copy.deepcopy(target)
        margin = 1.0      
        reduction = 1
        cpu_output = self.op_exec_cpu(cpu_input1, cpu_target, margin, reduction)
        npu_output = self.op_exec_npu(input1, target, margin, reduction)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_hinge_embedding_loss_float32_none(self, device):
        input1 = self.generate_data(0, 2, (5, 3), np.float32)
        target = self.generate_data(0, 2, (5, 3), np.int32)
        target[target < 1] = -1
        cpu_input1 = copy.deepcopy(input1)
        cpu_target = copy.deepcopy(target)
        margin = 1.0      
        reduction = 0
        cpu_output = self.op_exec_cpu(cpu_input1, cpu_target, margin, reduction)
        npu_output = self.op_exec_npu(input1, target, margin, reduction)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_hinge_embedding_loss_float32_sum(self, device):
        input1 = self.generate_data(0, 2, (5, 3), np.float32)
        target = self.generate_data(0, 2, (5, 3), np.int32)
        target[target < 1] = -1
        cpu_input1 = copy.deepcopy(input1)
        cpu_target = copy.deepcopy(target)
        margin = 1.2      
        reduction = 2
        cpu_output = self.op_exec_cpu(cpu_input1, cpu_target, margin, reduction)
        npu_output = self.op_exec_npu(input1, target, margin, reduction)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_hinge_embedding_loss_float16_mean(self, device):
        input1 = self.generate_data(-2, 2, (5, 3), np.float16)
        target = self.generate_data(0, 2, (5, 3), np.int32)
        target[target < 1] = -1
        cpu_input1 = copy.deepcopy(input1)
        cpu_input1 = cpu_input1.float()
        cpu_target = copy.deepcopy(target)
        margin = 1.0      
        reduction = 1
        cpu_output = self.op_exec_cpu(cpu_input1, cpu_target, margin, reduction).astype(np.float16)
        npu_output = self.op_exec_npu(input1, target, margin, reduction)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_hinge_embedding_loss_int32_sum(self, device):
        input1 = self.generate_data(-2, 2, (5, 3), np.int32)
        target = self.generate_data(0, 2, (5, 3), np.int32)
        target[target < 1] = -1
        cpu_input1 = copy.deepcopy(input1)
        cpu_target = copy.deepcopy(target)
        margin = 1.2      
        reduction = 2
        cpu_output = self.op_exec_cpu(cpu_input1, cpu_target, margin, reduction).astype(np.int32)
        npu_output = self.op_exec_npu(input1, target, margin, reduction)
        self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestHingeEmbeddingLoss, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:1")
    run_tests()
