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
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNormalizationFunctions(TestCase):
    def test_batch_norm(self):
        input1 = torch.rand(2, 3)
        running_mean = torch.rand(3)
        running_var = torch.rand(3)
        npu_input = input1.npu()
        npu_running_mean = running_mean.npu()
        npu_running_var = running_var.npu()

        cpu_output = F.batch_norm(input1, running_mean=running_mean, running_var=running_var)
        npu_output = F.batch_norm(npu_input, running_mean=npu_running_mean, running_var=npu_running_var)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_instance_norm(self):
        input1 = torch.randn(20, 100, 40)
        npu_input = input1.npu()

        cpu_output = F.instance_norm(input1)
        npu_output = F.instance_norm(npu_input)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_layer_norm(self):
        input1 = torch.randn(20, 100, 40)
        normalized_shape = (100, 40)
        npu_input = input1.npu()
        cpu_output = F.layer_norm(input1, normalized_shape=normalized_shape)
        npu_output = F.layer_norm(npu_input, normalized_shape=normalized_shape)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_local_response_norm(self):
        input1 = torch.randn(2, 3, 4)
        npu_input = input1.npu()

        cpu_output = F.local_response_norm(input1, size=2)
        npu_output = F.local_response_norm(npu_input, size=2)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_normalize(self):
        input1 = torch.randn(2, 3, 4)
        npu_input = input1.npu()

        cpu_output = F.normalize(input1)
        npu_output = F.normalize(npu_input)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())


if __name__ == "__main__":
    run_tests()
