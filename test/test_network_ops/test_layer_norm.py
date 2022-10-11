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
import numpy as np
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import skipIfNotRegistered
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLayerNorm(TestCase):
    @skipIfNotRegistered("LayerNorm", "Skipping as LayerNorm is not registered")
    def test_c10_layer_norm(self, device="npu"):
        # test that we can call c10 ops and they return a reasonable result
        X = torch.rand(5, 5, dtype=torch.float, device="cpu")
        X = X.to("npu")
        weight = torch.rand(*X.size()[1:], dtype=torch.float, device="cpu")
        weight = weight.to("npu")
        bias = torch.rand(*X.size()[1:], dtype=torch.float, device="cpu")
        bias = bias.to("npu")
        epsilon = 1e-4

        expected_norm = torch.nn.functional.layer_norm(
            X, X.size()[1:], weight=weight, bias=bias, eps=epsilon)
        expected_norm_cpu = torch.nn.functional.layer_norm(
            X.cpu(), X.size()[1:], weight=weight.cpu(), bias=bias.cpu(), eps=epsilon)
        self.assertRtolEqual(expected_norm.cpu().numpy(), expected_norm_cpu.numpy())

        actual_norm, actual_mean, actual_stdev = \
            torch.ops._caffe2.LayerNorm(torch.tensor(X.cpu()), torch.tensor(
                weight.cpu()), torch.tensor(bias.cpu()), 1, epsilon, True)
        self.assertRtolEqual(expected_norm.cpu().numpy(), actual_norm.numpy())

    def cpu_op_exec(self, input1):
        m = nn.LayerNorm(input1.size()[1:])
        output = m(input1)
        return output

    def npu_op_exec(self, input1):
        m = nn.LayerNorm(input1.size()[1:]).npu()
        output = m(input1)
        output = output.to("cpu")
        return output

    def test_layer_norm_shape_format(self, device="npu"):
        shape_format = [
                [np.float32, 0, (64, 10)],
                [np.float32, 0, (256, 2048, 7, 7)],
                [np.float32, 0, (32, 1, 3, 3)],
                [np.float32, 0, (10, 128)],
                [np.float32, 2, (46, 16)],
                [np.float32, 3, (2, 2, 2)],
                [np.float32, 29, (3, 4, 5, 6)]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())

    def test_layer_norm_float16_format(self, device="npu"):
        shape_format = [
                [np.float16, 0, (64, 10)],
                [np.float16, 0, (256, 2048, 7, 7)],
                [np.float16, 0, (32, 1, 3, 3)],
                [np.float16, 0, (10, 128)],
                [np.float16, 2, (46, 16)],
                [np.float16, 3, (2, 2, 2)],
                [np.float16, 29, (3, 4, 5, 6)] 
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 10)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.to(torch.float16)
            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())


if __name__ == "__main__":
    run_tests()
