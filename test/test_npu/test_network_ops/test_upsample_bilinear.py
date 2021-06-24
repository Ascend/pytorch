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
import torch.nn.functional as F
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestUpsampleBilinear(TestCase):
    def cpu_op_exec(self, input1, size):
        input1.requires_grad_(True)
        output = F.interpolate(input1, size, mode="bilinear")
        w = torch.ones_like(output)
        output.backward(w)
        res = input1.grad
        res = res.numpy()
        output = output.detach().numpy()
        return output, res

    def npu_op_exec(self, input1, size):
        input1.requires_grad_(True)
        output = F.interpolate(input1, size, mode="bilinear")
        w = torch.ones_like(output)
        w = w.to("npu")
        output.backward(w)
        output = output.to("cpu").detach().numpy()
        res = input1.grad
        res = res.to("cpu").numpy()
        return output, res

    def upsample_bilinear_backward_result(self, shape_format):
        for item in shape_format:
            input_cpu, input_npu = create_common_tensor(item[0], 0, 100)
            if input_cpu.dtype == torch.float16:
                input_cpu = input_cpu.to(torch.float32)

            cpu_output, cpu_grad = self.cpu_op_exec(input_cpu, item[1])
            npu_output, npu_grad = self.npu_op_exec(input_npu, item[1])
            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_grad = cpu_grad.astype(npu_grad.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)

    def test_upsample_bilinear_backward_shape_format_aicpu(self, device):
        format_list = [0, 3]
        size_list = [[10001, 2]]
        shape_format = [[[np.float32, i, [2, 10020, 100, 1]], s] for i in format_list for s in size_list]

        self.upsample_bilinear_backward_result(shape_format)
        
    def test_upsample_bilinear_backward_shape_format_aicore(self, device):
        format_list = [0, 3]
        size_list = [[100, 2]]
        shape_format = [[[np.float32, i, [2, 10020, 100, 1]], s] for i in format_list for s in size_list]

        self.upsample_bilinear_backward_result(shape_format)

instantiate_device_type_tests(TestUpsampleBilinear, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
