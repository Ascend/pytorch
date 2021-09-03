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
import torch.nn.functional as F
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestThresholdGradV2DBackward(TestCase):

    def cpu_op_exec(self, input1, val_0, val_1):
        input1.requires_grad = True
        input1_res = F.threshold(input1, val_0, val_1)
        input1_res.backward(torch.ones_like(input1_res))
        output = input1.grad.numpy()
        return output

    def npu_op_exec(self, input1, val_0, val_1):
        input1.requires_grad = True
        input1_res = F.threshold(input1, val_0, val_1)
        input1_res.backward(torch.ones_like(input1_res))
        output = input1.grad
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_threshold_grad_v2_d_common_shape_format(self, device):
        shape_format = [
            [[np.float32, -1, (4, 3)], 1, 100, 0.1, 1.0],
            [[np.float32, -1, (7, 5, 5)], 21474836, 21474837, -0.001, 1.001],
            [[np.float32, -1, (4, 44, 44)], 3450,34020, 3154, -2200],
            [[np.float32, -1, (65500,3,3)], -214748, -214746, -134, 0.001],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[1], item[2])
            cpu_output = self.cpu_op_exec(cpu_input1, item[3], item[4])
            npu_output = self.npu_op_exec(npu_input1, item[3], item[4])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_threshold_grad_v2_d_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input1, val_0, val_1):
            input1 = input1.to(torch.float32)
            input1.requires_grad = True
            input1_res = F.threshold(input1, val_0, val_1)
            input1_res.backward(torch.ones_like(input1_res))
            output = input1.grad.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, -1, (4, 3)], 1, 100, 0.1, 1.0],
            [[np.float16, -1, (7, 5, 5)], 21474836, 21474837, -0.001, 1.001],
            [[np.float16, -1, (4, 44, 44)], 3450,34020, 3154, -2200],
            [[np.float16, -1, (65500,3,3)], -214748, -214746, -134, 0.001],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[1], item[2])
            cpu_output = cpu_op_exec_fp16(cpu_input1, item[3], item[4])
            npu_output = self.npu_op_exec(npu_input1, item[3], item[4])
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestThresholdGradV2DBackward, globals(), except_for='cpu')

if __name__ == "__main__":
    run_tests()
