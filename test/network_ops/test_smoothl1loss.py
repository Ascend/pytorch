# Copyright (c) 2020 Huawei Technologies Co., Ltd
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


class TestSmoothL1loss(TestCase):
    def cpu_op_exec_new(self, input1, target, reduction):
        output = torch.nn.functional.smooth_l1_loss(input1, target, beta=2.0, reduction=reduction)
        return output.numpy()

    def npu_op_exec_new(self, input1, target, reduction):
        target = target.npu()
        output = torch.nn.functional.smooth_l1_loss(input1, target, beta=2.0, reduction=reduction)
        return output.cpu().numpy()

    def test_smoothl1loss_shape_format_fp32(self):
        format_list = [0]
        shape_list = [[256, 10], [256, 1000], [256, 10000],
                      [64, 10, 10], [64, 100, 100], [64, 200, 200],
                      [32, 3, 10, 10], [32, 3, 100, 100], [32, 3, 200, 200]]
        reduction_list = ['none', 'mean', 'sum']
        shape_format = [
            [[np.float32, i, j], [np.float32, 0, j], k] for i in format_list
            for j in shape_list for k in reduction_list
        ]
        for item in shape_format:
            np_target = np.random.uniform(0, 10, (item[1][2])).astype(item[1][0])
            target = torch.from_numpy(np_target)
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec_new(cpu_input1, target, item[2])
            npu_output = self.npu_op_exec_new(npu_input1, target, item[2])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_nllloss_shape_format_fp16(self):
        format_list = [0]
        shape_list = [[256, 10], [256, 1000], [256, 10000],
                      [64, 10, 10], [64, 100, 100], [64, 200, 200],
                      [32, 3, 10, 10], [32, 3, 100, 100], [32, 3, 200, 200]]
        reduction_list = ['none', 'mean']
        shape_format = [
            [[np.float16, i, j], [np.float16, 0, j], k] for i in format_list
            for j in shape_list for k in reduction_list
        ]

        for item in shape_format:
            np_target = np.random.uniform(0, 10, (item[1][2])).astype(item[1][0])
            target = torch.from_numpy(np_target)
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec_new(cpu_input1, target, item[2])
            npu_output = self.npu_op_exec_new(npu_input1, target, item[2])
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
