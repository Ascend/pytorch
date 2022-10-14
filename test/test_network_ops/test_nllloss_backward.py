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
from torch_npu.testing.decorator import graph_mode
from torch_npu.testing.common_utils import create_common_tensor


class TestNlllossbackward(TestCase):
    def cpu_op_exec_new(self, input1, target, reduction, ignore_index):
        if not ignore_index:
            ignore_index = -100 # 默认值
        input1.requires_grad_(True)
        output = torch.nn.functional.nll_loss(input1, target, reduction=reduction, ignore_index=ignore_index)
        input_cpu = output.detach().numpy()
        output.backward(torch.ones_like(output))
        res = input1.grad
        res = res.numpy()
        return input_cpu, res

    def npu_op_exec_new(self, input1, target, reduction, ignore_index):
        if not ignore_index:
            ignore_index = -100 # 默认值
        target = target.to(torch.int32)
        target = target.to("npu")
        input1.requires_grad_(True)
        output = torch.nn.functional.nll_loss(input1, target, reduction=reduction, ignore_index=ignore_index)
        output.backward(torch.ones_like(output))
        input_npu = output.to("cpu")
        input_npu = input_npu.detach().numpy()
        res = input1.grad.to("cpu")
        res = res.numpy()
        return input_npu, res

    @graph_mode
    def test_nllloss_shape_format_fp32(self):
        # 当前仅支持设置正数, 若np.sum(ignore_index == np_target) == 0,则ignore_index设置任意数值不影响
        ignore_index = 1 
        for reduction in ['mean', 'none', 'sum']:
            shape_format = [
                [[np.float32, 0, [256, 100]], [np.int32, 0, [256]], reduction, None],
                [[np.float32, 3, [256, 100]], [np.int32, 0, [256]], reduction, ignore_index],
                [[np.float32, 0, [4800, 3003]], [np.int32, 0, [4800]], reduction, ignore_index],
                [[np.float32, 3, [4800, 3003]], [np.int32, 0, [4800]], reduction, ignore_index],
                [[np.float32, 0, [4800, 3003]], [np.int32, 0, [4800]], reduction, None],
                ]
            for item in shape_format:
                np_target = np.random.randint(0, item[0][2][1], (item[1][2])).astype(np.long)
                target = torch.from_numpy(np_target)
                cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
                cpu_input, cpu_output = self.cpu_op_exec_new(cpu_input1, target, item[2], item[3])
                npu_input, npu_output = self.npu_op_exec_new(npu_input1, target, item[2], item[3])
                self.assertRtolEqual(cpu_input, npu_input)
                self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_nllloss_shape_format_fp16(self):
        # 当前仅支持设置正数, 若np.sum(ignore_index == np_target) == 0,则ignore_index设置任意数值不影响
        ignore_index = 1
        for reduction in ['mean', 'none', 'sum']:
            shape_format = [
                [[np.float16, 0, [256, 100]], [np.int32, 0, [256]], reduction, ignore_index],
                [[np.float16, 3, [256, 100]], [np.int32, 0, [256]], reduction, ignore_index],
                [[np.float16, 0, [4800, 3003]], [np.int32, 0, [4800]], reduction, ignore_index],
                [[np.float16, 3, [4800, 3003]], [np.int32, 0, [4800]], reduction, ignore_index],
                [[np.float16, 0, [4800, 3003]], [np.int32, 0, [4800]], reduction, None],
                ]
            for item in shape_format:
                np_target = np.random.uniform(0, item[0][2][1], (item[1][2])).astype(np.long)
                target = torch.from_numpy(np_target)
                cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input, cpu_output = self.cpu_op_exec_new(cpu_input1, target, item[2], item[3])
                npu_input, npu_output = self.npu_op_exec_new(npu_input1, target, item[2], item[3])
                cpu_input = cpu_input.astype(np.float16)
                cpu_output = cpu_output.astype(np.float16)
                self.assertRtolEqual(cpu_input, npu_input)
                self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
