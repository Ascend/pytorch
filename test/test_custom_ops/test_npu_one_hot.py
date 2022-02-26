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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuOneHot(TestCase):

    def create_target_lable(self, num_classes, size):
        label = torch.randint(0, num_classes, size)
        return label

    def cpu_op_exec(self, input1, num_classes, on_value=1, off_value=0):
        output = torch.nn.functional.one_hot(input1, num_classes=num_classes).float()
        output[output == 1] = on_value
        output[output == 0] = off_value
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, num_classes, on_value=1, off_value=0):
        output = torch_npu.npu_one_hot(input1, -1, num_classes, on_value, off_value)
        output = output.cpu().numpy()
        return output

    def test_one_hot_1(self, device="npu"):
        target = self.create_target_lable(10, (64, ))
        cpu_output = self.cpu_op_exec(target, 10, 0.9, 0.1)
        npu_output = self.npu_op_exec(target.npu(), 10, 0.9, 0.1)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()