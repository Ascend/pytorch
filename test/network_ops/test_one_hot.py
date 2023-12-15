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


class TestOneHot(TestCase):

    def generate_single_data(self, low, high):
        npu_input1 = torch.arange(low, high)
        return npu_input1

    def cpu_op_exec(self, input1, num_classes):
        output = torch.nn.functional.one_hot(input1, num_classes=num_classes)
        output = output.to(torch.int32)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, num_classes):
        input1 = input1.to(torch.int32)
        input1 = input1.to("npu")
        output = torch.nn.functional.one_hot(input1, num_classes=num_classes)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_one_hot_1(self, device="npu"):
        input1 = self.generate_single_data(0, 5)
        cpu_output = self.cpu_op_exec(input1, 5)
        npu_output = self.npu_op_exec(input1, 5)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_one_hot_2(self, device="npu"):
        input1 = self.generate_single_data(0, 5)
        npu_output = self.npu_op_exec(input1, -1)
        cpu_output = self.cpu_op_exec(input1, -1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_one_hot_3(self, device="npu"):
        input1 = self.generate_single_data(0, 5)
        npu_output = self.npu_op_exec(input1, 6)
        cpu_output = self.cpu_op_exec(input1, 6)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_one_hot_4(self, device="npu"):
        input1 = self.generate_single_data(0, 10)
        cpu_output = self.cpu_op_exec(input1, 10)
        npu_output = self.npu_op_exec(input1, 10)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_one_hot_5(self, device="npu"):
        input1 = self.generate_single_data(0, 10)
        cpu_output = self.cpu_op_exec(input1, -1)
        npu_output = self.npu_op_exec(input1, -1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_one_hot_6(self, device="npu"):
        input1 = self.generate_single_data(0, 10)
        cpu_output = self.cpu_op_exec(input1, 12)
        npu_output = self.npu_op_exec(input1, 12)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_one_hot_aicpu_int64(self):
        input1 = torch.randint(0, 4, size=(4, 64, 64, 64)).npu()
        cpu_output = self.cpu_op_exec(input1.cpu(), 4)
        npu_output = self.npu_op_exec(input1, 4)

        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
