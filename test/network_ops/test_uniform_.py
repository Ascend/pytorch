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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestUniform(TestCase):
    def test_uniform(self):
        shape_format = [
            [(20, 300), -100, 100, torch.float32],
            [(20, 300), -100, 100, torch.float16]
        ]

        for item in shape_format:
            input1 = torch.zeros(item[0], dtype=item[3]).npu()
            input1.uniform_(item[1], item[2])
            self.assertTrue(item[1] <= input1.min())
            self.assertTrue(item[2] >= input1.max())

    def test_uniform_trans(self):
        shape_format = [
            [(20, 300), -100, 100, torch.float32],
        ]

        for item in shape_format:
            input1 = torch.zeros(item[0], dtype=item[3]).npu()
            input1 = torch_npu.npu_format_cast(input1, 3)
            input1.uniform_(item[1], item[2])
            self.assertTrue(item[1] <= input1.min())
            self.assertTrue(item[2] >= input1.max())

    def test_uniform_seed(self):
        torch.manual_seed(123)
        input1 = torch.rand(2, 3, 4).npu()
        input1.uniform_(2, 10)
        torch.manual_seed(123)
        input2 = torch.rand(2, 3, 4).npu()
        input2.uniform_(2, 10)
        self.assertRtolEqual(input1.cpu(), input2.cpu())

    def test_uniform_seed_fp16(self):
        torch.manual_seed(13)
        input1 = torch.rand(2, 5, 4).half().npu()
        input1.uniform_(10, 100)
        torch.manual_seed(13)
        input2 = torch.rand(2, 5, 4).half().npu()
        input2.uniform_(10, 100)
        self.assertRtolEqual(input1.cpu(), input2.cpu())


if __name__ == "__main__":
    run_tests()
