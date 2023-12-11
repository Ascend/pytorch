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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestLinearLayers(TestCase):
    def test_Identity(self):
        m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False).npu()
        input1 = torch.randn(128, 20).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_Linear(self):
        m = nn.Linear(20, 30).npu()
        input1 = torch.randn(128, 20).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_Bilinear(self):
        m = nn.Bilinear(20, 30, 40).npu()
        input1 = torch.randn(128, 20).npu()
        input2 = torch.randn(128, 30).npu()
        output = m(input1, input2)
        self.assertEqual(output is not None, True)


if __name__ == "__main__":
    torch.npu.set_device(0)
    run_tests()
