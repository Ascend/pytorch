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


class TestDropoutLayers(TestCase):
    def test_Dropout(self):
        m = nn.Dropout(p=0.2).npu()
        input1 = torch.randn(20, 16).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_Dropout2d(self):
        m = nn.Dropout2d(p=0.2).npu()
        input1 = torch.randn(20, 16, 32, 32).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_Dropout3d(self):
        m = nn.Dropout3d(p=0.2).npu()
        input1 = torch.randn(20, 16, 4, 32, 32).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_AlphaDropout(self):
        m = nn.AlphaDropout(p=0.2).npu()
        input1 = torch.randn(20, 16).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

if __name__ == "__main__":
    torch.npu.set_device(0)
    run_tests()