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

    def test_native_dropout(self):
        for train in [True, False]:
            for p in [0.0, 1.0, 0.5]:
                cpu_x = torch.randn(5, 5)
                x = cpu_x.npu()
                x_ref = cpu_x.npu()
                x.requires_grad = True
                x_ref.requires_grad = True
                grad = torch.rand(5, 5).npu()
                torch.manual_seed(123)
                npu_o = torch.native_dropout(x, p, train)
                output = npu_o[0]
                output.backward(grad)
                torch.manual_seed(123)
                o_ref = torch.dropout(x_ref, p, train)
                o_ref.backward(grad)
                self.assertRtolEqual(output.cpu().detach(), o_ref.cpu().detach())
                self.assertRtolEqual(x.grad.cpu(), x_ref.grad.cpu())
                if p in [0.0, 1.0]:
                    cpu_o = torch.native_dropout(cpu_x, p, train)
                    self.assertRtolEqual(cpu_o[0], npu_o[0].cpu())
                    self.assertRtolEqual(cpu_o[1], npu_o[1].cpu())


if __name__ == "__main__":
    torch.npu.set_device(0)
    run_tests()
