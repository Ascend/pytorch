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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import graph_mode

class TestDropOutWithAddSoftMax(TestCase):
    def cpu_op_exec(self, x1, x2, alpha, axis):
        dropout = torch.nn.Dropout(p=0)
        add_out = torch.add(x1.float(), x2.float(), alpha=alpha)
        softmax_out = F.softmax(add_out, dim=axis).half()
        output = dropout(softmax_out.float()).half()
        output.backward(torch.ones_like(output).float() * 400)
        x2_grad = x2.grad
        return softmax_out.detach().numpy(), output.detach().numpy(), x2.grad.detach().numpy()

    def npu_op_exec(self, x1, x2, alpha, prod, dim):
        _, softmax_out, output = torch_npu.npu_dropout_with_add_softmax(x2, x1, alpha, prod, dim)
        output.backward(torch.ones_like(output) * 400)
        x2_grad = x2.grad
        return softmax_out.cpu().detach().numpy(), output.cpu().detach().numpy(), x2_grad.cpu().detach().numpy()

    @graph_mode
    def test_dropout_shape_format(self):
        cpu_input1 = torch.rand(96, 12, 384, 384).half()
        cpu_input2 = torch.rand(96, 12, 384, 384).half()
        npu_input1 = cpu_input1.npu()
        npu_input2 = cpu_input2.npu()
        cpu_input1.requires_grad = True
        cpu_input2.requires_grad = True
        npu_input1.requires_grad = True
        npu_input2.requires_grad = True
        alpha = 0.125
        axis = -1
        prod_npu = 0

        cpu_s_output, cpu_output, cpu_grad = self.cpu_op_exec(cpu_input1, cpu_input2, alpha, axis)
        npu_s_output, npu_output, npu_grad = self.npu_op_exec(npu_input1, npu_input2, alpha, prod_npu, axis)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_grad, npu_grad)

if __name__ == "__main__":
    run_tests()
