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
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestFastGelu(TestCase):
    def npu_op_exec(self, input1):
        input1.requires_grad = True
        output = torch_npu.fast_gelu(input1)
        output.backward(torch.ones_like(output))
        output_grad = input1.grad
        output_grad = output_grad.to("cpu")
        output_grad = output_grad.detach().numpy()
        output = output.cpu().detach().numpy()
        return output_grad, output

    def test_fastgelu(self, device="npu"):
        input1 = torch.tensor([1., 2., 3., 4.]).npu()
        exoutputgrad = torch.tensor([1.0677795, 1.0738151, 1.0245483, 1.0064018])
        exoutput = torch.tensor([0.8458, 1.9357, 2.9819, 3.9956])
        outputgrad, output = self.npu_op_exec(input1)
        self.assertRtolEqual(exoutputgrad.numpy(), outputgrad)
        self.assertRtolEqual(exoutput.numpy(), output)


if __name__ == "__main__":
    run_tests()
