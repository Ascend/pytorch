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


class TestMaxV1Backward(TestCase):
    def op_exec(self, npu_flag, data, dim):
        if npu_flag:
            data = data.to("npu")
        data.requires_grad = True
        if npu_flag:
            outputs, indices = torch_npu.npu_max(data, dim)
        else:
            outputs, indices = torch.max(data, dim)
        outputs.backward(torch.ones_like(outputs))
        gradoutput = data.grad
        out = outputs.detach()
        if npu_flag:
            out = out.cpu()
            gradoutput = gradoutput.cpu()
        return out, gradoutput

    def test_max_v1_backward_fp32(self):
        data = torch.randn(2, 2, 2, 2, dtype=torch.float32)
        npu_data = data.clone()
        cpu_out, cpu_grad_out = self.op_exec(0, data, 2)
        npu_out, npu_grad_out = self.op_exec(1, npu_data, 2)
        self.assertRtolEqual(cpu_grad_out, npu_grad_out)
        self.assertRtolEqual(cpu_out, npu_out)


if __name__ == "__main__":
    run_tests()
