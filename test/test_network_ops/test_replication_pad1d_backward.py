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
from torch_npu.testing.common_utils import create_common_tensor


class TestReplicationPad1dBackward(TestCase):
    def cpu_op_exec(self, input1, pad):
        m = torch.nn.ReplicationPad1d(pad)
        input1.requires_grad = True
        output = m(input1)
        output.backward(torch.ones_like(output))
        cpu_grad = input1.grad
        output = output.detach().numpy()
        cpu_grad = cpu_grad.detach().numpy()
        return output, cpu_grad

    def npu_op_exec(self, input1, pad):
        m = torch.nn.ReplicationPad1d(pad).npu()
        input1.requires_grad = True
        output = m(input1)
        output.backward(torch.ones_like(output))
        output = output.to("cpu")
        npu_grad = input1.grad
        npu_grad = npu_grad.to("cpu")
        output = output.detach().numpy()
        npu_grad = npu_grad.detach().numpy()
        return output, npu_grad

    def test_replication_pad1d_backward_shape_format_fp16(self):
        shape_format = [
            [[np.float16, 0, (16, 16, 4)], [3, 1]],
            [[np.float16, 2, (1, 2, 4)], [3, 1]]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input1, item[1])
            cpu_output = cpu_output.astype(np.float16)
            cpu_grad = cpu_grad.astype(np.float16)
            npu_output, npu_grad = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)

if __name__ == "__main__":
    run_tests()
