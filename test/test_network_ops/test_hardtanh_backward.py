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
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestHardtanhBackward(TestCase):
    def cpu_op_exec(self, input1, min1, max1):
        w = torch.ones_like(input1)
        input1.requires_grad_(True)
        output = torch.nn.functional.hardtanh(input1, min1, max1)
        output.backward(w)
        res = input1.grad
        return output, res

    def npu_op_exec(self, input1, min1, max1):
        w = torch.ones_like(input1)
        w = w.to("npu")
        input1.requires_grad_(True)
        output = torch.nn.functional.hardtanh(input1, min1, max1)
        output.backward(w)
        res = input1.grad.to("cpu")
        output = output.to("cpu")
        return output, res

    def test_floor_shape_format(self):
        shape_format = [
            [[np.float32, 0, (64, 10)], 0, 1],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 2)
            cpu_output, cpu_res = self.cpu_op_exec(cpu_input, item[1], item[2])
            npu_output, npu_res = self.npu_op_exec(npu_input, item[1], item[2])
            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())
            self.assertRtolEqual(cpu_res, npu_res)


if __name__ == "__main__":
    run_tests()
