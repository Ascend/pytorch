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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestHardtanhBackward(TestCase):
    def cpu_op_exec(self, input, min, max):
        w = torch.ones_like(input)
        input.requires_grad_(True)
        output = torch.nn.functional.hardtanh(input, min, max)
        output.backward(w)
        res = input.grad
        return output, res

    def npu_op_exec(self, input, min, max):
        w = torch.ones_like(input)
        w = w.to("npu")
        input.requires_grad_(True)
        output = torch.nn.functional.hardtanh(input, min, max)
        output.backward(w)
        res = input.grad.to("cpu")
        output = output.to("cpu")
        return output, res

    def test_floor_shape_format(self, device):
        shape_format = [
                [[np.float32, 0, (64, 10)], 0, 1],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 2)
            cpu_output, cpu_res = self.cpu_op_exec(cpu_input, item[1], item[2])
            npu_output, npu_res = self.npu_op_exec(npu_input, item[1], item[2])
            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())
            self.assertRtolEqual(cpu_res, npu_res)

instantiate_device_type_tests(TestHardtanhBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()