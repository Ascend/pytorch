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
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestUpsampleBilinear2d(TestCase):

    def cpu_op_exec(self, inputs, shapes):
        output = torch._C._nn.upsample_bilinear2d(inputs, shapes, True, 0, 0)
        output = output.numpy()
        return output

    def npu_op_exec(self, inputs, shapes):
        output = torch._C._nn.upsample_bilinear2d(inputs, shapes, True, 0, 0)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_UpsampleBilinear2d_common_shape_format(self, device):
        shape_format = [
            [[np.float32, -1, (1, 1, 3000, 3000)], (2500, 2500)],
            [[np.float32, -1, (4, 3, 1, 5)], (2, 2)],
            [[np.float32, -1, (2, 3, 2, 1)], (3, 3)],
            [[np.float32, -1, (1, 4, 2, 2)], (4, 4)],
            [[np.float16, -1, (4, 10, 16, 14)], (5, 5)],
            [[np.float16, -1, (8, 8, 8, 8)], (1, 2)],
            [[np.float16, -1, (10, 4, 3, 2)], (2, 4)]
        ]
        for item in shape_format:
            cpu_inputs, npu_inputs = create_common_tensor(item[0], 1, 100)
            if cpu_inputs.dtype == torch.float16:
                cpu_inputs = cpu_inputs.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_inputs, item[1])
            npu_output = self.npu_op_exec(npu_inputs, item[1])
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestUpsampleBilinear2d, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()