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
import numpy as np

from torch_npu.testing.common_utils import TestCase, run_tests
from torch_npu.testing.common_device_type import instantiate_device_type_tests
from torch_npu.testing.util_test import create_common_tensor

class TestTransepose(TestCase):
    def test_transepose(self, device):
        def cpu_op_exec(input1, perm):
            output = input1.permute(perm)
            output = output.numpy()
            return output

        def npu_op_exec(input1, perm):
            output = input1.npu_transpose(perm)
            output = output.to("cpu")
            output = output.numpy()
            return output

        shape_format = [
                        [[np.float32, 0, (5, 3, 6, 4)], [1, 0, 2, 3]],
                        [[np.float16, 0, (5, 3, 6, 4)], [0, 3, 2, 1]],
                        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_output = cpu_op_exec(cpu_input1, item[1])
            npu_output = npu_op_exec(npu_input1, item[1])
            
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestTransepose, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()