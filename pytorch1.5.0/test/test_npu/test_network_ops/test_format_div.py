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

import sys
import torch
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests, formats
from util_test import create_common_tensor, create_dtype_tensor


class TestDiv(TestCase):

    def cpu_op_exec(self, input1, input2):
        output = torch.div(input1, input2)
        return output.numpy()

    def npu_op_exec(self, input1, input2):
        output = torch.div(input1, input2)
        output = output.to("cpu")
        return output.numpy()
        
    @formats(0, 3)
    def test_div_shape_format(self, device, npu_format):
        shape_list = [6]
        shape_format = [
            [np.float16, npu_format, j] for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestDiv, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
