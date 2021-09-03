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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestViewAs(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = input1.view_as(input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = input1.view_as(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_viewas_shape_format(self, device):
        dtype_list = [np.float16, np.float32, np.int32, np.bool_]
        format_list = [0]
        shape_list = [[8,8], [2,4,8], [2,4,4,2]]
        shape_format = [
            [[i, j, k], [i,j,[4,16]]] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestViewAs, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
