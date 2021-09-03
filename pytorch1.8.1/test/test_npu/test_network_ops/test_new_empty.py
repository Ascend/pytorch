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
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestNewEmpty(TestCase):
    def cpu_op_exec(self, input1, size):
        output = input1.new_empty(size)
        output = output.numpy()
        return output
    
    def npu_op_exec(self, input1, size):
        output = input1.new_empty(size)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def test_new_empty_shape_format(self, device):
        shape_format = [
                [[np.float32, 0, (4, 3)], (2, 2)],
                [[np.float32, 0, (2, 3, 7)], (1, 2)]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertEqual(cpu_output.shape, npu_output.shape)


instantiate_device_type_tests(TestNewEmpty, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
    

