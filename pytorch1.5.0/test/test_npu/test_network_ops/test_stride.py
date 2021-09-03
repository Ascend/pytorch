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
from common_device_type import instantiate_device_type_tests
from util_test import create_common_tensor
 
class TestStride(TestCase):
    def test_stride_common_shape_format(self, device):
        def op_exec(input):
            output = input.stride()
            output = np.array(output, dtype=np.int32)
            return output
        
        shape_format = [
                [[np.float16, 0, (64)]],
                [[np.float32, 4, (32, 1, 3, 3)]],
                [[np.float32, 3, (10, 128)]]
        ]
        for shape in shape_format:
            cpu_input, npu_input = create_common_tensor(shape[0], -100, 100)
            cpu_output = op_exec(cpu_input)
            npu_output = op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)
    

instantiate_device_type_tests(TestStride, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()