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
 
class TestItem(TestCase):
    def test_item_common_shape_format(self, device):        
        shape_format = [
                [[np.float16, 0, 1]],
                [[np.float16, 0, 5]],
                [[np.float32, 4, 3]],
                [[np.float32, 29, 4]]
        ]
        for shape in shape_format:
            cpu_input, npu_input = create_common_tensor(shape[0], -1, 1)
            cpu_output = np.array([cpu_input[0].item()]).astype(np.float32)
            npu_output = np.array([npu_input[0].item()]).astype(np.float32)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestItem, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()