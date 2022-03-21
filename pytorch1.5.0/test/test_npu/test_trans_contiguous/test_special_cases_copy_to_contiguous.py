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

import os
import torch
import numpy as np

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests

os.environ["COMBINED_ENABLE"] = "1"  # Open combined-view cases optimization

class TestSpecialCasesCopyToContiguous(TestCase):
    def test_expand_copy_to_slice_discontiguous_tensor(self, device):
        dtype_list = [np.bool, np.int8, np.int16, np.float16, np.float32, np.int32, np.int64]
        index_list = [3, 8, 16, 32]
        shape_format = [
            [i, j] for i in dtype_list for j in index_list
        ]
        for item in shape_format: 
            np_input = np.zeros(40).astype(item[0])
            cpu_input = torch.from_numpy(np_input)
            cpu_out = cpu_input
            cpu_out[:item[1]] = 1
            npu_out = cpu_input.npu()
            npu_out[:item[1]] = 1
            self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy())
           
                
instantiate_device_type_tests(TestSpecialCasesCopyToContiguous, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()