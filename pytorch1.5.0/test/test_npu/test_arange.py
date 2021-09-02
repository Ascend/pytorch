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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from common.util_test_new import create_dtype_tensor
class TestArange(TestCase):
    def test_arange(self, device):
        shape_format = [
            [0, 100, 2, torch.float32],
            [1, 100, 1, torch.int32],
            [5, 9999, 4, torch.float32],
            [5, 9999, 4, torch.int32]
        ]
        for item in shape_format:
            cpu_output = torch.arange(item[0], item[1], item[2], dtype = item[3], device = 'cpu').numpy()
            npu_output = torch.arange(item[0], item[1], item[2], dtype=item[3], device='npu').cpu().numpy()
            self.assertRtolEqual(cpu_output, npu_output)

    def test_arange_out(self, device):
        shape_format = [
         [0, 100, 2, torch.float32],
            [1, 100, 1, torch.int32],
            [5, 9999, 4, torch.float32],
            [5, 9999, 4, torch.int32]
        ]

        for item in shape_format:
            cpu_output, npu_output = create_dtype_tensor(int((item[1] - item[0]+item[2]-0.000001)/item[2]), dtype= item[3],min_value = item[0], max_value = item[1])
            cpu_output = torch.arange(item[0], item[1], item[2], dtype = item[3], device = 'cpu',out = cpu_output).numpy()
            npu_output = torch.arange(item[0], item[1], item[2], dtype=item[3], device='npu',out = npu_output).cpu().numpy()
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestArange, globals(), except_for='cpu')
if __name__ == '__main__':
    run_tests()