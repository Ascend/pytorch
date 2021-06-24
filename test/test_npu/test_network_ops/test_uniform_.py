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
from common_device_type import instantiate_device_type_tests

class TestUniform(TestCase):
    def test_uniform(self, device):
        shape_format = [
           [(20,300), -100, 100, torch.float32],
           [(20,300), -100, 100, torch.float16]
        ]

        for item in shape_format:
            input1 = torch.zeros(item[0], dtype=item[3]).npu()
            input1.uniform_(item[1], item[2])
            self.assertTrue(item[1] <= input1.min())
            self.assertTrue(item[2] >= input1.max())
    
    def test_uniform_trans(self, device):
        shape_format = [
           [(20,300), -100, 100, torch.float32],
        ]

        for item in shape_format:
            input1 = torch.zeros(item[0], dtype=item[3]).npu()
            input1.npu_format_cast(3)
            input1.uniform_(item[1], item[2])
            self.assertTrue(item[1] <= input1.min())
            self.assertTrue(item[2] >= input1.max())


instantiate_device_type_tests(TestUniform, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
