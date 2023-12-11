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
from torch_npu.testing.testcase import TestCase, run_tests

device = 'npu:0'
torch.npu.set_device(device)


class TestParallelism(TestCase):
    def test_set_num_threads(self):
        torch.set_num_threads(2)

    def test_get_num_threads(self):
        output = torch.get_num_threads()
        print(output)

    def test_set_num_interop_threads(self):
        torch.set_num_interop_threads(2)

    def test_get_num_interop_threads(self):
        output = torch.get_num_interop_threads()
        print(output)


if __name__ == "__main__":
    run_tests()
