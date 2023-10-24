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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestSparseLayers(TestCase):
    def test_Embedding(self):
        embedding = nn.Embedding(10, 3).npu()
        input1 = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]).npu()
        output = embedding(input1)
        self.assertEqual(output is not None, True)


if __name__ == "__main__":
    run_tests()
