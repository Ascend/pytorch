# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
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


class TestCopy(TestCase):

    def test_copy_broadcast(self):
        x = torch.randn(10, 5)
        y = torch.randn(5).npu()
        x.copy_(y)
        self.assertEqual(x[3], y)

        x = torch.randn(10, 5).npu()
        y = torch.randn(5)
        x.copy_(y)
        self.assertEqual(x[3], y)


if __name__ == "__main__":
    run_tests()
