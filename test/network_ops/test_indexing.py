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
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestIndexing(TestCase):

    @Dtypes(torch.float, torch.half, torch.int32, torch.uint8, torch.int8, torch.int16, torch.long, torch.cfloat)
    def test_indexing(self, dtype):
        input1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype).to("npu")
        expect_output = torch.tensor([[1, 2], [5, 6]], dtype=dtype)
        output = torch.npu_indexing(input1, [0, 0], [2, 2], [1, 1])
        if dtype == torch.cfloat:
            self.assertRtolEqual(expect_output.float(), output.cpu().float())
        else:
            self.assertRtolEqual(expect_output, output.cpu())


if __name__ == "__main__":
    run_tests()
