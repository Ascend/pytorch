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

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestStrideAdd(TestCase):
    def npu_op_exec(self, input1, input2, offset1, offset2, c1_len):
        output = torch.npu_stride_add(input1, input2, offset1, offset2, c1_len)
        output = output.to("cpu")
        output = output.numpy()

        return output

    def test_StrideAdd(self):
        input1 = torch.tensor([[[[[1.]]]]]).npu()
        input2 = input1
        exoutput = torch.tensor([[[[[2.]]], [[[0.]]], [[[0.]]], [[[0.]]], [[[0.]]], [[[0.]]], [[[0.]]], [[[0.]]],
                                  [[[0.]]], [[[0.]]], [[[0.]]], [[[0.]]], [[[0.]]], [[[0.]]], [[[0.]]], [[[0.]]]]])
        output = self.npu_op_exec(input1, input2, 0, 0, 1)
        self.assertRtolEqual(exoutput.numpy(), output)


if __name__ == "__main__":
    run_tests()
