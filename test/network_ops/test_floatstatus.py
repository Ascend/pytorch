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


class TestFloatStatus(TestCase):
    def npu_op_exec(self, input1):
        output = torch_npu.npu_alloc_float_status(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_floatstatus(self):
        input1 = torch.randn([1, 2, 3]).npu()
        exoutput = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0.])
        output = self.npu_op_exec(input1)
        self.assertRtolEqual(exoutput.numpy(), output)


if __name__ == "__main__":
    run_tests()
