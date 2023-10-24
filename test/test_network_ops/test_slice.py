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

import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestSlice(TestCase):
    def npu_op_exec(self, input1, offset, sizes):
        output = torch.npu_slice(input1, offset, sizes)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @Dtypes(torch.float, torch.half, torch.int32, torch.uint8, torch.int8, torch.int16, torch.long,
            torch.cfloat)
    def test_slice(self, device, dtype):
        input_data = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).npu().to(dtype)
        exoutput = torch.tensor([[1, 2], [6, 7]]).to(dtype)
        output = self.npu_op_exec(input_data, [0, 0], [2, 2])
        if dtype == torch.cfloat:
            exoutput = exoutput.to(torch.float)
            output = output.astype(np.float32)
        self.assertRtolEqual(exoutput.numpy(), output)


if __name__ == "__main__":
    run_tests()
