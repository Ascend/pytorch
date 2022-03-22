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


class TestFlip(TestCase):
    def cpu_op_exec(self, input1, dims):
        output = torch.flip(input1, dims)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, dims):
        output = torch.flip(input1, dims)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_flip_shape_format(self, device="npu"):
        shape_format = [
                [[np.float32, 0, [2,2,2]], [0]],
                [[np.float32, 0, [2,2,2,4]], [-2]],
                [[np.int32, 0, [2,2,2]], [0,1]],
                [[np.int32, 0, [2,2,2,4]], [-1,1]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
	run_tests()