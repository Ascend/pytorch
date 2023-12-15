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
from torch_npu.testing.common_utils import create_common_tensor


class TestNpuReshape(TestCase):
    def custom_op_exec(self, input1, shape):
        output = self.npu_reshape(input1, shape)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, shape):
        output = torch_npu.npu_reshape(input1, shape)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_reshape(self, input1, shape, can_refresh=False, out=None):
        if can_refresh:
            out = torch.reshape(input1, shape)
        else:
            out = torch.reshape(input1, shape).clone()
        return out

    def test_reshape_shape_format(self):
        dtype_list = [np.float16, np.float32, np.int32, np.bool_, np.complex64]
        format_list = [0]
        shape_list = [[8, 8], [2, 4, 8], [2, 4, 4, 2]]
        shape_format = [
            [i, j, k]
            for i in dtype_list
            for j in format_list
            for k in shape_list
        ]

        for item in shape_format:
            _, npu_input = create_common_tensor(item, 0, 100)
            shape = [4, 16]
            custom_output = self.custom_op_exec(npu_input, shape)
            npu_output = self.npu_op_exec(npu_input, shape)
            if item[0] == np.complex64:
                custom_output = custom_output.astype(np.float)
                npu_output = npu_output.astype(np.float)
            self.assertRtolEqual(custom_output, npu_output)


if __name__ == "__main__":
    run_tests()
