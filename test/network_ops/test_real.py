# Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestReal(TestCase):
    def cpu_op_exec(self, input1):
        output = input1.real
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = input1.real
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_real_shape_format_complex(self, device="npu"):
        format_list = [0]
        shape_list = [[5], [5, 10], [1, 3, 2], [52, 15, 15, 20]]
        dtype_list = [np.complex64, np.complex128]
        shape_format = [
            [i, j, k]
            for i in dtype_list
            for j in format_list
            for k in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -10, 10)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
