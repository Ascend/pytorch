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
import torch.nn.functional as F
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestCopy(TestCase):

    def cpu_op_exec(self, input1, input2):
        input1.copy_(input2)
        return input1

    def npu_op_exec(self, input1, input2):
        input1.copy_(input2)
        return input1.cpu()

    def test_copy__(self):
        format_list = [0]
        shape_list = [(4, 1), (4, 3, 1)]
        dtype_list = [np.float32, np.int32, np.float16, np.cfloat]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            if item[0] == np.cfloat:
                cpu_output = cpu_output.float()
                npu_output = npu_output.float()
            self.assertRtolEqual(cpu_output, npu_output)

    def test_copy_memery_stampede(self):
        x = torch.randn((1, 6), device='npu:0').expand((6, 6))
        with self.assertRaises(RuntimeError):
            F.silu(x, inplace=True)


if __name__ == "__main__":
    run_tests()
