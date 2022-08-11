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


class TestMM(TestCase):
    def test_mm_mat1_mat2_transpose(self, device="npu"):
        dtype_list = [np.float16, np.float32]
        format_list = [2, 29]
        shape_list = [
                      [16, 32],
                      [16, 30],
                      [15, 32],
                      [15, 30]
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            npu_out = torch.mm(npu_input, npu_input.t())
            cpu_out = torch.mm(cpu_input, cpu_input.t())
            self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.to(npu_out.dtype).numpy())

    def test_mm_mat1_view_mat2_view_transpose(self, device="npu"):
        dtype_list = [np.float16, np.float32]
        format_list = [2, 29]
        shape_list = [
                      [2, 15, 10],
                      [2, 16, 10],
                      [2, 16, 16],
                      [2, 15, 16]
                      ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format: 
            cpu_input, npu_input = create_common_tensor(item, -10, 10)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            res_shape = [cpu_input.shape[0] * cpu_input.shape[1], cpu_input.shape[2]]
            npu_out = torch.mm(npu_input.view(res_shape[0], res_shape[1]), \
                npu_input.view(res_shape[0], res_shape[1]).t())
            cpu_out = torch.mm(cpu_input.view(res_shape[0], res_shape[1]), \
                cpu_input.view(res_shape[0], res_shape[1]).t())
            self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.to(npu_out.dtype).numpy())


if __name__ == "__main__":
    run_tests()