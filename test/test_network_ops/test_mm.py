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
    def test_mm_mat1_mat2_transpose(self):
        dtype_list = [np.float16, np.float32]
        format_list = [2, 29]
        shape_list = [
            [8, 16],
            [8, 15],
            [7, 16],
            [7, 15]
        ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 1)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            else:
                cpu_input = cpu_input.half().float()
                npu_input = npu_input.half().float()
            npu_out = torch.mm(npu_input, npu_input.t())
            cpu_out = torch.mm(cpu_input, cpu_input.t())
            if item[0] == np.float16:
                cpu_out = cpu_out.half()
            self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.to(npu_out.dtype).numpy(), prec=1.e-3, prec16=1.e-3)

    def test_mm_mat1_view_mat2_view_transpose(self):
        dtype_list = [np.float16, np.float32]
        format_list = [2, 29]
        shape_list = [
            [2, 7, 5],
            [2, 8, 5],
            [2, 8, 8],
            [2, 7, 8]
        ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 1)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            else:
                cpu_input = cpu_input.half().float()
                npu_input = npu_input.half().float()
            res_shape = [cpu_input.shape[0] * cpu_input.shape[1], cpu_input.shape[2]]
            npu_out = torch.mm(npu_input.view(res_shape[0], res_shape[1]),
                               npu_input.view(res_shape[0], res_shape[1]).t())
            cpu_out = torch.mm(cpu_input.view(res_shape[0], res_shape[1]),
                               cpu_input.view(res_shape[0], res_shape[1]).t())
            if item[0] == np.float16:
                cpu_out = cpu_out.half()
            self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.to(npu_out.dtype).numpy(), prec=1.e-3, prec16=1.e-3)


if __name__ == "__main__":
    run_tests()
