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
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestInverse(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.inverse(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.inverse(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_inverse_shape_format(self, device="npu"):
        dtype_list = [np.float16, np.float32]
        format_list = [0, 3]
        shape_list = [(4, 4), (0, 3, 29, 29), (1, 2, 4, 4)]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -100, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_inverse_out_fp16(self, device="npu"):
        cpu_x = torch.randn(5, 4, 9, 10, 10).uniform_(-2, 10).half()
        cpu_out = torch.randn(5, 4, 9, 10, 10).uniform_(-2, 10).float()
        npu_x = cpu_x.npu()
        npu_out = cpu_out.half().npu()

        torch.inverse(cpu_x.float(), out=cpu_out)
        torch.inverse(npu_x, out=npu_out)
        self.assertRtolEqual(cpu_out.half(), npu_out.cpu())


if __name__ == "__main__":
    run_tests()
