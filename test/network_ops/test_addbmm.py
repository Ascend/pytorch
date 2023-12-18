# Copyright (c) 2020 Huawei Technologies Co., Ltd
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


torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestAddbmm(TestCase):

    def cpu_op_exec(self, input1, input2, input3, scalar1, scalar2):
        output = torch.addbmm(input1, input2, input3, beta=scalar1, alpha=scalar2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, input3, scalar1, scalar2):
        output = torch.addbmm(input1, input2, input3, beta=scalar1, alpha=scalar2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, input3, scalar1, scalar2):
        output = torch.ones_like(input1)
        torch.addbmm(input1, input2, input3, beta=scalar1, alpha=scalar2, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_inplace(self, input1, input2, input3, scalar1, scalar2):
        input1.addbmm_(input2, input3, beta=scalar1, alpha=scalar2)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_transpose_exec(self, input1, input2, input3, scalar1, scalar2):
        input3_t = input3.permute(0, 2, 1)
        output = torch.addbmm(input1, input2, input3_t, beta=scalar1, alpha=scalar2)
        output = output.numpy()
        return output

    def npu_op_transpose_exec(self, input1, input2, input3, scalar1, scalar2):
        input3_t = input3.permute(0, 2, 1)
        output = torch.addbmm(input1, input2, input3_t, beta=scalar1, alpha=scalar2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_addbmm(self):
        shape_format = [
            [[np.float16, 0, [3, 5]], [np.float16, 0, [10, 3, 4]], [np.float16, 0, [10, 4, 5]]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 100)

            scalar1 = np.random.uniform(0, 10)
            scalar2 = np.random.uniform(0, 10)

            cpu_input1 = cpu_input1.float()
            cpu_input2 = cpu_input2.float()
            cpu_input3 = cpu_input3.float()
            npu_input1 = npu_input1.float()
            npu_input2 = npu_input2.float()
            npu_input3 = npu_input3.float()

            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3, scalar1, scalar2)

            npu_output1 = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3, scalar1, scalar2)
            npu_output2 = self.npu_op_exec_inplace(npu_input1, npu_input2, npu_input3, scalar1, scalar2)

            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-3)
            self.assertRtolEqual(cpu_output, npu_output1, prec=1.e-3)
            self.assertRtolEqual(cpu_output, npu_output2, prec=1.e-3)

    def test_addbmm_transpose(self):
        shape_format = [
            [[np.float16, 0, [4, 5]], [np.float16, 0, [10, 4, 7]], [np.float16, 0, [10, 5, 7]]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 100)

            scalar1 = np.random.uniform(0, 10)
            scalar2 = np.random.uniform(0, 10)

            cpu_transpose_output = self.cpu_op_transpose_exec(
                cpu_input1.float(), cpu_input2.float(), cpu_input3.float(), scalar1, scalar2)
            npu_transpose_output = self.npu_op_transpose_exec(
                npu_input1.float(), npu_input2.float(), npu_input3.float(), scalar1, scalar2)

            self.assertRtolEqual(cpu_transpose_output, npu_transpose_output, prec=1.e-3)


if __name__ == "__main__":
    run_tests()
