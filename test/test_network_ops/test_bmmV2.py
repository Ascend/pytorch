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


class TestBatchMatMulV2(TestCase):
    def cpu_op_exec(self, input1, input2):
        input1.requires_grad = True
        input2.requires_grad = True
        output = torch.bmm(input1, input2)
        output.backward(torch.ones_like(output))
        output = output.detach().numpy()
        input1_grad = input1.grad.numpy()
        input2_grad = input2.grad.numpy()
        return output, input1_grad, input2_grad

    def npu_op_exec(self, input1, input2):
        input1.requires_grad = True
        input2.requires_grad = True
        output = torch_npu.npu_bmmV2(input1, input2, [])
        output.backward(torch.ones_like(output))
        output = output.cpu().detach().numpy()
        input1_grad = input1.grad.cpu().numpy()
        input2_grad = input2.grad.cpu().numpy()
        return output, input1_grad, input2_grad

    def bmm_auto_list_exec(self, shape):
        for item in shape:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 10)
            if cpu_input1.dtype == torch.float16:
                cpu_output, cpu_grad1, cpu_grad2 = self.cpu_op_exec(cpu_input1.float(), cpu_input2.float())
                cpu_output = cpu_output.astype(np.float16)
                cpu_grad1 = cpu_grad1.astype(np.float16)
                cpu_grad2 = cpu_grad2.astype(np.float16)
            else:
                cpu_output, cpu_grad1, cpu_grad2 = self.cpu_op_exec(cpu_input1, cpu_input2)

            npu_output, npu_grad1, npu_grad2 = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-3, prec16=1.e-3)
            self.assertRtolEqual(cpu_grad1, npu_grad1, prec=1.e-3, prec16=1.e-3)
            self.assertRtolEqual(cpu_grad2, npu_grad2, prec=1.e-3, prec16=1.e-3)

    def test_batchmatmul_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        shape_list = [(1, 3, 2)]
        shape_format1 = [[np.float16, i, j]
                         for i in format_list for j in shape_list]
        format_list = [0, 3, 29]
        shape_list = [(1, 2, 3)]
        shape_format2 = [[np.float16, i, j]
                         for i in format_list for j in shape_list]
        shape_format = [[i, j] for i in shape_format1 for j in shape_format2]
        self.bmm_auto_list_exec(shape_format)

    def test_batchmatmul_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        shape_list = [(1, 3, 2)]
        shape_format1 = [[np.float32, i, j]
                         for i in format_list for j in shape_list]
        format_list = [0, 3, 29]
        shape_list = [(1, 2, 3)]
        shape_format2 = [[np.float32, i, j]
                         for i in format_list for j in shape_list]
        shape_format = [[i, j] for i in shape_format1 for j in shape_format2]
        self.bmm_auto_list_exec(shape_format)


if __name__ == "__main__":
    run_tests()
