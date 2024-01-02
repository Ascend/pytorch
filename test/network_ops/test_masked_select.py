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


class TestMaskedSelect(TestCase):

    def get_mask(self):
        mask = torch.tensor([[
            [True, False, True, True, False],
            [True, False, False, True, False],
            [False, False, False, False, False],
            [True, False, False, False, False]],

            [[True, False, False, False, True],
             [False, True, False, True, True],
             [False, True, False, True, True],
             [False, False, False, False, False]],

            [[False, True, True, False, True],
             [False, True, True, True, True],
             [False, True, False, True, False],
             [False, True, True, False, False]]])
        return mask

    def cpu_op_exec(self, input1, mask):
        output = torch.masked_select(input1, mask)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, mask):
        mask = mask.to("npu")
        output = torch.masked_select(input1, mask)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, mask, output):
        output = torch.masked_select(input1, mask, out=output)
        return output.detach().to("cpu").numpy()

    def test_maskedselect_out_result(self):
        shape_format = [
            [[np.float16, 2, [15, 15, 15, 16]], [np.float16, 2, [15, 15, 15, 16]]],
            [[np.float16, 2, [15, 15, 15, 16]], [np.float16, 2, [3, 3, 7, 7]]],
            [[np.float16, 0, [15, 15, 15, 16]], [np.float16, 0, [15, 15, 15, 16]]],
            [[np.float16, 0, [15, 15, 15, 16]], [np.float16, 0, [116, 116, 1, 1]]],
            [[np.float32, 2, [15, 15, 15, 16]], [np.float32, 2, [15, 15, 15, 16]]],
            [[np.float32, 2, [15, 15, 15, 16]], [np.float32, 2, [3, 3, 7, 7]]],
            [[np.float32, 0, [15, 15, 15, 16]], [np.float32, 0, [15, 15, 15, 16]]],
            [[np.float32, 0, [15, 15, 15, 16]], [np.float32, 0, [23, 23, 1, 1]]],
            [[np.float32, 0, [15, 15, 15, 16]], [np.float32, 0, [232, 232, 1, 1]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 2)
            cpu_input2, npu_input2 = create_common_tensor(item[0], -2, 2)
            cpu_input3, npu_input3 = create_common_tensor(item[1], -2, 2)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2.to(torch.int32) > 0)
            npu_output = self.npu_op_exec_out(npu_input1, npu_input2.to(torch.int32) > 0, npu_input3)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_maskedselect_shape_format_maskdiff(self):
        dtype_list = [np.int64, np.int32, np.float32]
        format_list = [0]
        shape_list = [[3, 4, 5]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            mask_cpu, mask_npu = create_common_tensor((np.int32, 0, (3, 4, 1)), 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, mask_cpu > 50)
            npu_output = self.npu_op_exec(npu_input, mask_npu > 50)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_maskedselect_shape_format_fp32(self):
        format_list = [0]
        shape_list = [[3, 4, 5]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        mask = self.get_mask()

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, mask)
            npu_output = self.npu_op_exec(npu_input, mask)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_maskedselect_shape_format_int(self):
        dtype_list = [np.int32, np.int64]
        format_list = [0]
        shape_list = [[3, 4, 5]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        mask = self.get_mask()

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, mask)
            npu_output = self.npu_op_exec(npu_input, mask)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_maskedselect_case_in_gaitset(self):
        cpu_in = torch.rand(1015808)
        npu_in = cpu_in.npu()
        cpu_mask = (torch.randn(1015808) > 0).byte()
        npu_mask = cpu_mask.npu()
        cpu_out = torch.masked_select(cpu_in, cpu_mask)
        npu_out = torch.masked_select(npu_in, npu_mask)
        self.assertRtolEqual(cpu_out, npu_out.cpu())

    def test_maskedselect_shape_mask_nobroadcast(self):
        cpu_input = torch.randn(2, 4, 1, 5)
        npu_input = cpu_input.npu()
        cpu_mask = torch.randn(2, 1, 3, 5) > 0.5
        npu_mask = cpu_mask.npu()
        cpu_out = torch.masked_select(cpu_input, cpu_mask)
        npu_out = torch.masked_select(npu_input, npu_mask)
        self.assertRtolEqual(cpu_out, npu_out.cpu())


if __name__ == "__main__":
    run_tests()
