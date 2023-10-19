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


class TestClamp(TestCase):

    def npu_op_exec(self, input1, min_val, max_val):
        output = torch.clamp(input1, min_val, max_val)
        output = output.cpu().numpy()
        return output

    def cpu_op_exec(self, input1, min_val, max_val):
        input_dtype = input1.dtype
        if input_dtype == torch.float16:
            input1 = input1.to(torch.float32)
        output = torch.clamp(input1, min_val, max_val)
        if input_dtype == torch.float16:
            output = output.to(torch.float16)
        output = output.numpy()
        return output

    def npu_inp_op_exec(self, input1, min_val, max_val):
        torch.clamp_(input1, min_val, max_val)
        output = input1.cpu().numpy()
        return output

    def cpu_inp_op_exec(self, input1, min_val, max_val):
        input_dtype = input1.dtype
        if input_dtype == torch.float16:
            input1 = input1.to(torch.float32)
        output = torch.clamp_(input1, min_val, max_val)
        if input_dtype == torch.float16:
            output = output.to(torch.float16)
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, min_val, max_val, output):
        torch.clamp(input1, min_val, max_val, out=output)
        output = output.cpu().numpy()
        return output

    def npu_inp_uncon_op_exec(self, input1, min_val, max_val):
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        torch.clamp_(input1, min_val, max_val)
        output = input1.cpu().numpy()
        return output

    def cpu_inp_uncon_op_exec(self, input1, min_val, max_val):
        input_dtype = input1.dtype
        if input_dtype == torch.float16:
            input1 = input1.to(torch.float32)
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        output = torch.clamp(input1, min_val, max_val)
        if input_dtype == torch.float16:
            output = output.to(torch.float16)
        output = output.numpy()
        return output

    def test_clamp_common(self):
        shape_format = [
            [np.float32, 0, (4, 3)],
            [np.int32, 0, (4, 3)],
            [np.int64, 0, (4, 3)],
            [np.float16, 0, (4, 3)]
        ]
        for item in shape_format:
            input_cpu, input_npu = create_common_tensor(item, 1, 100)
            _, out_npu = create_common_tensor(item, 1, 100)

            cpu_output = self.cpu_op_exec(input_cpu, 40, 60)
            npu_output = self.npu_op_exec(input_npu, 40, 60)

            cpu_inp_output = self.cpu_inp_op_exec(input_cpu, 40, 60)
            npu_inp_output = self.npu_inp_op_exec(input_npu, 40, 60)

            npu_out_output = self.npu_op_exec_out(input_npu, 40, 60, out_npu)

            cpu_inp_uncon_output = self.cpu_inp_uncon_op_exec(input_cpu, 40, 60)
            npu_inp_uncon_output = self.npu_inp_uncon_op_exec(input_npu, 40, 60)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_inp_output, npu_inp_output)
            self.assertRtolEqual(cpu_output, npu_out_output)
            self.assertRtolEqual(cpu_inp_uncon_output, npu_inp_uncon_output)

    def test_clamp_tensor(self):
        shape_format = [
            [[np.float32, 0, (4, 3)], [np.float32, 0, (4, 3)], [np.float32, 0, (4, 3)]],
            [[np.int32, 0, (24, 13)], [np.int32, 0, (24, 1)], [np.int32, 0, (1, 13)]],
            [[np.int64, 0, (41, 32, 23)], [np.int32, 0, (41, 32, 23)], [np.int32, 0, (41, 32, 23)]],
            [[np.float16, 0, (14, 3)], [np.float32, 0, (14, 3)], [np.float32, 0, (14, 3)]],
            [[np.float64, 0, (14, 3)], [np.float64, 0, (14, 1)], [np.float64, 0, (1, 3)]],
            [[np.int16, 0, (14, 3)], [np.int16, 0, (14, 1)], [np.int16, 0, (1, 3)]],
            [[np.int8, 0, (14, 3)], [np.int8, 0, (14, 1)], [np.int8, 0, (1, 3)]],
            [[np.uint8, 0, (14, 3)], [np.uint8, 0, (14, 1)], [np.uint8, 0, (1, 3)]]
        ]
        for item in shape_format:
            input_cpu, input_npu = create_common_tensor(item[0], 1, 100)
            min_cpu, min_npu = create_common_tensor(item[1], 1, 50)
            max_cpu, max_npu = create_common_tensor(item[2], 50, 100)
            _, out_npu = create_common_tensor(item[0], 1, 100)

            cpu_output = self.cpu_op_exec(input_cpu, min_cpu, max_cpu)
            npu_output = self.npu_op_exec(input_npu, min_npu, max_npu)

            cpu_inp_output = self.cpu_inp_op_exec(input_cpu, min_cpu, max_cpu)
            npu_inp_output = self.npu_inp_op_exec(input_npu, min_npu, max_npu)

            npu_out_output = self.npu_op_exec_out(input_npu, min_npu, max_npu, out_npu)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_inp_output, npu_inp_output)
            self.assertRtolEqual(cpu_output, npu_out_output)


if __name__ == "__main__":
    run_tests()
