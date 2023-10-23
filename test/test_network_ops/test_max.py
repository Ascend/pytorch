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


class TestMax(TestCase):

    def cpu_op_exec(self, input1):
        output = torch.max(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.max(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_other_exec(self, input1, input2):
        output = torch.max(input1, input2)
        output = output.numpy()
        return output

    def npu_op_other_exec(self, input1, input2):
        output = torch.max(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, out):
        torch.max(input1, input2, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_dim_exec(self, input1, dim, keepdim):
        output1, output2 = torch.max(input1, dim, keepdim)
        output1 = output1.numpy()
        output2 = output2.int().numpy()
        return output1, output2

    def npu_op_dim_exec(self, input1, dim, keepdim):
        output1, output2 = torch.max(input1, dim, keepdim)
        output1 = output1.to("cpu")
        output2 = output2.to("cpu")
        output1 = output1.numpy()
        output2 = output2.numpy()
        return output1, output2

    def _cpu_op_dim_exec(self, input1, dim, keepdim):
        output1, output2 = torch._max(input1, dim, keepdim)
        output1 = output1.numpy()
        output2 = output2.numpy()
        return output1, output2

    def _npu_op_dim_exec(self, input1, dim, keepdim):
        output1, output2 = torch._max(input1, dim, keepdim)
        output1 = output1.to("cpu")
        output2 = output2.to("cpu")
        output1 = output1.numpy()
        output2 = output2.numpy()
        return output1, output2

    def cpu_op_dim_exec_out(self, input1, dim, keepdim):
        out = torch.tensor(0).to(input1.dtype)
        indices = torch.tensor(0).to(torch.long)
        torch.max(input1, dim=dim, keepdim=keepdim, out=(out, indices))
        out = out.numpy()
        indices = indices.numpy()
        return out, indices

    def npu_op_dim_exec_out(self, input1, dim, keepdim):
        out = torch.tensor(0).to(input1.dtype).npu()
        indices = torch.tensor(0).to(torch.long).npu()
        torch.max(input1, dim=dim, keepdim=keepdim, out=(out, indices))
        out = out.to("cpu").numpy()
        indices = indices.to("cpu").numpy()
        return out, indices

    def cpu_op_amax_exec(self, input1, dim, keepdim):
        output = torch.amax(input1, dim, keepdim)
        output = output.numpy()
        return output

    def npu_op_amax_exec(self, input1, dim, keepdim):
        output = torch.amax(input1, dim, keepdim)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_amax_exec_out(self, input1, dim, keepdim, out):
        torch.amax(input1, dim, keepdim, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def max_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def max_result_dim(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_dim, _ = self.cpu_op_dim_exec(cpu_input1, item[1], item[2])
            npu_output_dim, _ = self.npu_op_dim_exec(npu_input1, item[1], item[2])
            cpu_output_dim = cpu_output_dim.astype(npu_output_dim.dtype)
            self.assertRtolEqual(cpu_output_dim, npu_output_dim)

    def max_result_other(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 0, 7)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output_other = self.cpu_op_other_exec(cpu_input1, cpu_input2)
            npu_output_other = self.npu_op_other_exec(npu_input1, npu_input2)
            cpu_output_other = cpu_output_other.astype(npu_output_other.dtype)
            self.assertRtolEqual(cpu_output_other, npu_output_other)

    def max_out_result_other(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], -100, 100)
            _, npu_input4 = create_common_tensor(item[1], -100, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_other_exec(cpu_input1, cpu_input2)
            npu_output_out1 = self.npu_op_other_exec(npu_input1, npu_input2)
            npu_output_out2 = self.npu_op_exec_out(npu_input1, npu_input2, npu_input4)
            cpu_output = cpu_output.astype(npu_output_out1.dtype)
            self.assertRtolEqual(cpu_output, npu_output_out1)
            self.assertRtolEqual(cpu_output, npu_output_out2)

            cpu_out_dim, cpu_out_indices = self.cpu_op_dim_exec_out(cpu_input1, dim=0, keepdim=True)
            npu_out_dim, npu_out_indices = self.npu_op_dim_exec_out(npu_input1, dim=0, keepdim=True)
            npu_output_dim, npu_output_indices = self.npu_op_dim_exec(npu_input1, dim=0, keepdim=True)
            cpu_out_dim = cpu_out_dim.astype(npu_out_dim.dtype)
            if npu_out_dim.dtype != np.float16:
                self.assertRtolEqual(npu_out_dim, cpu_out_dim)
                self.assertRtolEqual(npu_out_indices, cpu_out_indices)
            else:
                self.assertRtolEqual(npu_out_dim, npu_output_dim)
                self.assertRtolEqual(npu_out_indices, npu_output_indices)

    def max_name_result_other(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input1.names = item[0][3]
            npu_input1.names = item[0][3]
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_dim, cpu_output_indices = self.cpu_op_dim_exec(cpu_input1, item[1], item[2])
            npu_output_dim, npu_output_indices = self.npu_op_dim_exec(cpu_input1, item[1], item[2])

            if npu_output_dim.dtype != np.float16:
                self.assertRtolEqual(npu_output_dim, cpu_output_dim)
                self.assertRtolEqual(npu_output_indices.astype(np.int32), cpu_output_indices.astype(np.int32))
            else:
                self.assertRtolEqual(npu_output_dim, cpu_output_dim.astype(np.float16))
                self.assertRtolEqual(npu_output_indices.astype(np.int32), cpu_output_indices.astype(np.int32))

    def max_name_out_result_other(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input1.names = item[0][3]
            npu_input1.names = item[0][3]
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_dim, _ = self.cpu_op_dim_exec_out(cpu_input1, item[1], item[2])
            npu_output_dim, _ = self.npu_op_dim_exec_out(npu_input1, item[1], item[2])

            if npu_output_dim.dtype != np.float16:
                self.assertRtolEqual(npu_output_dim, cpu_output_dim)
            else:
                self.assertRtolEqual(npu_output_dim, cpu_output_dim.astype(np.float16))

    def amax_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            _, npu_input2 = create_common_tensor(item[0], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_amax = self.cpu_op_amax_exec(cpu_input1, item[1], item[2])
            npu_output_amax = self.npu_op_amax_exec(npu_input1, item[1], item[2])
            npu_output_amax_out = self.npu_op_amax_exec_out(npu_input1, item[1], item[2], npu_input2)
            cpu_output_amax = cpu_output_amax.astype(npu_output_amax.dtype)
            self.assertRtolEqual(cpu_output_amax, npu_output_amax)
            self.assertRtolEqual(cpu_output_amax, npu_output_amax_out)

    def test_max_out_result(self):
        shape_format = [
            [[np.float16, 0, [9, 10, 14, 14]], [np.float16, 0, [7, 10, 1, 1]]],
            [[np.float16, 0, [9, 7, 12, 12]], [np.float16, 0, [7, 7, 1, 1]]],
            [[np.float32, 0, [2, 3, 3, 3]], [np.float32, 0, [3, 1, 3]]],
            [[np.float32, 0, [9, 13, 7, 7]], [np.float32, 0, [9, 13, 7, 7]]],
        ]
        self.max_out_result_other(shape_format)

    def test_max_shape_format_fp16_1d(self):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self.max_result(shape_format)

    def test_max_shape_format_fp32_1d(self):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result(shape_format)

    def test_max_shape_format_fp16_2d(self):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result(shape_format)

    def test_max_shape_format_fp32_2d(self):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result(shape_format)

    def test_max_shape_format_fp16_3d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result(shape_format)

    def test_max_shape_format_fp32_3d(self):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result(shape_format)

    def test_max_shape_format_fp16_4d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.max_result(shape_format)

    def test_max_shape_format_fp32_4d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.max_result(shape_format)

    def test_max_dim_shape_format_fp16_1d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_dim_shape_format_fp32_1d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_dim_shape_format_fp16_2d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_dim_shape_format_fp32_2d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_dim_shape_format_fp16_3d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_dim_shape_format_fp32_3d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_dim_shape_format_fp16_4d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_dim_shape_format_fp32_4d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.max_result_dim(shape_format)

    def test_max_other_shape_format_fp16_1d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self.max_result_other(shape_format)

    def test_max_other_shape_format_fp32_1d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self.max_result_other(shape_format)

    def test_max_other_shape_format_fp16_2d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_other(shape_format)

    def test_max_other_shape_format_fp32_2d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_other(shape_format)

    def test_max_other_shape_format_fp16_3d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_other(shape_format)

    def test_max_other_shape_format_fp32_3d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.max_result_other(shape_format)

    def test_max_other_shape_format_fp16_4d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.max_result_other(shape_format)

    def test_max_other_shape_format_fp32_4d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.max_result_other(shape_format)

    def test_max_dimname_shape_format(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9, 10], ('N', 'C', 'H', 'W')],
                        np.random.choice(['N', 'C', 'H', 'W']), j] for i in format_list for j in keepdim_list
                        ]
        self.max_name_result_other(shape_format)

    def test_max_dimname_shape_format_fp16(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9, 10], ('N', 'C', 'H', 'W')],
                        np.random.choice(['N', 'C', 'H', 'W']), j] for i in format_list for j in keepdim_list
                        ]
        self.max_name_result_other(shape_format)

    def test_max_dimname_out_shape_format(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9, 10], ('N', 'C', 'H', 'W')],
                        np.random.choice(['N', 'C', 'H', 'W']), j] for i in format_list for j in keepdim_list
                        ]
        self.max_name_out_result_other(shape_format)

    def test_max_dimname_out_shape_format_fp16(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9, 10], ('N', 'C', 'H', 'W')],
                        np.random.choice(['N', 'C', 'H', 'W']), j] for i in format_list for j in keepdim_list
                        ]
        self.max_name_out_result_other(shape_format)

    def test_amax_shape_format_fp16_1d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self.amax_result(shape_format)

    def test_amax_shape_format_fp32_1d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self.amax_result(shape_format)

    def test_amax_shape_format_fp16_2d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.amax_result(shape_format)

    def test_amax_shape_format_fp32_2d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.amax_result(shape_format)

    def test_amax_shape_format_fp16_3d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.amax_result(shape_format)

    def test_amax_shape_format_fp32_3d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.amax_result(shape_format)

    def test_amax_shape_format_fp16_4d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.amax_result(shape_format)

    def test_amax_shape_format_fp32_4d(self):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.amax_result(shape_format)

    def test_max_out_5d_ncdhw(self):
        a_cpu = torch.rand(2, 2, 2, 2, 2)
        b_cpu = torch.rand(2, 2, 2, 2, 2)
        a_npu = a_cpu.clone().npu()
        b_npu = b_cpu.clone().npu()
        cpu_output = self.cpu_op_other_exec(a_cpu, b_cpu)
        npu_output_out2 = self.npu_op_exec_out(a_npu, b_npu, a_npu)
        self.assertRtolEqual(cpu_output, npu_output_out2)

    def test_max_different_dtype(self):
        cpu_input1, npu_input1 = create_common_tensor([np.int32, 0, [10, 20, 30]], -100, 100)
        cpu_input2, npu_input2 = create_common_tensor([np.int64, 0, [10, 20, 30]], -100, 100)
        cpu_output = self.cpu_op_other_exec(cpu_input1, cpu_input2)
        npu_output = self.npu_op_other_exec(npu_input1, npu_input2)
        npu_out = np.random.uniform(-100, 100, (10, 10)).astype(cpu_output.dtype)
        npu_out = torch.from_numpy(npu_out).npu()
        torch.max(npu_input1, npu_input2, out=npu_out)
        npu_out = npu_out.cpu().numpy()
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_output, npu_out)

    def test_max_slice_out(self):
        cpu_input = torch.randn(4, 4, 4, 4)
        npu_input = cpu_input.npu()

        cpu_val = torch.zeros(10, 10, 10)
        cpu_ind = torch.zeros(10, 10, 10).long()
        npu_val = cpu_val.npu()
        npu_ind = cpu_ind.npu()
        torch.max(cpu_input, 1, out=(cpu_val[0:4, 1:5, 2:6], cpu_ind[0:4, 1:5, 2:6]))
        torch.max(npu_input, 1, out=(npu_val[0:4, 1:5, 2:6], npu_ind[0:4, 1:5, 2:6]))
        self.assertRtolEqual(cpu_val.numpy(), npu_val.cpu().numpy())
        self.assertRtolEqual(cpu_ind.numpy(), npu_ind.cpu().numpy())

        cpu_val = torch.zeros(10, 10, 10)
        npu_val = cpu_val.npu()
        torch.amax(cpu_input, 1, out=cpu_val[0:4, 1:5, 2:6])
        torch.amax(npu_input, 1, out=npu_val[0:4, 1:5, 2:6])
        self.assertRtolEqual(cpu_val.numpy(), npu_val.cpu().numpy())

        cpu_input0 = torch.randn(2, 2, 2, 2)
        cpu_input1 = torch.randn(2, 2, 2, 2)
        npu_input0 = cpu_input0.npu()
        npu_input1 = cpu_input1.npu()
        cpu_val = torch.zeros(3, 3, 3, 3)
        npu_val = cpu_val.npu()
        torch.max(cpu_input0, cpu_input1, out=cpu_val[0:2, 0:2, 0:2, 0:2])
        torch.max(npu_input0, npu_input1, out=npu_val[0:2, 0:2, 0:2, 0:2])
        self.assertRtolEqual(cpu_val.numpy(), npu_val.cpu().numpy())


if __name__ == "__main__":
    run_tests()
