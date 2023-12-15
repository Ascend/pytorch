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


class TestMin(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.min(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.min(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_other_exec(self, input1, input2):
        output = torch.min(input1, input2)
        output = output.numpy()
        return output

    def npu_op_other_exec(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.min(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_other_exec_out(self, input1, input2, out):
        torch.min(input1, input2, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_dim_exec(self, input1, dim, keepdim):
        output1, output2 = torch.min(input1, dim, keepdim)
        output1 = output1.numpy()
        output2 = output2.int().numpy()
        return output1, output2

    def npu_op_dim_exec(self, input1, dim, keepdim):
        input1 = input1.to("npu")
        output1, output2 = torch.min(input1, dim, keepdim)
        output1 = output1.to("cpu")
        output2 = output2.to("cpu")
        output1 = output1.numpy()
        output2 = output2.numpy()
        return output1, output2

    def cpu_op_dim_exec_out(self, input1, dim, keepdim):
        out = torch.tensor(0).to(input1.dtype)
        indices = torch.tensor(0).to(torch.long)
        torch.min(input1, dim=dim, keepdim=keepdim, out=(out, indices))
        out = out.numpy()
        indices = indices.numpy()
        return out, indices

    def npu_op_dim_exec_out(self, input1, dim, keepdim):
        out = torch.tensor(0).to(input1.dtype).npu()
        indices = torch.tensor(0).to(torch.long).npu()
        torch.min(input1, dim=dim, keepdim=keepdim, out=(out, indices))
        out = out.to("cpu").numpy()
        indices = indices.to("cpu").numpy()
        return out, indices

    def cpu_op_amin_exec(self, input1, dim, keepdim):
        output = torch.amin(input1, dim, keepdim)
        output = output.numpy()
        return output

    def npu_op_amin_exec(self, input1, dim, keepdim):
        output = torch.amin(input1, dim, keepdim)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_amin_exec_out(self, input1, dim, keepdim, out):
        torch.amin(input1, dim, keepdim, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def min_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)

            self.assertRtolEqual(cpu_output, npu_output)

    def min_result_dim(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_dim, cpu_output_indices = self.cpu_op_dim_exec(cpu_input1, item[1], item[2])
            npu_output_dim, npu_output_indices = self.cpu_op_dim_exec(cpu_input1, item[1], item[2])
            cpu_output_dim = cpu_output_dim.astype(npu_output_dim.dtype)
            self.assertRtolEqual(cpu_output_dim, npu_output_dim)

    def min_result_other(self, shape_format):
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

    def min_out_result_other(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], -100, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[0], -100, 100)
            cpu_input4, npu_input4 = create_common_tensor(item[1], -100, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_other_exec(cpu_input1, cpu_input2)
            npu_output_out1 = self.npu_op_other_exec(npu_input1, npu_input2)
            npu_output_out2 = self.npu_op_other_exec_out(npu_input1, npu_input2, npu_input4)
            cpu_output = cpu_output.astype(npu_output_out1.dtype)
            self.assertRtolEqual(cpu_output, npu_output_out1)
            self.assertRtolEqual(cpu_output, npu_output_out2)
            cpu_out_dim, cpu_out_indices = self.cpu_op_dim_exec_out(cpu_input1, dim=0, keepdim=True)
            npu_out_dim, npu_out_indices = self.npu_op_dim_exec_out(npu_input1, dim=0, keepdim=True)
            npu_output_dim, npu_output_indices = self.npu_op_dim_exec(npu_input1, dim=0, keepdim=True)
            cpu_out_dim = cpu_out_dim.astype(npu_out_dim.dtype)
            if cpu_out_dim.dtype != np.float16:
                self.assertRtolEqual(npu_out_dim, cpu_out_dim)
                self.assertRtolEqual(npu_out_indices, cpu_out_indices)
            else:
                self.assertRtolEqual(npu_out_dim, npu_output_dim)
                self.assertRtolEqual(npu_out_indices, npu_output_indices)

    def min_name_result_other(self, shape_format):
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

    def min_name_out_result_other(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input1.names = item[0][3]
            npu_input1.names = item[0][3]
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_dim, cpu_output_indices = self.cpu_op_dim_exec_out(cpu_input1, item[1], item[2])
            npu_output_dim, npu_output_indices = self.npu_op_dim_exec_out(npu_input1, item[1], item[2])

            if npu_output_dim.dtype != np.float16:
                self.assertRtolEqual(npu_output_dim, cpu_output_dim)
                self.assertRtolEqual(npu_output_indices.astype(np.int32), cpu_output_indices.astype(np.int32))
            else:
                self.assertRtolEqual(npu_output_dim, cpu_output_dim.astype(np.float16))
                self.assertRtolEqual(npu_output_indices.astype(np.int32), cpu_output_indices.astype(np.int32))

    def amin_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_amin = self.cpu_op_amin_exec(cpu_input1, item[1], item[2])
            npu_output_amin = self.npu_op_amin_exec(npu_input1, item[1], item[2])
            npu_output_amin_out = self.npu_op_amin_exec_out(npu_input1, item[1], item[2], npu_input2)
            cpu_output_amin = cpu_output_amin.astype(npu_output_amin.dtype)
            self.assertRtolEqual(cpu_output_amin, npu_output_amin)
            self.assertRtolEqual(cpu_output_amin, npu_output_amin_out)

    def test_min_out_result(self, device="npu"):
        shape_format = [
            [[np.float16, 0, [9, 10, 14, 14]], [np.float16, 0, [7, 10, 1, 1]]],
            [[np.float16, 0, [9, 7, 12, 12]], [np.float16, 0, [7, 7, 1, 1]]],
            [[np.float32, 0, [2, 3, 3, 3]], [np.float32, 0, [3, 1, 3]]],
            [[np.float32, 0, [9, 13, 7, 7]], [np.float32, 0, [9, 13, 7, 7]]],
        ]
        self.min_out_result_other(shape_format)

    def test_min_shape_format_fp16_1d(self, device="npu"):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self.min_result(shape_format)

    def test_min_shape_format_fp32_1d(self, device="npu"):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result(shape_format)

    def test_min_shape_format_fp16_2d(self, device="npu"):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result(shape_format)

    def test_min_shape_format_fp32_2d(self, device="npu"):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result(shape_format)

    def test_min_shape_format_fp16_3d(self, device="npu"):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result(shape_format)

    def test_min_shape_format_fp32_3d(self, device="npu"):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result(shape_format)

    def test_min_shape_format_fp16_4d(self, device="npu"):
        format_list = [0, 4, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9, 10]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result(shape_format)

    def test_min_shape_format_fp32_4d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9, 10]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result(shape_format)

    def test_min_dim_shape_format_fp16_1d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self.min_result_dim(shape_format)

    def test_min_dim_shape_format_fp32_1d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result_dim(shape_format)

    def test_min_dim_shape_format_fp16_2d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result_dim(shape_format)

    def test_min_dim_shape_format_fp32_2d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result_dim(shape_format)

    def test_min_dim_shape_format_fp16_3d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result_dim(shape_format)

    def test_min_dim_shape_format_fp32_3d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result_dim(shape_format)

    def test_min_dim_shape_format_fp16_4d(self, device="npu"):
        format_list = [0, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.min_result_dim(shape_format)

    def test_min_dim_shape_format_fp32_4d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.min_result_dim(shape_format)

    def test_min_other_shape_format_fp16_1d(self, device="npu"):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self.min_result_other(shape_format)

    def test_min_other_shape_format_fp32_1d(self, device="npu"):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result_other(shape_format)

    def test_min_other_shape_format_fp16_2d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result_other(shape_format)

    def test_min_other_shape_format_fp32_2d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result_other(shape_format)

    def test_min_other_shape_format_fp16_3d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result_other(shape_format)

    def test_min_other_shape_format_fp32_3d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.min_result_other(shape_format)

    def test_min_other_shape_format_fp16_4d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.min_result_other(shape_format)

    def test_min_other_shape_format_fp32_4d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.min_result_other(shape_format)

    def test_min_dimname_shape_format(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9, 10], ('N', 'C', 'H', 'W')],
                         np.random.choice(['N', 'C', 'H', 'W']), j] for i in format_list for j
                        in
                        keepdim_list
                        ]
        self.min_name_result_other(shape_format)

    def test_min_dimname_shape_format_fp16(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9, 10], ('N', 'C', 'H', 'W')],
                         np.random.choice(['N', 'C', 'H', 'W']), j] for i in format_list for j
                        in
                        keepdim_list
                        ]
        self.min_name_result_other(shape_format)

    def test_min_dimname_out_shape_format(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9, 10], ('N', 'C', 'H', 'W')],
                         np.random.choice(['N', 'C', 'H', 'W']), j] for i in format_list for j
                        in
                        keepdim_list
                        ]
        self.min_name_out_result_other(shape_format)

    def test_min_dimname_out_shape_format_fp16(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9, 10], ('N', 'C', 'H', 'W')],
                         np.random.choice(['N', 'C', 'H', 'W']), j] for i in format_list for j
                        in
                        keepdim_list
                        ]
        self.min_name_out_result_other(shape_format)

    def test_amin_shape_format_fp16_1d(self, device="npu"):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in keepdim_list
                        ]
        self.amin_result(shape_format)

    def test_amin_shape_format_fp32_1d(self, device="npu"):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8]], np.random.randint(0, 1), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.amin_result(shape_format)

    def test_amin_shape_format_fp16_2d(self, device="npu"):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.amin_result(shape_format)

    def test_amin_shape_format_fp32_2d(self, device="npu"):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7]], np.random.randint(0, 2), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.amin_result(shape_format)

    def test_amin_shape_format_fp16_3d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.amin_result(shape_format)

    def test_amin_shape_format_fp32_3d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9]], np.random.randint(0, 3), j] for i in format_list for j in
                        keepdim_list
                        ]
        self.amin_result(shape_format)

    def test_amin_shape_format_fp16_4d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.amin_result(shape_format)

    def test_amin_shape_format_fp32_4d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [8, 7, 9, 10]], np.random.randint(0, 4), j] for i in format_list for j
                        in keepdim_list
                        ]
        self.amin_result(shape_format)


if __name__ == "__main__":
    run_tests()
