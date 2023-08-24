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


class TestProd(TestCase):

    def create_input_tensor(self, dtype, npu_format, shape, minValue, maxValue):
        input1 = np.random.uniform(minValue, maxValue, shape).astype(dtype)
        cpu_input = torch.from_numpy(input1)
        npu_input = torch.from_numpy(input1).npu()
        if npu_format != -1:
            npu_input = torch_npu.npu_format_cast(npu_input, npu_format)
        return cpu_input, npu_input

    def cpu_op_exec(self, input1):
        output = torch.prod(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.prod(input1)
        output = output.to("cpu").numpy()
        return output

    def cpu_op_dataType_exec(self, input1):
        output = torch.prod(input1, dtype=torch.float32)
        output = output.numpy()
        return output

    def npu_op_dataType_exec(self, input1):
        output = torch.prod(input1, dtype=torch.float32)
        output = output.to("cpu").numpy()
        return output

    def cpu_op_dim_exec(self, input1, dim, keepdim):
        output = torch.prod(input1, dim, keepdim)
        output = output.numpy()
        return output

    def npu_op_dim_exec(self, input1, dim, keepdim):
        output = torch.prod(input1, dim, keepdim)
        output = output.to("cpu").numpy()
        return output

    def cpu_op_dim_out_exec(self, input1, dim, keepdim, output):
        output = torch.prod(input1, dim, keepdim, out=output)
        output = output.numpy()
        return output

    def npu_op_dim_out_exec(self, input1, dim, keepdim, output):
        output = torch.prod(input1, dim, keepdim, out=output)
        output = output.to("cpu").numpy()
        return output

    def npu_op_dimname_exec(self, input1, dim, keepdim):
        output = torch.prod(input1, dim, keepdim)
        output = output.numpy()
        return output

    def npu_op_dimname_exec(self, input1, dim, keepdim):
        output = torch.prod(input1, dim, keepdim)
        output = output.to("cpu").numpy()
        return output

    def npu_op_dimname_out_exec(self, input1, dim, keepdim, output):
        output = torch.prod(input1, dim, keepdim, out=output)
        output = output.numpy()
        return output

    def npu_op_dimname_out_exec(self, input1, dim, keepdim, output):
        output = torch.prod(input1, dim, keepdim, out=output)
        output = output.to("cpu").numpy()
        return output

    def prod_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            cpu_output_dataType = self.cpu_op_dataType_exec(cpu_input1)
            npu_output_dataType = self.npu_op_dataType_exec(npu_input1)

            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_output_dataType = cpu_output_dataType.astype(npu_output_dataType.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output_dataType, npu_output_dataType)

    def prod_dim_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_dim = self.cpu_op_dim_exec(cpu_input1, item[1], item[2])
            npu_output_dim = self.npu_op_dim_exec(npu_input1, item[1], item[2])
            cpu_output_dim = cpu_output_dim.astype(npu_output_dim.dtype)
            self.assertRtolEqual(cpu_output_dim, npu_output_dim)

    def output_shape(self, item):
        output_size = list(item[0][2])
        dims = len(item[0][2])
        keepdim = item[2]
        dim = item[1]
        if dim < dims and keepdim is True:
            output_size[dim] = 1
        if dim < dims and keepdim is False:
            output_size.pop(dim)
        return output_size

    def prod_dim_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            shapes = self.output_shape(item)
            cpu_output, npu_output = self.create_input_tensor(item[0][0], item[0][1], shapes, 0, 1)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_output = cpu_output.to(torch.float32)
            cpu_output_dim = self.cpu_op_dim_out_exec(cpu_input1, item[1], item[2], cpu_output)
            npu_output_dim = self.npu_op_dim_out_exec(npu_input1, item[1], item[2], npu_output)

            cpu_output_dim = cpu_output_dim.astype(npu_output_dim.dtype)
            self.assertRtolEqual(cpu_output_dim, npu_output_dim)

    def prod_dim_name_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input1.names = item[0][3]
            npu_input1.names = item[0][3]

            cpu_output_dim = self.cpu_op_dim_exec(cpu_input1, item[1], item[2])
            npu_output_dim = self.npu_op_dim_exec(npu_input1, item[1], item[2])
            cpu_output_dim = cpu_output_dim.astype(npu_output_dim.dtype)
            self.assertRtolEqual(cpu_output_dim, npu_output_dim)

    def prod_dim_name_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            shapes = self.output_shape(item)
            cpu_output, npu_output = self.create_input_tensor(item[0][0], item[0][1], shapes, 0, 1)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_output = cpu_output.to(torch.float32)
            cpu_input1.names = item[0][3]
            npu_input1.names = item[0][3]
            cpu_output_dim = self.cpu_op_dim_out_exec(cpu_input1, item[1], item[2], cpu_output)
            npu_output_dim = self.npu_op_dim_out_exec(npu_input1, item[1], item[2], npu_output)
            cpu_output_dim = cpu_output_dim.astype(npu_output_dim.dtype)
            self.assertRtolEqual(cpu_output_dim, npu_output_dim)

    def test_prod_shape_format_fp16_1d(self):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18]], np.random.randint(0, 1), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_result(shape_format)

    def test_prod_shape_format_fp32_1d(self):
        format_list = [0, 3]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18]], np.random.randint(0, 1), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_result(shape_format)

    def test_prod_shape_format_fp16_2d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256]], np.random.randint(0, 2), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_result(shape_format)

    def test_prod_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 25]], np.random.randint(0, 2), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_result(shape_format)

    def test_prod_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 64, 32]], np.random.randint(0, 3), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_result(shape_format)

    def test_prod_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 64, 32]], np.random.randint(0, 3), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_result(shape_format)

    def test_prod_shape_format_fp16_4d(self):
        format_list = [0]
        keepdim_list = [True]
        shape_format = [[[np.float16, i, [18, 64, 32, 128]], np.random.randint(0, 4), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_result(shape_format)

    def test_prod_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 64, 32, 128]], np.random.randint(0, 4), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_result(shape_format)

    def test_prod_dim_shape_format_fp16_1d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18]], np.random.randint(0, 1), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_dim_result(shape_format)

    def test_prod_dim_shape_format_fp32_1d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18]], np.random.randint(0, 1), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_dim_result(shape_format)

    def test_prod_dim_shape_format_fp16_2d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256]], np.random.randint(0, 1), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_dim_result(shape_format)

    def test_prod_dim_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 64]], np.random.randint(0, 1), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_dim_result(shape_format)

    def test_prod_dim_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 64, 32]], np.random.randint(0, 3), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_dim_result(shape_format)

    def test_prod_dim_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 64, 32]], np.random.randint(0, 3), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_dim_result(shape_format)

    def test_prod_dim_shape_format_fp16_4d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 64, 32, 128]], np.random.randint(0, 4), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_dim_result(shape_format)

    def test_prod_dim_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 64, 32, 128]], np.random.randint(0, 4), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_dim_result(shape_format)

    def test_prod_dim_out_shape_format_fp16_1d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18]], np.random.randint(0, 1), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_dim_out_result(shape_format)

    def test_prod_dim_out_shape_format_fp32_1d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18]], np.random.randint(0, 1), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_dim_out_result(shape_format)

    def test_prod_dim_out_shape_format_fp16_2d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256]], np.random.randint(0, 1), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_dim_out_result(shape_format)

    def test_prod_dim_out_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 64]], np.random.randint(0, 1), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_dim_out_result(shape_format)

    def test_prod_dim_out_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 64, 32]], np.random.randint(0, 3), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_dim_out_result(shape_format)

    def test_prod_dim_out_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 64, 32]], np.random.randint(0, 3), j] for i in format_list
                        for j in keepdim_list
                        ]
        self.prod_dim_out_result(shape_format)

    def test_prod_dim_out_shape_format_fp16_4d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True]
        shape_format = [[[np.float16, i, [18, 64, 32, 128]], np.random.randint(0, 4), j]
                        for i in format_list for j in keepdim_list
                        ]
        self.prod_dim_out_result(shape_format)

    def test_prod_dim_out_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 64, 32, 128]], np.random.randint(0, 4), j]
                        for i in format_list for j in keepdim_list
                        ]
        self.prod_dim_out_result(shape_format)

    def test_prod_dim_name_shape_format_fp32_1d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18], ('N',)], np.random.randint(0, 1), j]
                        for i in format_list for j in keepdim_list
                        ]
        self.prod_dim_name_result(shape_format)

    def test_prod_dim_name_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 64], ('N', 'C')], np.random.randint(0, 2), j]
                        for i in format_list for j in keepdim_list
                        ]
        self.prod_dim_name_result(shape_format)

    def test_prod_dim_name_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 64, 32], ('N', 'C', 'H')], np.random.randint(0, 3), j]
                        for i in format_list for j in keepdim_list
                        ]
        self.prod_dim_name_result(shape_format)

    def test_prod_dim_name_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 64, 32, 128], ('N', 'C', 'H', 'W')], np.random.randint(0, 4), j]
                        for i in format_list for j in keepdim_list
                        ]
        self.prod_dim_name_result(shape_format)

    def test_prod_dim_name_out_shape_format_fp32_1d(self):
        format_list = [0, 3, 4]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18], ('N',)], np.random.randint(0, 1), j]
                        for i in format_list for j in keepdim_list
                        ]
        self.prod_dim_name_out_result(shape_format)

    def test_prod_dim_name_out_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 64], ('N', 'C')], np.random.randint(0, 1), j]
                        for i in format_list for j in keepdim_list
                        ]
        self.prod_dim_name_out_result(shape_format)

    def test_prod_dim_name_out_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 64, 32], ('N', 'C', 'H')], np.random.randint(0, 3), j]
                        for i in format_list for j in keepdim_list
                        ]
        self.prod_dim_name_out_result(shape_format)

    def test_prod_dim_name_out_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 64, 32, 128], ('N', 'C', 'H', 'W')], np.random.randint(0, 4), j]
                        for i in format_list for j in keepdim_list
                        ]
        self.prod_dim_name_out_result(shape_format)

    def test_prod_dtype(self):
        cpu_input1 = torch.empty(15, 15, 15, 16).uniform_(1.0, 2.0).to(torch.float16)
        npu_input1 = cpu_input1.npu()

        cpu_out = torch.prod(cpu_input1.float(), dim=0, keepdim=False, dtype=torch.float).half()
        npu_out = torch.prod(npu_input1, dim=0, keepdim=False, dtype=torch.float16).cpu()
        self.assertRtolEqual(cpu_out, npu_out)

        cpu_out1 = torch.rand(1)
        npu_out1 = cpu_out1.npu().half()
        torch.prod(cpu_input1.float(), dim=0, keepdim=False, dtype=torch.float, out=cpu_out1)
        torch.prod(npu_input1, dim=0, keepdim=False, dtype=torch.float16, out=npu_out1)
        self.assertRtolEqual(cpu_out1.half(), npu_out1.cpu())


if __name__ == "__main__":
    run_tests()
