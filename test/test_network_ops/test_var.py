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

import itertools
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestVar(TestCase):

    def cpu_op_exec(self, input1, unbiased=True):
        output = torch.var(input1, unbiased=unbiased)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, unbiased=True):
        output = torch.var(input1, unbiased=unbiased)
        output = output.cpu().numpy()
        return output

    def cpu_op_dim_exec(self, input1, dim, unbiased=True, keepdim=False):
        output = torch.var(input1, dim, unbiased=unbiased, keepdim=keepdim)
        output = output.numpy()
        return output

    def npu_op_dim_exec(self, input1, dim, unbiased=True, keepdim=False):
        output = torch.var(input1, dim, unbiased=unbiased, keepdim=keepdim)
        output = output.cpu().numpy()
        return output

    def cpu_op_out_exec(self, input1, dim, output1, unbiased=True, keepdim=False):
        torch.var(input1, dim, unbiased=unbiased, keepdim=keepdim, out=output1)
        output1 = output1.numpy()
        return output1

    def npu_op_out_exec(self, input1, dim, output1, unbiased=True, keepdim=False):
        torch.var(input1, dim, unbiased=unbiased, keepdim=keepdim, out=output1)
        output1 = output1.cpu().numpy()
        return output1

    def cpu_op_mean_exec(self, input1, unbiased=True):
        output = torch.var_mean(input1, unbiased=unbiased)
        output1 = output[0].numpy()
        output2 = output[1].numpy()
        return output1, output2

    def npu_op_mean_exec(self, input1, unbiased=True):
        output = torch.var_mean(input1, unbiased=unbiased)
        output1 = output[0].cpu().numpy()
        output2 = output[1].cpu().numpy()
        return output1, output2

    def cpu_op_mean_dim_exec(self, input1, dim, unbiased=True, keepdim=False):
        output = torch.var_mean(input1, dim, unbiased=unbiased, keepdim=keepdim)
        output1 = output[0].numpy()
        output2 = output[1].numpy()
        return output1, output2

    def npu_op_mean_dim_exec(self, input1, dim, unbiased=True, keepdim=False):
        output = torch.var_mean(input1, dim, unbiased=unbiased, keepdim=keepdim)
        output1 = output[0].cpu().numpy()
        output2 = output[1].cpu().numpy()
        return output1, output2

    def cpu_op_out_correction_exec(self, input1, dim, output, correction=1, keepdim=False):
        torch.var(input1, dim, correction=correction, keepdim=keepdim, out=output)
        output = output.numpy()
        return output

    def npu_op_out_correction_exec(self, input1, dim, output, correction=1, keepdim=False):
        torch.var(input1, dim, correction=correction, keepdim=keepdim, out=output)
        output = output.cpu().numpy()
        return output

    def cpu_op_correction_exec(self, input1, dim, correction=1, keepdim=False):
        output = torch.var(input1, dim, correction=correction, keepdim=keepdim)
        output = output.numpy()
        return output

    def npu_op_correction_exec(self, input1, dim, correction=1, keepdim=False):
        output = torch.var(input1, dim, correction=correction, keepdim=keepdim)
        output = output.cpu().numpy()
        return output

    def cpu_op_mean_correction_exec(self, input1, dim, correction=1, keepdim=False):
        output1, output2 = torch.var_mean(input1, dim, correction=correction, keepdim=keepdim)
        output1 = output1.numpy()
        output2 = output2.numpy()
        return output1, output2

    def npu_op_mean_correction_exec(self, input1, dim, correction=1, keepdim=False):
        output1, output2 = torch.var_mean(input1, dim, correction=correction, keepdim=keepdim)
        output1 = output1.cpu().numpy()
        output2 = output2.cpu().numpy()
        return output1, output2

    def output_shape(self, inputshape, dim, unbiased=True, keepdim=False):
        shape = list(inputshape)
        if dim < len(inputshape):
            if keepdim:
                shape[dim] = 1
            else:
                shape.pop(dim)
        return shape

    def create_output_tensor(self, minvalue, maxvalue, shape, npuformat, dtype):
        input1 = np.random.uniform(minvalue, maxvalue, shape).astype(dtype)
        cpu_input = torch.from_numpy(input1)
        npu_input = torch.from_numpy(input1).npu()
        if npuformat != -1:
            npu_input = npu_input.npu_format_cast(npuformat)
        return cpu_input, npu_input

    def test_var_shape_format(self):
        format_list = [-1]
        shape_list = [[32, 24], [32, 8, 24]]
        unbiased_list = [True, False]
        dtype_list = [np.float16, np.float32]

        for item in itertools.product(dtype_list, format_list, shape_list, unbiased_list):
            cpu_input, npu_input = create_common_tensor(item[:3], 0, 100)
            if item[0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input, item[3])
            if item[0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            cpu_output = cpu_output.astype(item[0])
            npu_output = self.npu_op_exec(npu_input, item[3])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_var_dim_shape_format(self):
        format_list = [-1]
        shape_list = [[32, 24], [32, 8, 24]]
        dim_list = [0]
        unbiased_list = [True, False]
        keepdim_list = [True, False]
        dtype_list = [np.float16, np.float32]

        for item in itertools.product(dtype_list, format_list, shape_list, dim_list, unbiased_list, keepdim_list):
            cpu_input, npu_input = create_common_tensor(item[:3], 0, 100)
            if item[0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_dim_exec(cpu_input, item[3], item[4], item[5])
            if item[0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            npu_output = self.npu_op_dim_exec(npu_input, item[3], item[4], item[5])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_var_names_dim_shape_format(self):
        shape = (1024, 8, 32)
        dimlist = ['N', 'C', 'H']
        cpu_input = torch.rand(shape, dtype=torch.float32, names=('N', 'C', 'H'))
        dim = np.random.choice(dimlist)
        for dtype in [np.float16, np.float32]:
            if dtype == np.float16:
                npu_input = cpu_input.to(torch.float16).npu()
            else:
                npu_input = cpu_input.npu()
            cpu_output = self.cpu_op_dim_exec(cpu_input, dim=dim)
            if dtype == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            npu_output = self.npu_op_dim_exec(npu_input, dim=dim)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_var_out_shape_format(self):
        format_list = [-1]
        shape_list = [[32, 24], [32, 8, 24]]
        dim_list = [0]
        dtype_list = [np.float16, np.float32]
        unbiased_list = [True, False]
        keepdim_list = [True, False]

        for item in itertools.product(dtype_list, format_list, shape_list, dim_list, unbiased_list, keepdim_list):
            cpu_input, npu_input = create_common_tensor(item[:3], 0, 100)
            outputshape = self.output_shape(item[2], item[3], item[5])
            cpu_output, npu_output = self.create_output_tensor(0, 1, outputshape, item[1], item[0])
            if item[0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
                cpu_output = cpu_output.to(torch.float32)
            cpu_output = self.cpu_op_out_exec(cpu_input, item[3], cpu_output, item[4], item[5])
            if item[0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            npu_output = self.npu_op_out_exec(npu_input, item[3], npu_output, item[4], item[5])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_var_mean_shape_format(self):
        format_list = [-1]
        shape_list = [[32, 24], [32, 8, 24]]
        unbiased_list = [True, False]
        dtype_list = [np.float16, np.float32]

        for item in itertools.product(dtype_list, format_list, shape_list, unbiased_list):
            cpu_input, npu_input = create_common_tensor(item[:3], 0, 100)
            if item[0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output1, cpu_output2 = self.cpu_op_mean_exec(cpu_input, item[3])
            if item[0] == np.float16:
                cpu_output1 = cpu_output1.astype(np.float16)
                cpu_output2 = cpu_output2.astype(np.float16)
            npu_output1, npu_output2 = self.npu_op_mean_exec(npu_input, item[3])
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_var_mean_dim_shape_format(self):
        format_list = [-1]
        shape_list = [[32, 24], [32, 8, 24]]
        dim_list = [0]
        unbiased_list = [True, False]
        keepdim_list = [True, False]
        dtype_list = [np.float16, np.float32]

        for item in itertools.product(dtype_list, format_list, shape_list, dim_list, unbiased_list, keepdim_list):
            cpu_input, npu_input = create_common_tensor(item[:3], 0, 100)
            if item[0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output1, cpu_output2 = self.cpu_op_mean_dim_exec(cpu_input, item[3], item[4], item[5])
            if item[0] == np.float16:
                cpu_output1 = cpu_output1.astype(np.float16)
                cpu_output2 = cpu_output2.astype(np.float16)
            npu_output1, npu_output2 = self.npu_op_mean_dim_exec(npu_input, item[3], item[4], item[5])
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_var_mean_names_dim_shape_format(self):
        shape = (1024, 8, 32)
        dimlist = ['N', 'C', 'H']
        cpu_input = torch.rand(shape, dtype=torch.float32, names=('N', 'C', 'H'))
        dim = np.random.choice(dimlist)
        for dtype in [np.float16, np.float32]:
            if dtype == np.float16:
                npu_input = cpu_input.to(torch.float16).npu()
            else:
                npu_input = cpu_input.npu()
            cpu_output1, cpu_output2 = self.cpu_op_mean_dim_exec(cpu_input, dim=dim)
            if dtype == np.float16:
                cpu_output1 = cpu_output1.astype(dtype)
                cpu_output2 = cpu_output2.astype(dtype)
            npu_output1, npu_output2 = self.npu_op_mean_dim_exec(npu_input, dim=dim)
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_var_dim_shape_format_5d_fp16(self):
        shape = [2, 94, 4, 52, 192]
        unbiased_list = [True, False]
        keepdim_list = [True, False]

        for item in itertools.product(unbiased_list, keepdim_list):
            cpu_input1, npu_input1 = create_common_tensor((np.float16, -1, shape), 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output1 = self.cpu_op_dim_exec(cpu_input1, 0, item[0], item[1])
            cpu_output1 = cpu_output1.astype(np.float16)
            npu_output1 = self.npu_op_dim_exec(npu_input1, 0, item[0], item[1])
            self.assertRtolEqual(cpu_output1, npu_output1, prec16=0.004)

    def test_var_out_correction_shape_format(self):
        format_list = [-1]
        shape_list = [[32, 24], [32, 8, 24]]
        dim_list = [0]
        dtype_list = [np.float16, np.float32]
        correction_list = [0, 1]
        keepdim_list = [True, False]

        for item in itertools.product(dtype_list, format_list, shape_list, dim_list, correction_list, keepdim_list):
            cpu_input, npu_input = create_common_tensor(item[:3], 0, 100)
            outputshape = self.output_shape(item[2], item[3], item[5])
            cpu_output, npu_output = self.create_output_tensor(0, 1, outputshape, item[1], item[0])
            if item[0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
                cpu_output = cpu_output.to(torch.float32)
            cpu_output = self.cpu_op_out_correction_exec(cpu_input, item[3], cpu_output, item[4], item[5])
            if item[0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            npu_output = self.npu_op_out_correction_exec(npu_input, item[3], npu_output, item[4], item[5])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_var_correction_shape_format(self):
        format_list = [-1]
        shape_list = [[32, 24], [32, 8, 24]]
        dim_list = [0]
        correction_list = [0, 1]
        keepdim_list = [True, False]
        dtype_list = [np.float16, np.float32]

        for item in itertools.product(dtype_list, format_list, shape_list, dim_list, correction_list, keepdim_list):
            cpu_input, npu_input = create_common_tensor(item[:3], 0, 100)
            if item[0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_correction_exec(cpu_input, item[3], item[4], item[5])
            if item[0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            npu_output = self.npu_op_correction_exec(npu_input, item[3], item[4], item[5])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_var_names_correction_shape_format(self):
        shape = (1024, 8, 32)
        dimlist = ['N', 'C', 'H']
        cpu_input = torch.rand(shape, dtype=torch.float32, names=('N', 'C', 'H'))
        dim = np.random.choice(dimlist)
        for dtype in [np.float16, np.float32]:
            if dtype == np.float16:
                npu_input = cpu_input.to(torch.float16).npu()
            else:
                npu_input = cpu_input.npu()
            cpu_output = self.cpu_op_correction_exec(cpu_input, dim=dim)
            if dtype == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            npu_output = self.npu_op_correction_exec(npu_input, dim=dim)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_var_mean_correction_shape_format(self):
        format_list = [-1]
        shape_list = [[32, 24], [32, 8, 24]]
        dim_list = [0]
        dtype_list = [np.float16, np.float32]
        correction_list = [0, 1]
        keepdim_list = [True, False]

        for item in itertools.product(dtype_list, format_list, shape_list, dim_list, correction_list, keepdim_list):
            cpu_input, npu_input = create_common_tensor(item[:3], 0, 100)
            if item[0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output1, cpu_output2 = self.cpu_op_mean_correction_exec(cpu_input, item[3], item[4], item[5])
            if item[0] == np.float16:
                cpu_output1 = cpu_output1.astype(np.float16)
                cpu_output2 = cpu_output2.astype(np.float16)
            npu_output1, npu_output2 = self.npu_op_mean_correction_exec(npu_input, item[3], item[4], item[5])
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_var_mean_names_correction_shape_format(self):
        shape = (1024, 8, 32)
        dimlist = ['N', 'C', 'H']
        cpu_input = torch.rand(shape, dtype=torch.float32, names=('N', 'C', 'H'))
        dim = np.random.choice(dimlist)
        for dtype in [np.float16, np.float32]:
            if dtype == np.float16:
                npu_input = cpu_input.to(torch.float16).npu()
            else:
                npu_input = cpu_input.npu()
            cpu_output1, cpu_output2 = self.cpu_op_mean_correction_exec(cpu_input, dim=dim)
            if dtype == np.float16:
                cpu_output1 = cpu_output1.astype(np.float16)
                cpu_output2 = cpu_output2.astype(np.float16)
            npu_output1, npu_output2 = self.npu_op_mean_correction_exec(npu_input, dim=dim)
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)


if __name__ == "__main__":
    run_tests()
