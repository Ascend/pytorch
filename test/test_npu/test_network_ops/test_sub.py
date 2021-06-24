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
import torch.nn as nn
import numpy as np
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestSub(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = input1 - input2
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = input1 - input2
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_t(self, input1, input2):
        output = torch.sub(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_t_out(self, input1, input2, input3):
        torch.sub(input1, input2, out=input3)
        output = input3.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_tensor(self, input1, input2):
        output = input1.sub(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_inp_tensor(self, input1, input2):
        input1.sub_(input2)
        output = input1.numpy()
        return output

    def npu_op_exec_inp_tensor(self, input1, input2):
        input1.sub_(input2)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def sub_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            npu_input3 = copy.deepcopy(cpu_input1).to("npu")
            npu_input4 = torch.randn(6).to("npu").to(npu_input3.dtype)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)

            if type(item[1]) == list:
                cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
                if cpu_input2.dtype == torch.float16:
                    cpu_input2 = cpu_input2.to(torch.float32)
            else:
                cpu_input2 = item[1]
                npu_input2 = item[1]

            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)

            npu_output_t = self.npu_op_exec_t(npu_input1, npu_input2)
            npu_output_t_out = self.npu_op_exec_t_out(npu_input1, npu_input2, npu_input3)
            npu_output_tensor = self.npu_op_exec_tensor(npu_input1, npu_input2)
            npu_output_t_out_chk = self.npu_op_exec_t_out(npu_input1, npu_input2, npu_input4)#out tensor shape not shape as self

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output_t)
            self.assertRtolEqual(cpu_output, npu_output_t_out)
            self.assertRtolEqual(cpu_output, npu_output_tensor)
            self.assertRtolEqual(cpu_output, npu_output_t_out_chk)

            # test for tensor
            cpu_input1_tensor, npu_input1_tensor = create_common_tensor(item[0], 0, 100)
            if cpu_input1_tensor.dtype == torch.float16:
                cpu_input1_tensor = cpu_input1_tensor.to(torch.float32)

            if type(item[1]) == list:
                cpu_input2_tensor, npu_input2_tensor = create_common_tensor(item[1], 0, 100)
                if cpu_input2_tensor.dtype == torch.float16:
                    cpu_input2_tensor = cpu_input2_tensor.to(torch.float32)
            else:
                cpu_input2_tensor = item[1]
                npu_input2_tensor = item[1]

            cpu_output_inp_tensor = self.cpu_op_exec_inp_tensor(cpu_input1_tensor, cpu_input2_tensor)
            npu_output_inp_tensor = self.npu_op_exec_inp_tensor(npu_input1_tensor, npu_input2_tensor)
            cpu_output_inp_tensor = cpu_output_inp_tensor.astype(npu_output_inp_tensor.dtype)
            self.assertRtolEqual(cpu_output_inp_tensor, npu_output_inp_tensor)

    def test_sub_scalar_shape_format_fp32_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float32, i, [448]], np.random.uniform(0, 100)] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_scalar_shape_format_fp32_2d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [1000, 1280]], np.random.uniform(0, 100)] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_scalar_shape_format_fp32_3d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [32, 3, 3]], np.random.uniform(0, 100)] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_scalar_shape_format_fp32_4d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [256, 480, 14, 14]], np.random.uniform(0, 100)] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_scalar_shape_format_int32_1d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [448]], np.random.randint(0, 100)] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_scalar_shape_format_int32_2d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [64, 7]], np.random.randint(0, 100)] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_scalar_shape_format_int32_3d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [64, 7, 58]], np.random.randint(0, 100)] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_scalar_shape_format_int32_4d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [256, 480, 14, 14]], np.random.randint(0, 100)] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_shape_format_fp16_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float16, i, [448]], [np.float16, i, [448]]] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_shape_format_fp16_2d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [1000, 1280]], [np.float16, i, []]] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_shape_format_fp16_3d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [32, 3, 3]], [np.float16, i, []]] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_shape_format_fp16_4d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [256, 480, 14, 14]], [np.float16, i, []]] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_shape_format_fp32_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float32, i, [448]], [np.float32, i, []]] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_shape_format_fp32_2d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [1000, 1280]], [np.float32, i, []]] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_shape_format_fp32_3d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [32, 3, 3]], [np.float32, i, []]] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_shape_format_fp32_4d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [256, 480, 14, 14]], [np.float32, i, []]] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_shape_format_int32_1d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [448]], [np.int32, i, []]] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_shape_format_int32_2d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [64, 7]], [np.int32, i, []]] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_shape_format_int32_3d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [64, 7, 58]], [np.int32, i, []]] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_shape_format_int32_4d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [256, 480, 14, 14]], [np.int32, i, []]] for i in format_list]
        self.sub_result(shape_format)
'''
    # unsupport
    def test_sub_scalar_shape_format_fp16_1d(self, device):
        format_list = [-1, 0, 3, 4]
        shape_format = [[[np.float16, i, [448]], np.random.uniform(0, 100)] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_scalar_shape_format_fp16_2d(self, device):
        format_list = [-1, 0, 3, 4, 29]
        shape_format = [[[np.float16, i, [1000, 1280]], np.random.uniform(0, 100)] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_scalar_shape_format_fp16_3d(self, device):
        format_list = [-1, 0, 3, 4, 29]
        shape_format = [[[np.float16, i, [32, 3, 3]], np.random.uniform(0, 100)] for i in format_list]
        self.sub_result(shape_format)

    def test_sub_scalar_shape_format_fp16_4d(self, device):
        format_list = [-1, 0, 3, 4, 29]
        shape_format = [[[np.float16, i, [256, 480, 14, 14]], np.random.uniform(0, 100)] for i in format_list]
        self.sub_result(shape_format)
'''
instantiate_device_type_tests(TestSub, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
