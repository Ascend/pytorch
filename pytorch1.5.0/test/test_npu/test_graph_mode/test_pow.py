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
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from graph_utils import RunFuncInGraphMode


class TestPow(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.pow(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.pow(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, out):
        torch.pow(input1, input2, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_exec(self, input1, input2):
        input1.pow_(input2)
        output = input1.numpy()
        return output

    def npu_op_inplace_exec(self, input1, input2):
        input1.pow_(input2)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_tensor_scalar(self, input1, n):
        output = torch.pow(input1, n)
        output = output.numpy()
        return output

    def npu_op_exec_tensor_scalar(self, input1, n):
        output = torch.pow(input1, n)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_tensor_scalar_out(self, input1, n, out):
        output = torch.pow(input1, n, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_scalar_tensor(self, n, input1):
        output = torch.pow(n, input1)
        output = output.numpy()
        return output

    def npu_op_exec_scalar_tensor(self, n, input1):
        output = torch.pow(n, input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar_tensor_out(self, n, input1, out):
        torch.pow(n, input1, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def pow_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 1)
            npu_input3 = copy.deepcopy(cpu_input1).to("npu")
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
            cpu_output_inp = self.cpu_op_inplace_exec(cpu_input1, cpu_input2)
            npu_output_inp = self.npu_op_inplace_exec(npu_input1, npu_input2)

            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_output_inp = cpu_output_inp.astype(npu_output_inp.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output_out)
            self.assertRtolEqual(cpu_output_inp, npu_output_inp)

    def pow_result_scalar_tensor(self, shape_format):
        for item in shape_format:
            scalar = np.random.randint(0, 1)
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 1)
            npu_input3 = copy.deepcopy(cpu_input1).to("npu")
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_scalar = self.cpu_op_exec_scalar_tensor(scalar, cpu_input1)
            npu_output_scalar = self.npu_op_exec_scalar_tensor(scalar, npu_input1)
            npu_output_scalar_out = self.npu_op_exec_scalar_tensor_out(scalar, npu_input1, npu_input3)

            cpu_output_scalar = cpu_output_scalar.astype(npu_output_scalar.dtype)
            self.assertRtolEqual(cpu_output_scalar, npu_output_scalar)
            self.assertRtolEqual(cpu_output_scalar, npu_output_scalar_out)

    def pow_result_tensor_scalar_(self, shape_format):
        for item in shape_format:
            scalar = np.random.randint(0, 1)
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 1)
            npu_input3 = copy.deepcopy(cpu_input1).to("npu")
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_tensor_scalar = self.cpu_op_exec_tensor_scalar(cpu_input1, scalar)
            npu_output_tensor_scalar = self.npu_op_exec_tensor_scalar(npu_input1, scalar)
            npu_output_tensor_scalar_out = self.npu_op_exec_tensor_scalar_out(npu_input1, scalar, npu_input3)

            cpu_output_tensor_scalar = cpu_output_tensor_scalar.astype(npu_output_tensor_scalar.dtype)
            self.assertRtolEqual(cpu_output_tensor_scalar, npu_output_tensor_scalar)
            self.assertRtolEqual(cpu_output_tensor_scalar, npu_output_tensor_scalar_out)

    # scalar_tensor-------------------------------------------------------
    @RunFuncInGraphMode
    def test_pow_shape_format_scalar_tensor_fp16_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[np.float16, i, [18]] for i in format_list]
        self.pow_result_scalar_tensor(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_scalar_tensor_fp32_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[np.float32, i, [18]] for i in format_list]
        self.pow_result_scalar_tensor(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_scalar_tensor_fp16_2d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float16, i, [18, 64]] for i in format_list]
        self.pow_result_scalar_tensor(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_scalar_tensor_fp32_2d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float32, i, [18, 64]] for i in format_list]
        self.pow_result_scalar_tensor(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_scalar_tensor_fp16_3d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float16, i, [18, 64, 128]] for i in format_list]
        self.pow_result_scalar_tensor(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_scalar_tensor_fp32_3d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float32, i, [18, 64, 128]] for i in format_list]
        self.pow_result_scalar_tensor(shape_format)

    # tensor_scalar-----------------------------------------------------------
    @RunFuncInGraphMode
    def test_pow_shape_format_tensor_scala_fp16_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[np.float16, i, [18]] for i in format_list]
        self.pow_result_tensor_scalar_(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_tensor_scalar_fp32_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[np.float32, i, [18]] for i in format_list]
        self.pow_result_tensor_scalar_(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_tensor_scala_fp16_2d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float16, i, [18, 64]] for i in format_list]
        self.pow_result_tensor_scalar_(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_tensor_scalar_fp32_2d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float32, i, [18, 64]] for i in format_list]
        self.pow_result_tensor_scalar_(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_tensor_scala_fp16_3d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float16, i, [18, 64, 128]] for i in format_list]
        self.pow_result_tensor_scalar_(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_tensor_scalar_fp32_3d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float32, i, [18, 64, 128]] for i in format_list]
        self.pow_result_tensor_scalar_(shape_format)

    # tensor_tensor-----------------------------------------------------------
    '''
    @RunFuncInGraphMode
    def test_pow_shape_format_fp16_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float16, i, [5]], [np.float16, i, []]] for i in format_list]
        self.pow_result(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_fp32_1d(self, device):
        format_list = [-1, 0, 3,]
        shape_format = [[[np.float32, i, [5]], [np.float32, i, []]] for i in format_list]
        self.pow_result(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_fp16_2d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [448, 1]], [np.float16, i, []]] for i in format_list]
        self.pow_result(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_fp32_2d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [448, 1]], [np.float32, i, []]] for i in format_list]
        self.pow_result(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_fp16_3d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [16, 640, 640]], [np.float16, i, []]] for i in format_list]
        self.pow_result(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_fp32_3d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [16, 640, 640]], [np.float32, i, []]] for i in format_list]
        self.pow_result(shape_format)
    '''
    #broadcast
    @RunFuncInGraphMode
    def test_pow_shape_format_fp16_2d_broadcast(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [448, 20]], [np.float16, i, [448,1]]] for i in format_list]
        self.pow_result(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_fp32_2d_broadcast(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [448, 20]], [np.float32, i, [448,1]]] for i in format_list]
        self.pow_result(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_fp16_3d_broadcast(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [16, 640, 640]], [np.float16, i, [16, 640, 1]]] for i in format_list]
        self.pow_result(shape_format)

    @RunFuncInGraphMode
    def test_pow_shape_format_fp32_3d_broadcast(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [16, 640, 640]], [np.float32, i, [16, 1, 1]]] for i in format_list]
        self.pow_result(shape_format)

instantiate_device_type_tests(TestPow, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
