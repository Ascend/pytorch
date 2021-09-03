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


class TestRsub(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = input2 - input1
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = input2 - input1
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar(self, input1, input2):
        output = input1 - input2
        output = output.to("cpu")
        output = output.numpy()
        output = -output
        return output

    def rsub_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def rsub_scalar_result(self, shape_format):
        for item in shape_format:
            scalar = np.random.uniform(0, 100)
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, scalar)
            npu_output_scalar = self.npu_op_exec_scalar(npu_input1, scalar)

            cpu_output = cpu_output.astype(npu_output_scalar.dtype)
            self.assertRtolEqual(cpu_output, npu_output_scalar)

    def test_sub_shape_format_fp16_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float16, i, [32]], [np.float16, i, [32]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_fp32_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float16, i, [32]], [np.float16, i, [32]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_fp16_2d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [5, 3]], [np.float16, i, [5, 3]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_fp32_2d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [5, 3]], [np.float16, i, [5, 3]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_fp16_3d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [256, 480, 14]], [np.float16, i, [256, 480, 14]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_fp32_3d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [256, 480, 14]], [np.float16, i, [256, 480, 14]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_fp16_4d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [32, 3, 3, 3]], [np.float16, i, [32, 3, 3, 3]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_fp32_4d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [32, 3, 3, 3]], [np.float16, i, [32, 3, 3, 3]]] for i in format_list]
        self.rsub_result(shape_format)

    # int-------------------------------------------------------------------------------
    def test_sub_shape_format_int32_1d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [32]], [np.int32, i, [32]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_int32_2d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [5, 3]], [np.int32, i, [5, 3]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_int32_3d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [256, 480, 14]], [np.int32, i, [256, 480, 14]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_int32_4d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [32, 3, 3, 3]], [np.int32, i, [32, 3, 3, 3]]] for i in format_list]
        self.rsub_result(shape_format)

    # scalar----------------------------------------------------------------------------
    def test_sub_scalar_shape_format_fp16_1d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.float16, i, [32]]] for i in format_list]
        self.rsub_scalar_result(shape_format)

    def test_sub_scalar_shape_format_fp32_1d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.float16, i, [32]]] for i in format_list]
        self.rsub_scalar_result(shape_format)

    def test_sub_scalar_shape_format_fp16_2d(self, device):
        format_list = []
        shape_format = [[[np.float16, i, [32, 64]]] for i in format_list]
        self.rsub_scalar_result(shape_format)

    def test_sub_scalar_shape_format_fp32_2d(self, device):
        format_list = []
        shape_format = [[[np.float16, i, [32, 64]]] for i in format_list]
        self.rsub_scalar_result(shape_format)

    def test_sub_scalar_shape_format_fp16_3d(self, device):
        format_list = []
        shape_format = [[[np.float16, i, [32, 64, 128]]] for i in format_list]
        self.rsub_scalar_result(shape_format)

    def test_sub_scalar_shape_format_fp32_3d(self, device):
        format_list = []
        shape_format = [[[np.float16, i, [32, 64, 128]]] for i in format_list]
        self.rsub_scalar_result(shape_format)

    def test_sub_scalar_shape_format_fp16_4d(self, device):
        format_list = []
        shape_format = [[[np.float16, i, [32, 64, 128, 28]]] for i in format_list]
        self.rsub_scalar_result(shape_format)

    def test_sub_scalar_shape_format_fp32_4d(self, device):
        format_list = []
        shape_format = [[[np.float16, i, [32, 64, 128, 28]]] for i in format_list]
        self.rsub_scalar_result(shape_format)


instantiate_device_type_tests(TestRsub, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
