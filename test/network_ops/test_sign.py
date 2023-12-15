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


class TestSign(TestCase):
    def cpu_op_exec(self, input1):
        cpu_output = torch.sign(input1)
        cpu_output = cpu_output.numpy()
        return cpu_output

    def npu_op_exec(self, input1):
        output = torch.sign(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2):
        torch.sign(input1, out=input2)
        output = input2.to("cpu")
        output = output.numpy()
        return output

    def cpu_inp_op_exec(self, input1):
        input1.sign_()
        output = input1.numpy()
        return output

    def npu_inp_op_exec(self, input1):
        input1.sign_()
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def sign_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, -100, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2)
            cpu_output_inp = self.cpu_inp_op_exec(cpu_input1)
            npu_output_inp = self.npu_inp_op_exec(npu_input1)

            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_output_inp = cpu_output_inp.astype(npu_output_inp.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output_out)
            self.assertRtolEqual(cpu_output_inp, npu_output_inp)

    def test_sign_shape_format_fp16_1d(self):
        format_list = [0, 3]
        shape_format = [[np.float16, i, [18]] for i in format_list]
        self.sign_result(shape_format)

    def test_sign_shape_format_fp16_2d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [5, 256]] for i in format_list]
        self.sign_result(shape_format)

    def test_sign_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [32, 3, 3]] for i in format_list]
        self.sign_result(shape_format)

    def test_sign_shape_format_fp16_4d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [64, 112, 7, 7]] for i in format_list]
        self.sign_result(shape_format)

    def test_sign_shape_format_fp32_1d(self):
        format_list = [0, 3]
        shape_format = [[np.float32, i, [18]] for i in format_list]
        self.sign_result(shape_format)

    def test_sign_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [5, 256]] for i in format_list]
        self.sign_result(shape_format)

    def test_sign_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [32, 3, 3]] for i in format_list]
        self.sign_result(shape_format)

    def test_sign_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [64, 112, 7, 7]] for i in format_list]
        self.sign_result(shape_format)

    def test_sign_shape_format_int32_1d(self):
        format_list = [0]
        shape_format = [[np.int32, i, [18]] for i in format_list]
        self.sign_result(shape_format)

    def test_sign_shape_format_int32_2d(self):
        format_list = [0]
        shape_format = [[np.int32, i, [5, 256]] for i in format_list]
        self.sign_result(shape_format)

    def test_sign_shape_format_int32_3d(self):
        format_list = [0]
        shape_format = [[np.int32, i, [32, 3, 3]] for i in format_list]
        self.sign_result(shape_format)

    def test_sign_shape_format_int32_4d(self):
        format_list = [0]
        shape_format = [[np.int32, i, [64, 112, 7, 7]] for i in format_list]
        self.sign_result(shape_format)


if __name__ == "__main__":
    run_tests()
