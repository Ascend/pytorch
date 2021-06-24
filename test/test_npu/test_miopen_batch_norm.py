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
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestBn(TestCase):
    def cpu_op_exec(self, input1, grad_tensor, dim, fun):
        input1.requires_grad_(True)
        grad_tensor = grad_tensor.to("cpu")
        if fun == "1d":
            m = torch.nn.BatchNorm1d(dim)
        elif fun == "2d":
            m = torch.nn.BatchNorm2d(dim)
        else:
            m = torch.nn.BatchNorm3d(dim)
        input_cpu = m(input1)
        input_cpu = input_cpu.detach().numpy()
        return input_cpu

    def npu_op_exec_new(self, input1, grad_tensor, dim, fun):
        grad_tensor = grad_tensor.to("npu")
        w = torch.ones_like(input1)
        w = w.to("npu")
        if fun == "1d":
            m = torch.nn.BatchNorm1d(dim)
        elif fun == "2d":
            m = torch.nn.BatchNorm2d(dim)
        else:
            m = torch.nn.BatchNorm3d(dim)
        m = m.to("npu")
        input_npu = m(input1)
        input_npu = input_npu.to("cpu")
        input_npu = input_npu.detach().numpy()
        return input_npu

    def do_test(self, item, prec, prec16, fun):
        cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
        if cpu_input1.dtype == torch.float16:
            cpu_input1 = cpu_input1.to(torch.float32)
        grad_tensor = torch.randn(item[0][2])
        cpu_output = self.cpu_op_exec(cpu_input1, grad_tensor, item[0][2][1], fun)
        npu_output = self.npu_op_exec_new(npu_input1, grad_tensor, item[0][2][1], fun)
        if (cpu_output.dtype != npu_output.dtype):
            cpu_output = cpu_output.astype(npu_output.dtype)
        self.assertRtolEqual(cpu_output, npu_output, prec, prec16)
    
    def test_batchnorm_shape_format(self, device):
        #pylint:disable=unused-argument
        shape_format_1d = [
                [[np.float32, 0, [25, 35, 40]]],
                [[np.float32, 0, [256, 672, 7]]],
                [[np.float32, 0, [256, 288, 14]]],
                [[np.float16, 0, [1024, 58, 56]]],
                [[np.float16, 0, [1024, 1024, 7]]],
                [[np.float16, 0, [1024, 24, 28]]],
         ]
        shape_format_2d = [
                [[np.float32, 3, [2, 3, 2, 2]]],
                [[np.float32, 3, [256, 672, 7, 7]]],
                [[np.float32, 3, [256, 288, 14, 14]]],
                [[np.float32, 3, [1024, 58, 28, 28]]],
                [[np.float32, 3, [1024, 116, 14, 14]]],
                [[np.float32, 3, [1024, 24, 112, 112]]],
                [[np.float16, 3, [1024, 58, 56, 56]]],
                [[np.float16, 3, [1024, 1024, 7, 7]]],
                [[np.float16, 3, [1024, 24, 28, 28]]],
                [[np.float16, 3, [1024, 116, 28, 28]]],
                [[np.float16, 3, [1024, 232, 7, 7]]],
                [[np.float16, 3, [1024, 232, 14, 14]]],
         ]
        shape_format_3d = [
                [[np.float32, -1, [2, 3, 2, 2, 5]]],
                [[np.float16, -1, [1024, 232, 14, 14, 4]]],
         ]
        # BatchNorm1d ok
        for item in shape_format_1d:
            self.do_test(item, prec = 0.001, prec16 = 0.01, fun = "1d")
        # BatchNorm2d ok
        for item in shape_format_2d:
            self.do_test(item, prec = 0.001, prec16 = 0.01, fun = "2d")


instantiate_device_type_tests(TestBn, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
