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

class Testqr(TestCase):

    def cpu_op_exec(self, input1, some):
        out = torch.qr(input1, some)
        output_q = out.Q
        output_r = out.R
        output_q = output_q.numpy()
        output_r = output_r.numpy()
        return output_q, output_r, out

    def npu_op_exec(self, input1, some):
        out = torch.qr(input1.to("npu"), some)
        output_q = out.Q
        output_r = out.R
        output_q = output_q.to("cpu")
        output_r = output_r.to("cpu")
        output_q = output_q.numpy()
        output_r = output_r.numpy()
        return output_q, output_r, out
# pylint: disable=W0613
    def test_qr_common_shape_format(self, device):

        shape_format = [
            [np.float32, -1, (5, 3)],
            [np.float32, -1, (1, 64, 147, 147)],
            [np.float32, -1, (65536, 14, 7, 1)],
            [np.int32, -1, (1000000, 3, 3, 1)],
            [np.int32, -1, (1024, 107, 31, 2)],
            [np.int32, -1, (1, 128, 1, 1)]
        ]
        for item in shape_format:
            some = True
            cpu_input1, npu_input1 = create_common_tensor(item, -0.001, 0.001)
            if cpu_input1.dtype == torch.int32:
                cpu_input1 = cpu_input1.to(torch.float32)
            if npu_input1.dtype == torch.int32:
                npu_input1 = npu_input1.to(torch.float32)       
            cpu_output_q, cpu_output_r, cpu_out = self.cpu_op_exec(cpu_input1, some)
            npu_output_q, npu_output_r, npu_out = self.npu_op_exec(npu_input1, some)
            npu_output = np.matmul(npu_output_q, npu_output_r)

            self.assertRtolEqual(cpu_output_q, npu_output_q)
            self.assertRtolEqual(cpu_output_r, npu_output_r)
            self.assertRtolEqual(cpu_input1.numpy(), npu_output)
            self.assertRtolEqual(cpu_out, npu_out)

    def test_qr_float16_shape_format(self, device):
        shape_format = [
            [np.float16, -1, (5, 3)],
            [np.float16, -1, (1, 64, 147, 147)],
            [np.float16, -1, (65536, 14, 7, 1)],
            [np.float16, -1, (1000000, 3, 3, 1)],
            [np.float16, -1, (1024, 107, 31, 2)],
            [np.float16, -1, (1, 128, 1, 1)]
        ]
        for item in shape_format:
            some = True
            cpu_input1, npu_input1 = create_common_tensor(item, -0.001, 0.001)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if npu_input1.dtype == torch.float16:
                npu_input1 = npu_input1.to(torch.float32)
            cpu_output_q, cpu_output_r, cpu_out = self.cpu_op_exec(cpu_input1, some)
            npu_output_q, npu_output_r, npu_out = self.npu_op_exec(npu_input1, some)
            npu_output = np.matmul(npu_output_q, npu_output_r)

            self.assertRtolEqual(cpu_output_q, npu_output_q)
            self.assertRtolEqual(cpu_output_r, npu_output_r)
            self.assertRtolEqual(cpu_input1.numpy(), npu_output)
            self.assertRtolEqual(cpu_out, npu_out)

    def test_qr_common_False_shape_format(self, device):

        shape_format = [
            [np.float32, -1, (5, 3)],
            [np.float32, -1, (1, 64, 147, 147)],
            [np.float32, -1, (65536, 14, 7, 1)],
            [np.int32, -1, (1000000, 3, 3, 1)],
            [np.int32, -1, (1024, 107, 31, 2)],
            [np.int32, -1, (1, 128, 1, 1)]
        ]
        for item in shape_format:
            some = False
            cpu_input1, npu_input1 = create_common_tensor(item, -0.001, 0.001)
            if cpu_input1.dtype == torch.int32:
                cpu_input1 = cpu_input1.to(torch.float32)
            if npu_input1.dtype == torch.int32:
                npu_input1 = npu_input1.to(torch.float32)       
            cpu_output_q, cpu_output_r, cpu_out = self.cpu_op_exec(cpu_input1, some)
            npu_output_q, npu_output_r, npu_out = self.npu_op_exec(npu_input1, some)
            npu_output = np.matmul(npu_output_q, npu_output_r)

            self.assertRtolEqual(cpu_output_q, npu_output_q)
            self.assertRtolEqual(cpu_output_r, npu_output_r)
            self.assertRtolEqual(cpu_input1.numpy(), npu_output)
            self.assertRtolEqual(cpu_out, npu_out)

    def test_qr_float16_False_shape_format(self, device):
        shape_format = [
            [np.float16, -1, (5, 3)],
            [np.float16, -1, (1, 64, 147, 147)],
            [np.float16, -1, (65536, 14, 7, 1)],
            [np.float16, -1, (1000000, 3, 3, 1)],
            [np.float16, -1, (1024, 107, 31, 2)],
            [np.float16, -1, (1, 128, 1, 1)]
        ]
        for item in shape_format:
            some = False
            cpu_input1, npu_input1 = create_common_tensor(item, -0.001, 0.001)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if npu_input1.dtype == torch.float16:
                npu_input1 = npu_input1.to(torch.float32)
            cpu_output_q, cpu_output_r, cpu_out = self.cpu_op_exec(cpu_input1, some)
            npu_output_q, npu_output_r, npu_out = self.npu_op_exec(npu_input1, some)
            npu_output = np.matmul(npu_output_q, npu_output_r)

            self.assertRtolEqual(cpu_output_q, npu_output_q)
            self.assertRtolEqual(cpu_output_r, npu_output_r)
            self.assertRtolEqual(cpu_input1.numpy(), npu_output)
            self.assertRtolEqual(cpu_out, npu_out)

instantiate_device_type_tests(Testqr, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:1")
    run_tests()
