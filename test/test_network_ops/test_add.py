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


class TestAdd(TestCase):

    def cpu_op_out_exec(self, input1, input2, output):
        torch.add(input1, input2, alpha=1, out=output)
        output = output.numpy()
        return output

    def npu_op_out_exec_new(self, input1, input2, output):
        torch.add(input1, input2, alpha=1, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec(self, input1, input2):
        output = torch.add(input1, input2, alpha=1)
        output = output.numpy()
        return output

    def npu_op_exec_new(self, input1, input2):
        output = torch.add(input1, input2, alpha=1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_alpha(self, input1, input2):
        output = torch.add(input1, input2, alpha=3)
        output = output.numpy()
        return output

    def npu_op_exec_new_alpha(self, input1, input2):
        output = torch.add(input1, input2, alpha=3)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_scalar_exec(self, input1, scalar):
        output = torch.add(input1, scalar, alpha=1)
        output = output.numpy()
        return output

    def npu_op_scalar_exec_new(self, input1, scalar):
        output = torch.add(input1, scalar, alpha=1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_scalar_exec_alpha(self, input1, scalar):
        output = torch.add(input1, scalar, alpha=3)
        output = output.numpy()
        return output

    def npu_op_scalar_exec_new_alpha(self, input1, scalar):
        output = torch.add(input1, scalar, alpha=3)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def add_scalar_result(self, shape_format):
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_scalar_exec(cpu_input, item[1])
            npu_output = self.npu_op_exec_new(npu_input, item[1])
            cpu_output = cpu_output.astype(npu_output.dtype)

            self.assertRtolEqual(cpu_output, npu_output)

    def add_scalar_alpha_result(self, shape_format):
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_scalar_exec_alpha(cpu_input, item[1])
            npu_output = self.npu_op_scalar_exec_new_alpha(npu_input, item[1])
            cpu_output = cpu_output.astype(npu_output.dtype)

            self.assertRtolEqual(cpu_output, npu_output)

    def add_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec_new(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)

            self.assertRtolEqual(cpu_output, npu_output)

    def add_out_result(self, shape_format):
        for item in shape_format:
            cpuout = torch.randn(3)
            npuout = torch.randn(3).to("npu")
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)

            cpu_output = self.cpu_op_out_exec(cpu_input1, cpu_input2, cpuout)
            npu_output = self.npu_op_out_exec_new(npu_input1, npu_input2, npuout)
            cpu_output = cpu_output.astype(npu_output.dtype)

            self.assertRtolEqual(cpu_output, npu_output)

    def add_alpha_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)

            cpu_output = self.cpu_op_exec_alpha(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec_new_alpha(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)

            self.assertRtolEqual(cpu_output, npu_output)

    def test_add_scalar_shape_format_fp16_1d(self):
        format_list = [0, 3]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float16, i, [18]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_result(shape_format)

    def test_add_scalar_shape_format_fp32_1d(self):
        format_list = [0, 3]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float32, i, [18]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_result(shape_format)

    def test_add_scalar_shape_format_fp16_2d(self):
        format_list = [0, 3, 29]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float16, i, [5, 256]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_result(shape_format)

    def test_add_scalar_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float32, i, [5, 256]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_result(shape_format)

    def test_add_scalar_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float16, i, [32, 3, 3]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_result(shape_format)

    def test_add_scalar_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float32, i, [32, 3, 3]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_result(shape_format)

    def test_add_scalar_shape_format_fp16_4d(self):
        format_list = [0, 3, 29]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float16, i, [64, 112, 7, 7]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_result(shape_format)

    def test_add_scalar_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float32, i, [64, 112, 7, 7]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_result(shape_format)

    def test_add_scalar_shape_format_fp16_1d(self):
        format_list = [0, 3]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float16, i, [18]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_alpha_result(shape_format)

    def test_add_scalar_shape_format_fp32_1d(self):
        format_list = [0, 3]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float32, i, [18]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_alpha_result(shape_format)

    def test_add_scalar_shape_format_fp16_2d(self):
        format_list = [0, 3, 29]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float16, i, [5, 256]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_alpha_result(shape_format)

    def test_add_scalar_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float32, i, [5, 256]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_alpha_result(shape_format)

    def test_add_scalar_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float16, i, [32, 3, 3]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_alpha_result(shape_format)

    def test_add_scalar_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float32, i, [32, 3, 3]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_alpha_result(shape_format)

    def test_add_scalar_shape_format_fp16_4d(self):
        format_list = [0, 3, 29]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float16, i, [64, 112, 7, 7]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_alpha_result(shape_format)

    def test_add_scalar_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        scalar_list = [0, 1]
        shape_format = [
            [[np.float32, i, [64, 112, 7, 7]], k] for i in format_list for k in scalar_list
        ]
        self.add_scalar_alpha_result(shape_format)

    def test_add_shape_format_fp16_1d(self):
        format_list = [0, 3]
        shape_format = [
            [np.float16, i, [64]] for i in format_list
        ]
        self.add_result(shape_format)

    def test_add_shape_format_fp32_1d(self):
        format_list = [0, 3]
        shape_format = [
            [np.float32, i, [64]] for i in format_list
        ]
        self.add_result(shape_format)

    def test_add_shape_format_fp16_2d(self):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float16, i, [5, 256]] for i in format_list
        ]
        self.add_result(shape_format)

    def test_add_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float32, i, [5, 256]] for i in format_list
        ]
        self.add_result(shape_format)

    def test_add_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float16, i, [32, 3, 3]] for i in format_list
        ]
        self.add_result(shape_format)

    def test_add_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float32, i, [32, 3, 3]] for i in format_list
        ]
        self.add_result(shape_format)

    def test_add_shape_format_fp16_4d(self):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float16, i, [64, 112, 7, 7]] for i in format_list
        ]
        self.add_result(shape_format)

    def test_add_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float32, i, [64, 112, 7, 7]] for i in format_list
        ]
        self.add_result(shape_format)

    def test_add_shape_format_fp16_1d(self):
        format_list = [0, 3]
        shape_format = [
            [np.float16, i, [64]] for i in format_list
        ]
        self.add_alpha_result(shape_format)

    def test_add_shape_format_fp32_1d(self):
        format_list = [0, 3]
        shape_format = [
            [np.float32, i, [64]] for i in format_list
        ]
        self.add_alpha_result(shape_format)

    def test_add_shape_format_fp16_2d(self):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float16, i, [5, 256]] for i in format_list
        ]
        self.add_alpha_result(shape_format)

    def test_add_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float32, i, [5, 256]] for i in format_list
        ]
        self.add_alpha_result(shape_format)

    def test_add_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float16, i, [32, 3, 3]] for i in format_list
        ]
        self.add_alpha_result(shape_format)

    def test_add_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float32, i, [32, 3, 3]] for i in format_list
        ]
        self.add_alpha_result(shape_format)

    def test_add_shape_format_fp16_4d(self):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float16, i, [64, 112, 7, 7]] for i in format_list
        ]
        self.add_alpha_result(shape_format)

    def test_add_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float32, i, [64, 112, 7, 7]] for i in format_list
        ]
        self.add_alpha_result(shape_format)

    def test_add_mix_dtype(self):
        cpu_input1, npu_input1 = create_common_tensor([np.int32, 0, (2, 3)], 1, 100)
        cpu_input2, npu_input2 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        cpu_output = torch.add(cpu_input1, cpu_input2)
        npu_output = torch.add(npu_input1, npu_input2)
        npu_output = npu_output.to("cpu")
        self.assertRtolEqual(cpu_output, npu_output)

    def test_add_scalar_check_5d_5d_match(self):
        ca = torch.randn(4)
        cb = ca.view(2, 2).transpose(1, 0)
        na = ca.npu()
        nb = cb.npu()
        caout = torch.add(ca, 1)
        cbout = torch.add(cb, 1)
        naout = torch.add(na, 1)
        nbout = torch.add(nb, 1)
        naout = naout.to("cpu")
        nbout = nbout.to("cpu")
        self.assertRtolEqual(caout, naout)
        self.assertRtolEqual(cbout, nbout)

    def test_add_different_dtype(self):
        cpu_x1 = torch.rand(2, 3, 4)
        cpu_other1 = torch.rand(2, 3, 4).uniform_(1, 10).long()
        npu_x1 = cpu_x1.npu()
        npu_other1 = cpu_other1.npu()
        cpu_out1 = cpu_x1 + cpu_other1
        npu_out1 = npu_x1 + npu_other1

        cpu_x2 = 1.5
        cpu_other2 = torch.rand(2, 3, 4).uniform_(1, 10).long()
        npu_x2 = 1.5
        npu_other2 = cpu_other2.npu()
        cpu_out2 = cpu_x2 + cpu_other2
        npu_out2 = npu_x2 + npu_other2

        cpu_x3 = torch.rand(2, 3, 4).int()
        cpu_other3 = 3
        npu_x3 = cpu_x3.npu()
        npu_other3 = 3
        cpu_out3 = cpu_x3 + cpu_other3
        npu_out3 = npu_x3 + npu_other3

        self.assertRtolEqual(cpu_out1, npu_out1.cpu())
        self.assertRtolEqual(cpu_out2, npu_out2.cpu())
        self.assertRtolEqual(cpu_out3, npu_out3.cpu())

    def test_add_inplace_and_out_mix_dtype(self):
        dtype_list = [
            [np.int32, np.int64, np.int64],
            [np.int64, np.int32, np.int64],
            [np.float32, np.float16, np.float32],
            [np.float16, np.float32, np.float32],
            [np.int64, np.float32, np.float32]
        ]
        for item in dtype_list:
            cpu_input1, npu_input1 = create_common_tensor([item[0], 0, (2, 3, 4)], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor([item[1], 0, (2, 3, 4)], -100, 100)
            _, npu_output = create_common_tensor([item[2], 0, (2, 3, 4)], 1, 100)

            if item[0] == np.int64 and item[1] == np.float32:
                try:
                    npu_input1.add_(npu_input2)
                except RuntimeError as e:
                    self.assertRegex(
                        str(e), "result type Float can't be cast to the desired output type Long")
            else:
                cpu_input1.add_(cpu_input2)
                npu_input1.add_(npu_input2)
                self.assertRtolEqual(cpu_input1, npu_input1.cpu())

            cpu_output = torch.add(cpu_input1, cpu_input2)
            torch.add(npu_input1, npu_input2, out=npu_output)
            self.assertRtolEqual(cpu_output, npu_output.cpu())


if __name__ == "__main__":
    run_tests()
