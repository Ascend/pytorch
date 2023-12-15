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

import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestRemainder(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.remainder(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.remainder(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, out):
        output = torch.remainder(input1, input2, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_exec(self, input1, input2):
        output = input1.remainder_(input2)
        output = output.numpy()
        return output

    def npu_op_inplace_exec(self, input1, input2):
        output = input1.remainder_(input2)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar(self, input1, input2):
        output = torch.remainder(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def remainder_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            npu_input3 = torch.randn(6).to("npu")
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def remainder_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            npu_input3 = copy.deepcopy(cpu_input1).to("npu")
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
            cpu_output_inplace = self.cpu_op_inplace_exec(cpu_input1, cpu_input2)
            npu_output_inplace = self.npu_op_inplace_exec(npu_input1, npu_input2)

            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_output_inplace = cpu_output_inplace.astype(npu_output_inplace.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output_inplace, npu_output_inplace)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def remainder_scalar_result(self, shape_format):
        for item in shape_format:
            scalar = np.random.uniform(0, 100)
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 2)
            npu_input3 = copy.deepcopy(cpu_input1).to("npu")
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, scalar)
            npu_output_scalar = self.npu_op_exec_scalar(npu_input1, scalar)
            npu_output_out = self.npu_op_exec_out(npu_input1, scalar, npu_input3)

            cpu_output = cpu_output.astype(npu_output_scalar.dtype)
            self.assertRtolEqual(cpu_output, npu_output_scalar)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def test_remainder_shape_format_fp16_1d(self, device="npu"):
        format_list = [0, 3]
        shape_format = [[np.float16, i, [4]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    def test_remainder_shape_format_fp32_1d(self, device="npu"):
        format_list = [0, 3]
        shape_format = [[np.float32, i, [4]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    def test_remainder_shape_format_fp16_2d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [4, 18]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    def test_remainder_shape_format_fp32_2d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [4, 18]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    def test_remainder_shape_format_fp16_3d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [4, 18, 32]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    def test_remainder_shape_format_fp32_3d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [4, 18, 32]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    def test_remainder_shape_format_fp16_4d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [4, 18, 32, 128]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    def test_remainder_shape_format_fp32_4d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [4, 18, 32, 128]] for i in format_list
                        ]
        self.remainder_result(shape_format)

    # scalar----------------------------------------------------------
    def test_remainder_scalar_shape_format_fp16_1d(self, device="npu"):
        format_list = [0, 3]
        shape_format = [[np.float16, i, [4]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_scalar_shape_format_fp32_1d(self, device="npu"):
        format_list = [0, 3]
        shape_format = [[np.float32, i, [4]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_scalar_shape_format_fp16_2d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [4, 18]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_scalar_shape_format_fp32_2d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [4, 18]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_scalar_shape_format_fp16_3d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [4, 18, 32]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_scalar_shape_format_fp32_3d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [4, 18, 32]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_scalar_shape_format_fp16_4d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [4, 18, 32, 128]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_scalar_shape_format_fp32_4d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [4, 18, 32, 128]] for i in format_list
                        ]
        self.remainder_scalar_result(shape_format)

    def test_remainder_mix_dtype_1(self, device="npu"):
        npu_input1, npu_input2 = create_common_tensor([np.int32, 0, (2, 3)], 1, 100)
        npu_input3, npu_input4 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input3)
        npu_output = self.npu_op_exec(npu_input1, npu_input3)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_remainder_mix_dtype_2(self, device="npu"):
        npu_input1, npu_input2 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        npu_input3 = torch.tensor(3).int()
        cpu_output = self.cpu_op_exec(npu_input1, npu_input3)
        npu_output = self.npu_op_exec(npu_input1, npu_input3)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_remainder_scalar_shape_format_fp32_out_4d(self, device="npu"):
        format_list = [0]
        shape_format = [[np.float32, i, [4, 18, 32, 128]] for i in format_list
                        ]
        self.remainder_out_result(shape_format)


if __name__ == "__main__":
    run_tests()
