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


class TestRenorm(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input = torch.from_numpy(input_x)
        return npu_input

    def get_p0_result_cpu(self, input_x, dim, maxnorm=1.0):
        input_x = input_x.numpy()
        dims = len(input_x.shape)
        shape_list = []
        for i in range(dims):
            if (i != dim):
                shape_list = shape_list + [i]
        shape_tuple = tuple(shape_list)
        tmp = (input_x != 0)
        N = np.sum(tmp, shape_tuple, keepdims=True)
        N = np.where(N > maxnorm, maxnorm / (N + 1e-7), 1.0)
        output = input_x * N
        return output

    def cpu_op_exec(self, input_x, p, dim, maxnorm):
        if (p == 0):
            output = self.get_p0_result_cpu(input_x, dim, maxnorm)
        else:
            output = torch.renorm(input_x, p, dim, maxnorm)
            output = output.numpy()
        return output.astype(np.float32)

    def npu_op_exec(self, input_x, p, dim, maxnorm):
        input1 = input_x.to("npu")
        output = torch.renorm(input1, p, dim, maxnorm)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input_x, p, dim, maxnorm, output_y):
        input_x = input_x.to("npu")
        output_y = output_y.to("npu")
        torch.renorm(input_x, p, dim, maxnorm, out=output_y)
        output_y = output_y.to("cpu")
        output_y = output_y.numpy()
        return output_y

    def npu_op_exec_inplace(self, input_x, p, dim, maxnorm):
        input_x = input_x.to("npu")
        input_x.renorm_(p, dim, maxnorm)
        output = input_x.to("cpu")
        output = output.numpy()
        return output

    def test_renorm_3_3_4_0_1(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 4, 0, 1)
        npu_output1 = self.npu_op_exec(input_x1, 4, 0, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_1_1_1(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 1, 1, 1)
        npu_output1 = self.npu_op_exec(input_x1, 1, 1, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_0_0_1_float16(self, device="npu"):
        input_x1 = self.generate_data(-10, 10, (3, 3), np.float16)
        input_x1_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec(input_x1_cpu, 0, 0, 1).astype(np.float16)
        npu_output1 = self.npu_op_exec(input_x1, 0, 0, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_0_0_1(self, device="npu"):
        input_x1 = self.generate_data(-10, 10, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 0, 0, 1)
        npu_output1 = self.npu_op_exec(input_x1, 0, 0, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_4_0_1_float16(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float16)
        input_x1_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec(input_x1_cpu, 4, 0, 1).astype(np.float16)
        npu_output1 = self.npu_op_exec(input_x1, 4, 0, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_1_1_1_float16(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float16)
        input_x1_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec(input_x1_cpu, 1, 1, 1).astype(np.float16)
        npu_output1 = self.npu_op_exec(input_x1, 1, 1, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_1_0_1(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 1, 0, 1)
        npu_output1 = self.npu_op_exec(input_x1, 1, 0, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_3_3_1_1(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 3, 1, 1)
        npu_output1 = self.npu_op_exec(input_x1, 3, 1, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_3_2_2_1(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 2, 2, 1)
        npu_output1 = self.npu_op_exec(input_x1, 2, 2, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_3_3_2_0_1(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 2, 0, 1)
        npu_output1 = self.npu_op_exec(input_x1, 2, 0, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_3_3_3_3_1(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 3, 3, 1)
        npu_output1 = self.npu_op_exec(input_x1, 3, 3, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_3_3_3_4_4_1(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 4, 4, 1)
        npu_output1 = self.npu_op_exec(input_x1, 4, 4, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_4_0_1_out(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        output_y = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 4, 0, 1)
        npu_output1 = self.npu_op_exec_out(input_x1, 4, 0, 1, output_y)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_1_1_1_out(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        output_y = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 1, 1, 1)
        npu_output1 = self.npu_op_exec_out(input_x1, 1, 1, 1, output_y)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_1_0_1_out(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        output_y = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 1, 0, 1)
        npu_output1 = self.npu_op_exec_out(input_x1, 1, 0, 1, output_y)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_3_3_1_1_out(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3), np.float32)
        output_y = self.generate_data(-1, 1, (3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 3, 1, 1)
        npu_output1 = self.npu_op_exec_out(input_x1, 3, 1, 1, output_y)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_30_40_50_2_1_1_out_fp16(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (30, 40, 50), np.float16)
        output_y = self.generate_data(-1, 1, (30, 40, 50), np.float16)
        input_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec(input_cpu, 2, 1, 1)
        cpu_output1 = cpu_output1.astype(np.float16)
        npu_output1 = self.npu_op_exec_out(input_x1, 2, 1, 1, output_y)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_30_40_50_2_0_2_out_fp16(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (30, 40, 50), np.float16)
        output_y = self.generate_data(-1, 1, (30, 40, 50), np.float16)
        input_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec(input_cpu, 2, 0, 2)
        cpu_output1 = cpu_output1.astype(np.float16)
        npu_output1 = self.npu_op_exec_out(input_x1, 2, 0, 2, output_y)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_3_2_2_1_out(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3), np.float32)
        output_y = self.generate_data(-1, 1, (3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 2, 2, 1)
        npu_output1 = self.npu_op_exec_out(input_x1, 2, 2, 1, output_y)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_3_3_2_0_1_out(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3), np.float32)
        output_y = self.generate_data(-1, 1, (3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 2, 0, 1)
        npu_output1 = self.npu_op_exec_out(input_x1, 2, 0, 1, output_y)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_3_3_3_3_1_out(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3), np.float32)
        output_y = self.generate_data(-1, 1, (3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 3, 3, 1)
        npu_output1 = self.npu_op_exec_out(input_x1, 3, 3, 1, output_y)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_3_3_3_4_4_1_out(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3, 3), np.float32)
        output_y = self.generate_data(-1, 1, (3, 3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 4, 4, 1)
        npu_output1 = self.npu_op_exec_out(input_x1, 4, 4, 1, output_y)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_4_0_1_inplace(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 4, 0, 1)
        npu_output1 = self.npu_op_exec_inplace(input_x1, 4, 0, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_1_1_1_inplace(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 1, 1, 1)
        npu_output1 = self.npu_op_exec_inplace(input_x1, 1, 1, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_1_0_1_inplace(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 1, 0, 1)
        npu_output1 = self.npu_op_exec_inplace(input_x1, 1, 0, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_3_3_1_1_inplace(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 3, 1, 1)
        npu_output1 = self.npu_op_exec_inplace(input_x1, 3, 1, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_3_2_2_1_inplace(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 2, 2, 1)
        npu_output1 = self.npu_op_exec_inplace(input_x1, 2, 2, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_3_3_2_0_1_inplace(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 2, 0, 1)
        npu_output1 = self.npu_op_exec_inplace(input_x1, 2, 0, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_3_3_3_3_1_inplace(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 3, 3, 1)
        npu_output1 = self.npu_op_exec_inplace(input_x1, 3, 3, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_renorm_3_3_3_3_3_4_4_1_inplace(self, device="npu"):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, 4, 4, 1)
        npu_output1 = self.npu_op_exec_inplace(input_x1, 4, 4, 1)
        self.assertRtolEqual(cpu_output1, npu_output1)


if __name__ == "__main__":
    run_tests()
