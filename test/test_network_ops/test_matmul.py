# Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

PrescsionTableFP16 = [
    [2, 1e2, 0.005], [2, 1e3, 0.005], [2, 1e4, 0.005], [2, 1e5, 0.005], [2, 1e6, 0.005],
    [10, 1e2, 0.005], [10, 1e3, 0.01], [10, 1e4, 0.02], [10, 1e5, 0.0305], [10, 1e6, 0.04],
    [50, 1e2, 0.03], [50, 1e3, 0.03], [50, 1e4, 0.03], [50, 1e5, 0.03], [50, 1e6, 0.04],
    [100, 1e2, 0.03], [100, 1e3, 0.03], [100, 1e4, 0.03], [100, 1e5, 0.03], [100, 1e6, 0.04],
    [1000, 1e2, 0.03], [1000, 1e3, 0.04], [1000, 1e4, 0.04], [1000, 1e5, 0.04], [1000, 1e6, 0.04],
    [10000, 1e2, 0.04], [10000, 1e3, 0.04], [10000, 1e4, 0.04], [10000, 1e5, 0.04], [10000, 1e6, 0.04],
]


class TestMatMul(TestCase):
    def assertRtolEqualMatmul(self, x, y):
        def getFp16Precsion(D_range, K_range):
            prec16 = 1e-3
            for elm in PrescsionTableFP16:
                if elm[0] == D_range and elm[1] == K_range:
                    return elm[2]
            return prec16

        D = np.amax(np.maximum(np.abs(x), np.abs(y))) if (x.size and y.size) else 1
        D_range = 10000
        D_range = 10000 if (D > 1000) else D_range
        D_range = 1000 if (D <= 1000) else D_range
        D_range = 100 if (D <= 100) else D_range
        D_range = 50 if (D <= 50) else D_range
        D_range = 2 if (D <= 2) else D_range

        Kx = max(x.shape) if x.shape else 1
        Ky = max(y.shape) if y.shape else 1
        K = max(Kx, Ky)
        K_range = 1e6
        K_range = 1e6 if (K > 1e5) else K_range
        K_range = 1e5 if (K <= 1e5) else K_range
        K_range = 1e4 if (K <= 1e4) else K_range
        K_range = 1e3 if (K <= 1e3) else K_range
        K_range = 1e2 if (K <= 1e2) else K_range

        prec16 = 1e-3
        if x.dtype == np.float16 or x.dtype == np.float32:
            prec16 = getFp16Precsion(D_range, K_range)

        self.assertRtolEqual(x, y, prec16, prec16)

    def op_exec_cpu(self, mat1, mat2):
        input1 = mat1
        input2 = mat2
        input1.requires_grad = True
        input2.requires_grad = True

        cpu_output = torch.matmul(input1, input2)
        tmp = torch.ones_like(cpu_output)
        cpu_output.backward(tmp)

        return cpu_output.detach().numpy(), input1.grad.numpy(), input2.grad.numpy()

    def op_exec_npu(self, mat1, mat2):
        input1 = mat1
        input2 = mat2
        input1.requires_grad = True
        input2.requires_grad = True

        npu_output = torch.matmul(input1, input2)
        tmp = torch.ones_like(npu_output)
        npu_output.backward(tmp)
        npu_output = npu_output.cpu()
        return npu_output.detach().cpu().numpy(), input1.grad.cpu().numpy(), input2.grad.cpu().numpy()

    def matmul_backward_result(self, shape_format):
        for item in shape_format:
            mat1_cpu, mat1_npu = create_common_tensor(item[0], -10, 10)
            if mat1_cpu.dtype == torch.float16:
                mat1_cpu = mat1_cpu.to(torch.float32)
            mat2_cpu, mat2_npu = create_common_tensor(item[1], -10, 10)
            if mat2_cpu.dtype == torch.float16:
                mat2_cpu = mat2_cpu.to(torch.float32)
            cpu_output, cpu_mat1_grad, cpu_mat2_grad = self.op_exec_cpu(mat1_cpu, mat2_cpu)
            npu_output, npu_mat1_grad, npu_mat2_grad = self.op_exec_npu(mat1_npu, mat2_npu)

            self.assertRtolEqualMatmul(cpu_output.astype(npu_output.dtype), npu_output)
            self.assertRtolEqualMatmul(cpu_mat1_grad.astype(npu_mat1_grad.dtype), npu_mat1_grad)
            self.assertRtolEqualMatmul(cpu_mat2_grad.astype(npu_mat2_grad.dtype), npu_mat2_grad)

    def test_matmul_backward_shape_format_fp16_case1(self):
        shape_format = [
            # mat1 1dim, mat2 1dim
            [[np.float16, 2, [5]], [np.float16, 2, [5]]],
            [[np.float16, 2, [16]], [np.float16, 2, [16]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case3(self):
        shape_format = [
            # mat1 1dim, mat2 2dim
            [[np.float16, 2, [5]], [np.float16, 2, [5, 6]]],
            [[np.float16, 2, [5]], [np.float16, 2, [5, 5]]],

        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case4(self):
        shape_format = [
            # mat1 1dim, mat2 2dim
            [[np.float16, 2, [5, 7]], [np.float16, 2, [7, 10]]],
            [[np.float16, 2, [5, 10]], [np.float16, 2, [10, 20]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case5(self):
        shape_format = [
            # mat1 1dim, mat2 2dim
            [[np.float16, 2, [4, 5, 10]], [np.float16, 2, [10]]],
            [[np.float16, 2, [5, 10, 20, 30]], [np.float16, 2, [30]]],
            [[np.float16, 2, [20, 30, 40, 50, 60]], [np.float16, 2, [60]]],
            [[np.float16, 2, [2, 3, 4, 5, 6, 8]], [np.float16, 2, [8]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case6(self):
        shape_format = [
            # mat1 >2dim, mat2 2dim
            [[np.float16, 2, [5, 7, 10]], [np.float16, 2, [10, 16]]],
            [[np.float16, 2, [5, 10, 20, 30]], [np.float16, 2, [30, 25]]],
            [[np.float16, 2, [2, 5, 7, 8, 9, 10]], [np.float16, 2, [10, 16]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case7(self):
        shape_format = [
            # mat1 1dim, mat2 >2dim
            [[np.float16, 2, [3, ]], [np.float16, 2, [2, 3, 2]]],
            [[np.float16, 2, [20]], [np.float16, 2, [5, 10, 20, 30]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case8(self):
        shape_format = [
            # mat1 2dim, mat2 >2dim
            [[np.float16, 2, [2, 3]], [np.float16, 2, [2, 3, 2]]],
            [[np.float16, 2, [44, 20]], [np.float16, 2, [5, 10, 20, 30]]],
            [[np.float16, 2, [75, 50]], [np.float16, 2, [2, 3, 40, 50, 60]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case9(self):
        shape_format = [
            [[np.float16, 2, [5, 7, 10]], [np.float16, 2, [5, 10, 15]]],
            [[np.float16, 2, [68, 75, 16]], [np.float16, 2, [68, 16, 43]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_allow_hf32(self):
        torch.npu.matmul.allow_hf32 = True
        shape_format = [
            # mat1 1dim, mat2 1dim
            [[np.float16, 2, [5]], [np.float16, 2, [5]]],
            [[np.float16, 2, [16]], [np.float16, 2, [16]]],
        ]
        self.matmul_backward_result(shape_format)
        torch.npu.matmul.allow_hf32 = False

    def test_matmul_opapi(self):
        torch.npu.matmul.allow_hf32 = True
        shape_format = [
            [[np.float16, 2, [1, 1, 10, 2, 16, 16]], [np.float16, 2, [1, 10, 1, 16, 16]]],
            [[np.float16, 2, [1, 11, 10, 10, 16, 5]], [np.float16, 2, [1, 10, 1, 5, 16]]],
            [[np.float16, 2, [400, 11, 10, 10, 16, 5]], [np.float16, 2, [1, 10, 1, 5, 16]]],
        ]
        self.matmul_backward_result(shape_format)
        torch.npu.matmul.allow_hf32 = False

    def test_matmul_backward_shape_diff_input_types(self):
        torch.npu.matmul.allow_hf32 = True
        shape_format = [
            [[np.float16, 2, [1, 7, 10]], [np.float32, 2, [5, 10, 15]]],
            [[np.float32, 2, [68, 75, 16]], [np.float16, 2, [16, 43]]],
        ]
        self.matmul_backward_result(shape_format)
        torch.npu.matmul.allow_hf32 = False


if __name__ == "__main__":
    run_tests()
