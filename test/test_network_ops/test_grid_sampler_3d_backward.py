# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
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

import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestGridSampler3DBackward(TestCase):
    def exec_grid_sampler3d_bk_fp32(self, interpolation_mode, padding_mode, align_corners):
        format_list = [2]
        shape_list = [[2, 100, 1, 28, 28], [2, 100, 64, 32, 28]]
        shape_format = [
            [np.float32, j, k] for j in format_list for k in shape_list
        ]
        sample_format = [np.float32, 2, [2, 100, 1, 1, 3]]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_sample, npu_sample = create_common_tensor(sample_format, -1, 1)
            cpu_output, cpu_dx, cpu_dgrad = self.op_exec_com(0,
                                                             cpu_input, cpu_sample, interpolation_mode, padding_mode, align_corners)
            npu_output, npu_dx, npu_dgrad = self.op_exec_com(1,
                                                             npu_input, npu_sample, interpolation_mode, padding_mode, align_corners)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_dx, npu_dx)
            self.assertRtolEqual(cpu_dgrad, npu_dgrad)

    @unittest.skip("skip test_grid_sampler3d_bk_fp32 now")
    def test_grid_sampler3d_bk_fp32(self, device="npu"):
        self.exec_grid_sampler3d_bk_fp32(0, 0, True)
        self.exec_grid_sampler3d_bk_fp32(0, 1, True)
        self.exec_grid_sampler3d_bk_fp32(1, 0, True)
        self.exec_grid_sampler3d_bk_fp32(1, 1, True)
        self.exec_grid_sampler3d_bk_fp32(0, 0, False)
        self.exec_grid_sampler3d_bk_fp32(0, 1, False)
        self.exec_grid_sampler3d_bk_fp32(1, 0, False)
        self.exec_grid_sampler3d_bk_fp32(1, 1, False)

    def test_grid_sampler3d_bk_fp16(self, device="npu"):
        format_list = [2]
        shape_list = [[2, 1, 1, 3, 3], [2, 1, 2, 3, 4]]
        shape_format = [
            [np.float16, j, k] for j in format_list for k in shape_list
        ]
        sample_format = [np.float16, 2, [2, 1, 2, 2, 3]]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 10)
            cpu_sample, npu_sample = create_common_tensor(sample_format, -1, 1)
            cpu_output, cpu_dx, cpu_dgrad = self.cpu_op_fp16_exec(
                cpu_input.to(torch.float32), cpu_sample.to(torch.float32), 0, 0, True)
            npu_output, npu_dx, npu_dgrad = self.op_exec_com(1, npu_input, npu_sample, 0, 0, True)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_dx, npu_dx)
            self.assertRtolEqual(cpu_dgrad, npu_dgrad)

    def op_exec_com(self, npu_flag, input1, sample, interpolation_mode, padding_mode, align_corners):
        input1.requires_grad = True
        output = torch.grid_sampler_3d(input1, sample, interpolation_mode, padding_mode, align_corners)
        output.backward(torch.ones_like(output))
        dx, dgrad = input1.grad
        if npu_flag:
            output = output.detach().to("cpu").numpy()
            return output, dx.to("cpu").numpy(), dgrad.to("cpu").numpy()
        output = output.detach().numpy()
        return output, dx.numpy(), dgrad.numpy()

    def cpu_op_fp16_exec(self, input1, sample, interpolation_mode, padding_mode, align_corners):
        input1 = input1.to(torch.float32)
        sample = sample.to(torch.float32)
        input1.requires_grad = True
        output = torch.grid_sampler_3d(input1, sample, interpolation_mode, padding_mode, align_corners)
        output.backward(torch.ones_like(output))
        dx, dgrad = input1.grad
        dx = dx.numpy().astype(np.float16)
        dgrad = dgrad.numpy().astype(np.float16)
        output = output.detach().numpy().astype(np.float16)
        return output, dx, dgrad


if __name__ == "__main__":
    run_tests()
