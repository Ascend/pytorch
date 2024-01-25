# Copyright (c) 2023 Huawei Technologies Co., Ltd
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

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNanToNum(TestCase):
    @staticmethod
    def cpu_op_exec(input1, nan=None, posinf=None, neginf=None):
        output = torch.nan_to_num(input1, nan=nan, posinf=posinf, neginf=neginf)
        if input1.dtype == torch.bfloat16:
            return output.float().numpy()
        return output.numpy()

    @staticmethod
    def npu_op_exec(input1, nan=None, posinf=None, neginf=None):
        output = torch.nan_to_num(input1, nan=nan, posinf=posinf, neginf=neginf)
        if input1.dtype == torch.bfloat16:
            return output.cpu().float().numpy()
        return output.cpu().numpy()

    @staticmethod
    def cpu_op_out_exec(input1, nan=None, posinf=None, neginf=None, out=None):
        torch.nan_to_num(input1, nan=nan, posinf=posinf, neginf=neginf, out=None)
        if input1.dtype == torch.bfloat16:
            return out.float().numpy()
        return out.numpy()

    @staticmethod
    def npu_op_out_exec(input1, nan=None, posinf=None, neginf=None, out=None):
        torch.nan_to_num(input1, nan=nan, posinf=posinf, neginf=neginf, out=None)
        if input1.dtype == torch.bfloat16:
            return out.cpu().float().numpy()
        return out.cpu().numpy()

    @staticmethod
    def cpu_op_exec_(input1, nan=None, posinf=None, neginf=None):
        torch.nan_to_num_(input1, nan=nan, posinf=posinf, neginf=neginf)
        if input1.dtype == torch.bfloat16:
            return input1.float().numpy()
        return input1.numpy()

    @staticmethod
    def npu_op_exec_(input1, nan=None, posinf=None, neginf=None):
        torch.nan_to_num_(input1, nan=nan, posinf=posinf, neginf=neginf)
        if input1.dtype == torch.bfloat16:
            return input1.cpu().float().numpy()
        return input1.cpu().numpy()

    @SupportedDevices(['Ascend910B'])
    def test_nan_to_num(self):
        dtype_list = [torch.float16, torch.float32, torch.bfloat16]
        nan_list = [None, 0, -1.1]
        posinf_list = [None, 100, 2.3]
        neginf_list = [None, -100, -2.3]
        params_list = [
            [dtype, [nan, posinf, neginf]]
            for dtype in dtype_list
            for nan in nan_list
            for posinf in posinf_list
            for neginf in neginf_list
        ]

        for dtype, param in params_list:
            cpu_input = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14]).to(dtype)
            npu_input = cpu_input.npu()
            cpu_output = self.cpu_op_exec(cpu_input, *param)
            npu_output = self.npu_op_exec(npu_input, *param)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_out = torch.rand((1, 4), dtype=dtype)
            npu_out = cpu_out.npu()
            cpu_output = self.cpu_op_out_exec(cpu_input, *param, out=cpu_out)
            npu_output = self.npu_op_out_exec(npu_input, *param, out=npu_out)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_output = self.cpu_op_exec_(cpu_input, *param)
            npu_output = self.npu_op_exec_(npu_input, *param)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
