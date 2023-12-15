# Copyright (c) 2023 Huawei Technologies Co., Ltd
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

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestRotaryMul(TestCase):
    @staticmethod
    def rotary_mul(x, r1, r2):
        x1, x2 = torch.chunk(x, 2, -1)
        x_new = torch.cat((-x2, x1), dim=-1)
        output = r1 * x + r2 * x_new
        return output

    @staticmethod
    def gen_data(shape, dtype):
        cpu_input = torch.rand(shape, dtype=dtype)
        npu_input = cpu_input.npu()
        return cpu_input, npu_input

    def cpu_to_exec(self, x, r1, r2):
        out = self.rotary_mul(x, r1, r2)
        return out.cpu().numpy()

    @staticmethod
    def npu_to_exec(x, r1, r2):
        out = torch_npu.npu_rotary_mul(x, r1, r2)
        return out.cpu().numpy()

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `RotaryMul` is only supported on 910B, skip this ut!")
    def test_rotary_mul(self):
        dtype_list = [torch.float16, torch.float32]
        shape_list = [
            [[2, 8192, 5, 128], [1, 8192, 1, 128], [1, 8192, 1, 128]],
            [[8192, 2, 5, 128], [8192, 1, 1, 128], [8192, 1, 1, 128]],
            [[2048, 4, 32, 64], [2048, 4, 1, 64], [2048, 4, 1, 64]],
        ]
        items = [
            [shape, dtype]
            for shape in shape_list
            for dtype in dtype_list
        ]
        for shape, dtype in items:
            cpu_x, npu_x = self.gen_data(shape[0], dtype)
            cpu_r1, npu_r1 = self.gen_data(shape[1], dtype)
            cpu_r2, npu_r2 = self.gen_data(shape[2], dtype)
            cpu_out = self.cpu_to_exec(cpu_x, cpu_r1, cpu_r2)
            npu_out = self.npu_to_exec(npu_x, npu_r1, npu_r2)
            self.assertRtolEqual(cpu_out, npu_out)


if __name__ == '__main__':
    run_tests()
