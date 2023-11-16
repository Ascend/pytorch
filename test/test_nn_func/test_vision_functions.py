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

import unittest

import torch
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestVisionFunctions(TestCase):
    def test_pixel_shuffle(self):
        input1 = torch.randn(1, 9, 4, 4)

        npu_input = input1.npu()

        cpu_output = F.pixel_shuffle(input1, 3)
        npu_output = F.pixel_shuffle(npu_input, 3)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    @unittest.skip("skip test_pad now")
    def test_pad(self):
        input1 = torch.empty(3, 3, 4, 2)
        p1d = (1, 1)
        npu_input = input1.npu()

        cpu_output = F.pad(input1, p1d, "constant", 0)
        npu_output = F.pad(npu_input, p1d, "constant", 0)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    @unittest.skip("skip test_interpolate now")
    def test_interpolate(self):
        input1 = torch.empty(3, 3, 4, 2)
        npu_input = input1.npu()

        cpu_output = F.interpolate(input1, 4)
        npu_output = F.interpolate(npu_input, 4)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    @unittest.skip("skip test_upsample now")
    def test_upsample(self):
        input1 = torch.empty(3, 3, 4, 2)
        npu_input = input1.npu()

        cpu_output = F.upsample(input1, 4)
        npu_output = F.upsample(npu_input, 4)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    @unittest.skip("skip test_upsample_nearest now")
    def test_upsample_nearest(self):
        input1 = torch.empty(3, 3, 4, 2)
        npu_input = input1.npu()

        cpu_output = F.upsample_nearest(input1, 4)
        npu_output = F.upsample_nearest(npu_input, 4)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_grid_sample(self):
        input1 = torch.empty(1, 1, 2, 2)
        grid = torch.empty(1, 1, 1, 2)

        npu_input = input1.npu()
        npu_grid = grid.npu()

        cpu_output = F.grid_sample(input1, grid)
        npu_output = F.grid_sample(npu_input, npu_grid)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    @unittest.skip("skip test_affine_grid now")
    def test_affine_grid(self):
        input1 = torch.empty(1, 2, 3)
        size = torch.Size([1, 1, 2, 2])

        npu_input = input1.npu()

        cpu_output = F.affine_grid(input1, size)
        npu_output = F.affine_grid(npu_input, size)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())


if __name__ == "__main__":
    run_tests()
