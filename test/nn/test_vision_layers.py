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
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestVisionLayers(TestCase):
    def test_PixelShuffle(self):
        pixel_shuffle = nn.PixelShuffle(3).npu()
        input1 = torch.randn(1, 9, 4, 4).npu()
        output = pixel_shuffle(input1)
        self.assertEqual(output is not None, True)

    def test_Upsample(self):
        input1 = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2).npu()
        m = nn.Upsample(scale_factor=2, mode='nearest').npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_UpsamplingNearest2d(self):
        input1 = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2).npu()
        m = nn.UpsamplingNearest2d(scale_factor=2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)


if __name__ == "__main__":
    run_tests()
