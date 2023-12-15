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


class TestRoiAlignBackward(TestCase):
    def test_roi_align_backward_fp32(self):
        _input = torch.FloatTensor([[[[1, 2, 3, 4, 5, 6],
                                      [7, 8, 9, 10, 11, 12],
                                      [13, 14, 15, 16, 17, 18],
                                      [19, 20, 21, 22, 23, 24],
                                      [25, 26, 27, 28, 29, 30],
                                      [31, 32, 33, 34, 35, 36]]]]).npu()
        rois = torch.tensor([[0, -2.0, -2.0, 22.0, 22.0]]).npu()
        expect_out = torch.tensor([[[[4.5000, 6.5000, 8.5000],
                                     [16.5000, 18.5000, 20.5000],
                                     [28.5000, 30.5000, 32.5000]]]], dtype=torch.float32)
        expect_gradout = torch.tensor([[[[1.0786, 1.0557, 1.4942, 1.5248, 1.9226, 2.4709],
                                         [1.0557, 1.0332, 1.4625, 1.4924, 1.8817, 2.4183],
                                         [3.6872, 3.6087, 4.0380, 4.0679, 4.4572, 5.6097],
                                         [3.8708, 3.7884, 4.2177, 4.2476, 4.6369, 5.8324],
                                         [6.2575, 6.1243, 6.5536, 6.5835, 6.9729, 8.7269],
                                         [8.2847, 8.1084, 8.6403, 8.6774, 9.1598, 11.4575]]]], dtype=torch.float32)
        out = torch_npu.npu_roi_align(_input, rois, 0.25, 3, 3, 2, 0)
        self.assertRtolEqual(expect_out, out.cpu())
        gradout = torch_npu.npu_roi_alignbk(out, rois, _input.size(), 3, 3, 0.25, 2)
        self.assertRtolEqual(expect_gradout, gradout.cpu())


if __name__ == "__main__":
    run_tests()
