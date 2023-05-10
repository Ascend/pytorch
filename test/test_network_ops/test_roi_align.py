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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import graph_mode


class TestRoiAlign(TestCase):
    @graph_mode
    def test_roi_align_fp32(self):
        _input = torch.FloatTensor([[[[1, 2, 3 , 4, 5, 6],
                                      [7, 8, 9, 10, 11, 12],
                                      [13, 14, 15, 16, 17, 18],
                                      [19, 20, 21, 22, 23, 24],
                                      [25, 26, 27, 28, 29, 30],
                                      [31, 32, 33, 34, 35, 36]]]]).npu()
        rois = torch.tensor([[0, -2.0, -2.0, 22.0, 22.0]]).npu()
        expect_out = torch.tensor([[[[4.5000, 6.5000, 8.5000],
                                     [16.5000, 18.5000, 20.5000],
                                     [28.5000, 30.5000, 32.5000]]]], dtype = torch.float32)
        out = torch_npu.npu_roi_align(_input, rois, 0.25, 3, 3, 2, 0)
        self.assertRtolEqual(expect_out, out.cpu())


if __name__ == "__main__":
    run_tests()