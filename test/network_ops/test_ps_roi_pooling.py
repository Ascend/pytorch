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


class TestPsRoiPooling(TestCase):
    def test_ps_roi_pooling_fp16(self):
        roi = torch.tensor([[[1], [2], [3], [4], [5]],
                            [[6], [7], [8], [9], [10]]
                            ], dtype=torch.float16).npu()
        _input = torch.tensor([[[[1]], [[2]], [[3]], [[4]],
                                [[5]], [[6]], [[7]], [[8]]],
                               [[[9]], [[10]], [[11]], [[12]],
                                [[13]], [[14]], [[15]], [[16]]]
                               ], dtype=torch.float16).npu()
        out = torch_npu.npu_ps_roi_pooling(_input, roi, 0.5, 2, 2)
        expect_out = torch.tensor([[[[0., 0.], [0., 0.]],
                                    [[0., 0.], [0., 0.]]],
                                   [[[0., 0.], [0., 0.]],
                                    [[0., 0.], [0., 0.]]]
                                   ], dtype=torch.float16)
        self.assertRtolEqual(expect_out, out.cpu())


if __name__ == "__main__":
    run_tests()
