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
from common_utils import TestCase, run_tests
from common_device_type import instantiate_device_type_tests

class TestPtIou(TestCase):
    def test_pt_iou_fp32(self, device):
        bboxs = torch.tensor([[ 1,  2,  3,  4],
                              [ 5,  6,  7,  8],
                              [ 9, 10, 11, 12],
                              [13, 14, 15, 16]], dtype = torch.float16).npu()
        gtboxes = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype = torch.float16).npu()
        expect_output = torch.tensor([[0.9902, 0.0000, 0.0000, 0.0000],
                                      [0.0000, 0.9902, 0.0000, 0.0000]], dtype = torch.float16)         
        output = torch.npu_ptiou(bboxs, gtboxes, 1)
        self.assertRtolEqual(expect_output, output.cpu(), 1.e-3)
        
instantiate_device_type_tests(TestPtIou, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
