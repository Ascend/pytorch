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
import copy
import sys
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
class TestYoloBoxesEncode(TestCase):
    def npu_op_exec(self, anchor_boxes, gt_bboxes, stride, impl_mode=False):
        out = torch.npu_yolo_boxes_encode(anchor_boxes, gt_bboxes, stride, impl_mode)
        out = out.to("cpu")
        return out.detach().numpy()
        
    def test_yolo_boxes_encode(self, device):
        anchor_boxes = torch.rand((2, 4), dtype=torch.float32).to("npu")
        gt_bboxes = torch.rand((2, 4), dtype=torch.float32).to("npu")
        stride = torch.tensor([2, 2], dtype=torch.int32).to("npu")
        expect_cpu = torch.tensor([[0.7921727, 0.5314963, -0.74224466, -13.815511],
                                   [0.7360072, 0.58343244, 4.3334002, -0.51378196]], dtype=torch.float32)
        npu_output = self.npu_op_exec(anchor_boxes, gt_bboxes, stride, False)
        self.assertRtolEqual(expect_cpu.numpy(), npu_output)


instantiate_device_type_tests(TestYoloBoxesEncode, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
