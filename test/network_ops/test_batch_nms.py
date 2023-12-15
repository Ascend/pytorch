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
from torch_npu.testing.common_utils import create_common_tensor


class TesBatchNms(TestCase):
    def test_batch_nms_shape_format(self):
        boxes = torch.randn(8, 4, 1, 4).npu()
        scores = torch.randn(8, 4, 1).npu()
        boxes_fp16 = boxes.half()
        scores_fp16 = scores.half()
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 4, 4)
        boxes1, scores1, classes1, num1 = torch_npu.npu_batch_nms(boxes_fp16, scores_fp16, 0.3, 0.5, 4, 4)
        expedt_nmsed_classes = torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
                                             [0.0000, 0.0000, 0.0000, 0.0000],
                                             [0.0000, 0.0000, 0.0000, 0.0000],
                                             [0.0000, 0.0000, 0.0000, 0.0000],
                                             [0.0000, 0.0000, 0.0000, 0.0000],
                                             [0.0000, 0.0000, 0.0000, 0.0000],
                                             [0.0000, 0.0000, 0.0000, 0.0000],
                                             [0.0000, 0.0000, 0.0000, 0.0000]], dtype=torch.float32)
        self.assertRtolEqual(expedt_nmsed_classes, nmsed_classes.cpu())
        self.assertRtolEqual(expedt_nmsed_classes.half(), classes1.cpu())


if __name__ == "__main__":
    run_tests()
