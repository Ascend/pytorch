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

class TesBatchNms(TestCase):
    def test_batch_nms_shape_format_fp32(self, device):
        boxes = torch.randn(8, 2, 4, 4, dtype = torch.float32).to("npu")
        scores = torch.randn(3, 2, 4, dtype = torch.float32).to("npu")
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch.npu_batch_nms(boxes, scores, 0.3, 0.5, 3, 4)
        expedt_nmsed_classes = torch.tensor([[0.0000, 2.1250, 0.0000, 1.8750],
                                             [0.0000, 0.0000, 0.0000, 0.0000],
                                             [0.0000, 1.8750, 0.0000, 2.0000],
                                             [0.0000, 0.0000, 0.0000, 0.0000],
                                             [0.0000, 2.0000, 0.0000, 2.1250],
                                             [0.0000, 2.1250, 0.0000, 1.8750],
                                             [0.0000, 0.0000, 0.0000, 0.0000],
                                             [0.0000, 0.0000, 0.0000, 0.0000]], dtype = torch.float16)
        self.assertRtolEqual(expedt_nmsed_classes, nmsed_classes.cpu())

instantiate_device_type_tests(TesBatchNms, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
