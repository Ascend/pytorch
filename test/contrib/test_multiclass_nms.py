# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
from torch_npu.contrib.function import npu_multiclass_nms, \
    npu_batched_multiclass_nms


class TestMultiClassNms(TestCase):
    @unittest.skip("skip test_npu_multiclass_nms_1 now")
    def test_npu_multiclass_nms_1(self):
        boxes = torch.randint(1, 255, size=(1000, 4))
        scores = torch.randn(1000, 81)
        boxes = boxes.npu().half()
        scores = scores.npu().half()
        det_bboxes, det_labels = npu_multiclass_nms(boxes, scores, score_thr=0.3, nms_thr=0.5, max_num=3)
        expedt_det_bboxes = torch.tensor([[57.0000, 198.8750, 45.9688, 221.8750, 4.1484],
                                          [215.0000, 155.0000, 236.8750, 137.0000, 3.9023],
                                          [208.8750, 221.0000, 228.0000, 17.0000, 3.8867]],
                                         dtype=torch.float16)
        expedt_det_labels = torch.tensor([59., 3., 75.], dtype=torch.float16)
        self.assertRtolEqual(expedt_det_bboxes, det_bboxes.cpu())
        self.assertRtolEqual(expedt_det_labels, det_labels.cpu())

    @unittest.skip("skip test_npu_multiclass_nms_2 now")
    def test_npu_multiclass_nms_2(self):
        boxes = torch.randn(1000, 4)
        scores = torch.randn(1000, 81)
        boxes = boxes.npu().half()
        scores = scores.npu().half()
        det_bboxes, det_labels = npu_multiclass_nms(boxes, scores, score_thr=0.3, nms_thr=0.5, max_num=3)
        expedt_det_bboxes = torch.tensor([[0.2231, -1.6943, -0.1172, -1.0547, 4.1484],
                                          [0.2891, 0.4897, -0.3809, -0.7129, 3.9023],
                                          [0.6694, -1.2266, -0.3027, 0.4639, 3.8867]],
                                         dtype=torch.float16)
        expedt_det_labels = torch.tensor([59., 3., 75.], dtype=torch.float16)
        self.assertRtolEqual(expedt_det_bboxes, det_bboxes.cpu())
        self.assertRtolEqual(expedt_det_labels, det_labels.cpu())

    @unittest.skip("skip test_npu_batched_multiclass_nms_1 now")
    def test_npu_batched_multiclass_nms_1(self):
        boxes = torch.randint(1, 255, size=(4, 200, 80, 4))
        scores = torch.randn(4, 200, 81)
        boxes = boxes.npu().half()
        scores = scores.npu().half()
        det_bboxes, det_labels = npu_batched_multiclass_nms(boxes, scores, score_thr=0.3, nms_thr=0.5, max_num=3)
        expedt_det_bboxes = torch.tensor([[[221.8750, 60.0000, 183.0000, 22.0000, 3.8867],
                                           [167.0000, 250.0000, 136.0000, 144.0000, 3.6445],
                                           [45.9688, 147.0000, 67.0000, 241.8750, 3.4844]],
                                          [[5.0000, 178.0000, 243.8750, 138.0000, 3.7344],
                                           [238.0000, 132.0000, 47.0000, 84.0000, 3.6836],
                                           [32.0000, 110.0000, 131.0000, 73.0000, 3.6309]],
                                          [[111.9375, 120.9375, 54.0000, 231.0000, 3.9219],
                                           [147.0000, 162.0000, 78.0000, 1.0010, 3.9219],
                                           [157.0000, 118.0000, 57.0000, 115.0000, 3.6523]],
                                          [[80.0000, 126.9375, 54.0000, 246.8750, 3.7344],
                                           [31.0000, 253.8750, 19.0000, 138.0000, 3.6328],
                                           [54.0000, 253.8750, 78.0000, 75.0000, 3.5586]]],
                                         dtype=torch.float16)
        expedt_det_labels = torch.tensor([[76., 3., 32.],
                                          [26., 66., 25.],
                                          [34., 41., 30.],
                                          [22., 27., 46.]], dtype=torch.float16)
        self.assertRtolEqual(expedt_det_bboxes, det_bboxes.cpu())
        self.assertRtolEqual(expedt_det_labels, det_labels.cpu())


if __name__ == "__main__":
    run_tests()
