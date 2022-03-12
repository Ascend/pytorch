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

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.contrib.function import npu_iou, npu_giou

class TestIou(TestCase):
    def test_npu_iou_1(self):
        box1 = torch.FloatTensor([[10, 55, 85, 160]])
        box2 = torch.FloatTensor([[18, 45, 80, 130], [38, 85, 70, 230]])
        box1 = box1.float().npu()
        box2 = box2.float().npu()
        iou1 = npu_iou(box1, box2, mode="iou")
        iou2 = npu_iou(box1, box2)
        expedt_iou1 = torch.tensor([[0.5469, 0.2373]], dtype=torch.float32)
        expedt_iou2 = torch.tensor([[0.5469, 0.2373]], dtype=torch.float32)
        self.assertRtolEqual(expedt_iou1, iou1.cpu())
        self.assertRtolEqual(expedt_iou2, iou2.cpu())
    
    def test_npu_iou_2(self):
        box1 = torch.randint(0, 256, size=(8, 4))
        box2 = torch.randint(0, 256, size=(8, 4))
        box1 = box1.float().npu()
        box2 = box2.float().npu()
        iou1 = npu_iou(box1, box2, mode="iou")
        iou2 = npu_iou(box1, box2)
        expedt_iou1 = torch.tensor([[0.0000, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000],
                                    [0.0000, -0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, -0.0000],
                                    [0.0000, -0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, -0.0000],
                                    [0.0000, 0.0000, -0.0000, 0.0000, 0.0238, 0.0575, -0.0000, -0.0000],
                                    [0.0000, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000],
                                    [0.0000, -0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, -0.0000],
                                    [0.0000, 0.0000, -0.0000, 0.0000, 0.0000, 0.0943, -0.0000, 0.0000],
                                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000]],
                                    dtype=torch.float32)
        expedt_iou2 = torch.tensor([[0.0000, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000],
                                    [0.0000, -0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, -0.0000],
                                    [0.0000, -0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, -0.0000],
                                    [0.0000, 0.0000, -0.0000, 0.0000, 0.0238, 0.0575, -0.0000, -0.0000],
                                    [0.0000, 0.0000, -0.0000, 0.0000, 0.0000, 0.0000, -0.0000, -0.0000],
                                    [0.0000, -0.0000, -0.0000, -0.0000, 0.0000, 0.0000, -0.0000, -0.0000],
                                    [0.0000, 0.0000, -0.0000, 0.0000, 0.0000, 0.0943, -0.0000, 0.0000],
                                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000]], 
                                    dtype=torch.float32)
        self.assertRtolEqual(expedt_iou1, iou1.cpu())
        self.assertRtolEqual(expedt_iou2, iou2.cpu())

    def test_npu_iou_3(self):
        box1 = torch.FloatTensor([[10, 55, 85, 160]])
        box2 = torch.FloatTensor([[18, 45, 80, 130], [38, 85, 70, 230]])
        box1 = box1.float().npu() / 100.
        box2 = box2.float().npu() / 100.
        iou1 = npu_iou(box1, box2, mode="iou", is_normalized=True, normalized_scale=100.)
        iou2 = npu_iou(box1, box2, is_normalized=True, normalized_scale=100.)
        expedt_iou1 = torch.tensor([[0.5469, 0.2373]], dtype=torch.float32)
        expedt_iou2 = torch.tensor([[0.5469, 0.2373]], dtype=torch.float32)
        self.assertRtolEqual(expedt_iou1, iou1.cpu())
        self.assertRtolEqual(expedt_iou2, iou2.cpu())

    def test_npu_giou_1(self):
        box1 = torch.randn(16, 4)
        box1.requires_grad = True
        box2 = torch.randn(16, 4)
        box1 = box1.float().npu()
        box2 = box2.float().npu()
        iou1 = npu_giou(box1, box2)
        expedt_iou1 = torch.tensor([[-1.1377e+00],
                                    [-7.3738e-01],
                                    [ 1.5935e-01],
                                    [-2.1271e+00],
                                    [ 1.2136e+03],
                                    [-7.2693e+00],
                                    [-4.7243e-01],
                                    [-2.0380e+00],
                                    [ 5.1004e+00],
                                    [-1.8952e+00],
                                    [-3.2175e+00],
                                    [-9.4184e-01],
                                    [ 5.3800e-01],
                                    [-7.9274e-01],
                                    [-1.0181e+00],
                                    [-1.6168e+00]], dtype=torch.float32)
        self.assertRtolEqual(expedt_iou1, iou1.cpu().detach())

if __name__ == "__main__":
    run_tests()