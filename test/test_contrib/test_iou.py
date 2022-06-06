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
from torch_npu.contrib.function import npu_iou, npu_giou, npu_diou, npu_ciou

class TestIou(TestCase):
    def test_npu_iou_diff_shape_input(self):
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
    
    def test_npu_iou_same_shape_input(self):
        box1 = torch.tensor([[0, 0, 10, 10],
                            [10, 10, 20, 20],
                            [32, 32, 38, 42]], dtype=torch.float32).to("npu")
        box2 = torch.tensor([[0, 0, 10, 20],
                            [0, 10, 10, 10],
                            [10, 10, 20, 20]], dtype=torch.float32).to("npu")
        expedt_iou = torch.tensor([[0.4985, 0.0000, 0.0000],
                                   [0.0000, 0.0000, 0.9961],
                                   [0.0000, 0.0000, 0.0000]], dtype=torch.float32)
        iou1 = npu_iou(box1, box2, mode="iou")
        iou2 = npu_iou(box1, box2, mode='ptiou')
        self.assertRtolEqual(expedt_iou, iou1.cpu())
        self.assertRtolEqual(expedt_iou, iou2.cpu())

    def test_npu_iou_multiparameters(self):
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

    def test_npu_giou(self):
        box1 = torch.tensor([[0, 0, 10, 10],
                            [10, 10, 20, 20],
                            [32, 32, 38, 42]], dtype=torch.float32).to("npu")
        box1.requires_grad = True
        box2 = torch.tensor([[0, 0, 10, 20],
                            [0, 10, 10, 10],
                            [10, 10, 20, 20]], dtype=torch.float32).to("npu")
        iou = npu_giou(box1, box2)
        expedt_iou = torch.tensor([[ 0.5000],
                                    [ 0.0111],
                                    [-0.2523]], dtype=torch.float32)
        self.assertRtolEqual(expedt_iou, iou.cpu().detach())
        
    def test_npu_diou(self):
        box1 = torch.tensor([[0, 0, 10, 10],
                            [10, 10, 20, 20],
                            [32, 32, 38, 42],
                            [8, 8, 4, 4]], dtype=torch.float32).to("npu")
        box1.requires_grad = True
        box2 = torch.tensor([[0, 0, 10, 20],
                            [0, 10, 10, 10],
                            [10, 10, 20, 20],
                            [8, 8, 4, 4]], dtype=torch.float32).to("npu")
        box2.requires_grad = True
        expedt_diou = torch.tensor([[-0.0083, -0.0450, -0.0007, -0.0567]], dtype=torch.float32)
        diou = npu_diou(box1, box2)
        self.assertRtolEqual(expedt_diou, diou.cpu().detach())
        
    def test_npu_ciou(self):
        box1 = torch.tensor([[0, 0, 10, 10],
                            [10, 10, 20, 20],
                            [32, 32, 38, 42],
                            [8, 8, 4, 4]], dtype=torch.float32).to("npu")
        box1.requires_grad = True
        box2 = torch.tensor([[0, 0, 10, 20],
                            [0, 10, 10, 10],
                            [10, 10, 20, 20],
                            [8, 8, 4, 4]], dtype=torch.float32).to("npu")
        expedt_ciou = torch.tensor([[-0.0794,  0.3052, -0.0610, -0.1021]], dtype=torch.float32)
        box2.requires_grad = True
        ciou = npu_ciou(box1, box2)
        self.assertRtolEqual(expedt_ciou, ciou.cpu().detach())
        
if __name__ == "__main__":
    run_tests()