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
from torch_npu.contrib.function import npu_bbox_coder_encode_yolo, \
    npu_bbox_coder_encode_xyxy2xywh, npu_bbox_coder_decode_xywh2xyxy


class TestBboxCoder(TestCase):
    @unittest.skip("skip test_npu_bbox_coder_encode_xyxy2xywh now")
    def test_npu_bbox_coder_encode_xyxy2xywh(self):
        bboxes = torch.randint(0, 512, size=(6, 4))
        gt_bboxes = torch.randint(0, 512, size=(6, 4))
        bboxes = bboxes.npu()
        gt_bboxes = gt_bboxes.npu()
        npuout_1 = npu_bbox_coder_encode_xyxy2xywh(bboxes, gt_bboxes)
        npuout_2 = npu_bbox_coder_encode_xyxy2xywh(bboxes / 512., gt_bboxes / 512.,
                                                   is_normalized=True, normalized_scale=512.)
        expect_cpu = torch.tensor([[-1.1562e+01, -1.4492e+00, 2.8105e+00, -1.1855e+00],
                                   [-3.1465e+00, -1.1826e+00, 1.2939e+00, 1.2314e+00],
                                   [-3.8696e-01, 1.1758e+00, 1.7346e-01, -2.8174e-01],
                                   [1.3086e+01, 1.6631e+00, 2.4902e+00, 5.6055e-01],
                                   [-1.6914e+00, 3.8188e+01, 7.7490e-01, 2.9453e+00],
                                   [3.2598e+00, -2.8019e-03, 1.5000e+00, -1.5342e+00]],
                                  dtype=torch.float32)
        self.assertRtolEqual(expect_cpu.numpy(), npuout_1.cpu().numpy())
        self.assertRtolEqual(expect_cpu.numpy(), npuout_2.cpu().numpy())

    @unittest.skip("skip test_npu_bbox_coder_encode_yolo now")
    def test_npu_bbox_coder_encode_yolo(self):
        bboxes = torch.randint(0, 512, size=(6, 4))
        gt_bboxes = torch.randint(0, 512, size=(6, 4))
        stride = torch.randint(0, 32, size=(6,))
        bboxes = bboxes.npu()
        gt_bboxes = gt_bboxes.npu()
        stride = stride.npu()
        npu_output = npu_bbox_coder_encode_yolo(bboxes, gt_bboxes, stride)
        expect_cpu = torch.tensor([[1.0000e+00, 1.0000e+00, -1.3816e+01, -1.3816e+01],
                                   [1.0000e+00, 1.0000e-06, -1.3816e+01, -1.3816e+01],
                                   [1.0000e-06, 1.0000e+00, -1.3816e+01, -2.8768e-01],
                                   [1.0000e+00, 1.0000e+00, 2.5421e+00, -1.3816e+01],
                                   [1.0000e-06, 1.0000e-06, -1.3816e+01, -1.3816e+01],
                                   [1.0000e-06, 4.0909e-01, 1.4889e+00, -1.3816e+01]],
                                  dtype=torch.float32)
        self.assertRtolEqual(expect_cpu.numpy(), npu_output.cpu().numpy())
    
    @unittest.skip("skip test_npu_bbox_coder_decode_xywh2xyxy now")
    def test_npu_bbox_coder_decode_xywh2xyxy(self):
        max_shape = 512
        bboxes = torch.randint(0, max_shape, size=(6, 4))
        pred_bboxes = torch.randn(6, 4)
        bboxes = bboxes.npu()
        pred_bboxes = pred_bboxes.npu()
        npu_output = npu_bbox_coder_decode_xywh2xyxy(bboxes, pred_bboxes,
                                                     max_shape=(max_shape, max_shape))
        expect_cpu = torch.tensor([[295.2500, 289.5000, 291.7500, 198.5000],
                                   [235.0000, 221.1250, 21.9375, 511.0000],
                                   [415.5000, 199.1250, 444.5000, 205.6250],
                                   [133.0000, 16.0000, 137.5000, 272.0000],
                                   [275.2500, 373.7500, 367.2500, 362.2500],
                                   [408.7500, 0.0000, 396.7500, 78.0000]],
                                  dtype=torch.float32)
        self.assertRtolEqual(expect_cpu.numpy(), npu_output.cpu().numpy())


if __name__ == "__main__":
    run_tests()
