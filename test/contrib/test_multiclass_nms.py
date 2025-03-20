import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.contrib.function import npu_multiclass_nms, \
    npu_batched_multiclass_nms


class TestMultiClassNms(TestCase):
    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_npu_multiclass_nms_1(self):
        np.random.seed(123)
        data1 = np.random.randint(low=1, high=255, size=(1000, 4))
        boxes = torch.tensor(data1, dtype=torch.int64)
        data2 = np.random.randn(1000, 81)
        scores = torch.tensor(data2, dtype=torch.float32)
        boxes = boxes.npu().half()
        scores = scores.npu().half()
        det_bboxes, det_labels = npu_multiclass_nms(boxes, scores, score_thr=0.3, nms_thr=0.5, max_num=3)
        expect_det_bboxes = torch.tensor([[81.9375, 183.8750, 35.0000, 172.0000, 4.1797],
                                          [52.0000, 47.0000, 196.8750, 67.0000, 3.8750],
                                          [76.0000, 140.0000, 42.0000, 132.0000, 3.8613]], dtype=torch.float16)
        expect_det_labels = torch.tensor([69., 31., 48.], dtype=torch.float16)
        self.assertRtolEqual(expect_det_bboxes, det_bboxes.cpu())
        self.assertRtolEqual(expect_det_labels, det_labels.cpu())

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_npu_multiclass_nms_2(self):
        np.random.seed(123)
        data1 = np.random.randn(1000, 4)
        boxes = torch.tensor(data1, dtype=torch.float32)
        data2 = np.random.randn(1000, 81)
        scores = torch.tensor(data2, dtype=torch.float32)
        boxes = boxes.npu().half()
        scores = scores.npu().half()
        det_bboxes, det_labels = npu_multiclass_nms(boxes, scores, score_thr=0.3, nms_thr=0.5, max_num=3)
        expect_det_bboxes = torch.tensor([[0.1643, 0.7480, 0.1807, -0.2734, 4.1836],
                                          [-0.9736, -0.1373, 0.9473, 1.0938, 4.1641],
                                          [1.5234, -0.6831, -1.8359, 1.1035, 4.0664]], dtype=torch.float16)
        expect_det_labels = torch.tensor([46., 8., 69.], dtype=torch.float16)
        self.assertRtolEqual(expect_det_bboxes, det_bboxes.cpu())
        self.assertRtolEqual(expect_det_labels, det_labels.cpu())

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_npu_batched_multiclass_nms_1(self):
        np.random.seed(339)
        data1 = np.random.randint(low=1, high=255, size=(4, 200, 80, 4))
        boxes = torch.tensor(data1, dtype=torch.int64)
        data2 = np.random.randn(4, 200, 81)
        scores = torch.tensor(data2, dtype=torch.float32)
        boxes = boxes.npu().half()
        scores = scores.npu().half()
        det_bboxes, det_labels = npu_batched_multiclass_nms(boxes, scores, score_thr=0.3, nms_thr=0.5, max_num=3)
        expect_det_bboxes = torch.tensor([[[195.0000, 133.0000, 123.0000, 36.0000, 4.3984],
                                           [70.0000, 113.0000, 149.0000, 22.9844, 3.8184],
                                           [203.0000, 250.0000, 9.0000, 81.9375, 3.5996]],

                                          [[60.0000, 19.0000, 233.8750, 142.0000, 3.7852],
                                           [147.0000, 218.0000, 223.0000, 86.9375, 3.6426],
                                           [19.0000, 113.0000, 91.9375, 238.8750, 3.5449]],

                                          [[163.8750, 47.9688, 221.8750, 218.0000, 3.8281],
                                           [5.9961, 81.9375, 110.0000, 171.0000, 3.7734],
                                           [155.0000, 133.0000, 138.0000, 108.0000, 3.4844]],

                                          [[238.8750, 78.0000, 188.0000, 17.0000, 3.9121],
                                           [79.0000, 115.9375, 83.0000, 246.0000, 3.5879],
                                           [11.0000, 94.0000, 243.0000, 226.8750, 3.4805]]], dtype=torch.float16)
        expect_det_labels = torch.tensor([[27., 36., 78.],
                                          [19., 27., 39.],
                                          [54., 18., 27.],
                                          [72., 43., 31.]], dtype=torch.float16)
        self.assertRtolEqual(expect_det_bboxes, det_bboxes.cpu())
        self.assertRtolEqual(expect_det_labels, det_labels.cpu())


if __name__ == "__main__":
    run_tests()
