import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.contrib.function import npu_bbox_coder_encode_yolo, \
    npu_bbox_coder_encode_xyxy2xywh, npu_bbox_coder_decode_xywh2xyxy


class TestBboxCoder(TestCase):
    @SupportedDevices(["Ascend910B"])
    def test_npu_bbox_coder_encode_xyxy2xywh_A2(self):
        bboxes = torch.tensor([[1., 2., 3., 4.], [3., 4., 5., 6.]], dtype=torch.float32).to("npu")
        gt_bboxes = torch.tensor([[5., 6., 7., 8.], [7., 8., 9., 6.]], dtype=torch.float32).to("npu")
        npuout_1 = npu_bbox_coder_encode_xyxy2xywh(bboxes, gt_bboxes)
        npuout_2 = npu_bbox_coder_encode_xyxy2xywh(bboxes / 512., gt_bboxes / 512., is_normalized=True,
                                                   normalized_scale=512.)
        expect_cpu = torch.tensor([[1.3330, 1.3330, 0.0000, 0.0000],
                                   [1.3330, 0.6665, 0.0000, np.nan]], dtype=torch.float32)

        self.assertRtolEqual(expect_cpu.numpy(), npuout_1.cpu().numpy())
        self.assertRtolEqual(expect_cpu.numpy(), npuout_2.cpu().numpy())

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_npu_bbox_coder_encode_xyxy2xywh(self):
        np.random.seed(123)
        data1 = np.random.randint(low=0, high=512, size=(6, 4))
        bboxes = torch.tensor(data1, dtype=torch.int64)
        data2 = np.random.randint(low=0, high=512, size=(6, 4))
        gt_bboxes = torch.tensor(data2, dtype=torch.int64)
        bboxes = bboxes.npu()
        gt_bboxes = gt_bboxes.npu()
        npuout_1 = npu_bbox_coder_encode_xyxy2xywh(bboxes, gt_bboxes)
        npuout_2 = npu_bbox_coder_encode_xyxy2xywh(bboxes / 512., gt_bboxes / 512.,
                                                   is_normalized=True, normalized_scale=512.)
        expect_cpu = torch.tensor([[1.1807, -0.7979, -0.4014, 0.7285],
                                   [1.1064, -2.4883, -2.2051, 0.2026],
                                   [1.3047, -6.5430, 2.0156, 0.8291],
                                   [0.7993, -0.1396, -0.9336, 0.9038],
                                   [-4.6250, -17.8906, 1.8389, 3.0938],
                                   [-0.3145, -0.5000, -2.7676, 0.2954]], dtype=torch.float32)
        self.assertRtolEqual(expect_cpu.numpy(), npuout_1.cpu().numpy())
        self.assertRtolEqual(expect_cpu.numpy(), npuout_2.cpu().numpy())

    def test_npu_bbox_coder_encode_yolo(self):
        np.random.seed(234)
        data1 = np.random.randint(low=0, high=512, size=(6, 4))
        bboxes = torch.tensor(data1, dtype=torch.int64)
        data2 = np.random.randint(low=0, high=512, size=(6, 4))
        gt_bboxes = torch.tensor(data2, dtype=torch.int64)
        stride = torch.randint(0, 32, size=(6,))
        bboxes = bboxes.npu()
        gt_bboxes = gt_bboxes.npu()
        stride = stride.npu()
        npu_output = npu_bbox_coder_encode_yolo(bboxes, gt_bboxes, stride)
        expect_cpu = torch.tensor([[1.0000e-06, 1.0000e-06, -1.1692e+00, -8.7373e-01],
                                   [1.0000e+00, 1.0000e-06, -1.3816e+01, -1.3816e+01],
                                   [1.0000e-06, 1.0000e-06, 7.8846e-01, -6.0799e-01],
                                   [8.5000e-01, 1.0000e-06, -1.3816e+01, -2.2460e+00],
                                   [1.0000e+00, 6.5000e-01, -1.3312e+00, -1.3816e+01],
                                   [1.0000e+00, 1.0000e-06, 4.1058e-01, 2.4875e+00]], dtype=torch.float32)
        self.assertRtolEqual(expect_cpu.numpy(), npu_output.cpu().numpy())

    def test_npu_bbox_coder_decode_xywh2xyxy(self):
        max_shape = 512
        np.random.seed(345)
        data1 = np.random.randint(low=0, high=max_shape, size=(6, 4))
        bboxes = torch.tensor(data1, dtype=torch.int64)
        data2 = np.random.randn(6, 4)
        pred_bboxes = torch.tensor(data2, dtype=torch.float32)
        bboxes = bboxes.npu()
        pred_bboxes = pred_bboxes.npu()
        npu_output = npu_bbox_coder_decode_xywh2xyxy(bboxes, pred_bboxes,
                                                     max_shape=(max_shape, max_shape))
        expect_cpu = torch.tensor([[69.6262, 511.0000, 116.1755, 274.9286],
                                   [50.0117, 511.0000, 28.3380, 456.6909],
                                   [354.6850, 227.5013, 275.1128, 219.0241],
                                   [0.0000, 301.9681, 0.0000, 291.4622],
                                   [201.7177, 84.3664, 325.5544, 87.1693],
                                   [283.7152, 407.1944, 0.0000, 511.0000]], dtype=torch.float32)
        self.assertRtolEqual(expect_cpu.numpy(), npu_output.cpu().numpy())


if __name__ == "__main__":
    run_tests()
