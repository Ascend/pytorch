import unittest
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuIou(TestCase):
    def npu_iou(self, bboxes, gtboxes, mode=0):
        def box_area(boxes):
            return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Logics here have some differents from torchvision.
        lt = torch.max(bboxes[:, :2], gtboxes[:, None, :2])
        rb = torch.min(bboxes[:, 2:], gtboxes[:, None, 2:])
        wh = torch.clamp(rb - lt, min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        eps = 1e-7
        if mode == 0:
            b_area = box_area(bboxes)
            gt_area = box_area(gtboxes)
            iou = inter / (gt_area[:, None] + b_area - inter + eps)
            return iou
        else:
            gt_area = box_area(gtboxes)
            iof = inter / (gt_area[:, None] + eps)
            return iof

    def custom_op_exec(self, bboxes, gtboxes, mode=0):
        output = self.npu_iou(bboxes, gtboxes, mode)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec(self, bboxes, gtboxes, mode=0):
        output = torch_npu.npu_iou(bboxes, gtboxes, mode)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @unittest.skip("skip test_iou_fp16 now")
    def test_iou_fp16(self):
        bboxes = torch.tensor([[0, 0, 10, 10],
                               [10, 10, 20, 20],
                               [32, 32, 38, 42]], dtype=torch.float16).to("npu")
        gtboxes = torch.tensor([[0, 0, 10, 20],
                                [0, 10, 10, 10],
                                [10, 10, 20, 20]], dtype=torch.float16).to("npu")

        output_npu = self.npu_op_exec(bboxes, gtboxes, 1)
        output_custom = self.custom_op_exec(bboxes, gtboxes, 1)
        self.assertRtolEqual(output_npu, output_custom)

        output_npu = self.npu_op_exec(bboxes, gtboxes)
        output_custom = self.custom_op_exec(bboxes, gtboxes)
        self.assertRtolEqual(output_npu, output_custom)

    @unittest.skip("skip test_iou_fp16_pt now")
    def test_iou_fp16_pt(self):
        bboxes = torch.tensor([[1, 2, 3, 4],
                               [5, 6, 7, 8],
                               [9, 10, 11, 12],
                               [13, 14, 15, 16]], dtype=torch.float16).npu()
        gtboxes = torch.tensor([[1, 2, 3, 4],
                                [5, 6, 7, 8]], dtype=torch.float16).npu()

        output_npu = self.npu_op_exec(bboxes, gtboxes, 1)
        output_custom = self.custom_op_exec(bboxes, gtboxes, 1)
        self.assertRtolEqual(output_npu, output_custom)

        output_npu = self.npu_op_exec(bboxes, gtboxes)
        output_custom = self.custom_op_exec(bboxes, gtboxes)
        self.assertRtolEqual(output_npu, output_custom)


if __name__ == "__main__":
    run_tests()
