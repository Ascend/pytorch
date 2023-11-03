import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestIou(TestCase):
    def test_iou_fp16(self):
        bboxes = torch.tensor([[0, 0, 10, 10],
                               [10, 10, 20, 20],
                               [32, 32, 38, 42]], dtype=torch.float16).to("npu")
        gtboxes = torch.tensor([[0, 0, 10, 20],
                               [0, 10, 10, 10],
                               [10, 10, 20, 20]], dtype=torch.float16).to("npu")
        expect_iof = torch.tensor([[0.4990, 0.0000, 0.0000],
                                   [0.0000, 0.0000, 0.0000],
                                   [0.0000, 0.9980, 0.0000]], dtype=torch.float16)
        output_iof = torch_npu.npu_iou(bboxes, gtboxes, 1)
        self.assertRtolEqual(expect_iof, output_iof.cpu())

        expect_iou = torch.tensor([[0.4985, 0.0000, 0.0000],
                                   [0.0000, 0.0000, 0.0000],
                                   [0.0000, 0.9961, 0.0000]], dtype=torch.float16)
        output_iou = torch_npu.npu_iou(bboxes, gtboxes, 0)
        self.assertRtolEqual(expect_iou, output_iou.cpu())

    def test_iou_fp16_pt(self):
        bboxs = torch.tensor([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12],
                              [13, 14, 15, 16]], dtype=torch.float16).npu()
        gtboxes = torch.tensor([[1, 2, 3, 4],
                                [5, 6, 7, 8]], dtype=torch.float16).npu()
        expect_iof = torch.tensor([[0.9902, 0.0000, 0.0000, 0.0000],
                                   [0.0000, 0.9902, 0.0000, 0.0000]], dtype=torch.float16)
        output_iof = torch_npu.npu_ptiou(bboxs, gtboxes, 1)
        self.assertRtolEqual(expect_iof, output_iof.cpu(), 1.e-3)

        expect_iou = torch.tensor([[0.9805, 0.0000, 0.0000, 0.0000],
                                   [0.0000, 0.9805, 0.0000, 0.0000]], dtype=torch.float16)
        output_iou = torch_npu.npu_ptiou(bboxs, gtboxes, 0)
        self.assertRtolEqual(expect_iou, output_iou.cpu(), 1.e-3)


if __name__ == "__main__":
    run_tests()
