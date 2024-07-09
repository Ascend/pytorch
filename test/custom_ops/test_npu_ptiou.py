import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuIou(TestCase):

    def custom_ptiou(self, bboxes, gtboxes, mode=0):
        output = torch_npu.npu_ptiou(bboxes, gtboxes, mode)
        return output

    def custom_op_exec(self, bboxes, gtboxes, mode=0):
        output = self.custom_ptiou(bboxes, gtboxes, mode)
        return output.cpu().numpy()

    def npu_op_exec(self, bboxes, gtboxes, mode=0):
        output = torch_npu.npu_ptiou(bboxes, gtboxes, mode)
        return output.cpu().numpy()

    def test_iou_fp16(self):
        bboxes = torch.tensor([[0, 0, 10, 10],
                               [10, 10, 20, 20],
                               [32, 32, 38, 42]], dtype=torch.float16).npu()
        gtboxes = torch.tensor([[0, 0, 10, 20],
                                [0, 10, 10, 10],
                                [10, 10, 20, 20]], dtype=torch.float16).npu()

        output_npu = self.npu_op_exec(bboxes, gtboxes, 1)
        output_custom = self.custom_op_exec(bboxes, gtboxes, 1)
        self.assertRtolEqual(output_npu, output_custom)

        output_npu = self.npu_op_exec(bboxes, gtboxes)
        output_custom = self.custom_op_exec(bboxes, gtboxes)
        self.assertRtolEqual(output_npu, output_custom)

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
