import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNmsWithMask(TestCase):
    def npu_op_exec(self, input1, iou_threshold):
        npu_output1, npu_output2, npu_output3, = torch_npu.npu_nms_with_mask(input1, iou_threshold)
        npu_output1 = npu_output1.to("cpu")
        npu_output2 = npu_output2.to("cpu")
        npu_output3 = npu_output3.to("cpu")

        return npu_output1, npu_output2, npu_output3

    def test_nms_with_mask_float32(self):
        input1 = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.6], [6.0, 7.0, 8.0, 9.0, 0.4]]).npu()
        iou_threshold = 0.5
        eq_output1 = torch.tensor([[0.0000, 1.0000, 2.0000, 3.0000, 0.6001],
                                   [6.0000, 7.0000, 8.0000, 9.0000, 0.3999]])
        eq_output2 = torch.tensor([0, 1], dtype=torch.int32)
        eq_output3 = torch.tensor([1, 1], dtype=torch.uint8)
        npu_output1, npu_output2, npu_output3 = self.npu_op_exec(input1, iou_threshold)
        self.assertRtolEqual(eq_output1, npu_output1)
        self.assertRtolEqual(eq_output2, npu_output2)
        self.assertRtolEqual(eq_output3, npu_output3)


if __name__ == "__main__":
    run_tests()
