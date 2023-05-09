import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestGridAssignPositive(TestCase):

    def supported_op_exec(self, input1, box_responsible_flags, max_overlaps, argmax_overlaps, pos_iou_thr):
        pos_inds = (max_overlaps > pos_iou_thr) & box_responsible_flags.type(torch.bool)
        argmax_overlaps = argmax_overlaps.to(input1.dtype)
        input1[pos_inds] = argmax_overlaps[pos_inds] + 1
        return input1.cpu().detach()

    def custom_op_exec(self, input1, overlaps, box_responsible_flags, max_overlaps, argmax_overlaps,
                       gt_max_overlaps, gt_argmax_overlaps, num_gts, pos_iou_thr, min_pos_iou, gt_max_assign_all):
        output = torch_npu.npu_grid_assign_positive(input1, overlaps, box_responsible_flags, max_overlaps,
                                                    argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps, num_gts,
                                                    pos_iou_thr, min_pos_iou, gt_max_assign_all)
        return output.cpu().detach()

    def test_npu_grid_assign_positive(self):
        npu_input = torch.rand((4,), dtype=torch.float32).to("npu")
        overlaps = torch.rand((2, 4), dtype=torch.float32).to("npu")
        box_responsible_flags = torch.tensor([1, 1, 1, 0], dtype=torch.uint8).to("npu")
        max_overlaps = torch.rand((4,), dtype=torch.float32).to("npu")
        argmax_overlaps = torch.tensor([1, 0, 1, 0], dtype=torch.int32).to("npu")
        gt_max_overlaps = torch.rand((2,), dtype=torch.float32).to("npu")
        gt_argmax_overlaps = torch.tensor([1, 0], dtype=torch.int32).to("npu")
        num_gts = 128
        pos_iou_thr = .5
        min_pos_iou = .0
        gt_max_assign_all = True

        supported_output = self.supported_op_exec(npu_input, box_responsible_flags, max_overlaps,
                                                  argmax_overlaps, pos_iou_thr)
        custom_output = self.custom_op_exec(npu_input, overlaps, box_responsible_flags, max_overlaps,
                                            argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps, num_gts,
                                            pos_iou_thr, min_pos_iou, gt_max_assign_all)
        self.assertRtolEqual(supported_output, custom_output)


if __name__ == "__main__":
    run_tests()
