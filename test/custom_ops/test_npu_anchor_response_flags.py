import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestNpuAnchorResponseFlags(TestCase):
    def custom_op_exec(self, gt_bboxes, featmap_size, strides, num_base_anchors):
        if gt_bboxes.dtype == torch.float16:
            gt_bboxes = gt_bboxes.to(torch.float32)
        feat_h, feat_w = featmap_size
        gt_bboxes_cx = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) * 0.5)
        gt_bboxes_cy = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) * 0.5)
        gt_bboxes_grid_x = torch.floor(gt_bboxes_cx / strides[0]).int()
        gt_bboxes_grid_y = torch.floor(gt_bboxes_cy / strides[1]).int()
        gt_bboxes_grid_idx = gt_bboxes_grid_y * feat_w + gt_bboxes_grid_x
        responsible_grid = torch.zeros(feat_h * feat_w, dtype=torch.uint8).npu()
        gt_bboxes_grid_idx = gt_bboxes_grid_idx.long()
        responsible_grid[gt_bboxes_grid_idx] = 1
        responsible_grid = responsible_grid[:, None].expand(
            responsible_grid.size(0), num_base_anchors).contiguous().view(-1)
        return responsible_grid.cpu().numpy()

    def npu_op_exec(self, input_npu, featmap_size, strides, num_base_anchors):
        out = torch_npu.npu_anchor_response_flags(input_npu, featmap_size, strides, num_base_anchors)
        out = out.cpu().numpy()
        return out

    def test_npu_anchor_response_flags(self):
        shape_format = [
            [[np.float32, -1, [100, 4]], [60, 60], [2, 2], 9],
            [[np.float16, -1, [200, 4]], [10, 10], [32, 32], 3],
            [[np.float16, -1, [500, 4]], [32, 32], [16, 16], 5]
        ]
        for item in shape_format:
            _, npu_input = create_common_tensor(item[0], 0, 100)
            custom_output = self.custom_op_exec(npu_input, *item[1:])
            npu_output = self.npu_op_exec(npu_input, *item[1:])
            self.assertRtolEqual(custom_output, npu_output)


if __name__ == "__main__":
    run_tests()
