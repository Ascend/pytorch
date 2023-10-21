import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.contrib.function import npu_single_level_responsible_flags


class TestAnchorGenerator(TestCase):

    def single_level_responsible_flags(self,
                                       featmap_size,
                                       gt_bboxes,
                                       stride,
                                       num_base_anchors,
                                       device='cpu'):
        """Generate the responsible flags of anchor in a single feature map.
        Args:
            featmap_size (tuple[int]): The size of feature maps.
            gt_bboxes (Tensor): Ground truth boxes, shape (n, 4).
            stride (tuple(int)): stride of current level
            num_base_anchors (int): The number of base anchors.
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.
        Returns:
            torch.Tensor: The valid flags of each anchor in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        gt_bboxes_cx = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) * 0.5).to(device)
        gt_bboxes_cy = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) * 0.5).to(device)
        gt_bboxes_grid_x = torch.floor(gt_bboxes_cx / stride[0]).long()
        gt_bboxes_grid_y = torch.floor(gt_bboxes_cy / stride[1]).long()

        # row major indexing
        gt_bboxes_grid_idx = gt_bboxes_grid_y * feat_w + gt_bboxes_grid_x
        responsible_grid = torch.zeros(
            feat_h * feat_w, dtype=torch.uint8, device=device)
        responsible_grid[gt_bboxes_grid_idx] = 1
        responsible_grid = responsible_grid[:, None].expand(
            responsible_grid.size(0), num_base_anchors).contiguous().view(-1)
        return responsible_grid

    def test_anchor_generator(self):
        featmap_sizes = [[10, 10], [20, 20], [40, 40]]
        stride = [[32, 32], [16, 16], [8, 8]]
        gt_bboxes = torch.randint(0, 100, size=(128, 4))
        num_base_anchors = 3
        featmap_level = len(featmap_sizes)
        for i in range(featmap_level):
            gt_bboxes = gt_bboxes.npu()
            cpuout = self.single_level_responsible_flags(featmap_sizes[i],
                                                         gt_bboxes,
                                                         stride[i],
                                                         num_base_anchors)
            npuout = npu_single_level_responsible_flags(featmap_sizes[i],
                                                        gt_bboxes,
                                                        stride[i],
                                                        num_base_anchors)
            self.assertRtolEqual(cpuout, npuout.cpu())


if __name__ == "__main__":
    run_tests()
