import torch
import torch_npu


def box_dtype_check(box):
    if box not in [torch.float, torch.half]:
        return box.float()
    return box


def npu_single_level_responsible_flags(featmap_size,
                                       gt_bboxes,
                                       stride,
                                       num_base_anchors):
    """Using NPU OP to generate the responsible flags of anchor in a single feature map.

    .. note::
        Because of the limitation of NPU op,
        output_size(featmap_size[0] * featmap_size[1] * num_base_anchors) must be smaller than 60000.

    Args:
        featmap_size (tuple[int]): The size of feature maps.
        gt_bboxes (Tensor): Ground truth boxes, shape (n, 4). Support dtype: float, half.
        stride (tuple(int)): stride of current level
        num_base_anchors (int): The number of base anchors.

    Returns:
        torch.Tensor: The valid flags of each anchor in a single level \
            feature map. Output size is [featmap_size[0] * featmap_size[1] * num_base_anchors].
    """

    gt_bboxes = box_dtype_check(gt_bboxes)

    flags = torch_npu.npu_anchor_response_flags(
        gt_bboxes,
        featmap_size,
        stride,
        num_base_anchors)
    return flags
