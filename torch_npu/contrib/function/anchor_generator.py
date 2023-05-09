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

    Reference implementation link:
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py#L821

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


def main():
    featmap_sizes = [[10, 10], [20, 20], [40, 40]]
    stride = [[32, 32], [16, 16], [8, 8]]
    gt_bboxes = torch.randint(0, 512, size=(128, 4))
    num_base_anchors = 3
    featmap_level = len(featmap_sizes)

    torch.npu.set_device(0)

    for i in range(featmap_level):
        gt_bboxes = gt_bboxes.npu()
        out = npu_single_level_responsible_flags(featmap_sizes[i],
                                                 gt_bboxes,
                                                 stride[i],
                                                 num_base_anchors)
        print(out.shape, out.max(), out.min())


if __name__ == "__main__":
    main()
