import torch
import torch_npu

def box_dtype_check(box):
    if box not in [torch.float, torch.half]:
        return box.float()
    return box


def stride_dtype_check(stride):
    if stride not in [torch.int]:
        return stride.int()
    return stride


def npu_bbox_coder_encode_yolo(bboxes, gt_bboxes, stride):
    """Using NPU OP to Get box regression transformation deltas
    that can be used to transform the ``bboxes`` into the ``gt_bboxes``.

    Reference implementation link:
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/yolo_bbox_coder.py#L26

    .. note::
        Does not support dynamic shape, because of the semantics of operators, only supports 
        2-dimensional (n, 4) scenes, bboxes and gt_bboxes only support the same shape and the 
        same dtype, dtype only supports f16 and fp32, The third input (stride) only supports 
        1D and the first dimension is the same as the first input(bboxes).

    Args:
        bboxes (torch.Tensor): Source boxes, e.g., anchors. Support dtype: float, half.
        gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
            ground-truth boxes. Support dtype: float, half.
        stride (torch.Tensor): Stride of bboxes. Only IntTensor is supported.

    Returns:
        torch.Tensor: Box transformation deltas
    """

    assert bboxes.size(0) == gt_bboxes.size(0)
    assert bboxes.size(-1) == gt_bboxes.size(-1) == 4

    bboxes = box_dtype_check(bboxes)
    gt_bboxes = box_dtype_check(gt_bboxes)
    stride = stride_dtype_check(stride)

    # Explanation of parameter performance_mode in npu_yolo_boxes_encode:
    # The mode parameter is recommended to be set to false.
    # When set to true, the speed will increase, but the accuracy may decrease
    output_tensor = torch_npu.npu_yolo_boxes_encode(bboxes,
                                                gt_bboxes,
                                                stride,
                                                performance_mode=False)
    return output_tensor


def npu_bbox_coder_encode_xyxy2xywh(bboxes,
                                    gt_bboxes,
                                    means=None,
                                    stds=None,
                                    is_normalized=False,
                                    normalized_scale=10000.,
                                    ):
    """ Applies an NPU based bboxes's format-encode operation from xyxy to xywh.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`.

    Reference implementation link:
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L98

    .. note::
        Because this interface on the NPU is provided for conventional coordinate values,
        if the coordinate values have been regularized,
        they need to be restored to the conventional coordinate values.

    Args:
        bboxes (Tensor): Boxes to be transformed, shape (N, 4). Support dtype: float, half.
        gt_bboxes (Tensor): Gt bboxes to be used as base, shape (N, 4). Support dtype: float, half.
        means (List[float]): Denormalizing means of target for delta coordinates.
        stds (List[float]): Denormalizing standard deviation of target for delta coordinates.
        is_normalized (Bool): Whether the value of coordinates has been normalized.
        normalized_scale (Float): Sets the normalization scale for restoring coordinates.

    Returns:
        torch.Tensor: Box transformation deltas
    """

    if means is None:
        means = [0., 0., 0., 0.]

    if stds is None:
        stds = [1., 1., 1., 1.]

    assert bboxes.size(0) == gt_bboxes.size(0)
    assert bboxes.size(-1) == gt_bboxes.size(-1) == 4

    bboxes = box_dtype_check(bboxes)
    gt_bboxes = box_dtype_check(gt_bboxes)

    if is_normalized:
        bboxes = bboxes * normalized_scale
        gt_bboxes = gt_bboxes * normalized_scale

    bboxes_encoded = torch_npu.npu_bounding_box_encode(
        bboxes, gt_bboxes, means[0], means[1], means[2],
        means[3], stds[0], stds[1], stds[2], stds[3])

    return bboxes_encoded


def npu_bbox_coder_decode_xywh2xyxy(bboxes,
                                    pred_bboxes,
                                    means=None,
                                    stds=None,
                                    max_shape=None,
                                    wh_ratio_clip=16 / 1000,
                                    ):
    """ Applies an NPU based bboxes's format-encode operation from xywh to xyxy.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`.

    Reference implementation link:
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L144

    .. note::
        Supports dynamic shape, because of the semantics of operators, only supports 2D (n,4) scenes, 
        max_shape must pass 2 numbers, dtype only supports f16 and fp32, the dtype of the first input 
        and the second input should be the same.

    Args:
        anchors (torch.Tensor): Basic boxes, shape (N, 4). Support dtype: float, half.
        pred_bboxes (torch.Tensor): Encoded boxes with shape, shape (N, 4). Support dtype: float, half.
        means (List[float]): Denormalizing means of target for delta coordinates.
            This parameter needs to be aligned with the encoding parameter.
        stds (List[float]): Denormalizing standard deviation of target for delta coordinates.
            This parameter needs to be aligned with the encoding parameter.
        max_shape (tuple[int], optional): Maximum shape of boxes specifies (H, W).
            This parameter generally corresponds to the size of the real picture where bbox is located.
            Defaults to [9999, 9999] as not limited.
        wh_ratio_clip (float, optional): The allowed ratio between width and height.

    Returns:
        Tensor: Boxes with shape (N, 4), where 4 represent tl_x, tl_y, br_x, br_y.
    """

    if means is None:
        means = [0., 0., 0., 0.]
    
    if stds is None:
        stds = [1., 1., 1., 1.]

    if max_shape is None:
        max_shape = [9999, 9999]

    assert bboxes.size(0) == pred_bboxes.size(0)
    assert bboxes.size(-1) == pred_bboxes.size(-1) == 4

    bboxes = box_dtype_check(bboxes)
    pred_bboxes = box_dtype_check(pred_bboxes)

    bboxes_decoded = torch_npu.npu_bounding_box_decode(
        bboxes, pred_bboxes,
        means[0], means[1], means[2], means[3], stds[0], stds[1], stds[2], stds[3],
        max_shape, wh_ratio_clip
    )

    return bboxes_decoded


def _npu_bbox_coder_encode_yolo():
    A = 1024
    bboxes = torch.randint(0, 512, size=(A, 4))
    gt_bboxes = torch.randint(0, 512, size=(A, 4))
    stride = torch.randint(0, 32, size=(A,))

    torch.npu.set_device(0)
    bboxes = bboxes.npu()
    gt_bboxes = gt_bboxes.npu()
    stride = stride.npu()

    out = npu_bbox_coder_encode_yolo(bboxes, gt_bboxes, stride)

    torch.npu.synchronize()
    print('_npu_bbox_coder_encode_yolo done. output shape is ', out.shape)


def _npu_bbox_coder_encode_xyxy2xywh():
    A = 1024
    bboxes = torch.randint(0, 512, size=(A, 4))
    gt_bboxes = torch.randint(0, 512, size=(A, 4))

    torch.npu.set_device(0)
    bboxes = bboxes.npu()
    gt_bboxes = gt_bboxes.npu()

    out = npu_bbox_coder_encode_xyxy2xywh(bboxes, gt_bboxes)
    out = npu_bbox_coder_encode_xyxy2xywh(bboxes/512., gt_bboxes/512., is_normalized=True, normalized_scale=512.)
    torch.npu.synchronize()
    print('_npu_bbox_coder_encode_xyxy2xywh done. output shape is ', out.shape)


def _npu_bbox_coder_decode_xywh2xyxy():
    A = 1024
    max_shape = 512
    bboxes = torch.randint(0, max_shape, size=(A, 4))
    pred_bboxes = torch.randn(A, 4)

    torch.npu.set_device(0)
    bboxes = bboxes.npu()
    pred_bboxes = pred_bboxes.npu()

    out = npu_bbox_coder_decode_xywh2xyxy(bboxes, pred_bboxes, max_shape=(max_shape, max_shape))
    torch.npu.synchronize()
    print('_npu_bbox_coder_decode_xywh2xyxy done. output shape is ', out.shape)


if __name__ == "__main__":
    _npu_bbox_coder_encode_yolo()
    _npu_bbox_coder_encode_xyxy2xywh()
    _npu_bbox_coder_decode_xywh2xyxy()
