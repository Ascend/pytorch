import torch
import torch_npu


def npu_multiclass_nms(multi_bboxes,
                       multi_scores,
                       score_thr=0.05,
                       nms_thr=0.45,
                       max_num=50,
                       score_factors=None):
    """NMS for multi-class bboxes using npu api.

    This interface is similar to the original interface, but not exactly the same.

    .. note::
        In the case of dynamic shape, because of the limitation of NPU op, only supports the maximum
        number of categories(nmsed_classes) is 20, and the maximum number of boxes(nmsed_boxes) is 10000.

    Args:
        multi_bboxes (Tensor): shape (n, #class, 4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
            On npu, in order to keep the semantics unblocked, we will unify the dimensions
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold. In the original implementation, a dictionary of {"iou_threshold": 0.45}
            was passed, which is simplified here.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept; if there are less than max_num bboxes after NMS,
            the output will zero pad to max_num. On the NPU, the memory needs to be requested in advance,
            so the current max_num cannot be set to -1 at present
        score_factors (Tensor): The factors multiplied to scores before applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels are 0-based.
    """

    num_classes = multi_scores.size(1) - 1
    num_boxes = multi_scores.size(0)
    if score_factors is not None:
        multi_scores = multi_scores[:, :-1] * score_factors[:, None]
    else:
        multi_scores = multi_scores[:, :-1]
    multi_bboxes = multi_bboxes.reshape(1, num_boxes, multi_bboxes.numel() // 4 // num_boxes, 4)
    multi_scores = multi_scores.reshape(1, num_boxes, num_classes)

    nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(
        multi_bboxes.half(), 
        multi_scores.half(), 
        score_thr, 
        nms_thr, 
        max_num, 
        max_num
    )

    nmsed_boxes = nmsed_boxes.reshape(nmsed_boxes.shape[1:])
    nmsed_scores = nmsed_scores.reshape(nmsed_scores.shape[1])
    nmsed_classes = nmsed_classes.reshape(nmsed_classes.shape[1])

    return torch.cat([nmsed_boxes, nmsed_scores[:, None]], -1), nmsed_classes


def npu_batched_multiclass_nms(
        multi_bboxes,
        multi_scores,
        score_thr=0.05,
        nms_thr=0.45,
        max_num=50,
        score_factors=None):
    """NMS for batched multi-class bboxes using npu api.

    This interface is similar to the original interface, but not exactly the same.
    This interface implements the nms method under batch.

    .. note::
        In the case of dynamic shape, because of the limitation of NPU op, only supports the maximum
        number of categories(nmsed_classes) is 20, and the maximum number of boxes(nmsed_boxes) is 10000.

    Args:
        multi_bboxes (Tensor): shape (bs, n, #class, 4) or (bs, n, 4)
        multi_scores (Tensor): shape (bs, n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
            On npu, in order to keep the semantics unblocked, we will unify the dimensions
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold. In the original implementation, a dictionary of {"iou_threshold": 0.45}
            was passed, which is simplified here.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept; if there are less than max_num bboxes after NMS,
            the output will zero pad to max_num. On the NPU, the memory needs to be requested in advance,
            so the current max_num cannot be set to -1 at present
        score_factors (Tensor): The factors multiplied to scores before applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (bs, k, 5) and (bs, k, 1). Labels are 0-based.
    """

    num_classes = multi_scores.size(2) - 1
    num_boxes = multi_scores.size(1)
    batch_size = multi_scores.size(0)
    if score_factors is not None:
        multi_scores = multi_scores[..., :-1] * score_factors[..., None]
    else:
        multi_scores = multi_scores[..., :-1]
    multi_bboxes = multi_bboxes.reshape(batch_size, num_boxes, multi_bboxes.numel() // 4 // num_boxes // batch_size, 4)
    multi_scores = multi_scores.reshape(batch_size, num_boxes, num_classes)

    nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(
        multi_bboxes.half(), 
        multi_scores.half(), 
        score_thr, 
        nms_thr, 
        max_num, 
        max_num
    )

    return torch.cat([nmsed_boxes, nmsed_scores[..., None]], -1), nmsed_classes
