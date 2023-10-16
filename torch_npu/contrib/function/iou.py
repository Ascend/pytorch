import torch
import torch_npu


def box_dtype_check(box):
    if box not in [torch.float, torch.half]:
        return box.float()
    return box


def npu_iou(boxes1,
            boxes2,
            mode="ptiou",
            is_normalized=False,
            normalized_scale=100.
            ):
    """ Applies an NPU based IOU operation.

    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between all N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Compute Function:
    iou = (overlap_area + 0.001) / (union_area + 0.001)
    ptiou = overlap_area / (union_area + 0.001)

    .. note::
        This function is commonly used when bbox and anchor match.
        Until now, this function has no corresponding backward operator,
        so it cannot be used in IOU_Loss.

        Since 0.001 is added to the denominator in the calculation formula to avoid dividing by 0,
        when the input boxes are normalized data, the component of 0.001 will be too heavy.
        At this time, it is necessary to enlarge the input value to avoid excessive influence of 0.001.

    Examples::
    >>> box1 = torch.randint(0, 256, size=(32, 4))
    >>> box2 = torch.randint(0, 256, size=(16, 4))
    >>> iou1 = npu_iou(box1, box2) # (32, 16)

    Args:
        boxes1(N,4),boxes2(M,4): two `Boxes`. Contains N & M boxes, respectively. Support dtype: float, half.
        mode (String): Select the calculation mode of iou. Default ptiou.
        is_normalized (Bool): Whether the value of coordinates has been normalized. Default False.
        normalized_scale (Float): Sets the normalization scale for restoring coordinates. Default 100.

    Returns:
        Tensor: IoU, sized [N,M].
    """

    if mode not in ["iou", "ptiou"]:
        raise ValueError("Expected mode in [iou, ptiou]")

    boxes1 = box_dtype_check(boxes1)
    boxes2 = box_dtype_check(boxes2)

    if is_normalized:
        boxes1 = boxes1 * normalized_scale
        boxes2 = boxes2 * normalized_scale

    if mode == "iou":
        out = torch_npu.npu_iou(boxes2, boxes1)
    elif mode == "ptiou":
        out = torch_npu.npu_ptiou(boxes2, boxes1)

    return out


npu_ptiou = npu_iou


def npu_giou(boxes1,
             boxes2,
             is_permuted=True
             ):
    """ Applies an NPU based GIOU operation.

    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between all N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Compute Function:
    iou = overlap_area / union_area
    enclose_area = (max(x2) - min(x1)) * (max(y2) - min(y1))
    giou = iou - (enclose_area - union_area) / enclose_area

    .. note::
        This function is corresponding to a backward operator,
        so it can be used in IOU_Loss.

        Util now, only trans=True(only support xywh, not support xyxy),
        is_cross=False(only support boxes1.shape == boxes2.shape -- One-to-one calculation, not support ((n,4), (m,4)))
        in torch_npu.npu_giou is supported, please don't use other pram.

    Examples::
    >>> box1 = torch.randn(32, 4)
    >>> box1.requires_grad = True
    >>> box2 = torch.randn(32, 4)
    >>> iou1 = npu_giou(box1, box2) # (32, 1)
    >>> l = iou1.sum()
    >>> l.backward()

    Args:
        boxes1 (Tensor): Predicted bboxes of format xywh, shape (n, 4).
        boxes2 (Tensor): Corresponding gt bboxes, shape (n, 4).
        is_permuted (Bool): Whether the value of coordinates has been normalized. Default True.

    Returns:
        Tensor: IoU, sized [n, 1].
    """

    if boxes1.shape != boxes2.shape:
        raise ValueError("Expected boxes1.shape == boxes2.shape")

    boxes1 = box_dtype_check(boxes1)
    boxes2 = box_dtype_check(boxes2)

    if is_permuted:
        boxes1 = boxes1.permute(1, 0)
        boxes2 = boxes2.permute(1, 0)

    out = torch_npu.npu_giou(boxes1, boxes2, trans=True, is_cross=False)

    return out


def npu_diou(boxes1, 
             boxes2, 
             trans=True, 
             is_cross=False, 
             mode=0
             ):
    """ Applies an NPU based DIOU operation.

    Taking into account the distance between the targets, 
    the overlap rate of the distance and the range, different targets or boundaries will tend to be stable.

    Compute Function:
    iou = overlap_area / union_area
    diou = iou - p * p(b,bgt) / c * c
    
    Among them, b and bgt represent the center points of the predicted frame and the real frame, respectively, 
    and ρ represents the Euclidean distance between the two center points. c represents the diagonal distance 
    of the smallest closure region that can contain both the predicted box and the ground-truth box.

    .. note::

        Util now, diou backward only support trans==True, is_cross==False, mode==0('iou') current version if you 
        need to back propagation, please ensure your parameter is correct!
        
    Examples::
    >>> box1 = torch.randn(4, 32)
    >>> box1.requires_grad = True
    >>> box2 = torch.randn(4, 32)
    >>> box2.requires_grad = True
    >>> diou = npu_diou(box1, box2) # (1, 32)
    >>> l = diou.sum()
    >>> l.backward()

    Args:
        boxes1 (Tensor): Predicted bboxes of format xywh, shape (4, n).
        boxes2 (Tensor): Corresponding gt bboxes, shape (4, n).
        trans (Bool): Whether there is an offset
        is_cross (Bool): Whether there is a cross operation between box1 and box2.
        mode (int):  Select the calculation mode of diou.

    Returns:
        Tensor: IoU, sized [1, n].
    """

    out = torch_npu.npu_diou(boxes1, boxes2, trans, is_cross, mode)

    return out


def npu_ciou(boxes1, 
             boxes2,
             trans=True, 
             is_cross=False, 
             mode=0
             ):
    """ Applies an NPU based CIOU operation.

    A penalty item is added on the basis of DIoU, and CIoU is proposed

    Compute Function:
    iou = overlap_area / union_area
    ciou = 1 - iou + p * p(b,bgt) / c * c + αv

    Among them, b and bgt represent the center points of the predicted frame and the real frame, respectively, 
    and ρ represents the Euclidean distance between the two center points. c represents the diagonal distance 
    of the smallest closure region that can contain both the predicted box and the ground-truth box. α is the 
    weight function, v is used to measure the similarity of the aspect ratio.
    
    .. note::

        Util now, ciou backward only support trans==True, is_cross==False, mode==0('iou') current version if you 
        need to back propagation, please ensure your parameter is correct!
        
    Examples::
    >>> box1 = torch.randn(4, 32)
    >>> box1.requires_grad = True
    >>> box2 = torch.randn(4, 32)
    >>> box2.requires_grad = True
    >>> ciou = npu_ciou(box1, box2) # (1, 32)
    >>> l = ciou.sum()
    >>> l.backward()

    Args:
        boxes1 (Tensor): Predicted bboxes of format xywh, shape (4, n).
        boxes2 (Tensor): Corresponding gt bboxes, shape (4, n).
        trans (Bool): Whether there is an offset
        is_cross (Bool): Whether there is a cross operation between box1 and box2.
        mode (int):  Select the calculation mode of diou.
        atan_sub_flag (Bool): whether to pass the second value of the forward to the reverse.

    Returns:
        Tensor: IoU, sized [1, n].

    """

    out = torch_npu.npu_ciou(boxes1, boxes2, trans, is_cross, mode, True)

    return out
