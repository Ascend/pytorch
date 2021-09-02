# Copyright (c) 2020, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

def box_dtype_check(box):
    if box not in [torch.float, torch.half]:
        return box.float()


def npu_ptiou(boxes1, boxes2):
    """ Applies an NPU based IOU operation.

    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Compute Function: insect_area / (union_area + 0.001)

    .. note::
        This function is commonly used when bbox and anchor match.
        Until now, this function has no corresponding backward operator,
        so it cannot be used in IOU_Loss.

    Args:
        boxes1(N,4),boxes2(M,4): two `Boxes`. Contains N & M boxes, respectively. Support dtype: float, half.

    Returns:
        Tensor: IoU, sized [N,M].
    """

    boxes1 = box_dtype_check(boxes1)
    boxes2 = box_dtype_check(boxes2)

    out = torch.npu_ptiou(boxes2, boxes1)
    return out


def npu_iou(boxes1, boxes2):
    """ Applies an NPU based IOU operation.

    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Compute Function: (insect_area + 0.001) / (union_area + 0.001)

    .. note::
        This function is commonly used when bbox and anchor match.
        Until now, this function has no corresponding backward operator,
        so it cannot be used in IOU_Loss.

    Args:
        boxes1(N,4),boxes2(M,4): two `Boxes`. Contains N & M boxes, respectively. Support dtype: float, half.

    Returns:
        Tensor: IoU, sized [N,M].
    """

    boxes1 = box_dtype_check(boxes1)
    boxes2 = box_dtype_check(boxes2)

    out = torch.npu_iou(boxes2, boxes1)
    return out


if __name__ == "__main__":
    torch.npu.set_device(0)

    boxes1 = torch.FloatTensor([[10,55,85,160]])
    boxes2 = torch.FloatTensor([[18,45,80,130], [38,85,70,230]])
    boxes1 = boxes1.float().npu()
    boxes2 = boxes2.float().npu()
    iou1 = npu_iou(boxes1, boxes2)
    iou2 = npu_ptiou(boxes1, boxes2)
    print(iou1.shape, iou1.max(), iou1.min())
    print(iou2.shape, iou2.max(), iou2.min())


    N = 32
    M = 32 * 32

    boxes1 = torch.randint(0, 256, size=(N, 4))
    boxes2 = torch.randint(0, 256, size=(M, 4))
    boxes1 = boxes1.float().npu()
    boxes2 = boxes2.float().npu()
    iou1 = npu_iou(boxes1, boxes2)
    iou2 = npu_ptiou(boxes1, boxes2)
    print(iou1.shape, iou1.max(), iou1.min())
    print(iou2.shape, iou2.max(), iou2.min())

