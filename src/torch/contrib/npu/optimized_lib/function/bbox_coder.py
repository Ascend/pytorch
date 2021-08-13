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

def stride_dtype_check(stride):
    if stride not in [torch.int]:
        return stride.int()

def npu_yolo_bbox_coder(bboxes, gt_bboxes, stride):
    """Using NPU OP to Get box regression transformation deltas
    that can be used to transform the ``bboxes`` into the ``gt_bboxes``.

    Reference implementation link:
    https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/yolo_bbox_coder.py#L26

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
    output_tensor = torch.npu_yolo_boxes_encode(bboxes,
                                                gt_bboxes,
                                                stride,
                                                performance_mode=False)
    return output_tensor


if __name__ == "__main__":
    A = 1024
    bboxes = torch.randint(0, 512, size=(A, 4))
    gt_bboxes = torch.randint(0, 512, size=(A, 4))
    stride = torch.randint(0, 32, size=(A,))

    torch.npu.set_device(0)
    bboxes = bboxes.npu()
    gt_bboxes = gt_bboxes.npu()
    stride = stride.npu()

    out = npu_yolo_bbox_coder(bboxes, gt_bboxes, stride)
    print(out.shape, out)
