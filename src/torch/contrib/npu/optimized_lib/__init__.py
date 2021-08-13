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

from .function import npu_iou, npu_ptiou, npu_multiclass_nms, npu_batched_multiclass_nms, \
    npu_single_level_responsible_flags, npu_yolo_bbox_coder
from .module import ChannelShuffle, Prefetcher, DropoutV2, LabelSmoothingCrossEntropy, ROIAlign, DCNv2, \
    ModulatedDeformConv

__all__ = [
    # from function
    "npu_iou",
    "npu_ptiou",
    "npu_multiclass_nms",
    "npu_batched_multiclass_nms",
    "npu_single_level_responsible_flags",
    "npu_yolo_bbox_coder",

    # from module
    "ChannelShuffle",
    "Prefetcher",
    "DropoutV2",
    "LabelSmoothingCrossEntropy",
    "ROIAlign",
    "DCNv2",
    "ModulatedDeformConv",
]
