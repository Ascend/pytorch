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

from .iou import npu_iou, npu_ptiou, npu_giou, npu_diou, npu_ciou
from .nms import npu_multiclass_nms, npu_batched_multiclass_nms
from .anchor_generator import npu_single_level_responsible_flags
from .bbox_coder import npu_bbox_coder_encode_yolo, npu_bbox_coder_encode_xyxy2xywh, npu_bbox_coder_decode_xywh2xyxy
from .index_op import npu_fast_condition_index_put
from .fuse_add_softmax_dropout import fuse_add_softmax_dropout
from .roll import roll
from .matmul_transpose import matmul_transpose
from .fused_attention import npu_fused_attention_with_layernorm, npu_fused_attention

__all__ = [
    "npu_iou",
    "npu_ptiou",
    "npu_giou",
    "npu_diou",
    "npu_ciou",
    "npu_multiclass_nms",
    "npu_batched_multiclass_nms",
    "npu_single_level_responsible_flags",
    "npu_fast_condition_index_put",
    "npu_bbox_coder_encode_yolo",
    "npu_bbox_coder_encode_xyxy2xywh",
    "npu_bbox_coder_decode_xywh2xyxy",
    "fuse_add_softmax_dropout",
    "roll",
    "matmul_transpose",
    "npu_fused_attention",
    "npu_fused_attention_with_layernorm",
]
