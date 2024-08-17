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

from .function import npu_iou, npu_ptiou, npu_giou, npu_diou, npu_ciou, npu_multiclass_nms, \
     npu_batched_multiclass_nms, npu_single_level_responsible_flags, npu_fast_condition_index_put, \
     npu_bbox_coder_encode_yolo, npu_bbox_coder_encode_xyxy2xywh, npu_bbox_coder_decode_xywh2xyxy, \
     roll, matmul_transpose, npu_fused_attention_with_layernorm, npu_fused_attention
from .module import ChannelShuffle, Prefetcher, LabelSmoothingCrossEntropy, ROIAlign, DCNv2, \
     ModulatedDeformConv, Mish, BiLSTM, PSROIPool, SiLU, Swish, NpuFairseqDropout, NpuCachedDropout, \
     MultiheadAttention, FusedColorJitter, NpuDropPath, Focus, FastBatchNorm1d, FastBatchNorm2d, \
     FastBatchNorm3d, FastSyncBatchNorm, LinearWeightQuant, LinearA8W8Quant, LinearQuant

__all__ = [
    # from function
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
    "npu_fused_attention_with_layernorm",
    "npu_fused_attention",

    # from module
    "ChannelShuffle",
    "Prefetcher",
    "LabelSmoothingCrossEntropy",
    "ROIAlign",
    "DCNv2",
    "ModulatedDeformConv",
    "Mish",
    "BiLSTM",
    "PSROIPool",
    "SiLU",
    "Swish",
    "NpuFairseqDropout",
    "NpuCachedDropout",
    "MultiheadAttention",
    "NpuDropPath",
    "Focus",
    "LinearWeightQuant",
    "LinearA8W8Quant",
    "LinearQuant",
]
