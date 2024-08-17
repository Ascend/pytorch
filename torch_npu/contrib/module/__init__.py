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

from .channel_shuffle import ChannelShuffle
from .prefetcher import Prefetcher
from .crossentropy import LabelSmoothingCrossEntropy
from .roi_align import ROIAlign
from .deform_conv import ModulatedDeformConv, DCNv2
from .activations import Mish, SiLU, Swish
from .bidirectional_lstm import BiLSTM
from .ps_roi_pooling import PSROIPool
from .ensemble_dropout import NpuFairseqDropout, NpuCachedDropout
from .ensemble_dropout import NpuPreGenDropout
from .multihead_attention import MultiheadAttention
from .fusedcolorjitter import FusedColorJitter
from .drop_path import NpuDropPath
from .focus import Focus
from .batchnorm_with_int32_count import FastBatchNorm1d, \
    FastBatchNorm2d, FastBatchNorm3d, FastSyncBatchNorm
from .linear_weight_quant import LinearWeightQuant
from .linear_a8w8_quant import LinearA8W8Quant
from .linear_quant import LinearQuant
from .npu_modules import DropoutWithByteMask

__all__ = [
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
    "NpuPreGenDropout",
    "MultiheadAttention",
    "FusedColorJitter",
    "NpuDropPath",
    "Focus",
    "LinearWeightQuant",
    "LinearA8W8Quant",
    "LinearQuant",
    "DropoutWithByteMask",
]
