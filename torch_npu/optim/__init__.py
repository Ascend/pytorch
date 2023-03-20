# Copyright (c) 2023 Huawei Technologies Co., Ltd
# All rights reserved.
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

__all__ = [
    "NpuFusedOptimizerBase", "NpuFusedSGD", "NpuFusedAdadelta", "NpuFusedLamb",
    "NpuFusedAdam", "NpuFusedAdamW", "NpuFusedAdamP",
    "NpuFusedBertAdam", "NpuFusedRMSprop", "NpuFusedRMSpropTF",
]

from .npu_fused_optim_base import NpuFusedOptimizerBase
from .npu_fused_sgd import NpuFusedSGD
from .npu_fused_adadelta import NpuFusedAdadelta
from .npu_fused_lamb import NpuFusedLamb
from .npu_fused_adam import NpuFusedAdam
from .npu_fused_adamw import NpuFusedAdamW
from .npu_fused_adamp import NpuFusedAdamP
from .npu_fused_bert_adam import NpuFusedBertAdam
from .npu_fused_rmsprop import NpuFusedRMSprop
from .npu_fused_rmsprop_tf import NpuFusedRMSpropTF
