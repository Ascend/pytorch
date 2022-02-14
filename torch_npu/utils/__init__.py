# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

from .module import LayerNorm, Module


def _get_monkey_patches():
    nn_modules = ["activation", "adaptive", "batchnorm", "channelshuffle", "container",
                  "conv", "distance", "dropout", "flatten", "fold", "instancenorm",
                  "linear", "loss", "module", "normalization", "padding", "pixelshuffle",
                  "pooling", "rnn", "sparse", "transformer", "upsampling"]
    _monkey_patches = []
    for module_name in nn_modules:
        _monkey_patches.append([f"nn.modules.{module_name}.Module", Module])

    _monkey_patches.append(["nn.Module", Module])
    _monkey_patches.append(["nn.modules.Module", Module])
    _monkey_patches.append(["nn.modules.normalization.LayerNorm", LayerNorm])
    _monkey_patches.append(["nn.modules.LayerNorm", LayerNorm])
    _monkey_patches.append(["nn.LayerNorm", LayerNorm])
    return _monkey_patches


nn_monkey_patches = _get_monkey_patches()
