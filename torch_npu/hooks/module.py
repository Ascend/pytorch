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


import torch.nn as nn

from .hooks import warp_acc_cmp_hook


class HOOKModule(nn.Module):

    def __init__(self) -> None:
        super(HOOKModule, self).__init__()
        self.register_forward_hook(warp_acc_cmp_hook("forward"))
        self.register_backward_hook(warp_acc_cmp_hook("backward"))


def register_acc_cmp_hook(model):
    for _, module in model.named_modules():
        if not hasattr(module, "named_modules") or len(list(module.named_modules())) > 1:
            continue

        module.register_forward_hook(warp_acc_cmp_hook("forward"))
        module.register_backward_hook(warp_acc_cmp_hook("backward"))
