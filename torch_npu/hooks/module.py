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

from .hooks import warp_acc_cmp_hook, set_dump_path


class HOOKModule(nn.Module):

    def __init__(self) -> None:
        super(HOOKModule, self).__init__()
        prefix = ""
        if hasattr(self, "prefix_op_name_"):
            prefix = self.prefix_op_name_
 
        self.register_forward_hook(warp_acc_cmp_hook(prefix + "forward"))
        self.register_backward_hook(warp_acc_cmp_hook(prefix + "backward"))


def register_acc_cmp_hook(model, dump_path=None):
    assert hasattr(model, "named_modules"), "Please register hooks to nn.Module."
    set_dump_path(dump_path)
    for _, module in model.named_modules():
        if not hasattr(module, "named_modules") or len(list(module.named_modules())) > 1:
            continue

        prefix = "Module_" + module.__class__.__name__ + "_"
        if hasattr(module, "prefix_op_name_"):
            prefix = module.prefix_op_name_

        module.register_forward_hook(warp_acc_cmp_hook(prefix + "forward"))
        module.register_backward_hook(warp_acc_cmp_hook(prefix + "backward"))
