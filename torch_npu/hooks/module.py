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

import functools

import torch
import torch.nn as nn
import torch.utils.hooks as full_hooks


class HOOKModule(nn.Module):

    def __init__(self, hook) -> None:
        super(HOOKModule, self).__init__()
        self.has_overflow = False
        prefix = ""
        if hasattr(self, "prefix_op_name_"):
            prefix = self.prefix_op_name_

        self.register_forward_hook(hook(prefix + "forward"))
        self.register_backward_hook(hook(prefix + "backward"))

    def __call__(self, *input, **kwargs):
        full_backward_hooks, non_full_backward_hooks = [], []
        if len(self._backward_hooks) > 0:
            full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()
        for hook in self._forward_pre_hooks.values():
            result = hook(self, input)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result, )
                input = result
        bw_hook = None
        if len(full_backward_hooks) > 0:
            bw_hook = full_hooks.BackwardHook(self, full_backward_hooks)
            input = bw_hook.setup_input_hook(input)
        if torch._C._get_tracing_state():
            result = self._slow_forward(*input, **kwargs)
        else:
            result = self.forward(*input, **kwargs)
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                result = hook_result
        if bw_hook:
            result = bw_hook.setup_output_hook(result)
        if len(non_full_backward_hooks) > 0:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                elif isinstance(var, (list, tuple)):
                    var = var[0]
                else:
                    return result
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in non_full_backward_hooks:
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
                self._maybe_warn_non_full_backward_hook(input, result, grad_fn)
        return result
