# Copyright (c) 2026 Huawei Technologies Co., Ltd
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
import weakref

import torch


original_profile_init = torch.profiler.profile.__init__


@functools.wraps(original_profile_init)
def _custom_profile_init(self, **kwargs):
    cfg = kwargs.get("experimental_config")
    handler = kwargs.get("on_trace_ready")
    npu_handler = getattr(handler, "_npu_handler", False)
    if getattr(cfg, "_npu_config", False) and (npu_handler or cfg._backend == "aclprof"):
        if npu_handler and handler.analyse_flag:
            raise ValueError("ACLprof native capture does not support automatic analysis; set analyse_flag=False")
        if npu_handler and handler._profile_ref is not None:
            raise ValueError("An NPU trace handler belongs to one profiler session")
        if any(kwargs.get(key, False) for key in ("with_stack", "with_modules", "with_flops")):
            raise ValueError("ACLprof native capture does not collect Python stacks, modules or FLOPs")
        overrides = {
            "backend": "aclprof",
            "profile_memory": kwargs.get("profile_memory", False),
            "record_shapes": kwargs.get("record_shapes", False),
        }
        if npu_handler:
            overrides["output_path"] = handler.prof_path
        kwargs["experimental_config"] = cfg.to_custom_profiler_config(overrides)
        kwargs["activities"] = [torch.profiler.ProfilerActivity.PrivateUse1]
    original_profile_init(self, **kwargs)
    if getattr(cfg, "_npu_config", False) and getattr(handler, "_npu_handler", False):
        handler._profile_ref = weakref.ref(self)


def _apply_native_entry_patch():
    torch.profiler.profile.__init__ = _custom_profile_init
