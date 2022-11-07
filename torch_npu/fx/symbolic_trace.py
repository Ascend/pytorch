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


from typing import Any, Dict, Optional, Callable, Union

import torch
from torch.fx import Tracer, GraphModule


class NpuTracer(Tracer):
    def is_leaf_module(self, m : torch.nn.Module, qualname : str):
        if m.__module__.startswith('torch_npu.contrib.module'):
            return True
        return super().is_leaf_module(m, qualname)


def symbolic_trace(root : Union[torch.nn.Module, Callable[..., Any]], concrete_args: Optional[Dict[str, Any]] = None,
                   enable_cpatching: bool = False) -> GraphModule:
    """
    Symbolic tracing API
 
    Given an ``nn.Module`` or function instance ``root``, this function will return a ``GraphModule``
    constructed by recording operations seen while tracing through ``root``.
 
    ...
    """
    tracer = NpuTracer()
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return GraphModule(tracer.root, graph, name)
