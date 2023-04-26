# Copyright (c) 2023 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
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

from codegen.api import cpp
from codegen.model import NativeFunction, SchemaKind

'''
Functions in this file come from gen_trace_type.py, and will be deleted once
we have gen_trace_type.py in torch_npu.
'''

# Note: we still register to dispatch key Profiler for these ops, keeping it untouched for now.
# You can find the manual registration in torch/csrc/autograd/VariableTypeManual.cpp
MANUAL_BACKEND = set([
    'options', 'data', 'set_data', 'is_leaf', 'output_nr', '_version', 'retain_grad',
    '_backward', 'requires_grad_',
])

# For these ops we want to skip the codegen-ed registration to both Autograd and Tracer keys.
# You can find the manual registration in torch/csrc/autograd/VariableTypeManual.cpp
MANUAL_AUTOGRAD_AND_TRACER = set([
    'resize_', 'resize_as_', 'detach', 'detach_', 'copy_', '_fw_primal', '_make_dual',
])


def type_wrapper_name(f: NativeFunction) -> str:
    if f.func.name.overload_name:
        return f'{cpp.name(f.func)}_{f.func.name.overload_name}'
    else:
        return cpp.name(f.func)


def get_return_value(f: NativeFunction) -> str:
    names = cpp.return_names(f)
    if len(f.func.returns) == 1:
        return names[0]
    if f.func.kind() == SchemaKind.out:
        return f'std::forward_as_tuple({", ".join(names)})'
    else:
        moved = ", ".join(f'std::move({name})' for name in names)
        return f'std::make_tuple({moved})'


def tie_return_values(f: NativeFunction) -> str:
    if len(f.func.returns) == 1:
        return f'auto {f.func.returns[0].name or "result"}'
    names = cpp.return_names(f)
    return f'std::tie({", ".join(names)})'
