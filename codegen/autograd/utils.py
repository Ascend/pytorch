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
import yaml
from pathlib import Path
from typing import List, Sequence

from codegen.api import cpp
from codegen.api.autograd import (
    match_differentiability_info, NativeFunctionWithDifferentiabilityInfo,
    DifferentiabilityInfo)
from codegen.utils import CUSTOM_YAML_NAME, enable_opplugin
from codegen.model import NativeFunction, SchemaKind
from codegen.gen_backend_stubs import parse_native_and_custom_yaml
from .load_derivatives import load_derivatives


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

AUTOGRAD_BLACK_LIST = {'npu_format_cast.Tensor', 'npu_format_cast_', 'npu_format_cast_.acl_format'}

def parse_derivatives(
    native_functions_path: str,
    autograd_dir: str,
    npu_native_functions_path: str
) :
    derivatives_file_path =  ('third_party/op-plugin/op_plugin/config/v1r11/derivatives.yaml'
        if enable_opplugin()
        else "codegen/autograd/derivatives.yaml")
        
    derivatives_path = \
    str(Path(autograd_dir).parents[1].joinpath(derivatives_file_path))
    differentiability_infos = load_derivatives(
        derivatives_path, native_functions_path, npu_native_functions_path)
    native_funcs = parse_native_and_custom_yaml(native_functions_path,npu_native_functions_path).native_functions
    funcs = filte_out_native_autograd_function(native_funcs, differentiability_infos)
    funcs_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo] = []
    funcs_with_diff_infos = match_differentiability_info(funcs, differentiability_infos)

    filt_funcs_with_diff_infos = [f for f in funcs_with_diff_infos if str(f.func.func.name) not in AUTOGRAD_BLACK_LIST]

    return (differentiability_infos, native_funcs, filt_funcs_with_diff_infos)


def filt_npu_autograd_functions(
    native_functions_path: str,
    funcs_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo]
) :
    npu_funcs_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo] = []
    torch_funcs_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo] = []
    torch_functions = set()
    with open(native_functions_path, 'r') as f:
        es = yaml.safe_load(f)
    for e in es:
        torch_functions.add(e.get('func').split('(')[0])

    npu_autograd_functions = set()
    for f in funcs_with_diff_infos:
        name = str(f.func.func.name)
        # f.info is differentiabilityinfo. Existence of variants ops with a differentiabilityinfo of none.
        if not f.info:
            continue
        if name in torch_functions:
            torch_funcs_with_diff_infos.append(f)
        else:
            npu_funcs_with_diff_infos.append(f)
            npu_autograd_functions.add(name)

    return torch_funcs_with_diff_infos, npu_funcs_with_diff_infos, npu_autograd_functions


def filte_out_native_autograd_function(
    native_funcs: List[NativeFunction],
    differentiability_infos: Sequence[DifferentiabilityInfo],
):
    result: List[NativeFunction] = []
    derivatives_name_list: List[str] = []
    
    for info in differentiability_infos:
        derivatives_name_list.append(str(info.func.func.name))
    for funcs in native_funcs:
        func_name = str(funcs.func.name)
        func_base_name = str(funcs.func.name.name.base)
        if (func_name in derivatives_name_list) or (func_base_name in derivatives_name_list):
            result.append(funcs)
    return result


NPU_AUTOGRAD_FUNCTION = filt_npu_autograd_functions(
    str(Path(__file__).parents[2].joinpath('codegen/native_functions.yaml')),
    parse_derivatives(
    str(Path(__file__).parents[2].joinpath('codegen/native_functions.yaml')),
    str(Path(__file__).parent),
    str(Path(__file__).parents[2].joinpath(f'torch_npu/csrc/aten/{CUSTOM_YAML_NAME}')))[-1]
)[-1]


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
