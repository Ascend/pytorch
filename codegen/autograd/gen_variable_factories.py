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

# Generates C++ functions that wrap ATen tensor factory methods to turn them into Variables.
#
# This writes one file: variable_factories.h

import re
from typing import Optional, List

import codegen.api.python as python
from codegen.api.signature import CppSignatureGroup
from codegen.api import cpp
from codegen.gen_backend_stubs import parse_native_and_custom_yaml
from codegen.context import with_native_function
from codegen.gen import FileManager
from codegen.utils import map_maybe
from codegen.model import NativeFunction, TensorOptionsArguments, Variant
from codegen.gen_python_functions import NPU_AUTOGRAD_FUNCTION

OPTIONAL_TYPE_PATTERN = re.compile(r"c10::optional<(.+)>")
TYPE_PATTERN = re.compile(r"(?:const\s+)?([A-Z]\w+)")


# Add 'at::' to types defined in ATen namespace, e.g. Tensor, TensorList, IntArrayRef and etc.
# TODO: maybe update the cpp argument API to take optional namespace argument?
def fully_qualified_type(argument_type: str) -> str:
    def maybe_optional_type(input_type: str, is_opt: bool) -> str:
        return f'c10::optional<{input_type}>' if is_opt else input_type

    opt_match = OPTIONAL_TYPE_PATTERN.match(argument_type)
    is_opt = opt_match is not None
    if opt_match:
        argument_type = argument_type[opt_match.start(1):opt_match.end(1)]
    match = TYPE_PATTERN.match(argument_type)
    if match is None:
        return maybe_optional_type(argument_type, is_opt)
    index = match.start(1)
    qualified_type = f'{argument_type[:index]}at::{argument_type[index:]}'
    return maybe_optional_type(qualified_type, is_opt)


def gen_variable_factories(
    out: str, 
    native_yaml_path: str, 
    npu_native_yaml_path: str, 
    template_path: str
) -> None:
    native_functions = parse_native_and_custom_yaml(native_yaml_path, 
                                                    npu_native_yaml_path).native_functions
    factory_functions = [fn for fn in native_functions if 
                         (is_factory_function(fn) and fn.func.name.name.base in NPU_AUTOGRAD_FUNCTION)]
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_with_template('variable_factories.h', 'variable_factories.h', lambda: {
        'generated_comment': '@' + f'generated from {fm.template_dir}/variable_factories.h',
        'ops_headers': [f'#include <ATen/ops/{fn.root_name}.h>' for fn in factory_functions],
        'function_definitions': list(map_maybe(process_function, factory_functions)),
    })


@with_native_function
def is_factory_function(f: NativeFunction) -> bool:
    if Variant.function not in f.variants:
        return False

    name = cpp.name(f.func)
    has_tensor_options = python.has_tensor_options(f)
    return has_tensor_options or name.endswith("_like")


@with_native_function
def process_function(f: NativeFunction) -> Optional[str]:
    name = cpp.name(f.func)
    has_tensor_options = python.has_tensor_options(f)
    is_factory = has_tensor_options or name.endswith("_like")

    if Variant.function not in f.variants or not is_factory:
        return None

    sig = CppSignatureGroup.from_native_function(f, method=False).signature
    formals: List[str] = []
    exprs: List[str] = []
    requires_grad = 'false'
    for arg in sig.arguments():
        qualified_type = fully_qualified_type(arg.type)
        if arg.default:
            formals.append(f'{qualified_type} {arg.name} = {arg.default}')
        else:
            formals.append(f'{qualified_type} {arg.name}')

        if isinstance(arg.argument, TensorOptionsArguments):
            # note: we remove the requires_grad setting from the TensorOptions because
            # it is ignored anyways (and we actually have an assertion that it isn't set
            # which would fail otherwise). We handle requires_grad explicitly here
            # instead of passing it through to the kernel.
            exprs.append(f'at::TensorOptions({arg.name}).requires_grad(c10::nullopt)')
            # Manually set the requires_grad bit on the result tensor.
            requires_grad = f'{arg.name}.requires_grad()'
        else:
            exprs.append(arg.name)

    return f"""\
inline at::Tensor {name}({', '.join(formals)}) {{
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::{name}({', '.join(exprs)}), /*requires_grad=*/{requires_grad});
}}
"""
