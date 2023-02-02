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

from typing import Tuple, List

from codegen.api.types import (BaseCType, TupleCType, OptionalCType, BaseCppType)

from codegen.api.signature import DispatcherSignature, NativeSignature
from codegen.model import SchemaKind, NativeFunction
from codegen.api.native import native_arguments

backend = None

def transfer_args_of_wrapper_func_to_cpu(sig: DispatcherSignature, func: NativeFunction) -> Tuple[str, List[str]]:
    convert: str = f"// Convert args to cpu in order to use at::native kernel \n  " \
                   f"static auto warn_once = [](){{ \n      " \
                   f"std::cout << \"Warning: kernel [{sig.func.name}] is not supported by NPU currently. " \
                   f"Now this kernel is running on CPU.\" << std::endl; \n      " \
                   f"return true; \n  " \
                   f"}}();  \n  "
    args_names: List[str] = []
    args = native_arguments(sig.func, func.use_c10_dispatcher)
    for arg in args:
        arg_type = str(arg.type)
        if 'Tensor' not in arg_type:
            args_names.append(f"{arg.nctype.name}")
            continue
        cpu_arg_name = f"{arg.nctype.name}_cpu"
        args_names.append(cpu_arg_name)
        if arg_type == 'at::Tensor &':
            convert += f"auto {cpu_arg_name} = {arg.nctype.name}.cpu();\n  "
        elif arg_type == 'const at::Tensor &':
            convert += f"const auto {cpu_arg_name} = {arg.nctype.name}.cpu();\n  "
        elif arg_type == 'const c10::optional<at::Tensor> &':
            convert += f"auto {cpu_arg_name} = (*{arg.nctype.name}).cpu();\n  "
        elif arg_type == 'at::TensorList':
            convert += f"""\
::std::vector<at::Tensor> {cpu_arg_name}({arg.nctype.name}.size());
  ::std::transform({arg.nctype.name}.begin(), {arg.nctype.name}.end(), {cpu_arg_name}.rbegin(),
                   [](const Tensor & temp) {{ return temp.cpu(); }});
"""
        else:
            assert False, f'Do not support cur type {arg.type}'

    return convert, args_names


def transfer_ret_of_wrapper_func_to_xla(sig: DispatcherSignature, func_call: str) -> str:
    ret_code = ''
    if sig.func.kind() == SchemaKind.functional:
        if sig.returns_type().cpp_type() == 'at::Tensor':
            ret_code = f"return {func_call}.toBackend(Backend::{backend});"
        elif sig.returns_type().cpp_type() == '::std::vector<at::Tensor>':
            ret_code += f"""\
auto cpu_ret = {func_call};
  ::std::vector<at::Tensor> ret_xla(cpu_ret.size());
  ::std::transform(cpu_ret.begin(), cpu_ret.end(), ret_xla.rbegin(),
                 [](const Tensor & temp) {{return temp.toBackend(Backend::{backend}); }});
  return ret_xla;
            """
        elif type(sig.returns_type()) == BaseCType:
            ret_code = f"return {func_call};"
        elif type(sig.returns_type()) == TupleCType:
            ret_code += f"auto cpu_ret = {func_call}; \n  "
            tuple_ele_names: List[str] = []
            for i, e in enumerate(sig.returns_type().elems):
                assert e.cpp_type() == 'at::Tensor' or type(e) == BaseCppType, f'do not support cur type {e.cpp_type()}'
                if str(e.type) == 'at::Tensor':
                    ret_code += f"auto xla_tuple_ele_{i} = ::std::get<{i}>(cpu_ret).toBackend(Backend::{backend}); \n  "
                    tuple_ele_names.append(f"xla_tuple_ele_{i}")
                else:
                    ret_code += f"const auto & tuple_ele_{i} = ::std::get<{i}>(cpu_ret); \n  "
                    tuple_ele_names.append(f"tuple_ele_{i}")
            tuple_ele_names_str = ','.join(_ for _ in tuple_ele_names)
            ret_code += f"return ::std::make_tuple({tuple_ele_names_str});"
        else:
            assert False, f'Do not support cur type {sig.returns_type()}'

    elif sig.func.kind() == SchemaKind.out:
        out_names = [_.name for _ in sig.func.arguments.out]
        ret_code = f"{func_call};\n  "
        for out_name in out_names:
            ret_code += f"{out_name}.copy_({out_name}_cpu);\n  "
        if sig.returns_type().cpp_type() == 'at::Tensor &':
            ret_code += f"return {out_names[0]};"
        elif type(sig.returns_type()) == TupleCType:
            return_types: List[str] = []
            for i, e in enumerate(sig.returns_type().elems):
                assert e.cpp_type() == 'at::Tensor &', f'Do not support cur type {e.cpp_type()}'
                return_types.append(e.cpp_type())
            tuple_args_str = ','.join(_ for _ in out_names)
            return_type_str = ','.join(_ for _ in return_types)
            ret_code += f"::std::tuple<{return_type_str}> ret_xla({tuple_args_str});\n  " \
                        f"return ret_xla;"
        else:
            assert False, f'Do not support cur type {sig.returns_type()}'

    elif sig.func.kind() == SchemaKind.inplace:
        ret_code = f"{func_call};\n  "
        self_arg_name = sig.func.arguments.self_arg.argument.name

        if sig.returns_type().cpp_type() == 'at::Tensor &':
            ret_code += f"{self_arg_name}.copy_({self_arg_name}_cpu);\n  "
            ret_code += f"return {self_arg_name};"
        elif sig.returns_type().cpp_type() == 'void':
            if str(sig.func.arguments.self_arg.argument.type) == 'Tensor[]':
                ret_code += f"""\n \
  for (int i = 0; i < {self_arg_name}.size(); i++) {{
   {self_arg_name}[i].copy_({self_arg_name}_cpu[i]);
  }}\n
"""
                ret_code += f"  return;"
            else:
                ret_code += f"{self_arg_name}.copy_({self_arg_name}_cpu);\n  "
                ret_code += f"return;"
        else:
            assert False, f'Do not support cur type {sig.returns_type()}'
    else:
        assert False, f'Do not support cur func type {sig.func.kind()}'

    return ret_code