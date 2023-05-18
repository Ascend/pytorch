# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

from dataclasses import dataclass
from typing import Optional, Union, Sequence, Set, List, Dict, Tuple

from torchgen.api.types import CppSignature, CppSignatureGroup, Binding
from torchgen.api import cpp
from torchgen.model import (Argument, BaseTy, BaseType, ListType,
                            NativeFunction, OptionalType, Return, Type,
                            Variant)
from torchgen.api.python import (argument, PythonArgument, PythonSignature, DispatchLambdaArgument,
                                 DispatchLambdaArgumentExprs, PythonSignatureGroup, PythonSignatureNativeFunctionPair,
                                 arg_parser_output_exprs, dispatch_lambda_return_str, has_tensor_options,
                                 namedtuple_fieldnames, signature, PythonSignatureDeprecated, PythonOutArgument)


def _cpp_signature(f: NativeFunction, *, method: bool = False, faithful: bool = False) -> CppSignature:
    faithful_signature = CppSignatureGroup.from_native_function(f, method=method).faithful_signature
    if faithful and faithful_signature:
        return faithful_signature

    return CppSignatureGroup.from_native_function(f, method=method).signature


def dispatch_lambda_args(ps: PythonSignature,
                         f: NativeFunction,
                         faithful: bool = False) -> Tuple[DispatchLambdaArgument, ...]:
    # Start with cpp arguments - dispatch lambda signature always include 'self'
    cpp_args: Sequence[Binding] = _cpp_signature(f, method=False, faithful=faithful).arguments()

    # Special reorder logic for deprecated python signature
    if isinstance(ps, PythonSignatureDeprecated):
        m: Dict[str, Binding] = dict((a.name, a) for a in cpp_args)
        # reorder according to the deprecated signature
        # ignore 'out' argument when binding to non-output function.
        ordered_args = filter(lambda n: n != 'out' or f.func.is_out_fn(),
                              ps.deprecated_args_names)
        cpp_args = list(map(lambda n: m[n], ordered_args))

    out_args: Set[str] = set(a.name for a in f.func.arguments.out)

    # Convert from cpp argument to lambda argument
    def dispatch_lambda_arg(cpp_arg: Binding) -> DispatchLambdaArgument:
        type_str = cpp_arg.type
        is_out_arg = cpp_arg.name in out_args
        if ps.method and cpp_arg.name == 'self':
            # For method's 'self', we can use 'const Tensor &' and simply ignore mutability!
            type_str = 'const at::Tensor &'
        else:
            # For other cases we need prevent dangling refs to temps (unless it's
            # unpacked scattered output)
            # The reason is explained in the comments above and in 'dispatch_lambda_return_str()'.
            # TODO: avoid this special handling?
            ensure_temp_safe = len(out_args) <= 1 or not is_out_arg
            if ensure_temp_safe:
                type_str = {
                    'at::Tensor &': 'at::Tensor',
                }.get(type_str, type_str)
        return DispatchLambdaArgument(
            name=cpp_arg.name,
            type_str=type_str,
            is_out_arg=is_out_arg,
        )

    return tuple(map(dispatch_lambda_arg, cpp_args))


def cpp_record_func(f: NativeFunction, custom=False) -> str:
    name = cpp.name(f.func)

    if Variant.function in f.variants:
        if custom:
            record_func = f'RECORD_FUNCTION("{name}", std::vector<c10::IValue>({{}}));'
        else:
            record_func = f'// RECORD_FUNCTION("{name}")'
        return record_func
    raise RuntimeError(f'could not dispatch, neither function nor method: {f.func}')

def cpp_dispatch_target(f: NativeFunction, custom=False, is_npu_autograd=False) -> str:
    name = cpp.name(f.func)
    if Variant.method in f.variants and not custom:
        return f'self.{name}'
    if Variant.function in f.variants:
        if custom:
            if is_npu_autograd:
                namespace = 'at_npu::autograd::VariableType'
            else:
                namespace = 'at_npu::native::NPUNativeFunctions'
        elif has_tensor_options(f) or f.func.name.name.base.endswith('_like'):
            namespace = 'torch'
        else:
            namespace = 'at'
        return f'{namespace}::{name}'
    raise RuntimeError(f'could not dispatch, neither function nor method: {f.func}')

def cpp_dispatch_exprs(f: NativeFunction, *,
                       python_signature: Optional[PythonSignature] = None,
                       faithful: bool = False,
                       ) -> Tuple[str, ...]:
    cpp_args: Sequence[Binding] = _cpp_signature(f, method=False, faithful=faithful).arguments()

    exprs: Tuple[str, ...] = tuple()
    if not isinstance(python_signature, PythonSignatureDeprecated):
        # By default the exprs are consistent with the C++ signature.
        exprs = tuple(map(lambda a: a.name, cpp_args))
    else:
        # For deprecated python signature we may need fill in some constants.
        exprs = tuple(filter(lambda n: n != 'out' or f.func.is_out_fn(),
                             python_signature.deprecated_args_exprs))

    if Variant.method in f.variants:
        exprs = tuple(filter('self'.__ne__, exprs))

    return exprs


def dispatch_lambda_exprs(
    ps: PythonSignature, f: NativeFunction, faithful: bool = False
) -> DispatchLambdaArgumentExprs:
    # This method is to bind 'arg_parser_outputs' and 'lambda_args' by producing
    # 'inits' and 'lambda_args_exprs' for each lambda argument using arg parser
    # outputs.
    arg_parser_outputs = arg_parser_output_exprs(ps, f)
    lambda_args = dispatch_lambda_args(ps, f, faithful=faithful)
    inits: List[str] = []
    lambda_args_exprs: Dict[str, str] = dict()

    has_toptions = has_tensor_options(f)

    # 1. special inits/unpacking to provide binding exprs for lambda arguments.
    for a in ps.arguments(skip_tensor_options=not faithful):
        name = a.name
        arg_parser_expr = arg_parser_outputs[a.name].expr

        if has_toptions and name == 'self':
            # TODO: why this needs to be special case?
            inits.extend([
                f'auto self = {arg_parser_expr.expr};',
            ])
            lambda_args_exprs[name] = name
        elif isinstance(a, PythonOutArgument) and len(a.outputs) > 1 and f.func.is_out_fn():
            inits.extend([
                f'auto out = {arg_parser_expr};',
            ])
            for i, out_arg in enumerate(a.outputs):
                lambda_args_exprs[out_arg.name] = f'out[{i}]'
        elif str(a.type) == 'Dimname[]?':
            # [old codegen]
            # TODO: make this part of something more general, or get rid of it.
            # optional<ArrayRef<T>> are special. The PythonArgParser returns an
            # optional<vector<T>>, which cannot be implicitly converted to
            # optional<ArrayRef<T>>. One needs to unwrap the optional and rewrap.
            inits.extend([
                f'auto __{name} = {arg_parser_expr};',
                (f'c10::optional<DimnameList> {name} = __{name} ?'
                 + f' c10::make_optional(DimnameList(__{name}.value())) : c10::nullopt;'),
            ])
            lambda_args_exprs[name] = name
        else:
            # default case - directly using PythonArgParser output expr
            lambda_args_exprs[name] = arg_parser_expr

    # method's self is passed directly to python binding, rather than parsed
    if ps.method:
        lambda_args_exprs['self'] = 'self'

    # 2. special packing/checking for TensorOptions.
    tensor_options_args_names = list(map(lambda a: a.name, ps.tensor_options_args))
    if has_toptions and not faithful:
        if f.func.is_out_fn():
            raise RuntimeError(f'{f.func}: tensor options with output arg')
        for a in ps.tensor_options_args:
            if a.name not in TENSOR_OPTIONS_FIELDS:
                raise RuntimeError(
                    f'{f.func}: unrecognized tensor options field \'{a.name}\' in python binding arguments')
            if str(a.type) != TENSOR_OPTIONS_FIELDS.get(a.name):
                raise RuntimeError(
                    f'{f.func}: unrecognized type \'{str(a.type)}\' for tensor options field \'{a.name}\'')
        if not all(map(lambda a: a in tensor_options_args_names, TENSOR_OPTIONS_FIELDS.keys())):
            raise RuntimeError(
                f'{f.func}: incomplete tensor options args: {tensor_options_args_names}')

        inits.append(f'''\
const auto options = TensorOptions()
    .dtype({arg_parser_outputs['dtype'].expr})
    .device({arg_parser_outputs['device'].expr})
    .layout({arg_parser_outputs['layout'].expr})
    .requires_grad({arg_parser_outputs['requires_grad'].expr})
    .pinned_memory({arg_parser_outputs['pin_memory'].expr});
torch_npu::utils::maybe_initialize_npu(options);
''')
        lambda_args_exprs['options'] = 'options'

    # 3. special case - access scattered TensorOptions fields without packing
    # TODO: maybe move to the generator side as it's not related to binding.
    if not has_toptions and tensor_options_args_names:
        if 'dtype' in tensor_options_args_names:
            # we're an output-arg variant, check these args against output tensor
            if not f.func.is_out_fn():
                raise RuntimeError(
                    f'{f.func}: dtype in tensor_options_args without output arg')
            if not all(map(lambda a: a in tensor_options_args_names, ('layout', 'device'))):
                raise RuntimeError(
                    f'{f.func}: incomplete tensor options for output check')

            inits.append(f"""\
check_out_type_matches({arg_parser_outputs['out'].expr}, {arg_parser_outputs['dtype'].expr},
                       {arg_parser_outputs['dtype'].is_none_expr}, {arg_parser_outputs['layout'].expr},
                       {arg_parser_outputs['device'].expr}, {arg_parser_outputs['device'].is_none_expr});
""")
        # we'll set requires_grad on outgoing tensor
        if 'requires_grad' not in tensor_options_args_names:
            raise RuntimeError(
                f'{f.func}: expected "requires_grad" in tensor_options_args absent,'
                + f' but found [{tensor_options_args_names}]')

    return DispatchLambdaArgumentExprs(
        exprs=tuple(map(lambda a: lambda_args_exprs[a.name], lambda_args)),
        inits=inits,
    )
