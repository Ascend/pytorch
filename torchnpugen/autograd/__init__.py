import os
import re
from typing import List, Dict

import torchgen.gen
from torchgen.code_template import CodeTemplate
from torchgen.api.autograd import NativeFunctionWithDifferentiabilityInfo, DifferentiabilityInfo

from torchgen.packaged.autograd.gen_inplace_or_view_type import (
    gen_inplace_or_view_type_env,
    extract_bindings,
    unpacked_name,
    is_tensor_list_type,
    unpack_args as ori_unpack_args
)
from torchgen.packaged.autograd.gen_autograd_functions import process_function
from torchgen.context import with_native_function
from torchgen.model import (
    NativeFunction,
    SelfArgument,
    TensorOptionsArguments
)
from torchgen.api.types import Binding
from torchnpugen.gen_backend_stubs import parse_native_and_custom_yaml
from torchnpugen.utils import CUSTOM_YAML_NAME


def parse_native_and_custom_yaml_(*args, **kwargs):

    ## if aclnn extension for customers is used:
    env_aclnn_extension_switch = os.getenv('ACLNN_EXTENSION_SWITCH')
    env_aclnn_extension_path = os.getenv('ACLNN_EXTENSION_PATH')
    if env_aclnn_extension_switch and os.path.exists(env_aclnn_extension_path):
        # if apply aclnn extension
        custom_path = env_aclnn_extension_path
    else:
        # original code logic
        custom_path = './torch_npu/csrc/aten/'

    return parse_native_and_custom_yaml(*args, **kwargs, custom_path=f'{custom_path}/{CUSTOM_YAML_NAME}')


def gen_inplace_or_view_type_env_for_npu(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> Dict[str, List[str]]:
    gen_code = gen_inplace_or_view_type_env(fn)

    if len(gen_code['inplace_or_view_method_definitions']):
        gen_code['ops_headers'] = []
        method_definitions = re.sub(pattern=r"at::_ops::(\w+)::redispatch",
                                    repl=r'at_npu::redispatch::\1',
                                    string=gen_code['inplace_or_view_method_definitions'][0])
        gen_code['inplace_or_view_method_definitions'] = [method_definitions]
    return gen_code


# A temporary solution, due to op coupling, temporarily removing symint
def process_function_(info: DifferentiabilityInfo, template: CodeTemplate) -> str:
    function_codegen = process_function(info, template)
    if '_symint' in function_codegen:
        function_codegen = function_codegen.replace('_symint', '')
    return function_codegen

UNPACK_TENSOR_UNABLE = CodeTemplate(
    """\
auto${ref} ${arg_name}_ = ${arg_name};"""
)


@with_native_function
def npu_unpack_args_list(f: NativeFunction) -> tuple[list[str], list[Binding]]:
    body: list[str] = []
    unpacked_bindings: list[Binding] = []

    for _, binding in enumerate(extract_bindings(f)):
        if isinstance(binding.argument, SelfArgument) or isinstance(binding.argument, TensorOptionsArguments):
            raise TypeError("VariableKernel shouldn't be SelfArgument or take TensorOptions")

        is_nullable = binding.argument.type.is_nullable()
        if not binding.argument.type.is_tensor_like() or is_nullable:
            unpacked_bindings.append(binding)
            continue

        is_tensor_list = is_tensor_list_type(binding.argument.type)
        ref = (not is_nullable) and not is_tensor_list
        body.append(
            UNPACK_TENSOR_UNABLE.substitute(
                arg_name=binding.name,
                ref="&" if ref else "",
            )
        )
        unpacked_bindings.append(
            Binding(
                name=unpacked_name(binding.name),
                nctype=binding.nctype,
                argument=binding.argument,
                default=binding.default,
            )
        )

    return body, unpacked_bindings

DISABLE_UNPACK_ATEN = ["matmul_backward", "matmul_backward.out"]


@with_native_function
def npu_unpack_args(f: NativeFunction) -> tuple[list[str], list[Binding]]:
    if str(f.func.name) in DISABLE_UNPACK_ATEN:
        return npu_unpack_args_list(f)
    return ori_unpack_args(f)


def apply_autograd_patches():
    torchgen.gen.parse_native_yaml = parse_native_and_custom_yaml_
    torchgen.packaged.autograd.gen_inplace_or_view_type.gen_inplace_or_view_type_env = \
        gen_inplace_or_view_type_env_for_npu
    torchgen.packaged.autograd.gen_autograd_functions.process_function = process_function_
    torchgen.packaged.autograd.gen_inplace_or_view_type.unpack_args = npu_unpack_args


apply_autograd_patches()
