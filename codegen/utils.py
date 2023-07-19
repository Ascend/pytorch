# Copyright (c) 2023 Huawei Technologies Co., Ltd
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

from collections import defaultdict
import os
import re
import sys
import stat
import traceback
from typing import List, Optional, Set
import yaml

from torchgen.context import native_function_manager
from torchgen.model import (
    Arguments,
    DispatchKey,
    is_cuda_dispatch_key,
    NativeFunction,
    NativeFunctionsGroup,
)
from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import Binding, CppSignatureGroup, kernel_signature
from torchgen.utils import Target
from torchgen.gen import LineLoader
from torchgen.yaml_utils import YamlLoader

GLOBAL_STRUCTURED_OP_INFO_CACHE = defaultdict(str)

def parse_npu_yaml(custom_path: str) -> List:
    from io import StringIO
    f_str = StringIO()
    with open(custom_path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            f_str.write(line)

    f_str.seek(0)
    global GLOBAL_STRUCTURED_OP_INFO_CACHE
    source_es = yaml.load(f_str, Loader=LineLoader)

    wrap_impl_keys = ['supported', 'autograd', 'symint', 'custom', 'custom_autograd']

    def _set_wrap_impl_state(key: str):
        if source_es[key]:
            for x in source_es[key]:
                if 'wrap_impl' in x:
                    GLOBAL_STRUCTURED_OP_INFO_CACHE[x['func'].split('(')[0]] = x['wrap_impl']

    for x in wrap_impl_keys:
        _set_wrap_impl_state(x)

    return source_es


def filt_npu_autograd_functions(path, custom_path, derivatives_path) -> set:
    torch_functions = set()
    with open(path, 'r') as f:
        es = yaml.load(f, Loader=LineLoader)
    assert isinstance(es, list)
    for e in es:
        torch_functions.add(e.get('func'))

    parse_npu_yaml(custom_path)

    with open(derivatives_path, 'r') as f:
        definitions = yaml.load(f, Loader=YamlLoader)
    npu_autograd_functions = set()
    for item in definitions:
        if item['name'] in torch_functions:
            continue
        name = item['name'].split('(')[0]
        suffixes = ['', '_', '.out']
        for suffix in suffixes:
            func_name = name + suffix
            if func_name in GLOBAL_STRUCTURED_OP_INFO_CACHE:
                npu_autograd_functions.add(func_name)

    return npu_autograd_functions


def rename_privateuse1_dispatch_key():
    # rename DispatcherKey about PrivateUse1
    custom_backend = "NPU"
    def PrivateUse1Str(self):
        return self.name.replace("PrivateUse1", custom_backend)

    @staticmethod
    def parse(value: str) -> "DispatchKey":
        for k, v in DispatchKey.__members__.items():
            if k == value.replace(custom_backend, "PrivateUse1"):
                return v
        raise AssertionError(f"unknown dispatch key {value}")

    DispatchKey.__str__ = PrivateUse1Str
    DispatchKey.parse = parse


def get_torchgen_dir():
    # get path of torchgen, then get tags.yaml and native_functions.yaml
    try:
        import torchgen
        return os.path.dirname(os.path.abspath(torchgen.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)


# This function is to add profiler information for each operator, which is later extended in the official
def gen_unstructured(
    self, f: NativeFunction, g: Optional[NativeFunctionsGroup] = None
) -> Optional[str]:
    with native_function_manager(f):
        inplace_meta = False
        gets_out_inplace_wrapper = False
        if not self.backend_index.has_kernel(f):
            return None
        if f.manual_kernel_registration:
            return None

        if (
            self.target is Target.REGISTRATION
            and not self.selector.is_native_function_selected(f)
        ):
            return None

        sig = self.wrapper_kernel_sig(f)

        name = sig.name()
        returns_type = sig.returns_type().cpp_type()
        args = sig.arguments()
        args_str = ", ".join(a.defn() for a in args)

        # See Note [Direct dispatch bindings]
        cpp_sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=False
        )

        if self.target is Target.NAMESPACED_DECLARATION:
            result = ""
            for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                result += f"TORCH_API {cpp_sig.decl()};\n"
            return result
        elif self.target is Target.NAMESPACED_DEFINITION:

            def generate_defn(cpp_sig: CppSignature) -> str:
                return f"""
{cpp_sig.defn()} {{
return {sig.name()}({', '.join(e.expr for e in translate(cpp_sig.arguments(), sig.arguments()))});
}}
"""

            result = ""
            for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                result += generate_defn(cpp_sig)
            return result

        elif self.target is Target.ANONYMOUS_DEFINITION:
            # short circuit for inplace_meta
            if inplace_meta:
                assert f.func.arguments.self_arg is not None
                self_arg_name = f.func.arguments.self_arg.argument.name
                return f"""
{returns_type} {name}({args_str}) {{
TORCH_CHECK_NOT_IMPLEMENTED({self_arg_name}.is_meta(),
"Cannot inplace into non-meta tensor with meta tensor argument");
return {self_arg_name};
}}
"""

            # short circuit for generated inplace/out wrappers
            if gets_out_inplace_wrapper:
                return self.gen_out_inplace_wrapper(f, g)

            metadata = self.backend_index.get_kernel(f)
            if metadata is None:
                return None
            if self.class_method_name is None:
                impl_name = f"{metadata.cpp_namespace}::{metadata.kernel}"
            else:
                impl_name = f"{metadata.cpp_namespace}::{self.class_method_name}::{metadata.kernel}"

            kernel_sig = kernel_signature(f, self.backend_index)

            args_exprs_str = ", ".join(
                e.expr
                for e in translate(
                    sig.arguments(), kernel_sig.arguments(), method=False
                )
            )

            device_check = "  // No device check\n"
            # Backends that require device guards presumably also require device checks.
            if self.backend_index.device_guard:
                device_check_args = itertools.chain(
                    f.func.arguments.out, f.func.arguments.flat_positional
                )
                device_check = RegisterDispatchKey.gen_device_check(
                    f.device_check, list(device_check_args), name
                )

            device_guard = "// DeviceGuard omitted"  # default
            record_func_def = """
#ifndef BUILD_LIBTORCH
torch_npu::profiler::NPURecordFunction guard;
#endif 
"""
            if f.device_guard and self.backend_index.device_guard:
                has_tensor_options = any(
                    isinstance(a, TensorOptionsArguments)
                    for a in f.func.arguments.non_out
                )
                if has_tensor_options:
                    # kernel is creating a tensor
                    device_guard = """
const DeviceGuard device_guard(device_or_default(device));"""

                    # CUDA requires special handling
                    if is_cuda_dispatch_key(self.backend_index.dispatch_key):
                        device_guard = (
                            f"globalContext().lazyInitCUDA();\n{device_guard}"
                        )
                else:
                    # kernel is operating on existing tensors

                    # There is precedence for which argument we use to do
                    # device guard.  This describes the precedence order.
                    self_arg = (
                        [f.func.arguments.self_arg.argument]
                        if f.func.arguments.self_arg is not None
                        else []
                    )
                    candidate_args = itertools.chain(
                        self_arg,
                        f.func.arguments.out,
                        f.func.arguments.flat_positional,
                    )

                    # Only tensor like arguments are eligible
                    device_of = next(
                        (
                            f"{a.name}"
                            for a in candidate_args
                            if a.type.is_tensor_like()
                        ),
                        None,
                    )
                    if device_of is not None:
                        device_guard = f"const OptionalDeviceGuard device_guard(device_of({device_of}));"

            if enable_opplugin():
                op_key = str(f.func.name)
                if op_key in GLOBAL_STRUCTURED_OP_INFO_CACHE:
                    impl_name = f"op_plugin::{GLOBAL_STRUCTURED_OP_INFO_CACHE[op_key]}"

            return f"""\
namespace {{

{returns_type} {name}({args_str}) {{
{device_check}

{device_guard}
{record_func_def}
return {impl_name}({args_exprs_str});
}}

}} // anonymous namespace
"""

        elif self.target is Target.REGISTRATION:
            if f.manual_kernel_registration or self.skip_dispatcher_op_registration:
                return None
            else:
                payload = f"TORCH_FN({name})"
                return f'm.impl("{f.func.name}",\n{payload});\n'
        else:
            assert_never(self.target)


def arguments(
    arguments: Arguments,
    *,
    faithful: bool,
    symint: bool = False,
    method: bool,
    cpp_no_default_args: Set[str],
) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    args.extend(arguments.non_out)
    args.extend(arguments.out)
    return [
        r.no_default() if faithful else r
        for a in args
        for r in cpp.argument(
            a,
            faithful=faithful,
            symint=symint,
            method=method,
            has_tensor_options=arguments.tensor_options is not None,
            cpp_no_default_args=cpp_no_default_args,
        )
    ]


def add_header_to_template_file():
    torchgen_path = get_torchgen_dir()
    template_dir = os.path.join(torchgen_path, "packaged/ATen/templates/DispatchKeyNativeFunctions.h")
    with open(template_dir, "r") as file:
        template_content = file.read()
    if "#include <ATen/ATen.h>" not in template_content:
        template_content = template_content.replace("#include <ATen/Tensor.h>",
                                                    "#include <ATen/Tensor.h>\n#include <ATen/ATen.h>")
        with os.fdopen(os.open(template_dir, os.O_WRONLY, stat.S_IWUSR | stat.S_IRUSR), "w") as file:
            file.write(template_content)


def enable_opplugin() -> bool:
    # enable op_plugin, if path of third_party/op-plugin is valid.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    op_plugin_path = os.path.join(base_dir, '../third_party/op-plugin/op_plugin')
    return os.path.exists(op_plugin_path)


def is_op_valid(op_key: str) -> bool:
    return True if op_key in GLOBAL_STRUCTURED_OP_INFO_CACHE else False
