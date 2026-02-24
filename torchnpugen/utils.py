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

import os
import sys
import stat
import traceback
import warnings
import itertools
from typing import List, Optional, Set, Dict, Union, Sequence, Iterator, Tuple
from collections import defaultdict
from dataclasses import dataclass
import yaml

import torch
from torchgen.api.types.signatures import NativeSignature, DispatcherSignature
from torchgen.context import native_function_manager
from torchgen.code_template import CodeTemplate
from torchgen.model import (
    Arguments,
    BackendIndex,
    BackendMetadata,
    DispatchKey,
    is_cuda_dispatch_key,
    NativeFunction,
    NativeFunctionsGroup,
    FunctionSchema,
    OperatorName,
    TensorOptionsArguments,
    SchemaKind,
    DeviceCheckType,
    Argument,
)
from torchgen.native_function_generation import pre_group_native_functions
from torchgen.utils import concatMap
from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import Binding, CppSignatureGroup, kernel_signature
from torchgen.utils import Target
from torchgen.dest.register_dispatch_key import RegisterDispatchKey

GLOBAL_STRUCTURED_OP_INFO_CACHE = defaultdict(str)
GLOBAL_OPAPI_INFO_CACHE = set()
GLOBAL_INTERNAL_FORMAT_OPAPI_INFO_CACHE = set()

CUSTOM_YAML_NAME = "npu_native_functions_by_codegen.yaml"
FIELDS_TO_USE = ["func", "tags", "dispatch", "device_check"]
DEVICE_NOCHECK_SET = set()
DEVICE_CHECK_NOTSUPPORT_TYPE = {"Tensor[]?"}


class PathManager:

    @classmethod
    def check_path_owner_consistent(cls, path: str):
        """
        Function Description:
            check whether the path belong to process owner
        Parameter:
            path: the path to check
        Exception Description:
            when invalid path, prompt the user
        """

        if not os.path.exists(path):
            msg = f"The path does not exist: {path}"
            raise RuntimeError(msg)
        if os.stat(path).st_uid != os.getuid():
            warnings.warn(f"Warning: The {path} owner does not match the current user.")

    @classmethod
    def check_directory_path_readable(cls, path):
        """
        Function Description:
            check whether the path is writable
        Parameter:
            path: the path to check
        Exception Description:
            when invalid data throw exception
        """
        cls.check_path_owner_consistent(path)
        if os.path.islink(path):
            msg = f"Invalid path is a soft chain: {path}"
            raise RuntimeError(msg)
        if not os.access(path, os.R_OK):
            msg = f"The path permission check failed: {path}"
            raise RuntimeError(msg)

    @classmethod
    def remove_path_safety(cls, path: str):
        if os.path.islink(path):
            raise RuntimeError(f"Invalid path is a soft chain: {path}")
        if os.path.exists(path):
            os.remove(path)


def parse_npu_yaml(custom_path: str) -> Dict:
    if not os.path.exists(custom_path):
        return {}
    from io import StringIO
    f_str = StringIO()
    PathManager.check_directory_path_readable(custom_path)
    with open(custom_path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            f_str.write(line)

    f_str.seek(0)
    source_es = yaml.safe_load(f_str)
    return source_es


def merge_yaml(base_data, additional_data):
    """Merge two YAML data structures. If there's a conflict, the base data will take precedence."""
    map_dict = {"official": "supported"}

    def key_map(x):
        if x in map_dict:
            return map_dict[x]
        else:
            return x
    if isinstance(base_data, dict):
        for key, value in additional_data.items():
            if key_map(key) not in base_data:
                base_data[key_map(key)] = value
            else:
                base_data[key_map(key)] = merge_yaml(base_data[key_map(key)], value)
    elif isinstance(base_data, list):
        for item in additional_data:
            if item not in base_data:
                base_data.append(item)
    return base_data


def merge_custom_yaml(pta_path, op_plugin_path):
    PathManager.check_directory_path_readable(pta_path)
    with open(pta_path, 'r') as pta_file:
        pta_es = yaml.safe_load(pta_file)
    PathManager.check_directory_path_readable(op_plugin_path)
    with open(op_plugin_path, 'r') as op_plugin_file:
        op_es = yaml.safe_load(op_plugin_file)

    merged_yaml = merge_yaml(pta_es, op_es)
    merged_yaml_path = gen_custom_yaml_path(pta_path)
    PathManager.remove_path_safety(merged_yaml_path)
    with os.fdopen(os.open(merged_yaml_path, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), "w") as outfile:
        yaml.dump(merged_yaml, outfile, default_flow_style=False, width=float("inf"))
    os.chmod(merged_yaml_path, stat.S_IRUSR | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
    return merged_yaml


def field_tag(custom_es):
    for i, es in enumerate(custom_es):
        if not isinstance(es, dict):
            continue
        custom_es[i] = {key: custom_es[i][key] for key in FIELDS_TO_USE if key in custom_es[i]}
    return custom_es


def filt_exposed_api(custom_path: str):
    source_es = parse_npu_yaml(custom_path)
    custom_es = source_es.get('custom', []) + source_es.get('custom_autograd', [])
    exposed_set = set()
    for es in custom_es:
        if es.get('exposed', False):
            exposed_set.add(es.get('func').split('(')[0].split('.')[0])
    return list(exposed_set)


# Different implements of ops from origin torch. 
# Native ops with dispatchkey CompositeImplicitAutograd but implemented as a kernel op in pta
COMPOSITEIMPLICITAUTOGRAD_EXCEPT_LIST = [
    'isclose',
    'isfinite',
]


def filt_dispath_key(api_name: str) -> List:
    dispatch_dump = torch._C._dispatch_dump(f"aten::{api_name}")
    return [dump.split(":")[0] for dump in dispatch_dump.split('\n')]


def filt_compositeimplicitautograd_api(native_yaml_path, npu_supported):
    PathManager.check_directory_path_readable(native_yaml_path)
    with open(native_yaml_path, 'r') as f:
        es = yaml.safe_load(f)

    from torchnpugen.autograd.utils import TORCH_AUTOGRAD_FUNCTION
    supported_autograd = []
    for e in es:
        api_name = e['func'].split('(')[0]
        dispatch_keys = filt_dispath_key(api_name)
        is_compositekey = "CompositeImplicitAutograd[alias]" in dispatch_keys and \
                          "Autograd[alias]" not in dispatch_keys and \
                          api_name not in TORCH_AUTOGRAD_FUNCTION
        is_npu_api = api_name in npu_supported and api_name not in COMPOSITEIMPLICITAUTOGRAD_EXCEPT_LIST 
        if is_npu_api and is_compositekey:
            supported_autograd.append(api_name)
    return supported_autograd


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
        return os.path.dirname(os.path.realpath(torchgen.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)


def gen_op_hook_post_code(sig: Union[NativeSignature, DispatcherSignature]) -> Tuple[str, str]:
    res_code: str = None
    return_code: str = None

    if sig.returns_type().cpp_type() == "void":
        res_code = ""
        return_code = f"""at_npu::native::OpHook::GetInstance().PostHook();
    return;"""
    else:
        res_code = f"""{sig.returns_type().cpp_type()} res = """
        return_code = f"""at_npu::native::OpHook::GetInstance().PostHook(res);
    return res;"""

    return res_code, return_code


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

        op_name = str(f.func.name.name)
        force_aclnn = f"at_npu::native::ForceAclnn::GetInstance().IsForceAclnnOp(\"{op_name}\")"
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
                if f.func.arguments.self_arg is None:
                    raise ValueError("f.func.arguments.self_arg is None")
                self_arg_name = f.func.arguments.self_arg.argument.name
                return f"""
{returns_type} {name}({args_str}) {{
TORCH_CHECK_NOT_IMPLEMENTED({self_arg_name}.is_meta(),
"Cannot inplace into non-meta tensor with meta tensor argument", OPS_ERROR(ErrCode::NOT_SUPPORT));
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
            if self.backend_index.device_guard and str(f.func.name) not in DEVICE_NOCHECK_SET:
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
            unsafe_tensor_check = """  // No data check"""
            if self.backend_index.device_guard:
                has_tensor_options = any(
                    isinstance(a, TensorOptionsArguments)
                    for a in f.func.arguments.non_out
                )

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

                candidate_tensor_args = []
                for a in candidate_args:
                    if a.type.is_tensor_like():
                        candidate_tensor_args.append(f"{a.name}")

                candidate_tensor_args = list(set(candidate_tensor_args))
                unsafe_tensor_check = """// No data check."""
                if len(candidate_tensor_args) > 0:
                    unsafe_tensor_check = """
if (c10_npu::get_npu_data_unsafe_flag()) {"""

                if name in ["wrapper_NPU__copy_", "wrapper_NPU___foreach_copy_"]:
                    tensor_arg = candidate_tensor_args[0] + ", " +\
                        candidate_tensor_args[1]
                    unsafe_tensor_check = unsafe_tensor_check + f"""
    c10_npu::check_and_update_npu_tensor_for_copy(self, src);"""
                else:
                    for tensor_arg in candidate_tensor_args:
                        unsafe_tensor_check = unsafe_tensor_check + f"""
    c10_npu::check_npu_tensor_is_safe({tensor_arg});"""
                
                if len(candidate_tensor_args) > 0:
                    unsafe_tensor_check = unsafe_tensor_check + """
}
"""
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
                if has_tensor_options and device_of is not None:
                    device_guard = f"""
OptionalDeviceGuard device_guard(device_of({device_of}));
if (device.has_value()) {{
    device_guard.reset_device(device_or_default(device));
}}
"""
                elif has_tensor_options:
                    # kernel is creating a tensor
                    device_guard = """
const DeviceGuard device_guard(device_or_default(device));"""
                elif device_of is not None:
                    # kernel is operating on existing tensors
                    device_guard = f"const OptionalDeviceGuard device_guard(device_of({device_of}));"

            op_key = str(f.func.name)
            is_aclnn_only = "c10_npu::IsAclnnOnly()"
            if enable_opplugin():
                if op_key in GLOBAL_STRUCTURED_OP_INFO_CACHE:
                    impl_name = f"op_plugin::{GLOBAL_STRUCTURED_OP_INFO_CACHE[op_key]}"

            # for op_hook_check
            res_of_op_hook_post_code, return_of_op_hook_post_code = gen_op_hook_post_code(sig)

            args_exprs_str_list = [
                e.expr
                for e in translate(
                    sig.arguments(), kernel_sig.arguments(), method=False
                )
            ]
            lvalue_ref_list = ["C10_AS_INTARRAYREF_SLOW", "expect_int", "guard_int"]
            auto_lvalue = ""
            for idx, args_expr in enumerate(args_exprs_str_list):
                if any(lvalue_ref in args_expr for lvalue_ref in lvalue_ref_list):
                    auto_lvalue += f"""
    auto {kernel_sig.arguments()[idx].name}_tmp = {args_expr};"""
                    args_exprs_str_list[idx] = f"""{kernel_sig.arguments()[idx].name}_tmp"""

            args_exprs_str_for_op_hook = ", ".join(e for e in args_exprs_str_list)

            op_hook_blacklist = ["is_pinned", "_pin_memory"]
            op_hook_check = ""

            if is_opapi(op_key) and not is_op_valid(op_key):
                op_api_impl_name = f"{metadata.cpp_namespace}::NPUNativeOpApiFunctions::{metadata.kernel}"
                tensor_check_str = ""
                tensor_check_list = []
                for a in args:
                    if a.argument.type.is_tensor_like():
                        tensor_check_list.append(f"at_npu::native::FormatHelper::IsOpInputBaseFormat({a.name})")
                if tensor_check_list:
                    tensor_check_str = f" && {' && '.join(tensor_check_list)}"

                if self.backend_index.dispatch_key is DispatchKey.PrivateUse1:
                    if op_key not in op_hook_blacklist:
                        if not is_opapi_support_internal_format(op_key):
                            op_hook_check += f"""\
if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {{
{auto_lvalue}
    at_npu::native::OpHook::GetInstance().PreHook(\"{op_key}\", {args_exprs_str_for_op_hook});
    if (({force_aclnn} || at_npu::native::env::CheckJitDisable()){tensor_check_str}) {{
        {res_of_op_hook_post_code}{op_api_impl_name}({args_exprs_str_for_op_hook});
        {return_of_op_hook_post_code}
    }} else {{
        {res_of_op_hook_post_code}{impl_name}({args_exprs_str_for_op_hook});
        {return_of_op_hook_post_code}
    }}
}}
"""
                        else:
                            op_hook_check += f"""\
if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {{
{auto_lvalue}
    at_npu::native::OpHook::GetInstance().PreHook(\"{op_key}\", {args_exprs_str_for_op_hook});
    if (({force_aclnn} || at_npu::native::env::CheckJitDisable())) {{
        {res_of_op_hook_post_code}{op_api_impl_name}({args_exprs_str_for_op_hook});
        {return_of_op_hook_post_code}
    }} else {{
        {res_of_op_hook_post_code}{impl_name}({args_exprs_str_for_op_hook});
        {return_of_op_hook_post_code}
    }}
}}
"""
                if not is_opapi_support_internal_format(op_key):
                    return_code = f"""\
if (({force_aclnn} || at_npu::native::env::CheckJitDisable()){tensor_check_str}) {{
        return {op_api_impl_name}({args_exprs_str});
    }} else {{
        if ({is_aclnn_only}) {{
            TORCH_CHECK(false,
                "Current device only support aclnn operator, and current operator {impl_name} do not support internal format.",
                PTA_ERROR(ErrCode::NOT_SUPPORT));
        }}
        return {impl_name}({args_exprs_str});
    }}
"""
                else:
                    return_code = f"""\
if (({force_aclnn} || at_npu::native::env::CheckJitDisable())) {{
        return {op_api_impl_name}({args_exprs_str});
    }} else {{
        return {impl_name}({args_exprs_str});
    }}
"""
            else:
                if self.backend_index.dispatch_key is DispatchKey.PrivateUse1:
                    if op_key not in op_hook_blacklist:
                        op_hook_check += f"""\
if (C10_UNLIKELY(at_npu::native::env::CheckOpHookEnable())) {{
{auto_lvalue}
    at_npu::native::OpHook::GetInstance().PreHook(\"{op_key}\", {args_exprs_str_for_op_hook});
    {res_of_op_hook_post_code}{impl_name}({args_exprs_str_for_op_hook});
    {return_of_op_hook_post_code}
}}
"""

                return_code = f"""\
    return {impl_name}({args_exprs_str});
"""

            return f"""\
namespace {{

{returns_type} {name}({args_str}) {{
{device_check}
{unsafe_tensor_check}
{device_guard}
{record_func_def}
{op_hook_check}
{return_code}
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


def gen_device_check(
    type: DeviceCheckType, args: List[Argument], method_name: str
) -> str:
    if type == DeviceCheckType.NoCheck:
        return "  // No device check\n"

    device_check = "c10::optional<at::Device> common_device = at::nullopt;\n"
    device_check += "(void)common_device; // Suppress unused variable warning\n"
    for arg in args:
        # Only tensor like arguments are eligible
        if arg.type.is_tensor_like() and str(arg.type) not in DEVICE_CHECK_NOTSUPPORT_TYPE:
            device_check += \
f"""c10::impl::check_and_update_common_device(common_device, {arg.name}, "{method_name}", "{arg.name}");\n"""
    return device_check


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
    result = []
    for a in args:
        for r in cpp.argument(
            a,
            faithful=faithful,
            symint=symint,
            method=method,
            has_tensor_options=arguments.tensor_options is not None,
            cpp_no_default_args=cpp_no_default_args,
        ):
            if faithful:
                result.append(r.no_default())
            else:
                result.append(r)
    return result


def add_header_to_template_file():
    torchgen_path = get_torchgen_dir()
    template_dir = os.path.join(torchgen_path, "packaged/ATen/templates/DispatchKeyNativeFunctions.h")
    PathManager.check_directory_path_readable(template_dir)
    with open(template_dir, "r") as file:
        template_content = file.read()
    if "#include <ATen/ATen.h>" not in template_content:
        template_content = template_content.replace("#include <ATen/Tensor.h>",
                                                    "#include <ATen/Tensor.h>\n#include <ATen/ATen.h>")
        with os.fdopen(os.open(template_dir, os.O_WRONLY, stat.S_IWUSR | stat.S_IRUSR), "w") as file:
            file.write(template_content)


def enable_opplugin() -> bool:
    # enable op_plugin, if path of third_party/op-plugin is valid.
    
    env_aclnn_extension_switch = os.getenv('ACLNN_EXTENSION_SWITCH')
    env_aclnn_extension_path = os.getenv('ACLNN_EXTENSION_PATH')
    # if apply aclnn extension
    if env_aclnn_extension_switch and os.path.exists(env_aclnn_extension_path):
        op_plugin_path = os.path.join(env_aclnn_extension_path, 'op_plugin')
    # original code logic
    else:
        base_dir = os.path.dirname(os.path.realpath(__file__))
        op_plugin_path = os.path.join(base_dir, '../third_party/op-plugin/op_plugin')
    
    return os.path.exists(op_plugin_path)


def is_op_valid(op_key: str) -> bool:
    return True if op_key in GLOBAL_STRUCTURED_OP_INFO_CACHE else False


def get_opplugin_wrap_name(func) -> str:
    op_key = str(func.func.name) if type(func) is NativeFunction else func
    return GLOBAL_STRUCTURED_OP_INFO_CACHE.get(op_key, None)


def gen_custom_yaml_path(original_path, codegen_yaml_filename=CUSTOM_YAML_NAME):
    new_path = os.path.join(os.path.dirname(original_path), codegen_yaml_filename)
    return new_path


def update_opapi_info(op_info):
    global GLOBAL_OPAPI_INFO_CACHE
    if isinstance(op_info, str):
        return
    elif isinstance(op_info, dict):
        if op_info.get("op_api", False):
            GLOBAL_OPAPI_INFO_CACHE.add(op_info.get("func").split("(")[0])
    else:
        print(f"Warning: Unsupported parameter types, only str and dict is supported, but input is {type(op_info)}")


def is_opapi(op_key):
    global GLOBAL_OPAPI_INFO_CACHE
    return op_key in GLOBAL_OPAPI_INFO_CACHE


def update_internal_format_opapi_info(op_info):
    global GLOBAL_INTERNAL_FORMAT_OPAPI_INFO_CACHE
    if isinstance(op_info, str):
        return
    elif isinstance(op_info, dict):
        if op_info.get("internal_format_opapi", False):
            GLOBAL_INTERNAL_FORMAT_OPAPI_INFO_CACHE.add(op_info.get("func").split("(")[0])
    else:
        print(f"Warning: Unsupported parameter types, only str and dict is supported, but input is {type(op_info)}")


def is_opapi_support_internal_format(op_key):
    global GLOBAL_INTERNAL_FORMAT_OPAPI_INFO_CACHE
    return op_key in GLOBAL_INTERNAL_FORMAT_OPAPI_INFO_CACHE


def get_target_functions(yaml_path: str, target_op_type: str = None) -> List:
    source_es = parse_npu_yaml(yaml_path)

    custom = source_es.pop('custom', [])
    if custom is None:
        custom = []  # Allow an empty list of supported ops
    official = source_es.pop('official', [])
    if official is None:
        official = []  # Allow an empty list of supported ops

    support_ops = custom + official

    symint = source_es.pop("symint", [])
    if symint is None:
        symint = []
    symint = [op['func'] if isinstance(op, Dict) else op for op in symint]
    symint_set = set([str(FunctionSchema.parse(op).name) for op in symint])

    global GLOBAL_STRUCTURED_OP_INFO_CACHE
    GLOBAL_STRUCTURED_OP_INFO_CACHE.clear()
    target_funcs = []
    for op in support_ops:
        funcs = op.get("func", None)
        if not isinstance(funcs, str):
            raise TypeError(f'not a str : {funcs}')
        func = FunctionSchema.parse(funcs)
        wrap_name = cpp.name(func)
        op_key = str(func.name)
        if op_key in symint_set:
            wrap_name += "_symint"
        if target_op_type is not None:
            if target_op_type not in op.keys():
                continue
            wrap_name += "_" + target_op_type
        cur_wrap_name = GLOBAL_STRUCTURED_OP_INFO_CACHE.get(op_key, "")
        if cur_wrap_name and cur_wrap_name != wrap_name:
            print(f"Find different wrap_name for {cur_wrap_name} and {wrap_name} between pta and opplugin, ",
                  f"with {wrap_name} being used as the actual wrap_name")
        GLOBAL_STRUCTURED_OP_INFO_CACHE[op_key] = wrap_name
        target_funcs.append(func)

    return target_funcs


def get_target_native_registration(
    dispatch_key: DispatchKey,
    backend_indices: Dict[DispatchKey, BackendIndex],
    metadata: Dict[OperatorName, BackendMetadata],
    native_functions: List[NativeFunction],
):
    if native_functions is None:
        return ""
    cpu_dispatch_key = DispatchKey.parse(dispatch_key.name.replace("PrivateUse1", "CPU"))
    cpu_backend_indices = backend_indices[cpu_dispatch_key]
    cuda_dispatch_key = DispatchKey.parse(dispatch_key.name.replace("PrivateUse1", "CUDA"))
    cuda_backend_indices = backend_indices[cuda_dispatch_key]

    target_native_functions_kernels = {}
    for op_name in cpu_backend_indices.index.keys():
        if op_name not in cuda_backend_indices.index.keys():
            continue
        if cpu_backend_indices.index[op_name].kernel == cuda_backend_indices.index[op_name].kernel \
                and op_name not in metadata.keys():
            target_native_functions_kernels[op_name] = cpu_backend_indices.index[op_name].kernel
    target_native_functions = []
    for f in native_functions:
        if f.func.name in target_native_functions_kernels.keys():
            target_native_functions.append(f)

    native_functions_registration_template = CodeTemplate(
        """\
namespace {

${dispatch_helpers}

TORCH_LIBRARY_IMPL(aten, ${dispatch_key}, m) {
${native_kernels}
}
}
""")

    def wrap_native_function(f):
        kernel_name = target_native_functions_kernels[f.func.name]
        wrap_func_name = f"wrap_{dispatch_key}_{str(f.func.name.name)}_{f.func.name.overload_name}"
        with native_function_manager(f):
            sig = NativeSignature(f.func, prefix='', symint=kernel_name.endswith('symint'))
            args_exprs_str = ', '.join(a.name for a in sig.arguments())
            return f"""{sig.decl(name=wrap_func_name)} {{
    return at::native::{kernel_name}({args_exprs_str});
}}
"""

    def register_wrap_native_function(f):
        wrap_func_name = f"wrap_{dispatch_key}_{str(f.func.name.name)}_{f.func.name.overload_name}"
        return f"m.impl(\"{f.func.name}\", TORCH_FN({wrap_func_name}));"

    return native_functions_registration_template.substitute(
        dispatch_helpers=[wrap_native_function(f) for f in target_native_functions],
        dispatch_key=dispatch_key.name,
        native_kernels=[register_wrap_native_function(f) for f in target_native_functions]
    )


# A structured kernel is guaranteed to have a functional, optionally out variant and
# optionally an inplace variant.
@dataclass(frozen=True)
class NativeFunctionsGroupOptionalOut:
    functional: NativeFunction
    inplace: Optional[NativeFunction]
    mutable: Optional[NativeFunction]
    out: Optional[NativeFunction]

    @property
    def root_name(self) -> str:
        return self.functional.root_name

    def functions(self) -> Iterator[NativeFunction]:
        yield self.functional
        if self.out is not None:
            yield self.out
        if self.inplace is not None:
            yield self.inplace
        if self.mutable is not None:
            yield self.mutable

    @staticmethod
    def from_dict(
        d: Dict[SchemaKind, NativeFunction]
    ) -> Optional["NativeFunctionsGroupOptionalOut"]:
        if len(d) == 0:
            raise RuntimeError('The variable d is empty')
        if len(d) == 1:
            return None
        d = dict(d)  # non-destructive updates please
        functional = d.pop(SchemaKind.functional, None)
        inplace = d.pop(SchemaKind.inplace, None)
        mutable = d.pop(SchemaKind.mutable, None)
        out = d.pop(SchemaKind.out, None)
        if len(d) != 0:
            raise RuntimeError('The variable d is not empty after popping keys')

        if functional is None:
            raise RuntimeError('The variable functional is None')

        return NativeFunctionsGroupOptionalOut(
            functional=functional,
            inplace=inplace,
            mutable=mutable,
            out=out,
        )


def get_grouped_native_functions_optional_out(
    native_functions: Sequence[NativeFunction],
) -> Sequence[Union[NativeFunction, NativeFunctionsGroupOptionalOut]]:

    def flatten_pre_group(
        d: Dict[SchemaKind, NativeFunction]
    ) -> Sequence[Union[NativeFunction, NativeFunctionsGroupOptionalOut]]:
        r = NativeFunctionsGroupOptionalOut.from_dict(d)
        if r is None:
            # Invariant: any NativeFunctions that are code-generated
            # should have been grouped into NativeFunctionsGroupOptionalOut objects
            if any("generated" in f.tags for f in d.values()):
                raise RuntimeError("The variable d contains 'generated' in function tags")
            return list(d.values())
        else:
            return [r]

    pre_grouped_native_functions = pre_group_native_functions(native_functions)
    return list(
        concatMap(flatten_pre_group, list(pre_grouped_native_functions.values()))
    )
