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

import argparse
import os
import pathlib
import re
import stat
from collections import Counter, defaultdict, namedtuple
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torchgen
import torchgen.api.dispatcher as dispatcher
import torchgen.api.native as native
import torchgen.dest as dest
import yaml
from torchgen.api.cpp import JIT_TO_CPP_DEFAULT
from torchgen.code_template import CodeTemplate
from torchgen.gen import (
    error_check_native_functions,
    FileManager,
    get_grouped_native_functions,
    parse_native_yaml,
    parse_tags_yaml,
)
from torchgen.gen_backend_stubs import gen_dispatchkey_nativefunc_headers
from torchgen.model import (
    BackendIndex,
    BackendMetadata,
    DispatchKey,
    is_cuda_dispatch_key,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
)
from torchgen.native_function_generation import add_generated_native_functions
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, context, NamespaceHelper, Target

from torchnpugen.custom_functions import (
    gen_custom_functions_dispatch,
    gen_custom_ops_patch,
    gen_custom_trace,
    parse_custom_yaml,
)
from torchnpugen.gen_functionalization_type import (
    gen_functionalization_definition,
    gen_functionalization_registration,
)
from torchnpugen.gen_npu_c_shim import gen_npu_c_shim_files
from torchnpugen.utils import (
    add_header_to_template_file,
    DEVICE_NOCHECK_SET,
    field_tag,
    filt_compositeimplicitautograd_api,
    filt_exposed_api,
    gen_custom_yaml_path,
    gen_device_check,
    gen_unstructured,
    get_grouped_native_functions_optional_out,
    get_opplugin_wrap_name,
    get_target_functions,
    get_target_native_registration,
    get_torchgen_dir,
    is_opapi,
    merge_custom_yaml,
    NativeFunctionsGroupOptionalOut,
    parse_npu_yaml,
    PathManager,
    rename_privateuse1_dispatch_key,
    update_internal_format_opapi_info,
    update_opapi_info,
)


torchgen.model.dispatch_keys.append(torchgen.model.DispatchKey.AutogradPrivateUse1)


# Create backend_indices map for func retrieval with the key of each func we supported.
def create_backend_index(
    backend_ops: list[str],
    symint_ops: set[str],
    dispatch_key: DispatchKey,
    native_funcs_map: dict[OperatorName, NativeFunction],
    cpp_namespace: str,
) -> BackendIndex:
    metadata: dict[OperatorName, BackendMetadata] = {}
    for op in backend_ops:
        op_name = OperatorName.parse(op)
        if op_name not in native_funcs_map:
            raise KeyError(f"Found an invalid operator name: {op_name}")
        # See Note [External Backends Follow Dispatcher API]
        kernel_name = dispatcher.name(native_funcs_map[op_name].func)
        if op in symint_ops:
            kernel_name += "_symint"
        m = BackendMetadata(
            kernel=kernel_name, structured=False, cpp_namespace=cpp_namespace
        )
        metadata[op_name] = m
    return BackendIndex(
        dispatch_key=dispatch_key,
        use_out_as_primary=False,
        external=True,
        device_guard=True,
        index=metadata,
    )


# Check whether the function is placed at the wrong place.
def check_grouped_native_functions(
    backend_key: DispatchKey,
    autograd_key: DispatchKey,
    backend_indices: dict[DispatchKey, BackendIndex],
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
):
    for g in grouped_native_functions:
        if isinstance(g, NativeFunction):
            forward_kernels = (
                []
                if backend_key is None
                else [
                    m
                    for m in [backend_indices[backend_key].get_kernel(g)]
                    if m is not None
                ]
            )
            backward_kernels = (
                []
                if autograd_key is None
                else [
                    m
                    for m in [backend_indices[autograd_key].get_kernel(g)]
                    if m is not None
                ]
            )
        else:
            if backend_key is None:
                forward_kernels = []
            else:
                forward_kernels = []
                for f in g.functions():
                    kernel = backend_indices[backend_key].get_kernel(f)
                    if kernel is not None:
                        forward_kernels.append(kernel)
            if autograd_key is None:
                backward_kernels = []
            else:
                backward_kernels = []
                for f in g.functions():
                    kernel = backend_indices[autograd_key].get_kernel(f)
                    if kernel is not None:
                        backward_kernels.append(kernel)

        forward_kernels = [f for f in forward_kernels if f is not None]
        backward_kernels = [f for f in backward_kernels if f is not None]

        if len(forward_kernels) != 0 and len(backward_kernels) != 0:
            raise ValueError(
                f'Currently, all variants of an op must either be registered to a backend key, \
                             or to a backend\'s autograd key. They cannot be mix and matched. If this is \
                             something you need, feel free to create an issue! {forward_kernels[0].kernel} \
                             is listed under "supported", but {backward_kernels[0].kernel} is listed under "autograd".'
            )


_GLOBAL_PARSE_NATIVE_YAML_CACHE = {}

# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
ParsedYaml = namedtuple("ParsedYaml", ["native_functions", "backend_indices"])


def modify_func_in_native_yaml(func: str) -> str:
    # func_to_modify: {old_value: new_value}
    func_to_modify = {
        "matmul_backward(Tensor grad, Tensor self, Tensor other, bool[2] mask) -> (Tensor, Tensor)": "matmul_backward(Tensor grad_out, Tensor self, Tensor other, bool[2] mask) -> (Tensor, Tensor)"  # noqa: E501, B950
    }
    if func in func_to_modify:
        return func_to_modify[func]
    return func


def parse_native_and_custom_yaml(
    path: str, tag_path: str, custom_path: str
) -> ParsedYaml:
    global _GLOBAL_PARSE_NATIVE_YAML_CACHE
    if path not in _GLOBAL_PARSE_NATIVE_YAML_CACHE:
        valid_tags = parse_tags_yaml(tag_path)
        PathManager.check_directory_path_readable(path)
        with open(path) as f:
            es = yaml.safe_load(f)
        if not isinstance(es, list):
            raise TypeError("es is not list")
        rs: list[NativeFunction] = []
        bs: dict[DispatchKey, dict[OperatorName, BackendMetadata]] = defaultdict(dict)
        for e in es:
            e["func"] = modify_func_in_native_yaml(e["func"])
            func, m = NativeFunction.from_yaml(e, "Location", valid_tags)
            rs.append(func)
            BackendIndex.grow_index(bs, m)

        source_es = parse_npu_yaml(custom_path)
        custom_es = source_es.get("custom", []) + source_es.get("custom_autograd", [])
        supported_es = (
            source_es.get("supported", []) + source_es.get("autograd", []) + custom_es
        )
        for es in supported_es:
            update_opapi_info(es)
            update_internal_format_opapi_info(es)
        custom_es = field_tag(custom_es)
        for e in custom_es:
            func, m = NativeFunction.from_yaml(e, "Location", valid_tags)
            rs.append(func)
            BackendIndex.grow_index(bs, m)

        error_check_native_functions(rs)
        # Default dict is to prevent the codegen from barfing when we have a dispatch key that has no kernels yet.
        indices: dict[DispatchKey, BackendIndex] = defaultdict(
            lambda: BackendIndex(
                dispatch_key=DispatchKey.Undefined,
                use_out_as_primary=True,
                device_guard=True,
                external=False,
                index={},
            )
        )
        add_generated_native_functions(rs, bs)
        for k, v in bs.items():
            # All structured in-tree operators are implemented in terms of their out operator.
            indices[k] = BackendIndex(
                dispatch_key=k,
                use_out_as_primary=True,
                external=False,
                device_guard=is_cuda_dispatch_key(k),
                index=v,
            )
        _GLOBAL_PARSE_NATIVE_YAML_CACHE[path] = ParsedYaml(rs, indices)

    return _GLOBAL_PARSE_NATIVE_YAML_CACHE[path]


# Parses the external backend's yaml, and adds a new BackendIndex for the backend's dispatch key.
# Returns a Tuple of (true_backend, backend_key, autograd_key, cpp_namespace, updated BackendIndex mapping)
ParsedExternalYaml = namedtuple(
    "ParsedExternalYaml",
    ["true_backend", "backend_key", "autograd_key", "cpp_namespace", "backend_indices"],
)


def parse_backend_yaml(
    native_yaml_path: str,
    backend_yaml_path: str,
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    backend_indices: dict[DispatchKey, BackendIndex],
) -> ParsedExternalYaml:
    native_functions_map = {}
    for f in grouped_native_functions:
        if isinstance(f, NativeFunction):
            native_functions_map[f.func.name] = f
        else:
            for func in f.functions():
                native_functions_map[func.func.name] = func

    PathManager.check_directory_path_readable(backend_yaml_path)
    with open(backend_yaml_path) as f:
        yaml_values = yaml.safe_load(f)
    if not isinstance(yaml_values, dict):
        raise TypeError("yaml_values is not dict")

    # NestedTensor is a stub for now. (Not yet implemented.)
    valid_keys = [
        "backend",
        "cpp_namespace",
        "supported",
        "autograd",
        "custom",
        "custom_autograd",
        "symint",
        "quant",
        "nestedtensor",
    ]

    yaml_backend = yaml_values.pop("backend", None)
    true_backend = "PrivateUse1" if yaml_backend == "NPU" else yaml_backend
    if true_backend is None:
        raise ValueError("You must provide a value for 'backend'")
    backend = "NPU"

    cpp_namespace = yaml_values.pop("cpp_namespace", None)
    if cpp_namespace is None:
        raise ValueError("You must provide a value for 'cpp_namespace'")

    supported = yaml_values.pop("supported", [])
    if supported is None:
        supported = []  # Allow an empty list of supported ops
    if not isinstance(supported, list):
        raise TypeError(
            f'expected "supported" to be a list, but got type {type(supported)}'
        )

    symint = yaml_values.pop("symint", [])
    if symint is None:
        symint = []
    if not (isinstance(symint, list)):
        raise RuntimeError(
            f'expected "symint" to be a list, but got: {supported} (of type {type(supported)})'
        )
    symint = [op["func"].split("(")[0] if isinstance(op, dict) else op for op in symint]
    symint_set = set(symint)

    supported_autograd = yaml_values.pop("autograd", [])
    if not isinstance(supported_autograd, list):
        raise TypeError(
            f'expected "autograd" to be a list, but got: {supported_autograd}'
        )

    supported_list = []
    for op in supported:
        if isinstance(op, dict) and op.get("device_check", None) == "NoCheck":
            DEVICE_NOCHECK_SET.add(op["func"].split("(")[0])
        if isinstance(op, dict) and (
            {"impl_ns", "op_api", "device_check"} & set(op.keys())
        ):
            supported_list.append(op["func"].split("(")[0])
        elif not isinstance(op, dict):
            supported_list.append(op)
    supported = supported_list

    supported_autograd = [
        op["func"].split("(")[0] if isinstance(op, dict) else op
        for op in supported_autograd
    ]
    supported_autograd += filt_compositeimplicitautograd_api(
        native_yaml_path, supported
    )

    custom = yaml_values.pop("custom", [])
    if not isinstance(custom, list):
        raise TypeError(f'expected "autograd" to be a list, but got: {custom}')

    for item in custom:
        try:
            supported.append(item["func"][: item["func"].index("(")])
        except ValueError as e:
            raise ValueError(f"Wrong format for function: {item['func']}") from e

    custom_autograd = yaml_values.pop("custom_autograd", [])
    if not isinstance(custom_autograd, list):
        raise TypeError(f'expected "autograd" to be a list, but got: {custom_autograd}')
    for item in custom_autograd:
        supported_autograd.append(item["func"][: item["func"].index("(")])

    quant = yaml_values.pop("quant", [])
    if not isinstance(quant, list):
        raise TypeError(f'expected "quant" to be a list, but got: {quant}')
    quant = [op["func"].split("(")[0] if isinstance(op, dict) else op for op in quant]

    nestedtensor = yaml_values.pop("nestedtensor", [])
    if not isinstance(nestedtensor, list):
        raise TypeError(
            f'expected "nestedtensor" to be a list, but got: {nestedtensor}'
        )
    nestedtensor = [
        op["func"].split("(")[0] if isinstance(op, dict) else op for op in nestedtensor
    ]

    # custom_supported is only supported for filt expose api, and is not useful here.
    yaml_values.pop("custom_supported", [])
    if len(yaml_values.keys()) > 0:
        raise KeyError(
            f"{backend_yaml_path} contains unexpected keys: {', '.join(yaml_values.keys())}. \
                       Only the following keys are supported: {', '.join(valid_keys)}"
        )

    backend_key: DispatchKey | None = None
    opapi_key = "OpApi"
    if len(supported) > 0:
        with context(
            lambda: f'The provided value for "backend" must be a valid DispatchKey, but got {backend}.'
        ):
            backend_key = DispatchKey.parse(backend)

        backend_idx = create_backend_index(
            supported, symint_set, backend_key, native_functions_map, cpp_namespace
        )
        opapi_backend_idx = create_backend_index(
            [op for op in supported if is_opapi(op)],
            symint_set,
            backend_key,
            native_functions_map,
            cpp_namespace,
        )
        if backend_key in backend_indices:
            raise KeyError("backend_key should not be in backend_indices.")
        backend_indices[backend_key] = backend_idx
        backend_indices[str(backend_key) + opapi_key] = opapi_backend_idx

    autograd_key: DispatchKey | None = None
    if len(supported_autograd) > 0:
        with context(
            lambda: f'The "autograd" key was specified, which indicates that you would like to override \
the behavior of autograd for some operators on your backend. However "Autograd{backend}" is not a valid DispatchKey.'
        ):
            autograd_key = DispatchKey.parse(f"Autograd{backend}")

        autograd_idx = create_backend_index(
            supported_autograd,
            symint_set,
            autograd_key,
            native_functions_map,
            cpp_namespace,
        )
        opapi_autograd_idx = create_backend_index(
            [op for op in supported_autograd if is_opapi(op)],
            symint_set,
            autograd_key,
            native_functions_map,
            cpp_namespace,
        )

        backend_indices[autograd_key] = autograd_idx
        backend_indices[str(autograd_key) + opapi_key] = opapi_autograd_idx

    quant_key = "Quantize"
    if len(quant) > 0:
        quant_idx = create_backend_index(
            quant, symint_set, backend_key, native_functions_map, cpp_namespace
        )
        if quant_key in backend_indices:
            raise KeyError("quant_key should not be in backend_indices.")
        backend_indices[str(backend_key) + quant_key] = quant_idx

    nestedtensor_key = "Nestedtensor"
    if len(nestedtensor) > 0:
        nestedtensor_idx = create_backend_index(
            nestedtensor, symint_set, backend_key, native_functions_map, cpp_namespace
        )
        if nestedtensor_key in backend_indices:
            raise KeyError("nestedtensor_key should not be in backend_indices.")
        backend_indices[str(backend_key) + nestedtensor_key] = nestedtensor_idx

    # check_grouped_native_functions(backend_key, autograd_key, backend_indices, grouped_native_functions)
    return ParsedExternalYaml(
        true_backend, backend_key, autograd_key, cpp_namespace, backend_indices
    )


def op_plugin_kernel_conut(op_plugin_ops_dir: str):
    actual_backend_kernel_name_counts = Counter()
    file_path = os.path.join(op_plugin_ops_dir, "OpInterface.h")
    PathManager.check_directory_path_readable(file_path)
    try:
        with open(file_path) as f:
            backend_defns = f.read()
    except OSError as e:
        raise AssertionError(
            f"Unable to read from the specified impl_path file: {file_path}"
        ) from e

    kernel_defn_regex = r"\w+(?=\()"
    actual_backend_kernel_name_counts += Counter(
        re.findall(kernel_defn_regex, backend_defns)
    )
    return actual_backend_kernel_name_counts


def pta_kernel_conut(class_name: str, pta_op_dir: str):
    actual_backend_kernel_name_counts = Counter()
    for cur_dir, _, filenames in os.walk(pta_op_dir):
        for filename in filenames:
            if not filename.endswith(".cpp"):
                continue
            file_path = os.path.join(cur_dir, filename)
            PathManager.check_directory_path_readable(file_path)
            try:
                with open(file_path) as f:
                    backend_defns = f.read()
            except OSError as err:
                raise AssertionError(
                    f"Unable to read from the specified impl_path file: {file_path}"
                ) from err

            kernel_defn_regex = rf"{class_name}::([\w\d]*)\([^\)]*\)\s*{{"
            actual_backend_kernel_name_counts += Counter(
                re.findall(kernel_defn_regex, backend_defns)
            )
    return actual_backend_kernel_name_counts


def check_op_plugin_kernels(
    native_functions: Sequence[NativeFunction],
    expected_kernel_counts: dict[str, list[NativeFunction]],
    actual_kernel_counts: dict[str, list[NativeFunction]],
):
    for f in native_functions:
        wrap_name = get_opplugin_wrap_name(f)
        expect_op_plugin_kernel_count = len(expected_kernel_counts[wrap_name])
        if expect_op_plugin_kernel_count > actual_kernel_counts[wrap_name]:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate backend stub files")
    parser.add_argument(
        "-s",
        "--source_yaml",
        help="path to source yaml file containing operator external definitions",
    )
    parser.add_argument("-o", "--output_dir", help="output directory")
    parser.add_argument("--dry_run", type=bool, default=False, help="output directory")
    parser.add_argument(
        "--impl_path",
        type=str,
        default=None,
        help="path to the source C++ file containing kernel definitions",
    )
    parser.add_argument(
        "--op_plugin_impl_path",
        type=str,
        default=None,
        help="path to the source C++ file containing kernel definitions in op_plugin",
    )
    parser.add_argument(
        "--op_plugin_yaml_path",
        type=str,
        default=None,
        help="path to the source yaml file containing kernel definitions in op_plugin",
    )
    parser.add_argument(
        "--update_aoti_c_shim",
        action="store_true",
        help="Update AOTInductor C shim after adding an entry to inductor_fallback_ops in torchgen/aoti/fallback_ops.py. "
        "WARNING: Do not use this unless you are sure what you are doing!!!",
    )
    options = parser.parse_args()

    run(
        options.source_yaml,
        options.output_dir,
        options.dry_run,
        options.impl_path,
        options.op_plugin_impl_path,
        options.op_plugin_yaml_path,
        options.update_aoti_c_shim,
    )


def gen_dispatcher_registrations(
    fm: FileManager,
    class_name: str,
    backend_indices: dict[DispatchKey, BackendIndex],
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    backend_dispatch_key: DispatchKey,
    dispatch_key: DispatchKey,
    selector: "SelectiveBuilder",
    dispatch_key_name: str,
    register_dispatch_key_func: Callable,
    native_function_registrations: str = "",
):
    backend_index = backend_indices[backend_dispatch_key]
    ns_helper = NamespaceHelper(namespace_str="at")

    # 获取环境变量
    env_aclnn_extension_switch = os.getenv("ACLNN_EXTENSION_SWITCH")

    native_func_header = """\
#include "torch_npu/csrc/core/npu/NPURecovery.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/profiler/utils.h"
#endif

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/NPUOpApiNativeFunctions.h"
"""

    # 根据环境变量决定是否包含FormatHelper.h
    if env_aclnn_extension_switch:
        native_func_header += ""
    else:
        native_func_header += """#include "torch_npu/csrc/framework/FormatHelper.h"
"""

    native_func_header += """#include "torch_npu/csrc/framework/utils/ForceAclnnList.h"
#include "torch_npu/csrc/framework/OpHook.h"
#include "op_plugin/OpInterface.h"
"""
    static_template = CodeTemplate(
        """\
TORCH_LIBRARY_IMPL(aten, $dispatch_key, m) {
$dispatch_registrations_body
};"""
    )
    static_init_dispatch_registrations = static_template.substitute(
        dispatch_key=dispatch_key_name,
        dispatch_registrations_body=list(
            concatMap(
                register_dispatch_key_func(
                    backend_index,
                    Target.REGISTRATION,
                    selector,
                    rocm=False,
                    symint=True,
                    class_method_name=f"{class_name}",
                    skip_dispatcher_op_registration=False,
                ),
                grouped_native_functions,
            )
        ),
    )
    fm.write_with_template(
        f"Register{dispatch_key}.cpp",
        "RegisterDispatchKey.cpp",
        lambda: {
            "extra_cuda_headers": "",
            "external_backend_headers": native_func_header,
            "namespaced_headers": "",
            "DispatchKey": dispatch_key,
            "dispatch_headers": dest.gen_registration_headers(
                backend_index, per_operator_headers=False, rocm=False
            ),
            "ops_headers": "",
            "dispatch_helpers": dest.gen_registration_helpers(backend_index),
            "dispatch_definitions": fm.substitute_with_template(
                "RegisterDispatchDefinitions.ini",
                lambda: {
                    "ns_prologue": ns_helper.prologue,
                    "ns_epilogue": ns_helper.epilogue,
                    "static_init_dispatch_registrations": static_init_dispatch_registrations,
                    "deferred_dispatch_registrations": "",
                    "dispatch_namespace": dispatch_key.lower(),
                    "dispatch_namespaced_definitions": native_function_registrations if native_function_registrations else list(
                        concatMap(
                            register_dispatch_key_func(
                                backend_index,
                                Target.NAMESPACED_DEFINITION,
                                selector,
                                rocm=False,
                                symint=True,
                                class_method_name=f'{class_name}',
                                skip_dispatcher_op_registration=False,
                            ),
                            grouped_native_functions,
                        )
                    ),
                    "dispatch_anonymous_definitions": list(
                        concatMap(
                            register_dispatch_key_func(
                                backend_index,
                                Target.ANONYMOUS_DEFINITION,
                                selector,
                                rocm=False,
                                symint=True,
                                class_method_name=f"{class_name}",
                                skip_dispatcher_op_registration=False,
                            ),
                            grouped_native_functions,
                        )
                    ),
                },
            ).split("\n"),
        },
    )


def get_supported_grouped_native_functions(
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    backend_index: BackendIndex,
) -> Sequence[NativeFunction | NativeFunctionsGroup]:
    supported_grouped_native_functions: Sequence[
        NativeFunction | NativeFunctionsGroup
    ] = []
    for funcs in grouped_native_functions:
        if isinstance(funcs, NativeFunctionsGroup) and not backend_index.has_kernel(
            funcs.out
        ):
            for f in funcs.functions():
                if backend_index.has_kernel(f):
                    supported_grouped_native_functions.append(f)
            continue
        supported_grouped_native_functions.append(funcs)
    return supported_grouped_native_functions


def gen_foreach_register(
    fm: FileManager,
    tags_yaml_path: str,
    native_yaml_path: str,
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    backend_indices: BackendIndex,
):
    cpu_backend_indices = parse_native_yaml(
        native_yaml_path, tags_yaml_path
    ).backend_indices[DispatchKey.CPU]
    foreach_dict: dict[str, str] = {}
    header_set = set()

    def get_foreach_kernel(func: NativeFunction):
        schema = func.func.name
        if not str(schema).startswith("_foreach"):
            return
        if schema in cpu_backend_indices.index and schema not in backend_indices.index:
            foreach_dict[str(schema)] = cpu_backend_indices.index[schema].kernel

    for f in grouped_native_functions:
        if isinstance(f, NativeFunctionsGroup):
            header_set.add(str(f.signature().name.name.base))
            for func in f.functions():
                get_foreach_kernel(func)
        else:
            header_set.add(str(f.func.name.name.base))
            get_foreach_kernel(f)

    kernel_template = CodeTemplate(
        """\
m.impl("${schema}", TORCH_FN(at::native::${kernel}));"""
    )
    header_template = CodeTemplate(
        """\
#include <ATen/ops/${function}_native.h>"""
    )
    # Sort for deterministic output
    fm.write_with_template(
        "ForeachRegister.cpp",
        "ForeachRegister.cpp",
        lambda: {
            "include_headers": [
                header_template.substitute(function=h)
                for h in sorted(header_set)
                if h.startswith("_foreach")
            ],
            "foreach_kernel": [
                kernel_template.substitute(schema=kv[0], kernel=kv[1])
                for kv in sorted(foreach_dict.items())
            ],
        },
    )


# 定义配置数据类
@dataclass
class SpecialRegisterConfig:
    dispatch_key: str
    filename: str
    header: str
    extra_impls: list[str]


KERNEL_TEMPLATE = CodeTemplate("""\
m.impl("${schema}", TORCH_FN(op_plugin::${kernel}));""")


def _gen_special_registration_body(
    backend_indices: BackendIndex,
    config: SpecialRegisterConfig,
) -> str:
    """生成特殊注册的主体内容,生成内容需保证确定性"""
    kernel_regs = [
        KERNEL_TEMPLATE.substitute(schema=op_name, kernel=metadata.kernel)
        for op_name, metadata in sorted(
            backend_indices.index.items(), key=lambda x: str(x[0])
        )
    ]

    template = CodeTemplate("""\
TORCH_LIBRARY_IMPL(aten, $dispatch_key, m) {
$kernel_registrations
$extra_impls
};""")

    return template.substitute(
        dispatch_key=config.dispatch_key,
        kernel_registrations=kernel_regs,
        extra_impls="\n".join(config.extra_impls),
    )


def _write_special_register(
    fm: FileManager,
    config: SpecialRegisterConfig,
    static_init_dispatch_registrations: str,
) -> None:
    """写入特殊注册文件"""
    ns_helper = NamespaceHelper(namespace_str="at")

    dispatch_definitions = fm.substitute_with_template(
        "RegisterDispatchDefinitions.ini",
        lambda: {
            "ns_prologue": ns_helper.prologue,
            "ns_epilogue": ns_helper.epilogue,
            "static_init_dispatch_registrations": static_init_dispatch_registrations,
            "deferred_dispatch_registrations": "",
            "dispatch_namespace": "",
            "dispatch_namespaced_definitions": "",
            "dispatch_anonymous_definitions": "",
        },
    ).split("\n")

    fm.write_with_template(
        f"{config.filename}.cpp",
        "RegisterDispatchKey.cpp",
        lambda: {
            "extra_cuda_headers": "",
            "external_backend_headers": config.header,
            "namespaced_headers": "",
            "DispatchKey": "NPU",
            "dispatch_headers": "",
            "ops_headers": "",
            "dispatch_helpers": "",
            "dispatch_definitions": dispatch_definitions,
        },
    )


SPECIAL_REGISTERS = {
    "quantize": SpecialRegisterConfig(
        dispatch_key="QuantizedPrivateUse1",
        filename="QuantizedRegister",
        header="""\
#include <ATen/ops/quantize_per_tensor.h>
#include "op_plugin/OpInterface.h"
""",
        extra_impls=[
            'm.impl("q_scale", TORCH_FN(at::native::q_scale_quant));',
            'm.impl("q_per_channel_scales", TORCH_FN(at::native::q_per_channel_scales));',
            'm.impl("q_zero_point", TORCH_FN(at::native::q_zero_point_quant));',
            'm.impl("q_per_channel_zero_points", TORCH_FN(at::native::q_per_channel_zero_points));',
            'm.impl("q_per_channel_axis", TORCH_FN(at::native::q_per_channel_axis));',
            'm.impl("qscheme", TORCH_FN(at::native::qscheme_quant));',
        ],
    ),
    "nestedtensor": SpecialRegisterConfig(
        dispatch_key="NestedTensorPrivateUse1",
        filename="NestedTensorRegister",
        header="",
        extra_impls=[
            'm.impl("unbind.int", TORCH_FN(at::native::NestedTensor_unbind));',
            'm.impl("values", TORCH_FN(at::native::values_nested));',
            'm.impl("_nested_tensor_size", TORCH_FN(at::native::_nested_tensor_size));',
        ],
    ),
}


def gen_quantize_register(
    fm: FileManager,
    backend_indices: BackendIndex,
) -> None:
    """生成量化注册"""
    config = SPECIAL_REGISTERS["quantize"]
    static_init = _gen_special_registration_body(
        backend_indices["NPUQuantize"],
        config,
    )
    _write_special_register(fm, config, static_init)


def gen_nestedtensor_register(
    fm: FileManager,
    backend_indices: BackendIndex,
) -> None:
    """生成嵌套张量注册"""
    config = SPECIAL_REGISTERS["nestedtensor"]
    static_init = _gen_special_registration_body(
        backend_indices["NPUNestedtensor"],
        config,
    )
    _write_special_register(fm, config, static_init)


def gen_functionalization(
    fm: FileManager,
    selector: "SelectiveBuilder",
    grouped_native_functions: Sequence[
        NativeFunction | NativeFunctionsGroupOptionalOut
    ],
):
    def key_func(fn: NativeFunction | NativeFunctionsGroupOptionalOut) -> str:
        return fn.root_name

    def functionalization_env_callable(g):
        definition = gen_functionalization_definition(selector, g)
        register = gen_functionalization_registration(selector, g)
        return {
            "func_definitions": definition,
            "func_registrations": register,
        }

    fm.write_sharded(
        "RegisterFunctionalization.cpp",
        grouped_native_functions,
        key_fn=key_func,
        env_callable=functionalization_env_callable,
        num_shards=2,
        sharded_keys={
            "func_definitions",
            "func_registrations",
        },
    )
    return


def gen_target_registration(
    target_op_type: str,
    dispatch_key: DispatchKey,
    backend_indices: dict[DispatchKey, BackendIndex],
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    op_plugin_yaml_path: str,
    fm: FileManager,
    selector: "SelectiveBuilder",
    native_functions: list[NativeFunction] | None = None,
):
    target_ops = get_target_functions(
        op_plugin_yaml_path, target_op_type=target_op_type
    )
    target_native_functions = []
    for f in grouped_native_functions:
        if isinstance(f, NativeFunctionsGroup):
            for func in f.functions():
                if func.func in target_ops:
                    target_native_functions.append(func)
        elif f.func in target_ops:
            target_native_functions.append(f)

    metadata: dict[OperatorName, BackendMetadata] = {}
    for op in target_ops:
        kernel_name = dispatcher.name(op)
        metadata[op.name] = BackendMetadata(
            kernel=kernel_name, structured=False, cpp_namespace=target_op_type
        )
    backend_indices[dispatch_key] = BackendIndex(
        dispatch_key=dispatch_key,
        use_out_as_primary=False,
        external=True,
        device_guard=True,
        index=metadata,
    )

    native_registration = get_target_native_registration(
        dispatch_key, backend_indices, metadata, native_functions
    )
    gen_dispatcher_registrations(
        fm,
        backend_indices[dispatch_key].native_function_class_name(),
        backend_indices,
        target_native_functions,
        dispatch_key,
        dispatch_key,
        selector,
        dispatch_key_name=dispatch_key.name,
        register_dispatch_key_func=dest.RegisterDispatchKey,
        native_function_registrations=native_registration,
    )


def gen_per_operator_headers(
    fm: FileManager,
    ops_fm: FileManager,
    native_functions: Sequence[NativeFunction],
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    backend_indices: dict[DispatchKey, BackendIndex],
    dispatch_keys: Sequence[DispatchKey],
    selector: "SelectiveBuilder",
):
    """
    Generate per-operator dispatch header files (*_npu_dispatch.h) for NPU.

    This mirrors upstream's gen_per_operator_headers which generates
    *_cuda_dispatch.h and *_cpu_dispatch.h in ATen/ops/.

    Uses dest.RegisterDispatchKey with Target.NAMESPACED_DECLARATION to
    generate TORCH_API function declarations in at::npu namespace,
    exactly matching upstream's dispatch header format.

    Also generates NPUFunctions.h/NPUFunctions_inl.h which includes all per-operator
    dispatch headers, mirroring upstream's CUDAFunctions.h/CUDAFunctions_inl.h.
    """

    functions_by_root_name: dict[str, list[NativeFunction]] = defaultdict(list)
    for fn in native_functions:
        functions_by_root_name[fn.root_name].append(fn)

    grouped_functions_by_root_name: dict[
        str, list[NativeFunction | NativeFunctionsGroup]
    ] = defaultdict(list)
    for group in grouped_native_functions:
        name = group.root_name
        grouped_functions_by_root_name[name].append(group)

    for dispatch_key in dispatch_keys:
        if dispatch_key not in backend_indices:
            continue

        dispatch_namespace = dispatch_key.lower()
        dispatch_names = []

        for name, functions in functions_by_root_name.items():
            grouped_functions = grouped_functions_by_root_name.get(name, [])
            declarations = list(
                concatMap(
                    dest.RegisterDispatchKey(
                        backend_indices[dispatch_key],
                        Target.NAMESPACED_DECLARATION,
                        selector,
                        rocm=False,
                        symint=True,
                        class_method_name=None,
                        skip_dispatcher_op_registration=False,
                    ),
                    grouped_functions,
                )
            )

            if len(declarations) == 0:
                continue

            dispatch_names.append(name)

            ops_fm.write_with_template(
                f"{name}_{dispatch_namespace}_dispatch.h",
                "DispatchKeyFunction.h",
                lambda: {
                    "dispatch_namespace": dispatch_namespace,
                    "dispatch_namespaced_declarations": declarations,
                },
            )

        inl_headers = f"#include <torch_npu/csrc/aten/{dispatch_key}Functions_inl.h>"

        fm.write_with_template(
            f"{dispatch_key}Functions.h",
            "DispatchKeyFunctions.h",
            lambda: {
                "dispatch_key": str(dispatch_key),
                "inline_headers": inl_headers,
            },
        )
        fm.write_with_template(
            f"{dispatch_key}Functions_inl.h",
            "DispatchKeyFunctions_inl.h",
            lambda: {
                "dispatch_namespace": dispatch_namespace,
                "DispatchKeyFunctions_inl_includes": [
                    f"#include <torch_npu/csrc/aten/ops/{name}_{dispatch_namespace}_dispatch.h>"
                    for name in sorted(dispatch_names)
                ],
                "dispatch_namespaced_declarations": [],
            },
        )


def run(
    source_yaml: str,
    output_dir: str,
    dry_run: bool,
    impl_path: str | None,
    op_plugin_impl_path: str | None,
    op_plugin_yaml_path: str | None,
    update_aoti_c_shim: bool,
) -> None:
    rename_privateuse1_dispatch_key()
    torchgen_path = get_torchgen_dir()

    template_dir = os.path.join(torchgen_path, "packaged/ATen/templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(
            install_dir=install_dir, template_dir=template_dir, dry_run=dry_run
        )

    fm = make_file_manager(output_dir)
    ops_output_dir = os.path.join(output_dir, "ops")
    if not os.path.exists(ops_output_dir):
        os.makedirs(ops_output_dir, exist_ok=True)
    ops_fm = make_file_manager(ops_output_dir)
    merge_custom_yaml(source_yaml, op_plugin_yaml_path)
    source_yaml = gen_custom_yaml_path(source_yaml)
    tags_yaml_path = os.path.join(torchgen_path, "packaged/ATen/native/tags.yaml")
    native_yaml_path = os.path.join(
        torchgen_path, "packaged/ATen/native/native_functions.yaml"
    )
    parsed_yaml = parse_native_and_custom_yaml(
        native_yaml_path, tags_yaml_path, source_yaml
    )
    get_target_functions(op_plugin_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )
    grouped_native_functions = get_grouped_native_functions(native_functions)
    parsed_backend_yaml = parse_backend_yaml(
        native_yaml_path, source_yaml, grouped_native_functions, backend_indices
    )
    true_backend = parsed_backend_yaml.true_backend
    backend_key = parsed_backend_yaml.backend_key
    autograd_key = parsed_backend_yaml.autograd_key
    cpp_namespace = parsed_backend_yaml.cpp_namespace
    backend_indices = parsed_backend_yaml.backend_indices
    selector = SelectiveBuilder.get_nop_selector()
    if backend_key is not None:
        backend_dispatch_key: DispatchKey = backend_key
        autograd_dispatch_key: DispatchKey = autograd_key
        class_name = backend_indices[backend_dispatch_key].native_function_class_name()
        gen_dispatchkey_nativefunc_headers(
            fm,
            class_name,
            cpp_namespace,
            backend_indices,
            grouped_native_functions,
            backend_key,
            None,
        )

        gen_dispatchkey_nativefunc_headers(
            fm,
            "NPUNativeOpApiFunctions",
            cpp_namespace,
            backend_indices,
            grouped_native_functions,
            str(backend_key) + "OpApi",
            None,
        )

        gen_per_operator_headers(fm, ops_fm, native_functions, grouped_native_functions, backend_indices, [backend_dispatch_key, autograd_dispatch_key], selector)

        for dispatch_key in [backend_dispatch_key, autograd_dispatch_key]:
            if not dispatch_key:
                continue
            gen_dispatcher_registrations(
                fm,
                class_name,
                backend_indices,
                get_supported_grouped_native_functions(
                    grouped_native_functions, backend_indices[dispatch_key]
                ),
                dispatch_key,
                dispatch_key,
                selector,
                dispatch_key_name=dispatch_key.name.replace("NPU", true_backend),
                register_dispatch_key_func=dest.RegisterDispatchKey,
            )

        gen_quantize_register(fm, backend_indices)
        gen_nestedtensor_register(fm, backend_indices)

        pta_template_dir = os.path.join(
            pathlib.Path(__file__).parent.absolute(), "templates"
        )
        fm = FileManager(
            install_dir=output_dir, template_dir=pta_template_dir, dry_run=dry_run
        )

        custom_functions, custom_backend_indices = parse_custom_yaml(
            source_yaml, tags_yaml_path
        )
        grouped_custom_functions = get_grouped_native_functions_optional_out(
            custom_functions
        )
        gen_functionalization(fm, selector, grouped_custom_functions)
        gen_custom_trace(fm, custom_functions, custom_backend_indices)
        gen_custom_functions_dispatch(fm, custom_functions)

        gen_foreach_register(
            fm,
            tags_yaml_path,
            native_yaml_path,
            grouped_native_functions,
            backend_indices[backend_dispatch_key],
        )

        custom_ops_patch_dir = os.path.join(output_dir, "../../utils/")
        fm = FileManager(
            install_dir=custom_ops_patch_dir,
            template_dir=pta_template_dir,
            dry_run=dry_run,
        )
        gen_custom_ops_patch(fm, custom_functions)

        filt_exposed_list = filt_exposed_api(source_yaml)

        env_aclnn_extension_switch = os.getenv("ACLNN_EXTENSION_SWITCH")
        env_aclnn_extension_path = os.getenv("ACLNN_EXTENSION_PATH")
        # if apply aclnn extension
        if env_aclnn_extension_switch and os.path.exists(env_aclnn_extension_path):
            exposed_path = pathlib.Path(
                os.path.join(env_aclnn_extension_path, "torch_npu/utils/exposed_api.py")
            )
        # original code logic
        else:
            exposed_path = (
                pathlib.Path(__file__)
                .parents[1]
                .joinpath("torch_npu/utils/exposed_api.py")
            )

        PathManager.remove_path_safety(exposed_path)
        with os.fdopen(
            os.open(exposed_path, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR),
            "w",
        ) as f:
            f.write(f"public_npu_functions = {filt_exposed_list}")
        os.chmod(
            exposed_path, stat.S_IRUSR | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP
        )
        fm = make_file_manager(output_dir)
        gen_target_registration(
            "sparse",
            DispatchKey.SparsePrivateUse1,
            backend_indices,
            grouped_native_functions,
            op_plugin_yaml_path,
            fm,
            selector,
            native_functions,
        )

        gen_target_registration(
            "sparse_csr",
            DispatchKey.SparseCsrPrivateUse1,
            backend_indices,
            grouped_native_functions,
            op_plugin_yaml_path,
            fm,
            selector,
            native_functions,
        )

        aoti_output_dir = os.path.join(output_dir, "../inductor/aoti_torch/generated")
        if not os.path.exists(aoti_output_dir):
            os.makedirs(aoti_output_dir, exist_ok=True)
        aoti_fm = make_file_manager(aoti_output_dir)
        structured_native_functions = [
            g for g in grouped_native_functions if isinstance(g, NativeFunctionsGroup)
        ]
        gen_npu_c_shim_files(
            aoti_fm, native_functions, backend_indices,
            [backend_dispatch_key, autograd_dispatch_key],
            structured_native_functions, update_aoti_c_shim,
        )


def apply_torchgen_patch():
    dest.RegisterDispatchKey.gen_unstructured = gen_unstructured
    dest.RegisterDispatchKey.gen_device_check = gen_device_check
    # generate default arguments
    JIT_TO_CPP_DEFAULT["contiguous_format"] = "c10::MemoryFormat::Contiguous"
    add_header_to_template_file()
    dispatcher.arguments = native.arguments


if __name__ == "__main__":
    apply_torchgen_patch()
    main()
