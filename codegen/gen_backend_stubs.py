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

import pathlib
import argparse
import os
import re
import copy
from collections import namedtuple, Counter, defaultdict
from typing import List, Dict, Union, Sequence, Optional, Set, Callable
import yaml

from torchgen.code_template import CodeTemplate
from torchgen.gen import (parse_tags_yaml, LineLoader, FileManager, parse_native_yaml,
                          get_grouped_native_functions, error_check_native_functions)
from torchgen.model import (BackendIndex, DispatchKey, Location,
                            NativeFunction, NativeFunctionsGroup, OperatorName,
                            BackendMetadata, is_cuda_dispatch_key)
from torchgen.native_function_generation import add_generated_native_functions
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import Target, concatMap, context, NamespaceHelper
from torchgen.context import native_function_manager
import torchgen.dest as dest
import torchgen.api.dispatcher as dispatcher
from torchgen.api.types import DispatcherSignature
import torchgen.api.native as native
from torchgen.api.cpp import JIT_TO_CPP_DEFAULT
from torchgen.gen_backend_stubs import gen_dispatchkey_nativefunc_headers

from codegen.utils import (get_torchgen_dir, rename_privateuse1_dispatch_key, gen_unstructured,
                           add_header_to_template_file, parse_npu_yaml, get_opplugin_wrap_name,
                           parse_opplugin_yaml, merge_custom_yaml, filed_tag, gen_custom_yaml_path,
                           update_opapi_info, is_opapi, is_op_valid, CompositeImplicitAutograd_except_list, PathManager)
from codegen.custom_functions import (parse_custom_yaml, gen_custom_trace, gen_custom_ops_patch,
                                      gen_custom_functions_dispatch)


# Create backend_indices map for func retrieval with the key of each func we supported.
def create_backend_index(backend_ops: List[str],
                         symint_ops: Set[str],
                         dispatch_key: DispatchKey,
                         native_funcs_map: Dict[OperatorName, NativeFunction],
                         cpp_namespace: str,
                         ) -> BackendIndex:
    metadata: Dict[OperatorName, BackendMetadata] = {}
    for op in backend_ops:
        op_name = OperatorName.parse(op)
        if op_name not in native_funcs_map:
            raise KeyError(f"Found an invalid operator name: {op_name}")
        # See Note [External Backends Follow Dispatcher API]
        kernel_name = dispatcher.name(native_funcs_map[op_name].func)
        if op in symint_ops:
            kernel_name += "_symint"
        m = BackendMetadata(kernel=kernel_name, structured=False, cpp_namespace=cpp_namespace)
        metadata[op_name] = m
    return BackendIndex(
        dispatch_key=dispatch_key,
        use_out_as_primary=False,
        external=True,
        device_guard=False,
        index=metadata)


# Check whether the function is placed at the wrong place.
def check_grouped_native_functions(
        backend_key: DispatchKey,
        autograd_key: DispatchKey,
        backend_indices: Dict[DispatchKey, BackendIndex],
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]]):
    for g in grouped_native_functions:
        if isinstance(g, NativeFunction):
            forward_kernels = [] if backend_key is None else \
                [m for m in [backend_indices[backend_key].get_kernel(g)] if m is not None]
            backward_kernels = [] if autograd_key is None else \
                [m for m in [backend_indices[autograd_key].get_kernel(g)] if m is not None]
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


_GLOBAL_PARSE_NATIVE_YAML_CACHE = {}

# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
ParsedYaml = namedtuple('ParsedYaml', ['native_functions', 'backend_indices'])


def parse_native_and_custom_yaml(path: str, tag_path: str, custom_path: str) -> ParsedYaml:
    global _GLOBAL_PARSE_NATIVE_YAML_CACHE
    if path not in _GLOBAL_PARSE_NATIVE_YAML_CACHE:
        valid_tags = parse_tags_yaml(tag_path)
        PathManager.check_directory_path_readable(path)
        with open(path, 'r') as f:
            es = yaml.safe_load(f)
        if not isinstance(es, list):
            raise TypeError("es is not list")
        rs: List[NativeFunction] = []
        bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(dict)
        for e in es:
            func, m = NativeFunction.from_yaml(e, "Location", valid_tags)
            rs.append(func)
            BackendIndex.grow_index(bs, m)

        source_es = parse_npu_yaml(custom_path)
        custom_es = source_es.get('custom', []) + source_es.get('custom_autograd', [])
        supported_es = source_es.get('supported', []) + source_es.get('autograd', []) + custom_es
        for es in supported_es:
            update_opapi_info(es)
        custom_es = filed_tag(custom_es)
        for e in custom_es:
            func, m = NativeFunction.from_yaml(e, "Location", valid_tags)
            rs.append(func)
            BackendIndex.grow_index(bs, m)

        error_check_native_functions(rs)
        # Default dict is to prevent the codegen from barfing when we have a dispatch key that has no kernels yet.
        indices: Dict[DispatchKey, BackendIndex] = defaultdict(lambda: BackendIndex(
            dispatch_key=DispatchKey.Undefined,
            use_out_as_primary=True,
            device_guard=False,
            external=False,
            index={}))
        add_generated_native_functions(rs, bs)
        for k, v in bs.items():
            # All structured in-tree operators are implemented in terms of their out operator.
            indices[k] = BackendIndex(dispatch_key=k,
                                      use_out_as_primary=True,
                                      external=False,
                                      device_guard=is_cuda_dispatch_key(k),
                                      index=v)
        _GLOBAL_PARSE_NATIVE_YAML_CACHE[path] = ParsedYaml(rs, indices)

    return _GLOBAL_PARSE_NATIVE_YAML_CACHE[path]


# Parses the external backend's yaml, and adds a new BackendIndex for the backend's dispatch key.
# Returns a Tuple of (true_backend, backend_key, autograd_key, cpp_namespace, updated BackendIndex mapping)
ParsedExternalYaml = namedtuple('ParsedExternalYaml', [
    'true_backend', 'backend_key', 'autograd_key', 'cpp_namespace', 'backend_indices', 'header_indices'])


def parse_backend_yaml(
        native_yaml_path: str,
        backend_yaml_path: str,
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        backend_indices: Dict[DispatchKey, BackendIndex]
) -> ParsedExternalYaml:

    native_functions_map = {}
    for f in grouped_native_functions:
        if isinstance(f, NativeFunction):
            native_functions_map[f.func.name] = f
        else:
            for func in f.functions():
                native_functions_map[func.func.name] = func

    PathManager.check_directory_path_readable(backend_yaml_path)
    with open(backend_yaml_path, 'r') as f:
        yaml_values = yaml.safe_load(f)
    if not isinstance(yaml_values, dict):
        raise TypeError("yaml_values is not dict")

    valid_keys = ['backend', 'cpp_namespace', 'tocpu', 'supported', 'autograd', 'custom', 'custom_autograd', 'symint']

    yaml_backend = yaml_values.pop('backend', None)
    true_backend = 'PrivateUse1' if yaml_backend == 'NPU' else yaml_backend
    if true_backend is None:
        raise ValueError("You must provide a value for 'backend'")
    backend = "NPU"

    cpp_namespace = yaml_values.pop('cpp_namespace', None)
    if cpp_namespace is None:
        raise ValueError("You must provide a value for 'cpp_namespace'")

    supported = yaml_values.pop('supported', [])
    if supported is None:
        supported = []  # Allow an empty list of supported ops
    if not isinstance(supported, list):
        raise TypeError(f'expected "supported" to be a list, but got type {type(supported)}')

    symint = yaml_values.pop("symint", [])
    if symint is None:
        symint = []
    if not (isinstance(symint, list)):
        raise RuntimeError(f'expected "symint" to be a list, but got: {supported} (of type {type(supported)})')
    symint = [op['func'].split("(")[0] if isinstance(op, Dict) else op for op in symint]
    symint_set = set(symint)

    supported_autograd = yaml_values.pop('autograd', [])
    if not isinstance(supported_autograd, list):
        raise TypeError(f'expected "autograd" to be a list, but got: {supported_autograd}')
    supported = [op['func'].split("(")[0] if isinstance(op, Dict) else op for op in supported]
    supported_autograd = [op['func'].split("(")[0] if isinstance(op, Dict) else op for op in supported_autograd]

    # Move the native ops with dispatchkey CompositeImplicitAutograd from supported into supported_autograd
    PathManager.check_directory_path_readable(native_yaml_path)
    with open(native_yaml_path, 'r') as f:
        native_yaml_values = yaml.safe_load(f)
    if isinstance(native_yaml_values, list):
        for op in native_yaml_values:
            if 'structured_delegate' not in op and \
                ('dispatch' not in op or 'CompositeImplicitAutograd' in op['dispatch']):
                op_name = op['func'].split('(')[0]
                if op_name in supported and op_name not in CompositeImplicitAutograd_except_list:
                    supported_autograd.append(op_name)

    supported_tocpu = yaml_values.pop('tocpu', [])
    if not isinstance(supported_tocpu, list):
        raise TypeError(f'expected "tocpu" to be a list, but got: {supported_tocpu}')

    custom = yaml_values.pop('custom', [])
    if not isinstance(custom, list):
        raise TypeError(f'expected "autograd" to be a list, but got: {custom}')

    for item in custom:
        try:
            supported.append(item['func'][:item['func'].index('(')])
        except ValueError as e:
            raise Exception(f'Wrong format for function: {item["func"]}') from e

    custom_autograd = yaml_values.pop('custom_autograd', [])
    if not isinstance(custom_autograd, list):
        raise TypeError(f'expected "autograd" to be a list, but got: {custom_autograd}')
    for item in custom_autograd:
        supported_autograd.append(item['func'][:item['func'].index('(')])

    header_op = list(set(supported + supported_autograd))

    if (len(yaml_values.keys()) > 0):
        raise KeyError(f'{backend_yaml_path} contains unexpected keys: {", ".join(yaml_values.keys())}. \
                       Only the following keys are supported: {", ".join(valid_keys)}')

    header_indices = copy.deepcopy(backend_indices)
    backend_key: Optional[DispatchKey] = None
    opapi_key = "OpApi"
    if len(supported) > 0:
        with context(lambda: f'The provided value for "backend" must be a valid DispatchKey, but got {backend}.'):
            backend_key = DispatchKey.parse(backend)

        backend_idx = create_backend_index(supported, symint_set, backend_key, native_functions_map, cpp_namespace)
        opapi_backend_idx = create_backend_index([op for op in supported if is_opapi(op)],
                                                 symint_set, backend_key, native_functions_map, cpp_namespace)
        if backend_key in backend_indices:
            raise KeyError("backend_key should not be in backend_indices.")
        backend_indices[backend_key] = backend_idx
        backend_indices[str(backend_key) + opapi_key] = opapi_backend_idx

    autograd_key: Optional[DispatchKey] = None
    if len(supported_autograd) > 0:
        with context(lambda: f'The "autograd" key was specified, which indicates that you would like to override \
the behavior of autograd for some operators on your backend. However "Autograd{backend}" is not a valid DispatchKey.'):
            autograd_key = DispatchKey.parse(f'Autograd{backend}')

        autograd_idx = create_backend_index(supported_autograd, symint_set,
                                            autograd_key, native_functions_map, cpp_namespace)
        opapi_autograd_idx = create_backend_index([op for op in supported_autograd if is_opapi(op)],
                                                  symint_set, autograd_key, native_functions_map, cpp_namespace)
        if autograd_key in backend_indices:
            raise KeyError("autograd_key should not be in backend_indices.")
        backend_indices[autograd_key] = autograd_idx
        backend_indices[str(autograd_key) + opapi_key] = opapi_autograd_idx

    if len(header_op) > 0:
        backend_idx = create_backend_index([op for op in header_op if (not is_op_valid(op))],
                                           symint_set, backend_key, native_functions_map, cpp_namespace)
        opapi_backend_idx = create_backend_index([op for op in header_op if (is_opapi(op) and (not is_op_valid(op)))],
                                                 symint_set, backend_key, native_functions_map, cpp_namespace)
        header_indices[backend_key] = backend_idx
        header_indices[str(backend_key) + opapi_key] = opapi_backend_idx

    check_op_on_cpu_kernels(supported_tocpu, backend_indices)
    check_grouped_native_functions(backend_key, autograd_key, backend_indices, grouped_native_functions)
    check_grouped_native_functions(backend_key, autograd_key, header_indices, grouped_native_functions)
    return ParsedExternalYaml(true_backend, backend_key, autograd_key, cpp_namespace, backend_indices, header_indices)


def check_op_on_cpu_kernels(
        expected_to_cpu: List,
        backend_indices: Dict[DispatchKey, BackendIndex]):
    op_names: List[OperatorName] = list(backend_indices[DispatchKey.CPU].index.keys())

    for op_name in op_names:
        if op_name.name.base not in expected_to_cpu:
            backend_indices[DispatchKey.CPU].index.pop(op_name, None)


def op_plugin_kernel_conut(op_plugin_ops_dir: str):
    actual_backend_kernel_name_counts = Counter()
    file_path = os.path.join(op_plugin_ops_dir, "OpInterface.h")
    PathManager.check_directory_path_readable(file_path)
    try:
        with open(file_path, 'r') as f:
            backend_defns = f.read()
    except IOError as e:
        raise AssertionError(f'Unable to read from the specified impl_path file: {file_path}') from e

    kernel_defn_regex = rf'\w+(?=\()'
    actual_backend_kernel_name_counts += Counter(re.findall(kernel_defn_regex, backend_defns))
    return actual_backend_kernel_name_counts


def pta_kernel_conut(class_name: str, pta_op_dir: str):
    actual_backend_kernel_name_counts = Counter()
    for cur_dir, _, filenames in os.walk(pta_op_dir):
        for filename in filenames:
            if not filename.endswith('.cpp'):
                continue
            file_path = os.path.join(cur_dir, filename)
            PathManager.check_directory_path_readable(file_path)
            try:
                with open(file_path, 'r') as f:
                    backend_defns = f.read()
            except IOError:
                raise AssertionError(f'Unable to read from the specified impl_path file: {file_path}')

            kernel_defn_regex = rf'{class_name}::([\w\d]*)\([^\)]*\)\s*{{'
            actual_backend_kernel_name_counts += Counter(re.findall(kernel_defn_regex, backend_defns))
    return actual_backend_kernel_name_counts


def check_op_plugin_kernels(
        native_functions: Sequence[NativeFunction],
        expected_kernel_counts: Dict[str, List[NativeFunction]],
        actual_kernel_counts: Dict[str, List[NativeFunction]]):
    for f in native_functions:
        wrap_name = get_opplugin_wrap_name(f)
        expect_op_plugin_kernel_count = len(expected_kernel_counts[wrap_name])
        if expect_op_plugin_kernel_count > actual_kernel_counts[wrap_name]:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate backend stub files')
    parser.add_argument(
        '--to_cpu', type=str, default="TRUE", help='move op which npu does not support to cpu')
    parser.add_argument(
        '-s',
        '--source_yaml',
        help='path to source yaml file containing operator external definitions')
    parser.add_argument(
        '-o', '--output_dir', help='output directory')
    parser.add_argument(
        '--dry_run', type=bool, default=False, help='output directory')
    parser.add_argument(
        '--impl_path', type=str, default=None, help='path to the source C++ file containing kernel definitions')
    parser.add_argument(
        '--op_plugin_impl_path', type=str, default=None,
        help='path to the source C++ file containing kernel definitions in op_plugin')
    parser.add_argument(
        '--op_plugin_yaml_path', type=str, default=None,
        help='path to the source yaml file containing kernel definitions in op_plugin')
    options = parser.parse_args()

    run(options.to_cpu, options.source_yaml, options.output_dir, options.dry_run,
        options.impl_path, options.op_plugin_impl_path, options.op_plugin_yaml_path)


def gen_dispatcher_registrations(
        fm: FileManager,
        class_name: str,
        backend_indices: Dict[DispatchKey, BackendIndex],
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        backend_dispatch_key: DispatchKey,
        dispatch_key: DispatchKey,
        selector: "SelectiveBuilder",
        dispatch_key_name: str,
        register_dispatch_key_func: Callable,
):
    backend_index = backend_indices[backend_dispatch_key]
    ns_helper = NamespaceHelper(namespace_str="at")
    native_func_header = """\
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/profiler/utils.h"
#endif

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/NPUOpApiNativeFunctions.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/utils/ForceAclnnList.h"
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
    fm.write_with_template(f'Register{dispatch_key}.cpp', 'RegisterDispatchKey.cpp', lambda: {
        'extra_cuda_headers': '',
        'external_backend_headers': native_func_header,
        'namespaced_headers': '',
        'DispatchKey': dispatch_key,
        'dispatch_headers': dest.gen_registration_headers(
            backend_index, per_operator_headers=False, rocm=False
        ),
        'ops_headers': '',
        'dispatch_definitions': fm.substitute_with_template(
            'RegisterDispatchDefinitions.ini',
            lambda: {
                'ns_prologue': ns_helper.prologue,
                'ns_epilogue': ns_helper.epilogue,
                'static_init_dispatch_registrations': static_init_dispatch_registrations,
                'deferred_dispatch_registrations': '',
                'dispatch_helpers': dest.gen_registration_helpers(backend_index),
                'dispatch_namespace': dispatch_key.lower(),
                'dispatch_namespaced_definitions': '',
                'dispatch_anonymous_definitions': list(
                    concatMap(
                        register_dispatch_key_func(
                            backend_index,
                            Target.ANONYMOUS_DEFINITION,
                            selector,
                            rocm=False,
                            symint=True,
                            class_method_name=f'{class_name}',
                            skip_dispatcher_op_registration=False,
                        ),
                        grouped_native_functions,
                    )
                ),
            },
        ).split('\n'),
    })


def get_supported_grouped_native_functions(
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        backend_index: BackendIndex,
        ) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
    supported_grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]] = []
    for funcs in grouped_native_functions:
        if isinstance(funcs, NativeFunctionsGroup) and not backend_index.has_kernel(funcs.out):
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
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    backend_indices: BackendIndex,
):
    cpu_backend_indices = parse_native_yaml(native_yaml_path, tags_yaml_path).backend_indices[DispatchKey.CPU]
    foreach_dict: Dict[str, str] = {}
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
    fm.write_with_template(f'ForeachRegister.cpp', 'ForeachRegister.cpp', lambda: {
        'include_headers': [header_template.substitute(function=h) for h in header_set if h.startswith("_foreach")],
        'foreach_kernel': [kernel_template.substitute(schema=kv[0], kernel=kv[1]) for kv in foreach_dict.items()]
    })


def run(to_cpu: str, source_yaml: str, output_dir: str, dry_run: bool,
        impl_path: Optional[str], op_plugin_impl_path: Optional[str], op_plugin_yaml_path: Optional[str]) -> None:
    rename_privateuse1_dispatch_key()
    torchgen_path = get_torchgen_dir()

    template_dir = os.path.join(torchgen_path, "packaged/ATen/templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=dry_run)

    fm = make_file_manager(output_dir)
    merge_custom_yaml(source_yaml, op_plugin_yaml_path)
    source_yaml = gen_custom_yaml_path(source_yaml)
    tags_yaml_path = os.path.join(torchgen_path, 'packaged/ATen/native/tags.yaml')
    native_yaml_path = os.path.join(torchgen_path, 'packaged/ATen/native/native_functions.yaml')
    parsed_yaml = parse_native_and_custom_yaml(native_yaml_path, tags_yaml_path, source_yaml)
    parse_opplugin_yaml(op_plugin_yaml_path)
    native_functions, backend_indices = parsed_yaml.native_functions, parsed_yaml.backend_indices
    grouped_native_functions = get_grouped_native_functions(native_functions)
    parsed_backend_yaml = parse_backend_yaml(native_yaml_path, source_yaml, grouped_native_functions, backend_indices)
    true_backend = parsed_backend_yaml.true_backend
    backend_key = parsed_backend_yaml.backend_key
    autograd_key = parsed_backend_yaml.autograd_key
    cpp_namespace = parsed_backend_yaml.cpp_namespace
    backend_indices = parsed_backend_yaml.backend_indices
    header_indices = parsed_backend_yaml.header_indices

    selector = SelectiveBuilder.get_nop_selector()
    if backend_key is not None:
        backend_dispatch_key: DispatchKey = backend_key
        autograd_dispatch_key: DispatchKey = autograd_key
        class_name = backend_indices[backend_dispatch_key].native_function_class_name()

        gen_dispatchkey_nativefunc_headers(
            fm,
            class_name,
            cpp_namespace,
            header_indices,
            grouped_native_functions,
            backend_key,
            autograd_key,
        )

        gen_dispatchkey_nativefunc_headers(
            fm,
            "NPUNativeOpApiFunctions",
            cpp_namespace,
            backend_indices,
            grouped_native_functions,
            str(backend_key) + "OpApi",
            str(autograd_key) + "OpApi",
        )

        for dispatch_key in [backend_dispatch_key, autograd_dispatch_key]:
            if not dispatch_key:
                continue
            gen_dispatcher_registrations(
                fm,
                class_name,
                backend_indices,
                get_supported_grouped_native_functions(grouped_native_functions, backend_indices[dispatch_key]),
                dispatch_key,
                dispatch_key,
                selector,
                dispatch_key_name=dispatch_key.name.replace("NPU", true_backend),
                register_dispatch_key_func=dest.RegisterDispatchKey,
            )

        template_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "templates")
        fm = FileManager(install_dir=output_dir, template_dir=template_dir, dry_run=dry_run)
        custom_functions = parse_custom_yaml(source_yaml, tags_yaml_path).native_functions

        gen_custom_trace(fm, custom_functions)
        gen_custom_functions_dispatch(fm, custom_functions)

        gen_foreach_register(fm,
                             tags_yaml_path,
                             native_yaml_path,
                             grouped_native_functions,
                             backend_indices[backend_dispatch_key])

        custom_ops_patch_dir = os.path.join(output_dir, "../../utils/")
        fm = FileManager(install_dir=custom_ops_patch_dir, template_dir=template_dir, dry_run=dry_run)
        gen_custom_ops_patch(fm, custom_functions)


def apply_torchgen_patch():
    dest.RegisterDispatchKey.gen_unstructured = gen_unstructured
    # generate default arguments
    JIT_TO_CPP_DEFAULT["contiguous_format"] = "c10::MemoryFormat::Contiguous"
    add_header_to_template_file()
    dispatcher.arguments = native.arguments


if __name__ == '__main__':
    apply_torchgen_patch()
    main()