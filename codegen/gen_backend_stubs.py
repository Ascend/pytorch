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
from collections import namedtuple, Counter, defaultdict
from typing import List, Dict, Union, Sequence, Optional, Set, Callable
import yaml

from torchgen.code_template import CodeTemplate
from torchgen.gen import (parse_tags_yaml, LineLoader, FileManager,
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
                           add_header_to_template_file, parse_npu_yaml)
from codegen.custom_functions import parse_custom_yaml, gen_custom_trace, gen_custom_ops_patch


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
        assert op_name in native_funcs_map, f"Found an invalid operator name: {op_name}"
        # See Note [External Backends Follow Dispatcher API]
        kernel_name = dispatcher.name(native_funcs_map[op_name].func)
        if op in symint_ops:
            kernel_name += "_symint"
        # TODO: allow structured external backends later.
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
            forward_kernels = [] if backend_key is None else [m for m in [
                backend_indices[backend_key].get_kernel(f) for f in g.functions()]
                if m is not None]
            backward_kernels = [] if autograd_key is None else [m for m in [
                backend_indices[autograd_key].get_kernel(f) for f in g.functions()]
                if m is not None]

        forward_kernels = [f for f in forward_kernels if f is not None]
        backward_kernels = [f for f in backward_kernels if f is not None]
        assert len(forward_kernels) == 0 or len(backward_kernels) == 0, \
            f'Currently, all variants of an op must either be registered to a backend key, or to a backend\'s \
autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! \
{forward_kernels[0].kernel} is listed under "supported", but {backward_kernels[0].kernel} is listed under "autograd".'


_GLOBAL_PARSE_NATIVE_YAML_CACHE = {}

# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
ParsedYaml = namedtuple('ParsedYaml', ['native_functions', 'backend_indices'])


def parse_native_and_custom_yaml(path: str, tag_path: str, custom_path: str) -> ParsedYaml:
    global _GLOBAL_PARSE_NATIVE_YAML_CACHE
    if path not in _GLOBAL_PARSE_NATIVE_YAML_CACHE:
        valid_tags = parse_tags_yaml(tag_path)
        with open(path, 'r') as f:
            es = yaml.load(f, Loader=LineLoader)
        assert isinstance(es, list)
        rs: List[NativeFunction] = []
        bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(dict)
        for e in es:
            funcs = e.get('func')
            loc = Location(path, e["__line__"])
            with context(lambda: f'in {loc}:\n  {funcs}'):
                func, m = NativeFunction.from_yaml(e, loc, valid_tags)
                rs.append(func)
                BackendIndex.grow_index(bs, m)

        source_es = parse_npu_yaml(custom_path)
        custom_es = source_es['custom'] + source_es['custom_autograd']

        for e in custom_es:
            if e.get('wrap_impl'):
                del e['wrap_impl']
            funcs = e.get('func')
            loc = Location(custom_path, e["__line__"])
            with context(lambda: f'in {loc}:\n  {funcs}'):
                func, m = NativeFunction.from_yaml(e, loc, valid_tags)
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
    'true_backend', 'backend_key', 'autograd_key', 'cpp_namespace', 'backend_indices'])


def parse_backend_yaml(
        backend_yaml_path: str,
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        backend_indices: Dict[DispatchKey, BackendIndex]
) -> ParsedExternalYaml:

    native_functions_map: Dict[OperatorName, NativeFunction] = {
        f.func.name: f
        for f in concatMap(lambda f: [f] if isinstance(f, NativeFunction)
                            else list(f.functions()), grouped_native_functions)
    }

    with open(backend_yaml_path, 'r') as f:
        yaml_values = yaml.safe_load(f)
    assert isinstance(yaml_values, dict)

    valid_keys = ['backend', 'cpp_namespace', 'tocpu', 'supported', 'autograd', 'custom', 'custom_autograd', 'symint']

    yaml_backend = yaml_values.pop('backend', None)
    true_backend = 'PrivateUse1' if yaml_backend == 'NPU' else yaml_backend
    assert true_backend is not None, 'You must provide a value for "backend"'
    backend = "NPU"

    cpp_namespace = yaml_values.pop('cpp_namespace', None)
    assert cpp_namespace is not None, 'You must provide a value for "cpp_namespace"'

    supported = yaml_values.pop('supported', [])
    if supported is None:
        supported = []  # Allow an empty list of supported ops
    assert isinstance(supported, list), f'expected "supported" to be a list, but got type {type(supported)}'

    symint = yaml_values.pop("symint", [])
    if symint is None:
        symint = []
    assert isinstance(
        symint, list
    ), f'expected "symint" to be a list, but got: {supported} (of type {type(supported)})'
    symint = [op['func'] if isinstance(op, Dict) else op for op in symint]  
    symint_set = set(symint)

    supported_autograd = yaml_values.pop('autograd', [])
    assert isinstance(supported_autograd, list), f'expected "autograd" to be a list, but got: {supported_autograd}'

    supported = [op['func'] if isinstance(op, Dict) else op for op in supported]
    supported_autograd = [op['func'] if isinstance(op, Dict) else op for op in supported_autograd]

    supported_tocpu = yaml_values.pop('tocpu', [])
    assert isinstance(supported_tocpu, list), f'expected "tocpu" to be a list, but got: {supported_tocpu}'

    custom = yaml_values.pop('custom', [])
    assert isinstance(custom, list), f'expected "autograd" to be a list, but got: {custom}'
    for item in custom:
        try:
            supported.append(item['func'][:item['func'].index('(')])
        except ValueError:
            raise Exception(f'Wrong format for function: {item["func"]}')

    custom_autograd = yaml_values.pop('custom_autograd', [])
    assert isinstance(custom_autograd, list), f'expected "autograd" to be a list, but got: {custom_autograd}'
    for item in custom_autograd:
        supported_autograd.append(item['func'][:item['func'].index('(')])

    assert len(yaml_values.keys()) == 0, \
        f'{backend_yaml_path} contains unexpected keys: {", ".join(yaml_values.keys())}. \
Only the following keys are supported: {", ".join(valid_keys)}'

    backend_key: Optional[DispatchKey] = None
    if len(supported) > 0:
        with context(lambda: f'The provided value for "backend" must be a valid DispatchKey, but got {backend}.'):
            backend_key = DispatchKey.parse(backend)

        backend_idx = create_backend_index(supported, symint_set, backend_key, native_functions_map, cpp_namespace)
        assert backend_key not in backend_indices
        backend_indices[backend_key] = backend_idx

    autograd_key: Optional[DispatchKey] = None
    if len(supported_autograd) > 0:
        with context(lambda: f'The "autograd" key was specified, which indicates that you would like to override \
the behavior of autograd for some operators on your backend. However "Autograd{backend}" is not a valid DispatchKey.'):
            autograd_key = DispatchKey.parse(f'Autograd{backend}')

        autograd_idx = create_backend_index(supported_autograd, symint_set,
                                            autograd_key, native_functions_map, cpp_namespace)
        assert autograd_key not in backend_indices
        backend_indices[autograd_key] = autograd_idx

    check_op_on_cpu_kernels(supported_tocpu, backend_indices)
    check_grouped_native_functions(backend_key, autograd_key, backend_indices, grouped_native_functions)
    return ParsedExternalYaml(true_backend, backend_key, autograd_key, cpp_namespace, backend_indices)


def check_op_on_cpu_kernels(
        expected_to_cpu: List,
        backend_indices: Dict[DispatchKey, BackendIndex]):
    op_names: List[OperatorName] = list(backend_indices[DispatchKey.CPU].index.keys())

    for op_name in op_names:
        if op_name.name.base not in expected_to_cpu:
            backend_indices[DispatchKey.CPU].index.pop(op_name, None)


# Double-check the functions we supported to see whether there exists something mismatch.
def error_on_missing_kernels(
        native_functions: Sequence[NativeFunction],
        backend_indices: Dict[DispatchKey, BackendIndex],
        backend_key: DispatchKey,
        autograd_key: DispatchKey,
        kernel_def_file_path: str,
) -> None:
    class_name: Optional[str] = backend_indices[backend_key].native_function_class_name()
    assert class_name is not None

    actual_backend_kernel_name_counts = Counter()
    for cur_dir, _, filenames in os.walk(kernel_def_file_path):
        for filename in filenames:
            if not filename.endswith('.cpp'):
                continue
            file_path = os.path.join(cur_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    backend_defns = f.read()
            except IOError:
                raise AssertionError(f'Unable to read from the specified impl_path file: {file_path}')

            kernel_defn_regex = rf'{class_name}::([\w\d]*)\([^\)]*\)\s*{{'
            actual_backend_kernel_name_counts += Counter(re.findall(kernel_defn_regex, backend_defns))

    expected_backend_op_names: Dict[OperatorName, str] = dict(
        list(
            concatMap(
                lambda index: [(op_name, metadata.kernel) for op_name, metadata in index.items()],
                [backend_indices[backend_key].index] + [backend_indices[autograd_key].index],
            )
        )
    )
    expected_backend_native_funcs: List[NativeFunction] = \
        [f for f in native_functions if f.func.name in expected_backend_op_names]
    expected_backend_kernel_name_counts: Dict[str, List[NativeFunction]] = defaultdict(list)
    for native_f in expected_backend_native_funcs:
        expected_backend_kernel_name_counts[expected_backend_op_names[native_f.func.name]].append(native_f)

    missing_kernels_err_msg = ""
    for expected_name, funcs in expected_backend_kernel_name_counts.items():
        expected_overload_count = len(funcs)
        actual_overload_count = actual_backend_kernel_name_counts[expected_name]
        if expected_overload_count != actual_overload_count:
            def create_decl(f: NativeFunction) -> str:
                with native_function_manager(f):
                    return DispatcherSignature.from_schema(f.func).decl()
            expected_schemas_str = '\n'.join([create_decl(f) for f in funcs])
            missing_kernels_err_msg += f"""
{class_name} is missing a kernel definition for {expected_name}. We found {actual_overload_count} kernel(s) with that name,
but expected {expected_overload_count} kernel(s). The expected function schemas for the missing operator are:
{expected_schemas_str}
"""
    assert missing_kernels_err_msg == "", missing_kernels_err_msg


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
    options = parser.parse_args()

    run(options.to_cpu, options.source_yaml, options.output_dir, options.dry_run, options.impl_path)


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
#ifdef USE_OPPLUGIN
#include "op_plugin/ops/OpInterface.h"
#endif
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


def run(to_cpu: str, source_yaml: str, output_dir: str, dry_run: bool, impl_path: Optional[str]) -> None:
    rename_privateuse1_dispatch_key()
    torchgen_path = get_torchgen_dir()

    template_dir = os.path.join(torchgen_path, "packaged/ATen/templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=dry_run)

    fm = make_file_manager(output_dir)

    tags_yaml_path = os.path.join(torchgen_path, 'packaged/ATen/native/tags.yaml')
    native_yaml_path = os.path.join(torchgen_path, 'packaged/ATen/native/native_functions.yaml')
    parsed_yaml = parse_native_and_custom_yaml(native_yaml_path, tags_yaml_path, source_yaml)
    native_functions, backend_indices = parsed_yaml.native_functions, parsed_yaml.backend_indices
    grouped_native_functions = get_grouped_native_functions(native_functions)
    parsed_backend_yaml = parse_backend_yaml(source_yaml, grouped_native_functions, backend_indices)
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

        if impl_path is not None:
            error_on_missing_kernels(native_functions, backend_indices, backend_key, autograd_key, impl_path)
        gen_dispatchkey_nativefunc_headers(
            fm,
            class_name,
            cpp_namespace,
            backend_indices,
            grouped_native_functions,
            backend_key,
            autograd_key,
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
        custom_trace_functions = parse_custom_yaml(source_yaml, tags_yaml_path).native_functions
        gen_custom_trace(fm, custom_trace_functions)

        fm = FileManager(install_dir=output_dir+"../../../utils/", template_dir=template_dir, dry_run=dry_run)
        gen_custom_ops_patch(fm, custom_trace_functions)


def apply_torchgen_patch():
    dest.RegisterDispatchKey.gen_unstructured = gen_unstructured
    # generate default arguments
    JIT_TO_CPP_DEFAULT["contiguous_format"] = "c10::MemoryFormat::Contiguous"
    add_header_to_template_file()
    dispatcher.arguments = native.arguments


if __name__ == '__main__':
    apply_torchgen_patch()
    main()
