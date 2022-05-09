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
from typing import List, Dict, Union, Sequence, Optional
import yaml

from codegen.gen import FileManager, get_grouped_native_functions, error_check_native_functions
from codegen.model import (BackendIndex, BackendMetadata, DispatchKey,
                           NativeFunction, NativeFunctionsGroup, OperatorName)
from codegen.selective_build.selector import SelectiveBuilder
from codegen.utils import Target, concat_map, context
from codegen.context import native_function_manager
import codegen.dest as dest
import codegen.api.dispatcher as dispatcher
from codegen.api.signature import DispatcherSignature


# Create backend_indices map for func retrieval with the key of each func we supported.
def create_backend_index(backend_ops: List[str],
                         dispatch_key: DispatchKey,
                         native_funcs_map: Dict[OperatorName, NativeFunction]) -> BackendIndex:
    metadata: Dict[OperatorName, BackendMetadata] = {}
    for op in backend_ops:
        op_name = OperatorName.parse(op)
        assert op_name in native_funcs_map, f"Found an invalid operator name: {op_name}"
        # See Note [External Backends Follow Dispatcher API]
        kernel_name = dispatcher.name(native_funcs_map[op_name].func)
        # TODO: allow structured external backends later.
        m = BackendMetadata(kernel=kernel_name, structured=False)
        metadata[op_name] = m
    return BackendIndex(
        dispatch_key=dispatch_key,
        use_out_as_primary=False,
        external=True,
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
def parse_native_and_custom_yaml(path: str, custom_path: str) -> ParsedYaml:
    global _GLOBAL_PARSE_NATIVE_YAML_CACHE
    if path not in _GLOBAL_PARSE_NATIVE_YAML_CACHE:
        with open(path, 'r') as f:
            es = yaml.safe_load(f)
        assert isinstance(es, list)
        rs: List[NativeFunction] = []
        bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(dict)
        for e in es:
            funcs = e.get('func')
            with context(lambda: f'in {path}:\n  {funcs}'):
                func, m = NativeFunction.from_yaml(e)
                rs.append(func)
                BackendIndex.grow_index(bs, m)

        # Filter the custom native yaml file, and extract the functions we defined.
        from io import StringIO
        f_str = StringIO()
        with open(custom_path, 'r') as f:
            for line in f:
                if line.split(':')[0] in ['backend', 'cpp_namespace', 'extra_headers',
                                          'supported', 'autograd', 'custom', 'custom_autograd']:
                    continue
                if ':' not in line:
                    continue
                f_str.write(line)

        f_str.seek(0)
        custom_es = yaml.safe_load(f_str)
        for e in custom_es:
            funcs = e.get('func')
            with context(lambda: f'in {custom_path}:\n  {funcs}'):
                func, m = NativeFunction.from_yaml(e)
                rs.append(func)
                BackendIndex.grow_index(bs, m)

        error_check_native_functions(rs)
        # Default dict is to prevent the codegen from barfing when we have a dispatch key that has no kernels yet.
        indices: Dict[DispatchKey, BackendIndex] = defaultdict(lambda: BackendIndex(
            dispatch_key=DispatchKey.Undefined, use_out_as_primary=True, external=False, index={}))
        for k, v in bs.items():
            # All structured in-tree operators are implemented in terms of their out operator.
            indices[k] = BackendIndex(dispatch_key=k, use_out_as_primary=True, external=False, index=v)
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
        for f in concat_map(lambda f: [f] if isinstance(f, NativeFunction)
                            else list(f.functions()), grouped_native_functions)
    }

    with open(backend_yaml_path, 'r') as f:
        yaml_values = yaml.safe_load(f)
    assert isinstance(yaml_values, dict)

    valid_keys = ['backend', 'cpp_namespace', 'extra_headers', 'supported', 'autograd', 'custom', 'custom_autograd']

    true_backend = yaml_values.pop('backend', None)
    assert true_backend is not None, 'You must provide a value for "backend"'
    backend = "NPU"

    cpp_namespace = yaml_values.pop('cpp_namespace', None)
    assert cpp_namespace is not None, 'You must provide a value for "cpp_namespace"'

    supported = yaml_values.pop('supported', [])
    if supported is None:
        supported = []  # Allow an empty list of supported ops
    assert isinstance(supported, list), f'expected "supported" to be a list, but got type {type(supported)}'

    supported_autograd = yaml_values.pop('autograd', [])
    assert isinstance(supported_autograd, list), f'expected "autograd" to be a list, but got: {supported_autograd}'

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

        backend_idx = create_backend_index(supported, backend_key, native_functions_map)
        assert backend_key not in backend_indices
        backend_indices[backend_key] = backend_idx

    autograd_key: Optional[DispatchKey] = None
    if len(supported_autograd) > 0:
        with context(lambda: f'The "autograd" key was specified, which indicates that you would like to override \
the behavior of autograd for some operators on your backend. However "Autograd{backend}" is not a valid DispatchKey.'):
            autograd_key = DispatchKey.parse(f'Autograd{backend}')

        autograd_idx = create_backend_index(supported_autograd, autograd_key, native_functions_map)
        assert autograd_key not in backend_indices
        backend_indices[autograd_key] = autograd_idx

    check_grouped_native_functions(backend_key, autograd_key, backend_indices, grouped_native_functions)
    return ParsedExternalYaml(true_backend, backend_key, autograd_key, cpp_namespace, backend_indices)

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

    expected_backend_op_names: List[OperatorName] = \
        list(backend_indices[backend_key].index.keys()) + list(backend_indices[autograd_key].index.keys())
    expected_backend_native_funcs: List[NativeFunction] = \
        [f for f in native_functions if f.func.name in expected_backend_op_names]
    expected_backend_kernel_name_counts: Dict[str, List[NativeFunction]] = defaultdict(list)
    for native_f in expected_backend_native_funcs:
        expected_backend_kernel_name_counts[dispatcher.name(native_f.func)].append(native_f)

    missing_kernels_err_msg = ""
    unsupported_ops_list = ""
    for expected_name, funcs in expected_backend_kernel_name_counts.items():
        expected_overload_count = len(funcs)
        actual_overload_count = actual_backend_kernel_name_counts[expected_name]
        if actual_overload_count == 0:
            for func in funcs:
                backend_indices[backend_key].index.pop(func.func.name, None)
                backend_indices[autograd_key].index.pop(func.func.name, None)
                unsupported_ops_list += f"{func.func.name}\n"
        elif expected_overload_count != actual_overload_count:
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
    if unsupported_ops_list != "":
        print(f"Unsupported Ops List:\n{unsupported_ops_list}")

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate backend stub files')
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

    run(options.source_yaml, options.output_dir, options.dry_run, options.impl_path)

def run(source_yaml: str, output_dir: str, dry_run: bool, impl_path: Optional[str]) -> None:

    template_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=dry_run)

    fm = make_file_manager(output_dir)

    native_yaml_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'native_functions.yaml')
    parsed_yaml = parse_native_and_custom_yaml(native_yaml_path, source_yaml)
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

        assert class_name is not None
        generated_comment = 'Autogenerated file by gen_backend_stubs.py. Do not edit directly!'
        fm.write_with_template(f'{backend_dispatch_key}NativeFunctions.h', 'DispatchKeyNativeFunctions.h', lambda: {
            'generated_comment': generated_comment,
            'cpp_namespace': cpp_namespace,
            'class_name': class_name,
            # Convert to a set first to remove duplicate kernel names.
            # Backends are allowed to repeat kernel names; only generate the declaration once!
            'dispatch_declarations': list(set(concat_map(
                lambda f: dest.compute_native_function_declaration(f, backend_indices[backend_dispatch_key]),
                grouped_native_functions
            ))) + list(set(concat_map(
                lambda f: dest.compute_native_function_declaration(f, backend_indices[autograd_dispatch_key]),
                grouped_native_functions
            ))) if autograd_dispatch_key else list(set(concat_map(
                lambda f: dest.compute_native_function_declaration(f, backend_indices[backend_dispatch_key]),
                grouped_native_functions
            ))),
        })

        for dispatch_key in [backend_dispatch_key, autograd_dispatch_key]:
            if not dispatch_key:
                continue

            native_func_header = f'#include "torch_npu/csrc/aten/NPUNativeFunctions.h"'
            fm.write_with_template(f'Register{dispatch_key}.cpp', 'RegisterDispatchKey.cpp', lambda: {
                'external_backend_headers': native_func_header,
                'namespaced_headers': '',
                'DispatchKey': dispatch_key.name.replace("NPU", true_backend),
                'dispatch_namespace': dispatch_key.lower(),
                'dispatch_helpers': dest.gen_registration_helpers(backend_indices[dispatch_key]),
                'dispatch_namespaced_definitions': list(concat_map(
                    dest.RegisterDispatchKey(
                        backend_indices[dispatch_key],
                        Target.NAMESPACED_DEFINITION,
                        selector,
                        rocm=False,
                        cpp_namespace=cpp_namespace,
                        class_method_name=f'{backend_dispatch_key}NativeFunctions'),
                    grouped_native_functions
                )),
                'dispatch_anonymous_definitions': list(concat_map(
                    dest.RegisterDispatchKey(
                        backend_indices[dispatch_key],
                        Target.ANONYMOUS_DEFINITION,
                        selector,
                        rocm=False,
                        cpp_namespace=cpp_namespace,
                        class_method_name=f'{backend_dispatch_key}NativeFunctions'),
                    grouped_native_functions
                )),
                'dispatch_registrations': list(concat_map(
                    dest.RegisterDispatchKey(
                        backend_indices[dispatch_key],
                        Target.REGISTRATION,
                        selector,
                        rocm=False,
                        cpp_namespace=cpp_namespace,
                        class_method_name=f'{backend_dispatch_key}NativeFunctions'),
                    grouped_native_functions
                )),
            })

if __name__ == '__main__':
    main()
