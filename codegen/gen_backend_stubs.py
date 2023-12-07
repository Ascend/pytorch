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
import stat
import re
from collections import namedtuple, Counter, defaultdict
from typing import List, Dict, Union, Sequence, Optional
import yaml

from codegen.gen import FileManager, get_grouped_native_functions, error_check_native_functions
from codegen.model import (BackendIndex, BackendMetadata, DispatchKey,
                           NativeFunction, NativeFunctionsGroup, OperatorName)
from codegen.selective_build.selector import SelectiveBuilder
from codegen.utils import (Target, concat_map, context, parse_npu_yaml, filt_exposed_api,
                           get_opplugin_wrap_name, parse_opplugin_yaml,
                           merge_custom_yaml, gen_custom_yaml_path, field_tag, PathManager)
import codegen.dest as dest
import codegen.dest.utils as utils
import codegen.api.dispatcher as dispatcher
from codegen.custom_functions import gen_custom_functions, gen_custom_registration, parse_custom_yaml


# Create backend_indices map for func retrieval with the key of each func we supported.
def create_backend_index(backend_ops: List[str],
                         dispatch_key: DispatchKey,
                         native_funcs_map: Dict[OperatorName, NativeFunction]) -> BackendIndex:
    metadata: Dict[OperatorName, BackendMetadata] = {}
    for op in backend_ops:
        op_name = OperatorName.parse(op)
        if op_name not in native_funcs_map:
            raise KeyError(f"Found an invalid operator name: {op_name}")
        # See Note [External Backends Follow Dispatcher API]
        kernel_name = dispatcher.name(native_funcs_map[op_name].func)
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
        if not (len(forward_kernels) == 0 or len(backward_kernels) == 0):
            raise ValueError(f'Currently, all variants of an op must either be registered to a backend key, or to a backend\'s \
                             autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! \
                             {forward_kernels[0].kernel} is listed under "supported", but {backward_kernels[0].kernel} is listed under "autograd".')


_GLOBAL_PARSE_NATIVE_YAML_CACHE = {}

# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
ParsedYaml = namedtuple('ParsedYaml', ['native_functions', 'backend_indices'])


def parse_native_and_custom_yaml(path: str, custom_path: str) -> ParsedYaml:
    global _GLOBAL_PARSE_NATIVE_YAML_CACHE
    if path not in _GLOBAL_PARSE_NATIVE_YAML_CACHE:
        # Filter the custom native yaml file, and extract the functions we defined.
        source_data = parse_npu_yaml(custom_path)
        custom_es = source_data.get('custom', []) + source_data.get('custom_autograd', [])
        custom_es = field_tag(custom_es)
        all_data = []
        need_key = ['supported', 'autograd', 'autograd', 'custom_autograd']
        for key in need_key:
            if source_data.get(key, []):
                all_data += source_data[key]
        all_data = [op for op in all_data if isinstance(op, dict)]
        PathManager.check_directory_path_readable(path)
        with open(path, 'r') as f:
            es = yaml.safe_load(f)
        if not isinstance(es, list):
            raise TypeError("es is not list.")
        rs: List[NativeFunction] = []
        bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(dict)
        for e in es:
            funcs = e.get('func')

            for op in all_data:
                if op['func'] == funcs.split('(')[0]:
                    op['func'] = funcs
                    e.update(op)
            with context(lambda: f'in {path}:\n  {funcs}'):
                func, m = NativeFunction.from_yaml(e)
                rs.append(func)
                BackendIndex.grow_index(bs, m)

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
    'true_backend', 'backend_key', 'autograd_key', 'unsupport_key', 'cpp_namespace', 'backend_indices'])


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

    PathManager.check_directory_path_readable(backend_yaml_path)
    with open(backend_yaml_path, 'r') as f:
        yaml_values = yaml.safe_load(f)
    if not isinstance(yaml_values, dict):
        raise TypeError("yaml_values is not dict.")

    valid_keys = ['backend', 'cpp_namespace', 'tocpu', 'supported', 'autograd',
                  'custom', 'custom_autograd', 'unsupported']

    yaml_backend = yaml_values.pop('backend', None)
    true_backend = 'XLA' if yaml_backend == 'NPU' else yaml_backend
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

    supported_autograd = yaml_values.pop('autograd', [])
    if supported_autograd is None:
        supported_autograd = []  # Allow an empty list of supported ops
    if not isinstance(supported_autograd, list):
        raise TypeError(f'expected "autograd" to be a list, but got: {supported_autograd}')

    supported_tocpu = yaml_values.pop('tocpu', [])
    if not isinstance(supported_tocpu, list):
        raise TypeError(f'expected "tocpu" to be a list, but got: {supported_tocpu}')

    custom = yaml_values.pop('custom', [])
    if not isinstance(custom, list):
        raise TypeError(f'expected "autograd" to be a list, but got: {custom}')

    for item in custom:
        try:
            supported.append(item['func'][:item['func'].index('(')])
        except ValueError:
            raise Exception(f'Wrong format for function: {item["func"]}')

    supported = [op['func'].split("(")[0] if isinstance(op, Dict) else op for op in supported]
    supported_autograd = [op['func'].split("(")[0] if isinstance(op, Dict) else op for op in supported_autograd]

    custom_autograd = yaml_values.pop('custom_autograd', [])
    if not isinstance(custom_autograd, list):
        raise TypeError(f'expected "autograd" to be a list, but got: {custom_autograd}')
    for item in custom_autograd:
        supported_autograd.append(item['func'][:item['func'].index('(')])

    unsupported = yaml_values.pop('unsupported', [])
    if not isinstance(unsupported, list):
        raise TypeError(f'expected "unsupported" to be a list, but got: {unsupported}')

    # Currently, symint is only supported for ops in opplugin, and is not useful here.
    yaml_values.pop('symint', [])
    # custom_supported is only supported for filt expose api, and is not useful here.
    yaml_values.pop('custom_supported', [])

    if (len(yaml_values.keys()) > 0):
        print(f'Waring: {backend_yaml_path} contains unexpected keys: {", ".join(yaml_values.keys())}. \
Only the following keys are supported: {", ".join(valid_keys)}')

    backend_key: Optional[DispatchKey] = None
    if len(supported) > 0:
        with context(lambda: f'The provided value for "backend" must be a valid DispatchKey, but got {backend}.'):
            backend_key = DispatchKey.parse(backend)

        backend_idx = create_backend_index(supported, backend_key, native_functions_map)
        if backend_key in backend_indices:
            raise KeyError("backend_key should not be in backend_indices.")
        backend_indices[backend_key] = backend_idx

    autograd_key: Optional[DispatchKey] = None
    if len(supported_autograd) > 0:
        with context(lambda: f'The "autograd" key was specified, which indicates that you would like to override \
the behavior of autograd for some operators on your backend. However "Autograd{backend}" is not a valid DispatchKey.'):
            autograd_key = DispatchKey.parse(f'Autograd{backend}')

        autograd_idx = create_backend_index(supported_autograd, autograd_key, native_functions_map)
        if autograd_key in backend_indices:
            raise KeyError("autograd_key should not be in backend_indices.")
        backend_indices[autograd_key] = autograd_idx

    unsupported_key: Optional[DispatchKey] = None
    if len(unsupported) > 0:
        with context(lambda: f'"Unsupport" is not a valid DispatchKey.'):
            unsupport_key = DispatchKey.parse('Unsupport')

        unsupported_idx = create_backend_index(unsupported, unsupported_key, native_functions_map)
        if unsupport_key in backend_indices:
            raise KeyError("unsupport_key should not be in backend_indices.")
        backend_indices[unsupported_key] = unsupported_idx

    check_op_on_cpu_kernels(supported_tocpu, backend_indices)
    check_grouped_native_functions(backend_key, autograd_key, backend_indices, grouped_native_functions)
    return ParsedExternalYaml(true_backend, backend_key, autograd_key, unsupported_key, cpp_namespace, backend_indices)


def check_op_on_cpu_kernels(
        expected_to_cpu: List,
        backend_indices: Dict[DispatchKey, BackendIndex]):

    op_names: List[OperatorName] = list(backend_indices[DispatchKey.CPU].index.keys())

    for op_name in op_names:
        if str(op_name) not in expected_to_cpu:
            backend_indices[DispatchKey.CPU].index.pop(op_name, None)


def op_plugin_kernel_conut(op_plugin_ops_dir: str):
    actual_backend_kernel_name_counts = Counter()
    file_path = os.path.join(op_plugin_ops_dir, "OpInterface.h")
    PathManager.check_directory_path_readable(file_path)
    try:
        with open(file_path, 'r') as f:
            backend_defns = f.read()
    except IOError:
        raise AssertionError(f'Unable to read from the specified impl_path file: {file_path}')

    kernel_defn_regex = rf'\w+(?=\()'
    actual_backend_kernel_name_counts += Counter(re.findall(kernel_defn_regex, backend_defns))
    return actual_backend_kernel_name_counts


def pta_kernel_conut(class_name: str, kernel_def_file_path: str):
    actual_backend_kernel_name_counts = Counter()
    opapi_actual_backend_kernel_name_counts = Counter()
    for cur_dir, _, filenames in os.walk(kernel_def_file_path):
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
            opapi_kernel_defn_regex = rf'NPUNativeOpApiFunctions::([\w\d]*)\([^\)]*\)\s*{{'
            opapi_actual_backend_kernel_name_counts += Counter(re.findall(opapi_kernel_defn_regex, backend_defns))

    return actual_backend_kernel_name_counts, opapi_actual_backend_kernel_name_counts


def check_op_plugin_kernels(
        native_functions: Sequence[NativeFunction],
        expected_kernel_counts: Dict[str, List[NativeFunction]],
        actual_kernel_counts: Dict[str, List[NativeFunction]]):
    for f in native_functions:
        wrap_name = get_opplugin_wrap_name(str(f.func.name))
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


def run(to_cpu: str, source_yaml: str, output_dir: str, dry_run: bool, impl_path: Optional[str],
        op_plugin_impl_path: Optional[str], op_plugin_yaml_path: Optional[str]) -> None:

    template_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=dry_run)

    fm = make_file_manager(output_dir)
    merge_custom_yaml(source_yaml, op_plugin_yaml_path)
    source_yaml = gen_custom_yaml_path(source_yaml)

    native_yaml_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'native_functions.yaml')
    parsed_yaml = parse_native_and_custom_yaml(native_yaml_path, source_yaml)
    parse_opplugin_yaml(op_plugin_yaml_path)
    native_functions, backend_indices = parsed_yaml.native_functions, parsed_yaml.backend_indices
    grouped_native_functions = get_grouped_native_functions(native_functions)
    parsed_backend_yaml = parse_backend_yaml(source_yaml, grouped_native_functions, backend_indices)
    true_backend = parsed_backend_yaml.true_backend
    utils.backend = true_backend
    backend_key = parsed_backend_yaml.backend_key
    autograd_key = parsed_backend_yaml.autograd_key
    unsupport_key = parsed_backend_yaml.unsupport_key
    cpp_namespace = parsed_backend_yaml.cpp_namespace
    backend_indices = parsed_backend_yaml.backend_indices

    selector = SelectiveBuilder.get_nop_selector()


    if backend_key is not None:
        backend_dispatch_key: DispatchKey = backend_key
        autograd_dispatch_key: DispatchKey = autograd_key
        class_name = backend_indices[backend_dispatch_key].native_function_class_name()

        if class_name is None:
            raise ValueError("class_name should not be None.")
        generated_comment = 'Autogenerated file by gen_backend_stubs.py. Do not edit directly!'
        fm.write_with_template(f'{backend_dispatch_key}NativeFunctions.h', 'DispatchKeyNativeFunctions.h', lambda: {
            'generated_comment': generated_comment,
            'cpp_namespace': cpp_namespace,
            'class_name': class_name,
            'macro': '__SCRIPTS_CODEGEN_TEMPLATES_DISPATCHKEYNATIVEFUNCTIONS__',
            'include_npu_native_functions': '',
            'static_define': '''
namespace at_npu {
namespace key {
static constexpr c10::DeviceType NativeDeviceType = c10::DeviceType::XLA;
static constexpr c10::DispatchKey NativeDispatchKey = c10::DispatchKey::XLA;
static constexpr c10::DispatchKey NativeAutogradDispatchKey = c10::DispatchKey::AutogradXLA;
static constexpr c10::Backend NativeBackend = c10::Backend::XLA;
static const std::string npu_device_str = "npu";
static const std::string default_device_str = "xla";

static bool isDeviceTensor(const at::Tensor &tensor) {
  return tensor.is_xla();
}

} // namespace key
} // namespace at_npu
''',
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

        fm.write_with_template(f'{backend_dispatch_key}NativeOpApiFunctions.h',
            'DispatchKeyNativeFunctions.h', lambda: {
            'generated_comment': generated_comment,
            'cpp_namespace': cpp_namespace,
            'class_name': 'NPUNativeOpApiFunctions',
            'macro': '__SCRIPTS_CODEGEN_TEMPLATES_DISPATCHKEYNATIVEFUNCTIONS_OPAPI__',
            'include_npu_native_functions': '''
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
''',
            'static_define': '',
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

            native_func_header = """
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/utils/ForceAclnnList.h"
#include "op_plugin/OpInterface.h"
"""
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

        dispatch_key = backend_dispatch_key
        native_func_header = f'#include "torch_npu/csrc/aten/NPUNativeFunctions.h"'
        fm.write_with_template(f'RegisterUnsupport{dispatch_key}.cpp', 'RegisterDispatchKey.cpp', lambda: {
            'external_backend_headers': native_func_header,
            'namespaced_headers': '',
            'DispatchKey': dispatch_key.name.replace("NPU", true_backend),
            'dispatch_namespace': dispatch_key.lower(),
            'dispatch_helpers': dest.gen_registration_helpers(backend_indices[unsupport_key]),
            'dispatch_namespaced_definitions': list(concat_map(
                dest.RegisterDispatchKey(
                    backend_indices[unsupport_key],
                    Target.NAMESPACED_DEFINITION,
                    selector,
                    rocm=False,
                    cpp_namespace=cpp_namespace,
                    class_method_name=f'{backend_dispatch_key}NativeFunctions'),
                grouped_native_functions
            )),
            'dispatch_anonymous_definitions': list(concat_map(
                dest.RegisterDispatchKey(
                    backend_indices[unsupport_key],
                    Target.ANONYMOUS_DEFINITION_UNSUPPORT,
                    selector,
                    rocm=False,
                    cpp_namespace=cpp_namespace,
                    class_method_name=f'{backend_dispatch_key}NativeFunctions'),
                grouped_native_functions
            )),
            'dispatch_registrations': list(concat_map(
                dest.RegisterDispatchKey(
                    backend_indices[unsupport_key],
                    Target.REGISTRATION,
                    selector,
                    rocm=False,
                    cpp_namespace=cpp_namespace,
                    class_method_name=f'{backend_dispatch_key}NativeFunctions'),
                grouped_native_functions
            )),
        })

        if to_cpu.upper() in {'OFF', '0', 'NO', 'FALSE', 'F', 'N'}:
            return

        dispatch_key = true_backend
        native_func_header = f'#include "torch_npu/csrc/aten/NPUNativeFunctions.h"\n'
        fm.write_with_template(f'RegisterCPU.cpp', 'RegisterDispatchKey.cpp', lambda: {
            'external_backend_headers': native_func_header,
            'namespaced_headers': '',
            'DispatchKey': dispatch_key,
            'dispatch_namespace': dispatch_key.lower(),
            'dispatch_helpers': dest.gen_registration_helpers(backend_indices[DispatchKey.CPU]),
            'dispatch_namespaced_definitions': list(concat_map(
                dest.RegisterDispatchKeyCPU(
                    backend_indices[DispatchKey.CPU],
                    Target.NAMESPACED_DEFINITION,
                    selector,
                    rocm=False,
                    cpp_namespace=cpp_namespace,
                    class_method_name=f'NPUNativeFunctions'),
                grouped_native_functions
            )),
            'dispatch_anonymous_definitions': list(concat_map(
                dest.RegisterDispatchKeyCPU(
                    backend_indices[DispatchKey.CPU],
                    Target.ANONYMOUS_DEFINITION,
                    selector,
                    rocm=False,
                    cpp_namespace=cpp_namespace,
                    class_method_name=f'NPUNativeFunctions'),
                grouped_native_functions
            )),
            'dispatch_registrations': list(concat_map(
                dest.RegisterDispatchKeyCPU(
                    backend_indices[DispatchKey.CPU],
                    Target.REGISTRATION,
                    selector,
                    rocm=False,
                    cpp_namespace=cpp_namespace,
                    class_method_name=f'NPUNativeFunctions'),
                grouped_native_functions
            )),
        })

        custom_functions = parse_custom_yaml(source_yaml).native_functions
        gen_custom_registration(fm, custom_functions)
        gen_custom_functions(fm, custom_functions)

        filt_exposed_list = filt_exposed_api(source_yaml)
        exposed_path = pathlib.Path(__file__).parents[1].joinpath('torch_npu/utils/exposed_api.py')
        PathManager.remove_path_safety(exposed_path)
        with os.fdopen(os.open(exposed_path, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as f:
            f.write(f'public_npu_functions = {filt_exposed_list}')
        os.chmod(exposed_path, stat.S_IRUSR | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)

if __name__ == '__main__':
    main()
