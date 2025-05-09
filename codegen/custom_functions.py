import re
import itertools
from collections import namedtuple, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Sequence

from torchgen.code_template import CodeTemplate
from torchgen.gen import (parse_tags_yaml, FileManager, cpp_string, error_check_native_functions)
from torchgen.model import (BackendIndex, DispatchKey, Variant,
                            NativeFunction, OperatorName, BackendMetadata, TensorOptionsArguments)
from torchgen.utils import concatMap, mapMaybe
from torchgen.context import with_native_function, native_function_manager, method_with_native_function
from torchgen.api.types import DispatcherSignature
from torchgen.api import cpp
from torchgen.dest.register_dispatch_key import RegisterDispatchKey
from codegen.utils import (enable_opplugin, is_op_valid, field_tag, get_opplugin_wrap_name, parse_npu_yaml)


# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
ParsedYaml = namedtuple('ParsedYaml', ['native_functions', 'backend_indices'])


CUSTOM_FUNCTIONS_DECLARATION = CodeTemplate("""\
${return_type} ${func_name}(${args_str});
""")

CUSTOM_FUNCTIONS_DEFINITION = CodeTemplate("""\
${return_type} ${func_name}(${args_str}) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::${base_name}", "${overload}").typed<${schema}>();
    return op.${func_type}(${args_exprs_str});
}
""")

SKIP_PYTHON_BINDINGS_SIGNATURES = []


@with_native_function
def should_generate_ops_patch(f: NativeFunction) -> bool:
    func_signature = str(f.func)

    for pattern in SKIP_PYTHON_BINDINGS_SIGNATURES:
        if pattern == func_signature:
            return False

    return True


def parse_custom_yaml(custom_path: str, tag_path: str) -> ParsedYaml:
    valid_tags = parse_tags_yaml(tag_path)
    rs: List[NativeFunction] = []
    bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(dict)
    # Filter the custom native yaml file, and extract the functions we defined.
    source_es = parse_npu_yaml(custom_path)
    custom_es = source_es.get('custom', []) + source_es.get('custom_autograd', [])
    custom_es = field_tag(custom_es)
    for e_with_vars in custom_es:
        func, m = NativeFunction.from_yaml(e_with_vars, "Location", valid_tags)
        func.variants.discard(Variant.method)
        rs.append(func)
        BackendIndex.grow_index(bs, m)

    error_check_native_functions(rs)
    # Default dict is to prevent the codegen from barfing when we have a dispatch key that has no kernels yet.
    indices: Dict[DispatchKey, BackendIndex] = defaultdict(lambda: BackendIndex(
        dispatch_key=DispatchKey.Undefined, use_out_as_primary=True, external=False, index={}))
    for k, v in bs.items():
        # All structured in-tree operators are implemented in terms of their out operator.
        indices[k] = BackendIndex(dispatch_key=k,
                                  use_out_as_primary=True,
                                  external=False,
                                  device_guard=False,
                                  index=v)
    return ParsedYaml(rs, indices)


METHOD_DEFINITION = CodeTemplate("""\
${return_type} ${name}(${args_str}) {
  ${unpack_out}
  ${unsafe_tensor_check}
  ${device_check}
  ${device_guard}
  ${type_definition_body}
}

""")

TRACE_DISPATCH = CodeTemplate("""\
return ${impl_name}(${args_exprs_str});""")


@with_native_function
def compute_op_definition(f: NativeFunction):
    out_num = len(f.func.arguments.out)
    sig = DispatcherSignature.from_schema(f.func, prefix=f'wrapper_{f.func.name.overload_name}_')
    name = sig.name()
    args = sig.arguments()
    args_str = ', '.join(a.defn() for a in args)

    args_exprs_str = ', '.join(a.name for a in args)

    impl_name = f"at_npu::native::NPUNativeFunctions::{cpp.name(f.func)}"

    if enable_opplugin() and is_op_valid(str(f.func.name)):
        impl_name = f"op_plugin::{get_opplugin_wrap_name(f)}"

    check_out = [f'TORCH_CHECK(out.size() == {out_num}, "expected tuple of {out_num} elements but got ", out.size(), '
                 f'OPS_ERROR(ErrCode::PARAM));']
    unpack_out = check_out + [f'at::Tensor {args[-out_num + i].name} = out[{i}];' for i in range(out_num)] \
        if out_num > 1 else ''
    out_return_type = '::std::tuple<{}>'.format(', '.join(['at::Tensor'] * out_num))

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

    unsafe_tensor_check = """ // No unsafe tensor check"""
    if len(candidate_tensor_args) > 0:
        unsafe_tensor_check = \
"""if (c10_npu::get_npu_data_unsafe_flag()) {"""
        for tensor_arg in candidate_tensor_args:
            unsafe_tensor_check = unsafe_tensor_check + f"""
    c10_npu::check_npu_tensor_is_safe({tensor_arg});"""
        unsafe_tensor_check = unsafe_tensor_check + """
}
"""
    candidate_args = itertools.chain(
        f.func.arguments.out,
        f.func.arguments.flat_positional,
        f.func.arguments.flat_kwarg_only,
    )
    device_check = RegisterDispatchKey.gen_device_check(
        f.device_check, list(candidate_args), name
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

    device_guard = ""
    if has_tensor_options and device_of is not None:
        device_guard = f"""
c10::OptionalDeviceGuard device_guard(device_of({device_of}));
if (device.has_value()) {{
device_guard.reset_device(device_or_default(device));
}}
"""
    elif has_tensor_options:
        # kernel is creating a tensor
        device_guard = """
const c10::DeviceGuard device_guard(device_or_default(device));"""
    elif device_of is not None:
        # kernel is operating on existing tensors
        device_guard = f"const c10::OptionalDeviceGuard device_guard(device_of({device_of}));"

    return [METHOD_DEFINITION.substitute(
        return_type=out_return_type if out_num > 1 else cpp.returns_type(f.func.returns).cpp_type(),
        name=name,
        args_str=','.join(a.defn() for a in args[:-out_num]) + ', at::TensorList out' if out_num > 1 else args_str,
        unpack_out=unpack_out,
        unsafe_tensor_check=unsafe_tensor_check,
        device_check=device_check,
        device_guard=device_guard,
        type_definition_body=[TRACE_DISPATCH.substitute(impl_name=impl_name, args_exprs_str=args_exprs_str)]
    )]


@dataclass(frozen=True)
class RegisterCustomSchema:
    known_tags: Dict[str, int] = field(default_factory=dict)

    @method_with_native_function
    def __call__(self, f: NativeFunction):
        out_num = len(f.func.arguments.out)
        if out_num > 1:
            decl = re.compile(r"(?P<name>[^\(]+)\((?P<args>.*)\) -> (?P<returns>.*)").findall(str(f.func))[0]
            func_schema = decl[0] + '(' + ','.join(decl[1].split(',')[:-out_num]) + ', Tensor[] out) -> (' + ', '.join(
                ['Tensor'] * out_num) + ')'
        else:
            func_schema = str(f.func)

        tags = "{" + ", ".join(f"at::Tag::{tag}" for tag in sorted(f.tags)) + "}"
        maybe_tags = ""
        if tags not in self.known_tags:
            idx = len(self.known_tags)
            self.known_tags[tags] = idx
            maybe_tags = f"const std::vector<at::Tag> tags_{idx} = {tags};\n"
        tag_index = f", tags_{self.known_tags[tags]}"
        if tags == "{}":
            tag_index = ""

        pattern = r'\bself\b(?=[,\)])'
        func_schema = re.sub(pattern, 'input', func_schema)

        if f.has_composite_explicit_autograd_kernel:
            name = DispatcherSignature.from_schema(f.func, prefix=f'wrapper_{f.func.name.overload_name}_').name()
            return f'{maybe_tags}m.def({cpp_string(func_schema)}, TORCH_FN(at_npu::native::{name}){tag_index});\n'
        else:
            return f'{maybe_tags}m.def({cpp_string(func_schema)}{tag_index});\n'


@with_native_function
def compute_register_impl(f: NativeFunction):
    if f.has_composite_explicit_autograd_kernel:
        return []
    else:
        name = DispatcherSignature.from_schema(f.func, prefix=f'wrapper_{f.func.name.overload_name}_').name()
        return [f'm.impl("{f.func.name}", TORCH_FN(at_npu::native::{name}));\n']


def gen_custom_trace(fm: FileManager, custom_trace_functions: Sequence[NativeFunction]):

    fm.write_with_template(f'CustomRegisterSchema.cpp', 'CustomRegisterSchema.cpp', lambda: {
        'custom_op_definitions': list(concatMap(
            lambda f: compute_op_definition(f),
            custom_trace_functions
        )),
        'custom_schema_registrations': list(mapMaybe(
            RegisterCustomSchema(),
            custom_trace_functions
        )),
        'custom_impl_registrations': list(concatMap(
            lambda f: compute_register_impl(f),
            custom_trace_functions
        )),
    })


def gen_custom_ops_patch(fm: FileManager, custom_trace_functions: Sequence[NativeFunction]):

    valid_native_functions = list(filter(should_generate_ops_patch, custom_trace_functions))
    fm.write_with_template(f'custom_ops.py', 'custom_ops.py', lambda: {
        'custom_ops': [f'torch_npu.{ops} = torch.ops.npu.{ops}'
                       for ops in set([f.func.name.name for f in valid_native_functions])],
    })


def compute_custom_functions_declaration(f: NativeFunction, func_type: str):
    with native_function_manager(f):
        sig = DispatcherSignature.from_schema(f.func)
        name = sig.name()
        args = sig.arguments()
        if func_type == 'call':
            args_str = ', '.join(a.decl() for a in args)
        if func_type == 'redispatch':
            args_str = 'c10::DispatchKeySet dispatchKeySet, ' + ', '.join(a.decl() for a in args)

        return [CUSTOM_FUNCTIONS_DECLARATION.substitute(
                return_type=cpp.returns_type(f.func.returns).cpp_type(),
                func_name=name,
                args_str=args_str,)]


def compute_custom_functions_definition(f: NativeFunction, func_type: str):
    with native_function_manager(f):
        sig = DispatcherSignature.from_schema(f.func)
        name = sig.name()
        args = sig.arguments()
        if func_type == 'call':
            args_str = ', '.join(a.defn() for a in args)
            args_exprs_str = ', '.join(a.name for a in args)
        if func_type == 'redispatch':
            args_str = 'c10::DispatchKeySet dispatchKeySet, ' + ', '.join(a.defn() for a in args)
            args_exprs_str = 'dispatchKeySet, ' + ', '.join(a.name for a in args)

        return [CUSTOM_FUNCTIONS_DEFINITION.substitute(
                return_type=cpp.returns_type(f.func.returns).cpp_type(),
                base_name=f.func.name.name,
                func_name=name,
                overload=f.func.name.overload_name,
                args_str=args_str,
                func_type=func_type,
                schema=sig.type(),
                args_exprs_str=args_exprs_str,)]


def gen_custom_functions_dispatch(
    fm: FileManager,
    custom_functions: Sequence[NativeFunction]
) -> None:
    func_type_list = ['call', 'redispatch']
    file_name_list = ['CustomFunctions', 'CustomRedispatch']

    for func_type, file_name in zip(func_type_list, file_name_list):
        fm.write_with_template(
        f'{file_name}.h', f'{file_name}.h', lambda:{
        'custom_function_declarations':list(concatMap(
            lambda f: compute_custom_functions_declaration(f, func_type),
            custom_functions
            ))}
        )

        fm.write_with_template(
        f'{file_name}.cpp', f'{file_name}.cpp', lambda:{
        'custom_function_definitions':list(concatMap(
            lambda f: compute_custom_functions_definition(f, func_type),
            custom_functions
            ))}
        )
