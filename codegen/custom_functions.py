from collections import namedtuple, defaultdict
from typing import List, Dict, Sequence
import yaml

from codegen.code_template import CodeTemplate
from codegen.gen import FileManager, cpp_string, error_check_native_functions
from codegen.model import (BackendIndex, DispatchKey, Variant,
                            NativeFunction, OperatorName, BackendMetadata)
from codegen.utils import concat_map, context, field_tag, parse_npu_yaml
from codegen.context import with_native_function
from codegen.api.signature import DispatcherSignature
from codegen.api import cpp

# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
ParsedYaml = namedtuple('ParsedYaml', ['native_functions', 'backend_indices'])
ExposeFuncList = ['npu_dtype_cast', 'npu_slice_out', 'npu_format_cast']


CUSTOM_FUNCTIONS_DECLARATION = CodeTemplate("""\
${return_type} ${func_name}(${args_str});
""")

CUSTOM_FUNCTIONS_DEFINITION = CodeTemplate("""\
${return_type} ${func_name}(${args_str}) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::${base_name}", "${overload}").typed<${schema}>();
    return op.call(${args_exprs_str});
}
""")


def parse_custom_yaml(custom_path: str) -> ParsedYaml:
    rs: List[NativeFunction] = []
    bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(dict)
    # Filter the custom native yaml file, and extract the functions we defined.
    source_es = parse_npu_yaml(custom_path)
    custom_es = source_es.get('custom', []) + source_es.get('custom_autograd', [])
    custom_es = field_tag(custom_es)
    for e_with_vars in custom_es:
        funcs = e_with_vars.get('func')
        with context(lambda: f'in {custom_path}:\n  {funcs}'):
            func, m = NativeFunction.from_yaml(e_with_vars)
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
                                  index=v)
    return ParsedYaml(rs, indices)


def gen_custom_registration(fm: FileManager, custom_functions: Sequence[NativeFunction]):
    fm.write_with_template(f'CustomRegisterSchema.cpp', 'CustomRegisterSchema.cpp', lambda: {
        'custom_function_registrations': [f'm.def({cpp_string(str(f.func))});\n' for f in custom_functions]
    })


@with_native_function
def compute_custom_functions_declaration(f: NativeFunction):
    sig = DispatcherSignature.from_schema(f.func)
    name = sig.name()
    args = sig.arguments()
    args_str = ', '.join(a.decl() for a in args)

    return [CUSTOM_FUNCTIONS_DECLARATION.substitute(
            return_type=cpp.returns_type(f.func.returns).cpp_type(),
            func_name=name,
            args_str=args_str,)]


@with_native_function
def compute_custom_functions_definition(f: NativeFunction):
    sig = DispatcherSignature.from_schema(f.func)
    name = sig.name()
    args = sig.arguments()
    args_str = ', '.join(a.defn() for a in args)
    args_exprs_str = ', '.join(a.name for a in args)
    return [CUSTOM_FUNCTIONS_DEFINITION.substitute(
            return_type=cpp.returns_type(f.func.returns).cpp_type(),
            base_name=f.func.name.name,
            func_name=name,
            overload=f.func.name.overload_name,
            args_str=args_str,
            schema=sig.type(),
            args_exprs_str=args_exprs_str,)]


def gen_custom_functions(
    fm: FileManager,
    custom_functions: Sequence[NativeFunction]
) -> None:
    fm.write_with_template(
    f'CustomFunctions.h', 'CustomFunctions.h', lambda:{
    'custom_function_declarations':list(concat_map(
        lambda f: compute_custom_functions_declaration(f),
        custom_functions
        ))}
    )

    fm.write_with_template(
        f'CustomFunctions.cpp', 'CustomFunctions.cpp', lambda:{
        'custom_function_definitions':list(concat_map(
            lambda f: compute_custom_functions_definition(f),
            custom_functions
        ))}
    )
