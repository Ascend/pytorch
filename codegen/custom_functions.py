from collections import namedtuple, defaultdict
from typing import List, Dict, Sequence
import yaml

from torchgen.code_template import CodeTemplate
from torchgen.gen import (parse_tags_yaml, LineLoader, FileManager, cpp_string, error_check_native_functions)
from torchgen.model import (BackendIndex, DispatchKey, Location, Variant,
                            NativeFunction, OperatorName, BackendMetadata)
from torchgen.utils import concatMap, context
from torchgen.context import with_native_function
from torchgen.api.types import DispatcherSignature
from torchgen.api import cpp

# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
ParsedYaml = namedtuple('ParsedYaml', ['native_functions', 'backend_indices'])


def parse_custom_yaml(custom_path: str, tag_path: str) -> ParsedYaml:
    valid_tags = parse_tags_yaml(tag_path)
    rs: List[NativeFunction] = []
    bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(dict)
    # Filter the custom native yaml file, and extract the functions we defined.
    from io import StringIO
    f_str = StringIO()
    with open(custom_path, 'r') as f:
        for line in f:
            if line.split(':')[0] in ['backend', 'cpp_namespace', 'extra_headers',
                                      'supported', 'autograd']:
                flag = False
                continue
            if line.split(':')[0] in ['custom', 'custom_autograd']:
                flag = True
                continue
            if ':' not in line or not flag:
                continue
            f_str.write(line)

    f_str.seek(0)
    custom_es = yaml.load(f_str, Loader=LineLoader)
    for e_with_vars in custom_es:
        funcs = e_with_vars.get('func')
        loc = Location(custom_path, e_with_vars["__line__"])
        with context(lambda: f'in {loc}:\n  {funcs}'):
            func, m = NativeFunction.from_yaml(e_with_vars, loc, valid_tags)
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
  ${type_definition_body}
}

""")


TRACE_DISPATCH = CodeTemplate("""\
return ${impl_name}(${args_exprs_str});""")


@with_native_function
def compute_trace_method_definition(f: NativeFunction):
    sig = DispatcherSignature.from_schema(f.func, prefix=f'wrapper_{f.func.name.overload_name}_')
    name = sig.name()
    args = sig.arguments()
    args_str = ', '.join(a.defn() for a in args)

    args_exprs_str = ', '.join(a.name for a in args)
    impl_name = f"at_npu::native::NPUNativeFunctions::{cpp.name(f.func)}"

    return [METHOD_DEFINITION.substitute(
        return_type=cpp.returns_type(f.func.returns).cpp_type(),
        name=name,
        args_str=args_str,
        type_definition_body=[TRACE_DISPATCH.substitute(impl_name=impl_name, args_exprs_str=args_exprs_str)]
    )]


@with_native_function
def compute_register_symbol(f: NativeFunction):
    name = DispatcherSignature.from_schema(f.func, prefix=f'wrapper_{f.func.name.overload_name}_').name()
    return [f'm.def({cpp_string(str(f.func))}, TORCH_FN(at_npu::native::{name}));\n']


def gen_custom_trace(fm: FileManager, custom_trace_functions: Sequence[NativeFunction]):

    fm.write_with_template(f'CustomRegisterSchema.cpp', 'CustomRegisterSchema.cpp', lambda: {
        'custom_trace_definitions': list(concatMap(
            lambda f: compute_trace_method_definition(f),
            custom_trace_functions
        )),
        'custom_trace_registrations': list(concatMap(
            lambda f: compute_register_symbol(f),
            custom_trace_functions
        )),
    })


MULTI_OUT = CodeTemplate("""\
def ${name}(${args_all}):
    warnings.warn('out=() will be replaced var=,m=,v=')
    if out is None:
        return torch.ops.npu.${ops}(${args})
    else:
        return torch.ops.npu.${ops}(${args_out})


""")


def warp_multi_out(f: NativeFunction):
    name = DispatcherSignature.from_schema(f.func, prefix=f'wrapper_{f.func.name.overload_name}_').name()
    args_real = list(f.func.schema_order_arguments())
    out_num = len(f.func.arguments.out)
    args_all = ', '.join([arg.name if arg.default is None else f'{arg.name}={arg.default}'
                          for arg in args_real[:-out_num]]) + ', out=None'
    args = ', '.join([arg.name for arg in args_real[:-out_num]])
    ops = f.func.name.name
    args_out = args + ', ' + ', '.join([f'{arg.name}=out[{i}]' for i, arg in enumerate(args_real[-out_num:])])

    return [MULTI_OUT.substitute(
        name=name,
        args_all=args_all,
        ops=ops,
        args=args,
        args_out=args_out
    )]


def patch_multi_out(f: NativeFunction):
    ops = f.func.name.name
    name = DispatcherSignature.from_schema(f.func, prefix=f'wrapper_{f.func.name.overload_name}_').name()

    return [f'torch_npu.{ops} = {name}']


def gen_custom_ops_patch(fm: FileManager, custom_trace_functions: Sequence[NativeFunction]):
    multi_out_functions = [f for f in custom_trace_functions if len(f.func.arguments.out) > 1]
    general_functions = [f for f in custom_trace_functions if f.func.name.name not in
                         set([f.func.name.name for f in multi_out_functions])]
    fm.write_with_template(f'custom_ops.py', 'custom_ops.py', lambda: {
        'custom_ops': [f'torch_npu.{ops} = torch.ops.npu.{ops}'
                       for ops in set([f.func.name.name for f in general_functions])],
        'warp_multi_out': list(concatMap(
            lambda f: warp_multi_out(f),
            multi_out_functions
        )),
        'patch_multi_out': list(concatMap(
            lambda f: patch_multi_out(f),
            multi_out_functions
        )),
    })
