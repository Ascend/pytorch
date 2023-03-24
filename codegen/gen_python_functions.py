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

# Generates Python bindings for ATen functions
#
# The bindings are generated as methods on functions on the
# torch_npu._C._nn. torch_npu._C._fft, or torch_npu._C._linalg objects.


import argparse
import os
import re
from collections import defaultdict, namedtuple
from typing import Dict, Optional, List, Tuple, Set, Sequence, Callable

import yaml
from codegen.code_template import CodeTemplate
from codegen.api import cpp
from codegen.api.python import (PythonSignature,
                                PythonSignatureGroup,
                                PythonSignatureNativeFunctionPair,
                                arg_parser_output_exprs,
                                cpp_dispatch_exprs,
                                cpp_record_func,
                                cpp_dispatch_target,
                                dispatch_lambda_args,
                                dispatch_lambda_exprs,
                                dispatch_lambda_return_str,
                                has_tensor_options,
                                namedtuple_fieldnames, signature)
from codegen.gen import cpp_string, FileManager, error_check_native_functions
from codegen.context import with_native_function
from codegen.model import (BaseOperatorName, NativeFunction,
                           Type, Variant, BackendIndex,
                           BackendMetadata, DispatchKey, OperatorName)
from codegen.utils import context


# These functions require manual Python bindings or are not exposed to Python
_SKIP_PYTHON_BINDINGS = [
    'alias', 'contiguous', 'is_cuda', 'is_sparse', 'is_sparse_csr', 'size', 'stride',
    '.*_backward', '.*_backward_(out|input|weight|bias)', '.*_forward',
    '.*_forward_out', '_unsafe_view', 'tensor', '_?sparse_coo_tensor.*',
    '_?sparse_csr_tensor.*',
    '_arange.*', '_range.*', '_linspace.*', '_logspace.*',
    '_sparse_add_out', '_sparse_div.*', '_sparse_mul.*', '_sparse_sub.*', '_sparse_dense_add_out',
    'index', 'unique_dim_consecutive',
    '_cumsum.*', '_cumprod.*', '_sum.*', '_prod.*',
    '_th_.*', '_thnn_.*',
    'arange.*', 'range.*', '_solve.*', '_inverse.*',
    'full(_out)?',
    '_cholesky.*', '_triangular_solve.*', '_qr.*', '_symeig.*', '_svd.*',
    'slice', 'randint(_out)?',
    'item', '_local_scalar_dense', 'to',
    '_to_copy',
    'copy_sparse_to_sparse_', 'copy_',
    'numpy_T',  # this needs to be an attribute in Python, not a function
    'nonzero(_(out|numpy))?',
    'set_data',
    '.*_overrideable',  # overrideable functions for backend extension
    'data', 'is_leaf', 'output_nr', '_version', 'requires_grad_', 'retains_grad', 'set_',
    '_fw_primal', 'fake_quantize_per_tensor_affine_cachemask',
    'fake_quantize_per_channel_affine_cachemask',
    '_reshape_alias',
    '_cudnn.*', '.*_quantized', 'fft_.*',
]

SKIP_PYTHON_BINDINGS = list(map(lambda pattern: re.compile(rf'^{pattern}$'), _SKIP_PYTHON_BINDINGS))

# These function signatures are not exposed to Python. Note that this signature
# list does not support regex.
SKIP_PYTHON_BINDINGS_SIGNATURES = []

DONT_RECORD_TRACE = []

NPU_AUTOGRAD_FUNCTION = []

def should_trace(f: NativeFunction) -> bool:
    # Operations involving Storage or Type are not traceable at the moment
    if any(str(arg.type) in {'Storage', 'Type', 'ConstQuantizerPtr'}
           for arg in f.func.schema_order_arguments()):
        return False
    # We can't trace functions which don't have any Tensor or TensorList returns
    if not any(r.type.is_tensor_like() for r in f.func.returns):
        return False
    return f.func.name.name.base not in DONT_RECORD_TRACE

# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
ParsedYaml = namedtuple('ParsedYaml', ['native_functions', 'backend_indices'])
def parse_custom_yaml(custom_path: str) -> ParsedYaml:
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
    custom_es = yaml.safe_load(f_str)
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
        indices[k] = BackendIndex(dispatch_key=k, use_out_as_primary=True, external=False, index=v)
    return ParsedYaml(rs, indices)


@with_native_function
def should_generate_py_binding(f: NativeFunction) -> bool:
    if os.environ.get('BSCPP_OPS_ENABLE') is None and f.bscpp_op:
        return False

    name = cpp.name(f.func)
    for skip_regex in SKIP_PYTHON_BINDINGS:
        if skip_regex.match(name):
            return False

    func_signature = str(f.func)
    for pattern in SKIP_PYTHON_BINDINGS_SIGNATURES:
        if pattern == func_signature:
            return False

    return True

def get_pycname(name: BaseOperatorName) -> str:
    return f'THPVariable_{name}'

def group_filter_overloads(
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool]
) -> Dict[BaseOperatorName, List[PythonSignatureNativeFunctionPair]]:
    grouped: Dict[BaseOperatorName, List[PythonSignatureNativeFunctionPair]] = defaultdict(list)
    for pair in pairs:
        if pred(pair.function):
            grouped[pair.function.func.name.name].append(pair)
    return grouped

def create_python_bindings(
    fm: FileManager,
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pairs_device: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
    module: Optional[str],
    filename: str,
    *,
    method: bool,
) -> None:
    """Generates Python bindings to ATen functions"""
    py_methods: List[str] = []
    py_method_defs: List[str] = []
    py_forwards: List[str] = []
    py_device_methods: List[str] = []
    py_device_forwards: List[str] = []
    py_device_method_defs: List[str] = []

    grouped = group_filter_overloads(pairs, pred)
    grouped_device = group_filter_overloads(pairs_device, pred)

    for name in sorted(grouped.keys(), key=lambda x: str(x)):
        overloads = grouped[name]
        py_methods.append(method_impl(name, module, overloads, method=method))
        py_method_defs.append(method_def(name, module, overloads, method=method))
        py_forwards.extend(forward_decls(name, overloads, method=method))

    for name in sorted(grouped_device.keys(), key=lambda x: str(x)):
        overloads = grouped_device[name]
        py_device_methods.append(method_impl(name, module, overloads, method=method, custom=False))
        py_device_method_defs.append(method_def(name, module, overloads, method=method))
        py_device_forwards.extend(forward_decls(name, overloads, method=method))

    fm.write_with_template(filename, filename, lambda: {
        'generated_comment': '@' + f'generated from {fm.template_dir}/{filename}',
        'py_forwards': py_forwards,
        'py_methods': py_methods,
        'py_device_methods': py_device_methods,
        'py_method_defs': py_method_defs,
        'py_device_forwards': py_device_forwards,
        'py_device_method_defs': py_device_method_defs,
    })


def create_python_device_bindings(
    fm: FileManager,
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
    module: Optional[str],
    filename: str,
    *,
    method: bool,
) -> None:
    """Generates Python bindings to ATen functions"""
    py_device_method_defs: List[str] = []
    device_methods_def_py_dispatch: List[str] = []

    grouped = group_filter_overloads(pairs, pred)

    PY_DEVICE_METHOD_DEF = CodeTemplate("""\
    torch.${name} = _${name}
""")

    PY_DEVICE_METHOD_DEF_DISPATCH = CodeTemplate("""\

@torch_device_guard
def _${name}(*args, **kwargs):
    return torch_npu.${name}(*args, **kwargs)

""")

    def method_device_def(name):
        return PY_DEVICE_METHOD_DEF.substitute(name=name)

    def method_device_def_dispatch(name):
        return PY_DEVICE_METHOD_DEF_DISPATCH.substitute(name=name)

    for name in sorted(grouped.keys(), key=lambda x: str(x)):
        py_device_method_defs.append(method_device_def(name))
        device_methods_def_py_dispatch.append(method_device_def_dispatch(name))
        

    fm.write_with_template(filename, filename, lambda: {
        'device_methods_def_py': py_device_method_defs,
        'device_methods_def_py_dispatch': device_methods_def_py_dispatch
    })

    def query_methods(filepath):
        with open(filepath, 'r', encoding='UTF-8') as f:
            read_lines = f.readlines()
        def_methods = []
        for read_line in read_lines:
            if read_line.startswith("def"):
                def_methods.append((read_line[4:read_line.index("(")]).strip())
        return def_methods

    device_methods = query_methods(fm.install_dir + filename)
    if len(device_methods) != len(set(device_methods)):
        raise RuntimeError("In device methods file " + 
                    str(fm.install_dir + filename) + " has multi-definition function.")

def load_signatures(
    native_functions: List[NativeFunction],
    *,
    method: bool,
    pyi: bool = False,
) -> Sequence[PythonSignatureNativeFunctionPair]:

    @with_native_function
    def gen_signature_pairs(f: NativeFunction) -> PythonSignatureNativeFunctionPair:
        return PythonSignatureNativeFunctionPair(
            signature=signature(f, method=method, pyi=pyi),
            function=f,
        )

    pairs = list(map(gen_signature_pairs, native_functions))
    return pairs

@with_native_function
def gen_namedtuple_typename_key(f: NativeFunction) -> str:
    name = cpp.name(f.func)
    fieldnames = namedtuple_fieldnames(f.func.returns)
    return '_'.join([name] + fieldnames)

def emit_namedtuple_typedefs(
    overloads: Sequence[PythonSignatureNativeFunctionPair]
) -> Tuple[List[str], Dict[str, str]]:
    """
    Generate block of named tuple type def inits, and add typeref snippets
    to declarations that use them
    """
    field_def_names: Dict[str, str] = {}  # map from unique field name lists to field def name
    field_defs: List[str] = []           # field def declarations
    type_names: Dict[str, str] = {}    # map from unique name + field name lists to typedef name
    type_defs: List[str] = []          # typedef declarations and init code

    for overload in overloads:
        fieldnames = namedtuple_fieldnames(overload.function.func.returns)
        if not fieldnames:
            continue

        fn_key = '_'.join(fieldnames)
        fieldsname = field_def_names.get(fn_key)
        if fieldsname is None:
            fieldsname = f'NamedTuple_fields{"" if not field_defs else len(field_defs)}'
            field_def_names[fn_key] = fieldsname
            fields = ', '.join(f'{{"{fn}", ""}}' for fn in fieldnames)
            field_defs.append(f"""\
static PyStructSequence_Field {fieldsname}[] = {{ {fields},  {{nullptr}} }};
""")

        name = cpp.name(overload.function.func)  # use @with_native_function?
        tn_key = gen_namedtuple_typename_key(overload.function)
        typename = type_names.get(tn_key)
        if typename is None:
            typename = f'NamedTuple{"" if not type_defs else len(type_defs)}'
            type_names[tn_key] = typename
            type_defs.append(f"""\
static PyTypeObject {typename};
static bool {typename}_initialized = false;
if (!{typename}_initialized) {{
  {typename}_initialized = true;
  static PyStructSequence_Desc desc = {{ "torch.return_types.{name}", nullptr, {fieldsname}, {len(fieldnames)} }};
  PyStructSequence_InitType(&{typename}, &desc);
  {typename}.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
}}
""")

    return field_defs + type_defs, type_names

# python binding for all overloads of a particular function/method
PY_VARIABLE_METHOD_VARARGS = CodeTemplate(r"""\
// ${name}
static PyObject * ${pycname}(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  ${method_header}
  static torch::PythonArgParser parser({
    ${signatures}
  }, /*traceable=*/${traceable});
  torch::ParsedArgs<${max_args}> parsed_args;
  auto _r = parser.parse(${self_}, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return torch::handle_torch_function(_r, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    ${dispatch}
  }
  ${method_footer}
}
""")

# handler for a single parsed signature - may be a single overload or
# a pair of overloads that whose signatures only differ in output params
# (plugged into PY_VARIABLE_METHOD_VARARGS as an item in ${dispatch})
PY_VARIABLE_CASE = CodeTemplate("""\
case ${overload_index}: {
  ${body}
}
""")

# python binding for single-overload function/method
PY_VARIABLE_METHOD_VARARGS_SINGLETON = CodeTemplate("""\
// ${name}
static PyObject * ${pycname}(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  ${method_header}
  static torch::PythonArgParser parser({
    ${signatures}
  }, /*traceable=*/${traceable});
  torch::ParsedArgs<${max_args}> parsed_args;
  auto _r = parser.parse(${self_}, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return torch::handle_torch_function(_r, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  ${dispatch}
  ${method_footer}
}
""")

# python binding for a method with no args, shortcuts parsing
PY_VARIABLE_METHOD_NOARGS = CodeTemplate("""\
// ${name}
static PyObject * ${pycname}(PyObject* self_, PyObject* args)
{
  ${method_header}
  ${dispatch}
  ${method_footer}
}
""")

def method_impl(
    name: BaseOperatorName,
    module: Optional[str],
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool,
    custom: bool = True
) -> str:
    """
    Generate a python binding for all overloads of an op.
    """
    pycname = get_pycname(name)
    namedtuple_inits, namedtuple_typenames = emit_namedtuple_typedefs(overloads)

    method_header = ['HANDLE_TH_ERRORS']
    method_header += namedtuple_inits
    method_header += [
        "const Tensor& self = THPVariable_Unpack(self_);"
    ] if method else []

    method_footer = ['Py_RETURN_NONE;'] + ['END_HANDLE_TH_ERRORS']

    traceable = 'true' if all(should_trace(o.function) for o in overloads) else 'false'

    grouped_overloads: Sequence[PythonSignatureGroup] = group_overloads(overloads)
    is_singleton = len(grouped_overloads) == 1
    signatures: List[str] = []
    dispatch: List[str] = []
    for overload_index, overload in enumerate(grouped_overloads):
        overload_signature = overload.signature.signature_str()
        signatures.append(f'{cpp_string(str(overload_signature))},')
        dispatch_body = emit_dispatch_case(overload, namedtuple_typenames, custom)
        dispatch.append(
            PY_VARIABLE_CASE.substitute(overload_index=overload_index, body=dispatch_body)
            if not is_singleton else dispatch_body)

    if is_singleton:
        template = PY_VARIABLE_METHOD_VARARGS_SINGLETON
    else:
        template = PY_VARIABLE_METHOD_VARARGS

    return template.substitute(
        name=name,
        pycname=pycname,
        method_header=method_header,
        max_args=max(map(lambda o: o.signature.arguments_count(), overloads)),
        signatures=signatures,
        traceable=traceable,
        dispatch=dispatch,
        method_footer=method_footer,
        self_="self_" if method else "nullptr",
    )


# handler for output/no-output overload pair
PY_VARIABLE_OUT = CodeTemplate("""\
if (_r.isNone(${out_idx})) {
  ${call_dispatch}
} else {
  ${call_dispatch_out}
}
""")


def emit_dispatch_case(
    overload: PythonSignatureGroup,
    namedtuple_typenames: Dict[str, str],
    custom: bool = True
) -> str:
    """
    Emit dispatching code for a single parsed signature. This corresponds to either
    a single native function, or a pair that differs only in output params. In the
    latter case, a single python signature is used for dispatching switches on the
    presence/absence of passed output args.
    """
    if overload.outplace is not None:
        # dispatch output and no-output variants, branch on _r.isNone(<out_idx>)
        return PY_VARIABLE_OUT.substitute(
            out_idx=overload.signature.output_idx(),
            call_dispatch=emit_single_dispatch(
                overload.signature, overload.base, namedtuple_typenames, custom),
            call_dispatch_out=emit_single_dispatch(
                overload.signature, overload.outplace, namedtuple_typenames, custom),
        )
    else:
        # no-output version only
        return emit_single_dispatch(
            overload.signature, overload.base, namedtuple_typenames, custom)


def forward_decls(
    name: BaseOperatorName,
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool
) -> Tuple[str, ...]:
    if method:
        return ()

    pycname = get_pycname(name)

    return (f"""\
static PyObject * {pycname}(PyObject* self_, PyObject* args, PyObject* kwargs);
""",)


def method_def(
    name: BaseOperatorName,
    module: Optional[str],
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool
) -> str:
    """
    Generate method def entry.
    """
    pycname = get_pycname(name)
    pyfunc_cast = 'castPyCFunctionWithKeywords'
    flags = 'METH_VARARGS | METH_KEYWORDS'

    if module == "torch_npu":
        flags += ' | METH_STATIC'

    if name.dunder_method:
        # PyMethodDef entry for binary op, throws not implemented error
        return f"""\
{{"{name}", {pyfunc_cast}(TypeError_to_NotImplemented_<{pycname}>), {flags}, NULL}},"""
    else:
        # PyMethodDef entry
        return f"""\
{{"{name}", {pyfunc_cast}({pycname}), {flags}, NULL}},"""


def group_overloads(
    overloads: Sequence[PythonSignatureNativeFunctionPair],
) -> Sequence[PythonSignatureGroup]:
    bases: Dict[str, PythonSignatureNativeFunctionPair] = {}
    outplaces: Dict[str, PythonSignatureNativeFunctionPair] = {}

    # first group by signature ignoring out arguments
    for overload in overloads:
        sig = overload.signature.signature_str(skip_outputs=True)
        if overload.function.func.is_out_fn():
            if sig in outplaces:
                raise RuntimeError(
                    f'Found duplicated function definition:\n- {overload.function.func}.\n'
                    f'Existing definition:\n- {outplaces[sig].function.func}.'
                )
            outplaces[sig] = overload
        else:
            if sig in bases:
                raise RuntimeError(
                    f'Found duplicated function definition:\n- {overload.function.func}.\n'
                    f'Existing definition:\n- {bases[sig].function.func}.'
                )
            bases[sig] = overload

    for sig, out in outplaces.items():
        if sig in bases:
            continue

        candidates: List[str] = []
        for overload in overloads:
            if str(overload.function.func.name.name) == str(out.function.func.name.name) \
                    and not overload.function.func.is_out_fn():
                candidates.append(overload.signature.signature_str(skip_outputs=True))
        out_sig = out.signature.signature_str()
        raise RuntimeError(
            f'While identifying overloads, we found an out schema {out_sig} without a corresponding non-out '
            f'variant. We expected the non-out variant to have schema: \n- {sig}\nPlease check that you '
            f'spelled the schema correctly in native_functions.yaml. We discovered the following candidate(s): \n'
            + '\n'.join(f'- {candidate}' for candidate in candidates))

    grouped: List[PythonSignatureGroup] = []
    for sig, base in bases.items():
        outplace = outplaces.get(sig)
        grouped.append(PythonSignatureGroup(
            # prefer the signature with optional out=... arguments because it's the
            # superset that can be used to parse input for both base and outplace.
            signature=outplace.signature if outplace is not None else base.signature,
            base=base.function,
            outplace=outplace.function if outplace is not None else None,
        ))

    return sort_overloads(grouped)


def sort_overloads(
    grouped_overloads: Sequence[PythonSignatureGroup]
) -> Sequence[PythonSignatureGroup]:

    def is_arg_smaller(t1: Type, t2: Type) -> bool:
        return (str(t1) == 'Scalar' and str(t2) == 'Tensor' or
                'Dimname' in str(t1) and 'Dimname' not in str(t2) or
                # In the discussion https://github.com/pytorch/pytorch/issues/54555 it has been
                # discussed why it is important to prioritize int/int? over int[]
                str(t1) == 'int[]' and (str(t2) == 'int' or str(t2) == 'int?') or
                # TensorList currently throws an error during argument parsing, that's why it needs to be
                # last in signature ordering. See discussion: https://github.com/pytorch/pytorch/issues/58087
                str(t1) == 'Tensor[]' and str(t2).find("[]") != -1)


    def is_smaller(s1: PythonSignature, s2: PythonSignature) -> bool:
        """Returns True if s1 < s2 in the partial order."""
        args1, args2 = s1.arguments(skip_outputs=True), s2.arguments(skip_outputs=True)
        if len(args1) != len(args2):
            return False
        equal = all(arg1.type == arg2.type for arg1, arg2 in zip(args1, args2))
        smaller_or_equal = all(str(arg1.type) == str(arg2.type)
                               or is_arg_smaller(arg1.type, arg2.type)
                               for arg1, arg2 in zip(args1, args2))
        return smaller_or_equal and not equal

    # First sort by signature
    grouped_overloads = sorted(grouped_overloads, key=lambda x: x.signature.signature_str())

    # Construct the relation graph
    larger_than: Dict[int, Set[int]] = defaultdict(set)
    for i1, overload1 in enumerate(grouped_overloads):
        for i2, overload2 in enumerate(grouped_overloads):
            if is_smaller(overload1.signature, overload2.signature):
                larger_than[i1].add(i2)

    if not larger_than:
        return list(grouped_overloads)

    # Use a topological sort to sort overloads according to the partial order.
    N = len(grouped_overloads)
    sorted_ids: List[int] = list(filter(lambda x: x not in larger_than, range(N)))

    for idx in range(N):
        # The size of sorted_ids will grow to N eventually.
        i = sorted_ids[idx]
        for j in sorted(larger_than.keys()):
            larger = larger_than[j]
            larger.discard(i)
            if not larger:
                del larger_than[j]
                sorted_ids.append(j)

    return list(map(lambda x: grouped_overloads[x], sorted_ids))


def emit_single_dispatch(
    ps: PythonSignature, f: NativeFunction, namedtuple_typenames: Dict[str, str], custom = True
) -> str:
    """
    Emit dispatch code for a single native function.
    """
    @with_native_function
    def go(f: NativeFunction) -> str:
        # header comments
        schema_comment = f'// aten::{f.func}'

        # dispatch lambda signature
        name = cpp.name(f.func)
        lambda_formals = ', '.join(map(lambda a: f"{a.type_str} {a.name}",
                                    dispatch_lambda_args(ps, f, custom)))
        lambda_return = dispatch_lambda_return_str(f)

        # device init
        if custom and ("Device" in str(f.func)):
            init_npu_device = f"torch_npu::utils::maybe_initialize_npu(device);"
        else:
            init_npu_device = f"//"

        # dispatch lambda body
        is_npu_autograd = str(f.func.name) in NPU_AUTOGRAD_FUNCTION
        record_func_define = cpp_record_func(f, custom=custom)
        dispatch_key_set = '' if not is_npu_autograd else 'auto ks_set = ' \
            'c10::DispatchKeySet().add(c10::DispatchKey::AutogradXLA).add(c10::DispatchKey::XLA);'
        dispatch_callee = cpp_dispatch_target(f, custom=custom, is_npu_autograd=is_npu_autograd)
        dispatch_args = ', '.join(cpp_dispatch_exprs(f, python_signature=ps, faithful=custom))
        if is_npu_autograd:
            dispatch_args = 'ks_set, ' + dispatch_args

        # from arg parser outputs to dispatch lambda arguments
        parser_outputs = arg_parser_output_exprs(ps, f)
        lambda_arg_exprs = dispatch_lambda_exprs(ps, f, custom)
        inits = '\n'.join(lambda_arg_exprs.inits)
        lambda_args = ', '.join(lambda_arg_exprs.exprs)

        need_set_requires_grad = ps.tensor_options_args and (not has_tensor_options(f) or (
            ps.method and ('requires_grad' in parser_outputs)))
        set_requires_grad = f'.set_requires_grad({parser_outputs["requires_grad"].expr})' \
            if need_set_requires_grad else ''

        if lambda_return == 'void':
            return f"""\
{schema_comment}
{inits}
auto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{
  {init_npu_device}
  pybind11::gil_scoped_release no_gil;
  {record_func_define}
  {dispatch_key_set}
  {dispatch_callee}({dispatch_args});
}};
dispatch_{name}({lambda_args}){set_requires_grad};
Py_RETURN_NONE;
"""
        else:
            typename = namedtuple_typenames.get(gen_namedtuple_typename_key(f))
            namedtuple_typeref = f'&{typename}, ' if typename is not None else ''
            return f"""\
{schema_comment}
{inits}
auto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{
  {init_npu_device}
  pybind11::gil_scoped_release no_gil;
  {record_func_define}
  {dispatch_key_set}
  return {dispatch_callee}({dispatch_args});
}};
return wrap({namedtuple_typeref}dispatch_{name}({lambda_args}){set_requires_grad});
"""

    return go(f)

# Parse native_functions.yaml into a sequence of NativeFunctions
def parse_native_yaml(path: str) -> List[NativeFunction]:
    from io import StringIO
    f_str = StringIO()
    with open(path, 'r') as f:
        for line in f:
            f_str.write(line)
    f_str.seek(0)
    es = yaml.safe_load(f_str)
    assert isinstance(es, list)
    rs: List[NativeFunction] = []
    with_device_base_operator = set()

    for e in es:
        funcs = e.get('func')
        with context(lambda: f'in {path}:\n  {funcs}'):
            func, m = NativeFunction.from_yaml(e)
            if "Device" in funcs:
                with_device_base_operator.add(func.func.name.name.base)

    for e in es:
        funcs = e.get('func')
        with context(lambda: f'in {path}:\n  {funcs}'):
            func, m = NativeFunction.from_yaml(e)
            if func.func.name.name.base not in with_device_base_operator:
                continue
            func.variants.discard(Variant.method)
            rs.append(func)
    return rs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate functions binding files')
    parser.add_argument(
        '-s',
        '--source_yaml',
        help='path to source yaml file containing operator external definitions')
    parser.add_argument(
        '-n',
        '--native_yaml',
        help='path to native yaml file containing operator external definitions with device arugment')
    parser.add_argument(
        '-o', '--output_dir', help='output directory')
    parser.add_argument(
        '-t', '--template_path', type=str, default=None, help='path of the templates')
    options = parser.parse_args()

    file_manager = FileManager(install_dir=options.output_dir, template_dir=options.template_path, dry_run=False)
    parsed_native_functions = parse_custom_yaml(options.source_yaml).native_functions
    valid_native_functions = list(filter(should_generate_py_binding, parsed_native_functions))

    functions = load_signatures(valid_native_functions, method=False)
    torch_native_functions = list(filter(should_generate_py_binding, parse_native_yaml(options.native_yaml)))
    device_native_functions = load_signatures(torch_native_functions, method=False)

    create_python_bindings(file_manager, functions, device_native_functions, lambda f: Variant.function in f.variants,
                           'torch_npu', 'python_custom_functions.cpp', method=False)
    
    file_device_manager=FileManager(install_dir=options.output_dir +"../../utils/", 
                                    template_dir=options.template_path, dry_run=False)
    create_python_device_bindings(file_device_manager, device_native_functions, 
                            lambda f: Variant.function in f.variants, 'torch_npu', 'torch_funcs.py', method=False)
