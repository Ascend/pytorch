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
from typing import Optional, Sequence, Union, List, Set

from codegen.model import (Argument, Arguments, BaseTy, BaseType,
                           FunctionSchema, ListType, NativeFunction,
                           OptionalType, Return, SelfArgument,
                           TensorOptionsArguments, Type, assert_never)
from codegen.api.types import (ArgName, BaseCType, Binding, ConstRefCType, NamedCType, CType,
                               MutRefCType, ArrayCType, ListCType, VectorCType, ArrayRefCType,
                               OptionalCType, TupleCType, SpecialArgName, boolT, scalarT,
                               tensorListT, dimnameListT, tensorT, voidT, iTensorListRefT,
                               BaseTypeToCppMapping, intArrayRefT, tensorOptionsT, optionalIntArrayRefT,
                               longT, SymIntT, symIntArrayRefT, optionalSymIntArrayRefT)
from codegen import local

# This file describes the translation of JIT schema to the public C++
# API, which is what people use when they call functions like at::add.
#
# Prominent characteristics of the C++ API:
#
#   - dtype, layout, device and pin_memory are collected into
#     a single C++ type TensorOptions  (the native functions API
#     also has this, but tensor options is really most relevant
#     for the C++ API; it makes calling kwarg factory functions
#     pleasant)
#
#   - defaulting lives here (in fact, the dispatcher is completely
#     oblivious of defaults!)
#
# BTW: policy on name collisions: we try not to have types with
# collisions, but functions are fair game to collide


def name(func: FunctionSchema, *, faithful_name_for_out_overloads: bool = False, symint_overload: bool = False) -> str:
    func_name = str(func.name.name)
    if symint_overload:
        func_name += "_symint"
    if func.is_out_fn():
        if faithful_name_for_out_overloads:
            func_name += '_outf'
        else:
            func_name += '_out'

    return func_name


# Translation of "value types" in JIT schema to C++ API type.  Value
# types look the same no matter if they are argument types or return
# types.  Returns None if the type in question is not a value type.
def valuetype_type(t: Type, *, binds: ArgName, symint: bool = False) -> Optional[NamedCType]:
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor or t.name == BaseTy.Scalar:
            return None
        elif str(t) == "SymInt":
            if symint:
                return NamedCType(binds, BaseCType(SymIntT))
            else:
                return NamedCType(binds, BaseCType(longT))
        # All other BaseType currently map directly to BaseCppTypes.
        return NamedCType(binds, BaseCType(BaseTypeToCppMapping[t.name]))
    elif isinstance(t, OptionalType):
        elem = valuetype_type(t.elem, binds=binds, symint=symint)
        if elem is None:
            return None
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        if str(t.elem) == 'bool':
            if t.size is None:
                raise ValueError("t.size is None")
            return NamedCType(binds, ArrayCType(BaseCType(boolT), t.size))
        else:
            return None
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


# Translation of types occuring in JIT arguments to a C++ argument type.
def argumenttype_type(t: Type, *, mutable: bool, binds: ArgName, symint: bool = False) -> NamedCType:
    # If it's a value type, do the value type translation
    r = valuetype_type(t, binds=binds, symint=symint)
    if r is not None:
        return r

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable and not local.use_const_ref_for_mutable_tensors():
                return NamedCType(binds, MutRefCType(BaseCType(tensorT)))
            else:
                return NamedCType(binds, ConstRefCType(BaseCType(tensorT)))
        elif t.name == BaseTy.Scalar:
            return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
        else:
            raise AssertionError(f"base type should have been value type {t}")
    elif isinstance(t, OptionalType):
        if str(t.elem) == 'Tensor':
            if mutable and not local.use_const_ref_for_mutable_tensors():
                return NamedCType(binds, MutRefCType(BaseCType(tensorT)))  # TODO: fix this discrepancy
            else:
                return NamedCType(binds, ConstRefCType(OptionalCType(BaseCType(tensorT))))
        elif str(t.elem) == 'Scalar':
            return NamedCType(binds, ConstRefCType(OptionalCType(BaseCType(scalarT))))
        elif isinstance(t.elem, ListType) and str(t.elem.elem) == "int":
            return NamedCType(binds, BaseCType(optionalIntArrayRefT))
        elif isinstance(t.elem, ListType) and str(t.elem.elem) == "SymInt":
            if symint:
                return NamedCType(binds, BaseCType(optionalSymIntArrayRefT))
            else:
                return NamedCType(binds, BaseCType(optionalIntArrayRefT))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds, symint=symint)
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        # TODO: remove these special cases, ArrayRef fallthrough works fine
        type_dict = {
            "int": BaseCType(intArrayRefT),
            "Scalar": ArrayRefCType(BaseCType(scalarT)),
            "Dimname": BaseCType(dimnameListT),
            "Tensor?": ConstRefCType(ListCType(OptionalCType(BaseCType(tensorT))))
        }
        if str(t.elem) == "Tensor":
            if local.use_ilistref_for_tensor_lists():
                return NamedCType(binds, ConstRefCType(BaseCType(iTensorListRefT)))
            else:
                return NamedCType(binds, BaseCType(tensorListT))
        if str(t.elem) == "SymInt":
            if symint:
                return NamedCType(binds, BaseCType(symIntArrayRefT))
            else:
                return NamedCType(binds, BaseCType(intArrayRefT))
        if str(t.elem) in type_dict:
            return NamedCType(binds, type_dict[str(t.elem)])
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds, symint=symint)
        return NamedCType(binds, ArrayRefCType(elem.type))
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


# Translate a JIT argument into its C++ type
def argument_type(a: Argument, *, binds: ArgName, symint: bool = False) -> NamedCType:
    return argumenttype_type(a.type, mutable=a.is_write, symint=symint, binds=binds)


# Translation of a (non-multi) return type from JIT to C++
# N.B: returntype_type returns a CType, not a NamedCType.
# This is mostly because of the mismatch between return types and return names.
# e.g. a function with a return type of 'void' has 0 return names,
# and a function with a return type of 'std::tuple' has >1 return name.
def returntype_type(t: Type, *, mutable: bool, symint: bool = False) -> CType:
    # placeholder is ignored
    r = valuetype_type(t, binds="__placeholder__", symint=symint)
    if r is not None:
        return r.type

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable and local.use_const_ref_for_mutable_tensors():
                return ConstRefCType(BaseCType(tensorT))
            elif mutable:
                return MutRefCType(BaseCType(tensorT))
            else:
                # Note [Tensor Copy Returns]
                # Currently, we use "Argument.is_write" to determine
                # whether or not Tensor return types should be copies or references.
                # If that ever changes, take a look at other locations of this note!
                return BaseCType(tensorT)
        elif t.name == BaseTy.Scalar:
            return BaseCType(scalarT)
    elif isinstance(t, ListType):
        if mutable:
            raise ValueError("Native functions should never return a mutable tensor list. They should return void.")
        elem = returntype_type(t.elem, mutable=False, symint=symint)
        if t.size is not None:
            raise ValueError(f"fixed size list returns not supported: {t}")
        return VectorCType(elem)

    raise AssertionError(f"unrecognized return type {t}")


# Translation of a single return to its C++ type
def return_type(r: Return, *, symint: bool = False) -> CType:
    return returntype_type(r.type, mutable=r.is_write, symint=symint)


# Translation of a full (possibly multi) return from JIT to its C++ type
def returns_type(rs: Sequence[Return], *, symint: bool = False) -> CType:
    if len(rs) == 0:
        return BaseCType(voidT)
    elif len(rs) == 1:
        return return_type(rs[0], symint=symint)
    else:
        return TupleCType([return_type(r, symint=symint) for r in rs])


def return_names(f: NativeFunction, *, fallback_name: str = 'result') -> Sequence[str]:
    returns: List[str] = []
    for i, r in enumerate(f.func.returns):
        # If we have an inplace function, the return argument is
        # implicitly named self.
        # TODO: Consider incorporating this into the data model
        if f.func.name.name.inplace:
            if i != 0:
                raise ValueError("illegal inplace function with multiple returns")
            func_name = 'self'
        # If we are out function, the func_name is the name of the
        # corresponding output function (r.name will get recorded
        # in field_name later.)
        elif f.func.is_out_fn():
            func_name = f.func.arguments.out[i].name
        # If the return argument is explicitly named...
        elif r.name:
            name_conflict = any(r.name == a.name for a in f.func.schema_order_arguments())
            if name_conflict and not f.func.is_out_fn():
                func_name = f'{r.name}_return'
            else:
                func_name = r.name
        # If there is no explicit name and no fallback name was passed in, we just name the output result,
        # unless it's a multi-return, in which case it's result0,
        # result1, etc (zero-indexed)
        else:
            func_name = fallback_name if len(f.func.returns) == 1 else f'{fallback_name}{i}'
        returns.append(func_name)
    return returns


JIT_TO_CPP_DEFAULT = {
    'False': 'false',
    'True': 'true',
    'None': 'c10::nullopt',  # UGH this one is type directed
    'Mean': 'at::Reduction::Mean',
    '[]': '{}',
    'contiguous_format': 'c10::MemoryFormat::Contiguous',
    'long': 'at::kLong',
}


# Convert a JIT default into C++ expression representing the default
def default_expr(d: str, t: Type) -> str:
    def deal_str_basetype(d):
        s = ''
        i = 1
        while i + 1 < len(d):
            if d[i] != '\\':
                if d[i] == '"':
                    s += '\\"'
                else:
                    s += d[i]
                i += 1
            else:
                if d[i + 1] == "'":
                    s += "'"
                else:
                    s += d[i:i + 2]
                i += 2

        return f'"{s}"'

    if d == 'None' and str(t) == 'Tensor?':
        return '{}'
    if isinstance(t, BaseType) and t.name is BaseTy.str:
        # Schema allows single quotes but C++ needs double
        if len(d) >= 2 and d[0] == "'" and d[-1] == "'":
            return deal_str_basetype(d)

    if isinstance(t, OptionalType):
        if d == 'None':
            return 'c10::nullopt'

        return default_expr(d, t.elem)

    if isinstance(t, ListType):
        if d.startswith('[') and d.endswith(']'):
            return '{' + d[1:-1] + '}'
        elif t.size is None:
            # NOTE: Sized lists can have scalar defaults
            raise ValueError(f"Expected a list default '[...]' but found: '{d}'")

    return JIT_TO_CPP_DEFAULT.get(d, d)

# Convert an argument into its C++ API form


def argument(
    a: Union[Argument, TensorOptionsArguments, SelfArgument],
    *, cpp_no_default_args: Set[str], method: bool, faithful: bool, symint: bool = False,
    has_tensor_options: bool
) -> List[Binding]:
    def sub_argument(a: Union[Argument, TensorOptionsArguments, SelfArgument]) -> List[Binding]:
        return argument(
            a, cpp_no_default_args=cpp_no_default_args, method=method, faithful=faithful, symint=symint,
            has_tensor_options=has_tensor_options)

    if isinstance(a, Argument):
        binds: ArgName
        if a.name == "memory_format" and has_tensor_options:
            binds = SpecialArgName.possibly_redundant_memory_format
        else:
            binds = a.name
        default: Optional[str] = None
        if a.name not in cpp_no_default_args and a.default is not None:
            default = default_expr(a.default, a.type)
        return [Binding(
            nctype=argument_type(a, binds=binds, symint=symint),
            name=a.name,
            default=default,
            argument=a,
        )]
    elif isinstance(a, TensorOptionsArguments):
        if faithful:
            return sub_argument(a.dtype) + sub_argument(a.layout) + \
                sub_argument(a.device) + sub_argument(a.pin_memory)
        else:
            default = None
            # Enforced by NativeFunction.__post_init__
            if 'options' in cpp_no_default_args:
                raise KeyError("'options' in cpp_no_default_args")
            if all(x.default == "None" for x in a.all()):
                default = '{}'
            elif a.dtype.default == "long":
                default = 'at::kLong'  # TODO: this is wrong
            return [Binding(
                nctype=NamedCType('options', BaseCType(tensorOptionsT)),
                name='options',
                default=default,
                argument=a,
            )]
    elif isinstance(a, SelfArgument):
        if method:
            # Caller is responsible for installing implicit this in context!
            return []
        else:
            return sub_argument(a.argument)
    else:
        assert_never(a)


def arguments(
    func_arguments: Arguments,
    *, faithful: bool, symint: bool = False, method: bool, cpp_no_default_args: Set[str]
) -> List[Binding]:
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    if faithful:
        args.extend(func_arguments.non_out)
        args.extend(func_arguments.out)
    else:
        args.extend(func_arguments.out)
        args.extend(func_arguments.non_out)
    return [
        r.no_default() if faithful else r for a in args
        for r in argument(
            a, faithful=faithful, symint=symint, method=method,
            has_tensor_options=func_arguments.tensor_options is not None,
            cpp_no_default_args=cpp_no_default_args)
    ]
