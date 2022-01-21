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

from typing import Dict, Sequence, List, NoReturn, Union
from codegen.api.types import (BaseCType, Binding, ConstRefCType,
                               Expr, MutRefCType, OptionalCType,
                               NamedCType, SpecialArgName, tensorT,
                               memoryFormatT, tensorOptionsT, scalarTypeT,
                               boolT, deviceT, layoutT, optionalTensorRefT,
                               scalarT, optionalScalarRefT)

# This file implements a small program synthesis engine that implements
# conversions between one API to another.
#
# The key data type in this file in NamedCType, short for Named C++ semantic type.  A NamedCType
# represents a C++ type, plus semantic information about what it represents.
# For example, consider the argument "bool pin_memory"; its normal C++ type is
# "bool", but its C++ semantic type also keeps track that this represents a
# "pin_memory"; you can't just use a random other boolean in a context where you
# need a "pin_memory"!
#
# The translator takes a list of needed NamedCTypes, and then figures out how
# to construct expressions with these NamedCTypes from the given bindings.  Many
# of these expressions are trivial (I need a Tensor other; there's a Tensor
# other scope); others are more nontrivial and may require packing/unpacking.
# Some examples of non-trivial action:
#
#   - Need the "dtype" binding?  Well, maybe "dtype" isn't available
#     in the context, instead, "options" is, and you need to extract
#     it from there.  (Gather)
#
#   - Need the "context" binding?  Well, maybe "context" isn't available
#     in the context, and you need to construct it from "dtype", "device",
#     etc.  (Scatter)
#
#   - Need the "memory_format" binding?  Well, actually, it's available
#     from both "memory_format" and "options", so you had better make sure
#     they are consistent.  (Join)

options_ctype = NamedCType("options", ConstRefCType(BaseCType(tensorOptionsT)))

class UnsatError(RuntimeError):
    pass

# Given a set of in-scope bindings and a set of target bindings, synthesize
# a list of expressions that uses only the in-scope bindings (bindings) that
# have all of the types of goals.  You may want to use this function if
# you're generating code for a function like:
#
#   void f({args}) {
#     g({exprs}); // g is a different API
#   }
#
# and you need to generate "exprs".
#
# Typically, a list of Bindings is convenient to get (you usually call something
# like arguments() to get them); but technically you only need less information:
# for 'bindings' an (un-ordered) list of Exprs is sufficient; similarly, for
# 'goals', an (ordered) list of NamedCType goals is sufficient.  If you are doing
# something more complicated, e.g., tracking the set of bindings in a context,
# you may find using these smaller types more convenient.
def translate(
    bindings: Sequence[Union[Expr, Binding]],
    goals: Sequence[Union[NamedCType, Binding]],
    *, method: bool = False
) -> List[Expr]:

    binding_exprs: List[Expr] = []
    for b in bindings:
        if isinstance(b, Binding):
            binding_exprs.append(Expr(
                expr=b.name,
                type=b.nctype,
            ))
        else:
            binding_exprs.append(b)

    goal_ctypes: List[NamedCType] = []
    for g in goals:
        if isinstance(g, Binding):
            goal_ctypes.append(g.nctype)
        else:
            goal_ctypes.append(g)

    # Add all the bindings to the context
    ctx: Dict[NamedCType, str] = {}
    for b in binding_exprs:
        ctx[b.type] = b.expr

        # While we're at it, do some simple forward inference, looking through
        # constructors.
        # TODO: My kingdom for a pattern matcher
        # https://www.python.org/dev/peps/pep-0634/
        # TODO: This could get us in recomputation trouble if b.expr is nontrivial
        t = b.type
        if isinstance(t, ConstRefCType) and isinstance(t.elem, OptionalCType) and \
                isinstance(t.elem.elem, BaseCType) and str(t.elem.elem.type) == 'at::Tensor':
            ctx[NamedCType(t.elem.elem.name, ConstRefCType(BaseCType(tensorT)))] = \
                f'({b.expr}.has_value() ? *{b.expr} : at::Tensor())'

        if t.type == ConstRefCType(OptionalCType(BaseCType(tensorT))):
            ctx[NamedCType(t.name, BaseCType(optionalTensorRefT))] = \
                f'(({b.expr}.has_value() && (*{b.expr}).defined()) ? at::OptionalTensorRef(*{b.expr}) : at::OptionalTensorRef())'

        if t.type == ConstRefCType(OptionalCType(BaseCType(scalarT))):
            ctx[NamedCType(t.name, BaseCType(optionalScalarRefT))] = \
                f'({b.expr}.has_value() ? at::OptionalScalarRef(&({b.expr}.value())) : at::OptionalScalarRef())'

    # Add implicit bindings if the generated code is inside a Tensor method
    if method:
        ctx[NamedCType("self", MutRefCType(BaseCType(tensorT)))] = "const_cast<Tensor&>(*this)"
        ctx[NamedCType("self", ConstRefCType(BaseCType(tensorT)))] = "const_cast<Tensor&>(*this)"

    def unsat(goal: NamedCType) -> NoReturn:
        ctx_desc = '\n'.join(f"  {t.cpp_type()} {t.name}; // {e}" for t, e in ctx.items())
        raise UnsatError(f'''
Failed to synthesize the expression "{goal.cpp_type()} {goal.name}".
When I failed, the following bindings were available in the context:

{ctx_desc}

This probably means there is a missing rule in the rules of tools.codegen.api.translate.
Check this module for more information.
''')

    # A shitty backtracking search implementation.  It's shitty because it
    # doesn't actually do backtracing or search. In particular, if
    # direct=True, we won't try to do any fancy synthesis, just trivial
    # conversions (e.g., "T a" is OK for "const T& a").  So all of the
    # existing rules in this function simply try to solve immediately,
    # and bail if things don't work out.
    def solve(goal: NamedCType, *, direct: bool) -> str:
        def direct_solve(goal: NamedCType) -> str:
            return solve(goal, direct=True)

        if goal in ctx:
            # Trivial
            return ctx[goal]

        # const & is satisfied with mutable &
        if isinstance(goal.type, ConstRefCType):
            try:
                # WARNING: not strictly decreasing; be careful not
                # to add a direct conversion that goes satisfies
                # mutable& with const&
                return solve(NamedCType(goal.name, MutRefCType(goal.type.elem)), direct=direct)
            except UnsatError:
                pass

        # mutable & is satisfied with value
        if isinstance(goal.type, MutRefCType):
            try:
                return solve(NamedCType(goal.name, goal.type.elem), direct=direct)
            except UnsatError:
                pass

        if direct:
            unsat(goal)

        # For now, all of these rules are mutually exclusive.
        if goal == NamedCType("memory_format", OptionalCType(BaseCType(memoryFormatT))):
            memory_format = direct_solve(
                NamedCType(SpecialArgName.possibly_redundant_memory_format, OptionalCType(BaseCType(memoryFormatT)))
            )
            # No need to join "memory_format" and "options" if the target API takes "options" directly.
            # Otherwise it will cause the redundant memory_format error.
            if options_ctype in goal_ctypes:
                return memory_format
            try:
                options = direct_solve(options_ctype)
                return f"c10::impl::check_tensor_options_and_extract_memory_format({options}, {memory_format})"
            except UnsatError:
                return memory_format

        elif goal == NamedCType("options", BaseCType(tensorOptionsT)):
            dtype = direct_solve(NamedCType("dtype", OptionalCType(BaseCType(scalarTypeT))))
            pin_memory = direct_solve(NamedCType("pin_memory", OptionalCType(BaseCType(boolT))))
            device = direct_solve(NamedCType("device", OptionalCType(BaseCType(deviceT))))
            layout = direct_solve(NamedCType("layout", OptionalCType(BaseCType(layoutT))))
            return f'TensorOptions().dtype({dtype}).layout({layout}).device({device}).pinned_memory({pin_memory})'

        elif goal == NamedCType("dtype", OptionalCType(BaseCType(scalarTypeT))):
            options = direct_solve(options_ctype)
            return f'optTypeMetaToScalarType({options}.dtype_opt())'

        elif goal == NamedCType("layout", OptionalCType(BaseCType(layoutT))):
            options = direct_solve(options_ctype)
            return f'{options}.layout_opt()'

        elif goal == NamedCType("device", OptionalCType(BaseCType(deviceT))):
            options = direct_solve(options_ctype)
            return f'{options}.device_opt()'

        elif goal == NamedCType("pin_memory", OptionalCType(BaseCType(boolT))):
            options = direct_solve(options_ctype)
            return f'{options}.pinned_memory_opt()'

        unsat(goal)

    return [Expr(solve(g, direct=False), g) for g in goal_ctypes]
