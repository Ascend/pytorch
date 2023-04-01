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

import itertools
from typing import Sequence, List, Union

from codegen.model import (Argument, FunctionSchema, Return,
                           SelfArgument, TensorOptionsArguments, Type,
                           assert_never)

from codegen.api.types import ArgName, Binding, NamedCType, CType
from codegen.api import cpp
from torchgen.utils import concatMap


# This file describes the translation of JIT schema to the dispatcher
# API, the *unboxed* calling convention by which invocations through
# the dispatcher are made.  Historically, the dispatcher API matched
# the C++ API, but with the establishment of the boxed API, we've
# made changes to the dispatcher API to so that the unboxed API
# better aligns with the boxed API.  The dispatcher API hooks heavily
# into our template based boxing/unboxing machinery, so changes
# to this convention will usually need template updates too.
#
# Prominent characteristics of the dispatcher API:
#
#   - dtype, layout, device and pin_memory are represented as separate
#     arguments.
#

def name(func: FunctionSchema) -> str:
    return cpp.name(func)


def argumenttype_type(t: Type, *, mutable: bool, binds: ArgName, symint: bool = True) -> NamedCType:
    # This is a faux amis.  If it makes sense in the future to add
    # more special cases here, or invert things so cpp.argument_type
    # calls this, or just completely inline the function, please do
    # it.
    return cpp.argumenttype_type(t, mutable=mutable, binds=binds, symint=symint)


def argument_type(a: Argument, *, binds: ArgName, symint: bool = True) -> NamedCType:
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds, symint=symint)


def returns_type(rs: Sequence[Return], *, symint: bool = True) -> CType:
    # At present, there is no difference. But there could be!
    return cpp.returns_type(rs, symint=symint)


def jit_arguments(func: FunctionSchema) -> List[Argument]:
    def to_argument(a: Union[Argument, TensorOptionsArguments, SelfArgument]) -> List[Argument]:
        if isinstance(a, Argument):
            return [a]
        elif isinstance(a, SelfArgument):
            return [a.argument]
        elif isinstance(a, TensorOptionsArguments):
            return [a.dtype, a.layout, a.device, a.pin_memory]
        else:
            assert_never(a)
    return list(concatMap(to_argument, itertools.chain(
        func.arguments.positional,
        func.arguments.kwarg_only,
        func.arguments.out)))


def argument(
    a: Argument, *, symint: bool = True
) -> Binding:
    return Binding(
        nctype=argument_type(
            a,
            binds=a.name,
            symint=symint,),
        name=a.name,
        argument=a,)


def arguments(func: FunctionSchema, *, symint: bool = True) -> List[Binding]:
    return [argument(a, symint=symint) for a in jit_arguments(func)]
