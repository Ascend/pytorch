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

from codegen.model import FunctionSchema, NativeFunction, BackendIndex
from dataclasses import dataclass
from typing import Optional, Union, Sequence, List, Set
from codegen.api.types import Binding, CType, Expr
from codegen.api import cpp, dispatcher, native, translate


# A CppSignature represents a single overload in the C++ API.  For
# any given function schema, there may be multiple CppSignatures
# corresponding to it, based on how we desugar to C++.  See also
# CppSignatureGroup.
@dataclass(frozen=True)
class CppSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    # Is this a C++ signature for a method, i.e. Tensor::my_op(...)?
    method: bool

    # Is this a faithful C++ signature (i.e. following the JIT schema) or a convenience API
    # (i.e. with a potential TensorOptions argument and out arguments in the front)
    faithful: bool

    # The set of C++ arguments which should not have defaults applied to them
    cpp_no_default_args: Set[str]

    # Is this a fallback C++ binding?  Fallback bindings are enabled by
    # manual_cpp_binding: True and are alternate, non-public API that
    # lets manual C++ binding implementors access the binding that would
    # have been automatically generated
    fallback_binding: bool = False

    # Return the unpacked argument structure of this signature,
    # discarding information about which arguments are semantically
    # related to each other.
    def arguments(self) -> Sequence[Binding]:
        return cpp.arguments(
            self.func.arguments, faithful=self.faithful,
            method=self.method, cpp_no_default_args=self.cpp_no_default_args)

    def name(self) -> str:
        n = cpp.name(self.func, faithful_name_for_out_overloads=self.faithful)
        if self.fallback_binding:
            n = f"__dispatch_{n}"
        return n

    # Render the C++ declaration for this signature
    def decl(self, *, name: Optional[str] = None, prefix: str = "", is_redispatching_fn: bool = False) -> str:
        returns_type = cpp.returns_type(self.func.returns).cpp_type()
        cpp_args = [a.decl() for a in self.arguments()]
        if is_redispatching_fn:
            cpp_args = ['c10::DispatchKeySet dispatchKeySet'] + cpp_args
        cpp_args_str = ', '.join(cpp_args)
        if name is None:
            name = prefix + self.name()
        return f"{returns_type} {name}({cpp_args_str})"

    # Render the C++ definition for this signature, not including
    # the body (with curly braces)
    def defn(self, *, name: Optional[str] = None, prefix: str = "", is_redispatching_fn: bool = False) -> str:
        returns_type = cpp.returns_type(self.func.returns).cpp_type()
        cpp_args = [a.defn() for a in self.arguments()]
        if is_redispatching_fn:
            cpp_args = ['c10::DispatchKeySet dispatchKeySet'] + cpp_args
        cpp_args_str = ', '.join(cpp_args)
        if name is None:
            name = prefix + self.name()
        return f"{returns_type} {name}({cpp_args_str})"

    def ptr_type(self) -> str:
        args_types_str = ', '.join(a.type for a in self.arguments())
        return f'{cpp.returns_type(self.func.returns).cpp_type()} (*)({args_types_str})'

    # Return the C++ function type, e.g., something like int(bool)
    def type(self) -> str:
        args_types_str = ', '.join(a.type for a in self.arguments())
        return f'{cpp.returns_type(self.func.returns).cpp_type()} ({args_types_str})'


# Represents group of all CppSignatures associated with a
# FunctionSchema.  Right now, that's the regular, user-visible
# signature, as well as a "faithful" signature which doesn't
# have grouping.
@dataclass(frozen=True)
class CppSignatureGroup:
    func: FunctionSchema
    signature: CppSignature
    faithful_signature: Optional[CppSignature]

    def most_faithful_signature(self) -> CppSignature:
        if self.faithful_signature:
            return self.faithful_signature
        else:
            return self.signature

    @staticmethod
    def from_native_function(f: NativeFunction, *, method: bool, fallback_binding: bool = False) -> 'CppSignatureGroup':
        func = f.func
        faithful_signature: Optional[CppSignature]
        if func.arguments.tensor_options is not None or len(func.arguments.out) > 0:
            faithful_signature = CppSignature(
                func=func,
                faithful=True,
                method=method,
                fallback_binding=fallback_binding,
                cpp_no_default_args=f.cpp_no_default_args
            )
        else:
            faithful_signature = None
        signature = CppSignature(
            func=func,
            faithful=False,
            method=method,
            fallback_binding=fallback_binding,
            cpp_no_default_args=f.cpp_no_default_args
        )
        return CppSignatureGroup(
            func=func,
            signature=signature,
            faithful_signature=faithful_signature,
        )

@dataclass(frozen=True)
class DispatcherSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    # Allows you to prepend an arbitrary prefix to the signature name.
    # This is useful for parts of the codegen that generate wrappers around kernels,
    # and need to avoid naming collisions.
    prefix: str = ""

    def arguments(self) -> List[Binding]:
        return dispatcher.arguments(self.func)

    def name(self) -> str:
        return self.prefix + dispatcher.name(self.func)

    def decl(self, name: Optional[str] = None) -> str:
        args_str = ', '.join(a.decl() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    def defn(self, name: Optional[str] = None, *, is_redispatching_fn: bool = False) -> str:
        args = [a.defn() for a in self.arguments()]
        if is_redispatching_fn:
            args = ['c10::DispatchKeySet dispatchKeySet'] + args
        args_str = ', '.join(args)
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    def exprs(self) -> List[Expr]:
        return [Expr(a.name, a.nctype) for a in self.arguments()]

    def returns_type(self) -> CType:
        return dispatcher.returns_type(self.func.returns)

    def ptr_type(self) -> str:
        dispatcher_args_types_str = ', '.join(a.type for a in self.arguments())
        return f'{self.returns_type().cpp_type()} (*)({dispatcher_args_types_str})'

    # Return the C++ function type, e.g., something like int(bool)
    def type(self) -> str:
        dispatcher_args_types_str = ', '.join(a.type for a in self.arguments())
        return f'{self.returns_type().cpp_type()} ({dispatcher_args_types_str})'

    @staticmethod
    def from_schema(func: FunctionSchema, *, prefix: str = '') -> 'DispatcherSignature':
        return DispatcherSignature(func, prefix)

@dataclass(frozen=True)
class NativeSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    prefix: str = ""

    def name(self) -> str:
        return self.prefix + native.name(self.func)

    def decl(self, name: Optional[str] = None) -> str:
        args_str = ', '.join(a.decl() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{native.returns_type(self.func.returns).cpp_type()} {name}({args_str})"

    def defn(self, name: Optional[str] = None) -> str:
        args_str = ', '.join(a.defn() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{native.returns_type(self.func.returns).cpp_type()} {name}({args_str})"

    def ptr_type(self) -> str:
        # don't include defaults in type signature!
        args_str = ', '.join(a.defn() for a in self.arguments())
        return f'{native.returns_type(self.func.returns).cpp_type()} (*)({args_str})'

    def arguments(self) -> List[Binding]:
        return native.arguments(self.func)

    def returns_type(self) -> CType:
        return native.returns_type(self.func.returns)

    def dispatcher_exprs(self) -> List[Expr]:
        return translate.translate(self.arguments(), dispatcher.arguments(self.func), method=False)


# Helper functions

def kernel_signature(
        f: NativeFunction, backend_index: BackendIndex, *, prefix: str = '') -> Union['NativeSignature', 'DispatcherSignature']:
    # Note [External Backends Follow Dispatcher API]
    # Kernel signatures for in-tree backends follow the "native" API,
    # while kernels for out-of-tree backends follow the dispatcher API.
    # See the comments in `native.py` for details, but historically there have been
    # some small differences in schema convention between them and the Dispatcher API.
    # Any differences that require translating between the two will results in a runtime cost,
    # so we'd like to keep the differences as small as possible.
    # With external backends, we'd like to enforce that they write their kernels with schemas
    # that match the Dispatcher API directly, if they can.
    if backend_index.external:
        return DispatcherSignature.from_schema(f.func, prefix=prefix)
    else:
        return NativeSignature(f.func, prefix)
