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

from dataclasses import dataclass
from typing import Optional, Union, TypeVar, List, Dict
from enum import Enum

from codegen.model import Argument, SelfArgument, TensorOptionsArguments, BaseTy

_T = TypeVar('_T')

# An ArgName is just the str name of the argument in schema;
# but in some special circumstances, we may add a little extra
# context.  The Enum SpecialArgName covers all of these cases;
# grep for their construction sites to see when they can occr.

SpecialArgName = Enum('SpecialArgName', (
    'possibly_redundant_memory_format',
))
ArgName = Union[str, SpecialArgName]

# This class shouldn't be created directly; instead, use/create one of the singletons below.
@dataclass(frozen=True)
class BaseCppType:
    ns: Optional[str]
    name: str

    def __str__(self) -> str:
        if self.ns is None or self.ns == '':
            return self.name
        return f"{self.ns}::{self.name}"

# The set of all non-templated, valid, fully-qualified names of C++ types that are used in the codegen.
# Templated types get their own dataclass, mainly to make namespace parsing easier.
intT = BaseCppType('', 'int64_t')
doubleT = BaseCppType('', 'double')
boolT = BaseCppType('', 'bool')
voidT = BaseCppType('', 'void')
longT = BaseCppType('', 'int64_t')
stringT = BaseCppType('c10', 'string_view')
generatorT = BaseCppType('at', 'Generator')
scalarTypeT = BaseCppType('at', 'ScalarType')
tensorT = BaseCppType('at', 'Tensor')
optionalTensorRefT = BaseCppType('at', 'OptionalTensorRef')
tensorListT = BaseCppType('at', 'TensorList')
dimnameT = BaseCppType('at', 'Dimname')
dimnameListT = BaseCppType('at', 'DimnameList')
layoutT = BaseCppType('at', 'Layout')
deviceT = BaseCppType('at', 'Device')
scalarT = BaseCppType('at', 'Scalar')
optionalScalarRefT = BaseCppType('at', 'OptionalScalarRef')
memoryFormatT = BaseCppType('at', 'MemoryFormat')
qschemeT = BaseCppType('at', 'QScheme')
storageT = BaseCppType('at', 'Storage')
streamT = BaseCppType('at', 'Stream')
intArrayRefT = BaseCppType('at', 'IntArrayRef')
tensorOptionsT = BaseCppType('at', 'TensorOptions')
typeAndSizeT = BaseCppType('torch::autograd::generated', 'TypeAndSize')
tensorGeometryT = BaseCppType('at', 'TensorGeometry')

BaseTypeToCppMapping: Dict[BaseTy, BaseCppType] = {
    BaseTy.int: intT,
    BaseTy.float: doubleT,
    BaseTy.bool: boolT,
    BaseTy.str: stringT,
    BaseTy.Generator: generatorT,
    BaseTy.ScalarType: scalarTypeT,
    BaseTy.Tensor: tensorT,
    BaseTy.Dimname: dimnameT,
    BaseTy.Layout: layoutT,
    BaseTy.Device: deviceT,
    BaseTy.Scalar: scalarT,
    BaseTy.MemoryFormat: memoryFormatT,
    BaseTy.QScheme: qschemeT,
    BaseTy.Storage: storageT,
    BaseTy.Stream: streamT,
}

# CTypes encode C++ type structure as needed for translation.

@dataclass(frozen=True)
class BaseCType:
    type: BaseCppType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return str(self.type)

    # For BC reasons, we don't want to introduce at:: namespaces to RegistrationDeclarations.yaml
    # TODO: Kill this when we eventually remove it!
    def cpp_type_registration_declarations(self) -> str:
        return str(self.type).replace('at::', '')

    def remove_const_ref(self) -> 'CType':
        return self

@dataclass(frozen=True)
class ConstRefCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        if strip_ref:
            return self.elem.cpp_type(strip_ref=strip_ref)
        return f'const {self.elem.cpp_type()} &'

    def cpp_type_registration_declarations(self) -> str:
        return f'const {self.elem.cpp_type_registration_declarations()} &'

    def remove_const_ref(self) -> 'CType':
        return self.elem.remove_const_ref()

@dataclass(frozen=True)
class MutRefCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        if strip_ref:
            return self.elem.cpp_type(strip_ref=strip_ref)
        return f'{self.elem.cpp_type()} &'

    def cpp_type_registration_declarations(self) -> str:
        return f'{self.elem.cpp_type_registration_declarations()} &'

    def remove_const_ref(self) -> 'CType':
        return self.elem.remove_const_ref()

@dataclass(frozen=True)
class OptionalCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'c10::optional<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'c10::optional<{self.elem.cpp_type_registration_declarations()}>'

    def remove_const_ref(self) -> 'CType':
        return OptionalCType(self.elem.remove_const_ref())

@dataclass(frozen=True)
class ListCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'c10::List<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'c10::List<{self.elem.cpp_type_registration_declarations()}>'

    def remove_const_ref(self) -> 'CType':
        return ListCType(self.elem.remove_const_ref())

@dataclass(frozen=True)
class ArrayRefCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'at::ArrayRef<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'ArrayRef<{self.elem.cpp_type_registration_declarations()}>'

    def remove_const_ref(self) -> 'CType':
        return ArrayRefCType(self.elem.remove_const_ref())

@dataclass(frozen=True)
class VectorCType:
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'::std::vector<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'::std::vector<{self.elem.cpp_type_registration_declarations()}>'

    def remove_const_ref(self) -> 'CType':
        return VectorCType(self.elem.remove_const_ref())

@dataclass(frozen=True)
class ArrayCType:
    elem: 'CType'
    size: int

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'::std::array<{self.elem.cpp_type()},{self.size}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'::std::array<{self.elem.cpp_type_registration_declarations()},{self.size}>'

    def remove_const_ref(self) -> 'CType':
        return ArrayCType(self.elem.remove_const_ref(), self.size)

@dataclass(frozen=True)
class TupleCType:
    elems: List['CType']

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f'::std::tuple<{",".join([e.cpp_type() for e in self.elems])}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'::std::tuple<{",".join([e.cpp_type_registration_declarations() for e in self.elems])}>'

    def remove_const_ref(self) -> 'CType':
        return TupleCType([e.remove_const_ref() for e in self.elems])

CType = Union[
    BaseCType,
    OptionalCType,
    ConstRefCType,
    MutRefCType,
    ListCType,
    ArrayRefCType,
    ArrayCType,
    VectorCType,
    TupleCType
]

# A NamedCType is short for Named C++ semantic type.  A NamedCType represents a C++ type, plus
# semantic information about what it represents.  For example, consider the
# argument "bool pin_memory"; its normal C++ type is "bool", but its C++
# semantic type also keeps track that this represents a "pin_memory"; you can't
# just use a random other boolean in a context where you need a "pin_memory"!
#

@dataclass(frozen=True)
class NamedCType:
    name: ArgName
    type: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return self.type.cpp_type(strip_ref=strip_ref)

    # For BC reasons, we don't want to introduce at:: namespaces to RegistrationDeclarations.yaml
    # TODO: Kill this when we eventually remove it!
    def cpp_type_registration_declarations(self) -> str:
        return self.type.cpp_type_registration_declarations()

    def remove_const_ref(self) -> 'NamedCType':
        return NamedCType(self.name, self.type.remove_const_ref())

    def with_name(self, name: str) -> 'NamedCType':
        return NamedCType(name, self.type)

# A binding represents any C++ binding site for a formal parameter.
# We don't distinguish between binding sites for different APIs;
# instead, all of the important distinctions are encoded in CType,
# which you can use to figure out if a given Binding is appropriate
# for use in another context.  (See codegen.api.translate)

@dataclass(frozen=True)
class Binding:
    name: str
    nctype: NamedCType
    argument: Union[Argument, TensorOptionsArguments, SelfArgument]
    # TODO: maybe don't represent default here
    default: Optional[str] = None

    @property
    def type(self) -> str:
        return self.nctype.cpp_type()

    def no_default(self) -> 'Binding':
        return Binding(
            name=self.name,
            nctype=self.nctype,
            default=None,
            argument=self.argument,
        )

    def decl(self, *, func_ptr_cast: bool = False) -> str:
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"

        # casting only needs to know the type
        if func_ptr_cast:
            return f"{self.type}"
        else:
            return f"{self.type} {self.name}{mb_default}"

    # For BC reasons, we don't want to introduce at:: namespaces to RegistrationDeclarations.yaml
    # TODO: Kill this when we eventually remove it!
    def decl_registration_declarations(self) -> str:
        type_s = self.nctype.cpp_type_registration_declarations()
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"
        return f"{type_s} {self.name}{mb_default}"

    def defn(self) -> str:
        return f"{self.type} {self.name}"

# An Expr is a C++ expression.  It has a C++ string representing its syntax,
# as well as a CType saying what it provides.

@dataclass(frozen=True)
class Expr:
    expr: str
    type: NamedCType
