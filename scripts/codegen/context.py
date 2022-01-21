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

from codegen.utils import S, T, context
from codegen.model import (NativeFunction, NativeFunctionsGroup, BackendIndex, DispatchKey)
import codegen.local as local

import functools
from typing import TypeVar, Union, Iterator, Callable, Dict
import contextlib

# Helper functions for defining generators on things in the model

F = TypeVar(
    'F',
    NativeFunction,
    NativeFunctionsGroup,
    Union[NativeFunction, NativeFunctionsGroup],
)

@contextlib.contextmanager
def native_function_manager(g: Union[NativeFunctionsGroup, NativeFunction]) -> Iterator[None]:
    if isinstance(g, NativeFunctionsGroup):
        # By default, we associate all errors with structured native functions
        # with the out variant.  In some cases, it might be better to have
        # a more specific place to hang things; if so, use
        # native_function_manager again on the inside
        f = g.out
    else:
        f = g
    with context(lambda: f'in native_functions.yaml line {f.loc}:\n  {f.func}'):
        with local.parametrize(new_use_const_ref_for_mutable_tensors=f.use_const_ref_for_mutable_tensors):
            yield

# Given a function that operates on NativeFunction, wrap it into a new function
# that sets some appropriate context managers for that native function.
# YOU MUST WRAP FUNCTIONS IN THIS for calls to api modules to be sound
# (you will get an error if we try to access the local variables without having
# set them).
def with_native_function(func: Callable[[F], T]) -> Callable[[F], T]:
    @functools.wraps(func)
    def wrapper(f: F) -> T:
        with native_function_manager(f):
            return func(f)
    return wrapper

def method_with_native_function(func: Callable[[S, F], T]) -> Callable[[S, F], T]:
    @functools.wraps(func)
    def wrapper(slf: S, f: F) -> T:
        with native_function_manager(f):
            return func(slf, f)
    return wrapper

# Convenience decorator for functions that explicitly take in a BackendIndex,
# instead of indirectly taking one in as a closure
def with_native_function_and_index(func: Callable[[F, BackendIndex], T]) -> Callable[[F, BackendIndex], T]:
    @functools.wraps(func)
    def wrapper(f: F, backend_index: BackendIndex) -> T:
        with native_function_manager(f):
            return func(f, backend_index)
    return wrapper

def with_native_function_and_indices(
        func: Callable[[F, Dict[DispatchKey, BackendIndex]], T]
) -> Callable[[F, Dict[DispatchKey, BackendIndex]], T]:
    @functools.wraps(func)
    def wrapper(f: F, backend_indices: Dict[DispatchKey, BackendIndex]) -> T:
        with native_function_manager(f):
            return func(f, backend_indices)
    return wrapper
