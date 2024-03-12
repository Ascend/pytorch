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

import os
import stat
import functools
import hashlib
from typing import (List, Dict, Optional, Set, Callable, Any, 
                    Union, Sequence, TypeVar, Iterable)
from collections import defaultdict

from codegen.code_template import CodeTemplate
from codegen.model import (FunctionSchema, NativeFunction,
                           NativeFunctionsGroup, OperatorName,
                           SchemaKind, assert_never)
from codegen.utils import concat_map, PathManager

T = TypeVar('T')

# Welcome to the ATen code generator v2!  The ATen code generator is
# responsible for parsing native_functions.yaml and then generating
# various generated files (e.g., TypeDefault.cpp) based on the operators
# defined in this file.  This means that the code generator knows how to
# parse function schema, and then translate this into various C++ types
# and boilerplate code.
#
# Some things to know about this file when you modify it:
#
# - This file has STRICT mypy typechecking.  Typecheck it with
#   `mypy --config mypy-strict.ini` in the root source directory
#
# - Most of the heavy lifting lives in external modules:
#   - 'model' has the data model for native_functions.yaml.  The classes
#     in those file represent what you see when you look at
#     a native_functions.yaml
#   - 'api' has conversions for how to translate JIT schema into
#     the various C++ APIs that the codegen interacts with.  There
#     are in fact THREE different C++ APIs: the public C++ API,
#     the dispatcher API, and the legacy disaptcher API.  See each
#     of these respective files for more information

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                         HELPER FUNCTIONS
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# Some assertions are already performed during parsing, but those are only within a single NativeFunction.
# Assertions here are meant to be performed across NativeFunctions.
def error_check_native_functions(funcs: Sequence[NativeFunction]) -> None:
    func_map: Dict[OperatorName, NativeFunction] = {}
    for f in funcs:
        func_map[f.func.name] = f
    for f in funcs:
        if f.structured_delegate is not None:
            delegate_func = func_map.get(f.structured_delegate)
            if not delegate_func.structured:
                raise ValueError(f"{f.func.name} is marked as a structured_delegate pointing to " \
                                 f"{f.structured_delegate}, but {f.structured_delegate} is not marked as structured. " \
                                 f"Consider adding 'structured=True' to the delegated operator")


def cpp_string(s: str) -> str:
    """Convert a python string into a c++ string literal """
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\a', '\\a')
    s = s.replace('\b', '\\b')
    s = s.replace('\f', '\\f')
    s = s.replace('\n', '\\n')
    s = s.replace('\v', '\\v')
    s = s.replace('\t', '\\t')
    return f'"{s}"'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           RUN IT ALL
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

@functools.lru_cache(maxsize=None)
def _read_template(template_fn: str) -> CodeTemplate:
    return CodeTemplate.from_file(template_fn)


# String hash that's stable across different executions, unlike builtin hash
def string_stable_hash(s: str) -> int:
    sha1 = hashlib.sha256(s.encode('latin1')).digest()
    return int.from_bytes(sha1, byteorder='little')


# A small abstraction for writing out generated files and keeping track
# of what files have been written (so you can write out a list of output
# files)
class FileManager:
    install_dir: str
    template_dir: str
    dry_run: bool
    filenames: Set[str]

    def __init__(self, install_dir: str, template_dir: str, dry_run: bool) -> None:
        self.install_dir = install_dir
        self.template_dir = template_dir
        self.filenames = set()
        self.dry_run = dry_run

    @staticmethod
    def _write_if_changed(filename: str, contents: str) -> None:
        old_contents: Optional[str]
        filepath = os.path.realpath(filename)
        try:
            with open(filepath, 'r') as f:
                old_contents = f.read()
        except IOError:
            old_contents = None
        if contents != old_contents:
            PathManager.remove_path_safety(filepath)
            with os.fdopen(os.open(filepath, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), "w") as f:
                f.write(contents)
            os.chmod(filepath, stat.S_IRUSR | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)

    def write_with_template(self, filename: str, template_fn: str,
                            env_callable: Callable[[], Union[str, Dict[str, Any]]]) -> None:
        filename = '{}/{}'.format(self.install_dir, filename)
        if filename in self.filenames:
            raise ValueError("duplicate file write {filename}")
        self.filenames.add(filename)
        if not self.dry_run:
            env = env_callable()
            if isinstance(env, dict):
                if 'generated_comment' not in env:
                    comment = "@" + "generated by tools/codegen/gen.py"
                    comment += " from {}".format(os.path.basename(template_fn))
                    env['generated_comment'] = comment
                env['legacy_th_headers'] = []
                template = _read_template(os.path.join(self.template_dir, template_fn))
                self._write_if_changed(filename, template.substitute(env))
            elif isinstance(env, str):
                self._write_if_changed(filename, env)
            else:
                assert_never(env)

    def write(self, filename: str, env_callable: Callable[[], Union[str, Union[str, Dict[str, Any]]]]) -> None:
        self.write_with_template(filename, filename, env_callable)

    def write_sharded(
            self,
            filename: str,
            items: Iterable[T],
            *,
            key_fn: Callable[[T], str],
            env_callable: Callable[[T], Dict[str, List[str]]],
            num_shards: int,
            base_env: Optional[Dict[str, Any]] = None,
            sharded_keys: Set[str]
    ) -> None:

        everything: Dict[str, Any] = {'shard_id': 'Everything'}
        shards: List[Dict[str, Any]] = [{'shard_id': f'_{i}'} for i in range(num_shards)]
        all_shards = [everything] + shards

        if base_env is not None:
            for shard in all_shards:
                shard.update(base_env)

        for key in sharded_keys:
            for shard in all_shards:
                if key in shard:
                    if not isinstance(shard[key], list):
                        raise TypeError("sharded keys in base_env must be a list")
                    shard[key] = shard[key].copy()
                else:
                    shard[key] = []

        def merge_env(into: Dict[str, List[str]], from_: Dict[str, List[str]]) -> None:
            for k, v in from_.items():
                if k not in sharded_keys:
                    raise KeyError(f"undeclared sharded key {k}")
                into[k] += v

        for item in items:
            key = key_fn(item)
            sid = string_stable_hash(key) % num_shards
            env = env_callable(item)

            merge_env(shards[sid], env)
            merge_env(everything, env)

        dot_pos = filename.rfind('.')
        if dot_pos == -1:
            dot_pos = len(filename)
        base_filename = filename[:dot_pos]
        extension = filename[dot_pos:]

        for shard in all_shards:
            shard_id = shard['shard_id']
            self.write_with_template(f"{base_filename}{shard_id}{extension}",
                                     filename,
                                     lambda: shard)

        # filenames is used to track compiled files, but FooEverything.cpp isn't meant to be compiled
        self.filenames.discard(
            f"{self.install_dir}/{base_filename}Everything{extension}")

    def write_outputs(self, filename: str) -> None:
        """Write a file containing the list of all outputs which are
        generated by this script.
        """
        self._write_if_changed(
            filename,
            ''.join(name + ";" for name in sorted(self.filenames)))


def get_grouped_native_functions(
        native_functions: Sequence[NativeFunction]) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
    pre_grouped_native_functions: Dict[FunctionSchema, Dict[SchemaKind, NativeFunction]] = defaultdict(dict)
    for f in native_functions:
        d = pre_grouped_native_functions[f.func.signature()]
        if f.func.kind() in d:
            raise KeyError("f.func.kind() in d")
        d[f.func.kind()] = f

    def flatten_pre_group(d: Dict[SchemaKind, NativeFunction]) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
        r = NativeFunctionsGroup.from_dict(d)
        if r is None:
            return list(d.values())
        else:
            return [r]

    return list(concat_map(flatten_pre_group, list(pre_grouped_native_functions.values())))
