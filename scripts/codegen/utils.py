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

import re
import functools
import os
import stat
from typing import Tuple, List, Iterable, Iterator, Callable, Sequence, TypeVar, Optional, Dict, Any, Union, Set, NoReturn
from enum import Enum
import contextlib
import textwrap

from codegen.code_template import CodeTemplate

# Safely load fast C Yaml loader/dumper if they are available
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore[misc]

try:
    from yaml import CSafeDumper as Dumper
except ImportError:
    from yaml import SafeDumper as Dumper  # type: ignore[misc]
YamlDumper = Dumper

# Many of these functions share logic for defining both the definition
# and declaration (for example, the function signature is the same), so
# we organize them into one function that takes a Target to say which
# code we want.
#
# This is an OPEN enum (we may add more cases to it in the future), so be sure
# to explicitly specify with Union[Literal[Target.XXX]] what targets are valid
# for your use.
Target = Enum('Target', (
    # top level namespace (not including at)
    'DEFINITION',
    'DECLARATION',
    # TORCH_LIBRARY(...) { ... }
    'REGISTRATION',
    # namespace { ... }
    'ANONYMOUS_DEFINITION',
    # namespace cpu { ... }
    'NAMESPACED_DEFINITION',
    'NAMESPACED_DECLARATION',
))

# Matches "foo" in "foo, bar" but not "foobar". Used to search for the
# occurrence of a parameter in the derivative formula
IDENT_REGEX = r'(^|\W){}($|\W)'

# TODO: Use a real parser here; this will get bamboozled
def split_name_params(schema: str) -> Tuple[str, List[str]]:
    m = re.match(r'(\w+)(\.\w+)?\((.*)\)', schema)
    if m is None:
        raise RuntimeError(f'Unsupported function schema: {schema}')
    name, _, params = m.groups()
    return name, params.split(', ')

T = TypeVar('T')
S = TypeVar('S')

# These two functions purposely return generators in analogy to map()
# so that you don't mix up when you need to list() them

# Map over function that may return None; omit Nones from output sequence
def map_maybe(func: Callable[[T], Optional[S]], xs: Iterable[T]) -> Iterator[S]:
    for x in xs:
        r = func(x)
        if r is not None:
            yield r

# Map over function that returns sequences and cat them all together
def concat_map(func: Callable[[T], Sequence[S]], xs: Iterable[T]) -> Iterator[S]:
    for x in xs:
        for r in func(x):
            yield r

# Conveniently add error context to exceptions raised.  Lets us
# easily say that an error occurred while processing a specific
# context.
@contextlib.contextmanager
def context(msg_fn: Callable[[], str]) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        # TODO: this does the wrong thing with KeyError
        msg = msg_fn()
        msg = textwrap.indent(msg, '  ')
        msg = f'{e.args[0]}\n{msg}' if e.args else msg
        e.args = (msg,) + e.args[1:]
        raise

# A little trick from https://github.com/python/mypy/issues/6366
# for getting mypy to do exhaustiveness checking
# TODO: put this somewhere else, maybe
def assert_never(x: NoReturn) -> NoReturn:
    raise AssertionError("Unhandled type: {}".format(type(x).__name__))

@functools.lru_cache(maxsize=None)
def _read_template(template_fn: str) -> CodeTemplate:
    return CodeTemplate.from_file(template_fn)

# Map over function that may return None; omit Nones from output sequence
def mapMaybe(func: Callable[[T], Optional[S]], xs: Iterable[T]) -> Iterator[S]:
    for x in xs:
        r = func(x)
        if r is not None:
            yield r

# A custom loader for YAML that errors on duplicate keys.
# This doesn't happen by default: see https://github.com/yaml/pyyaml/issues/165
class YamlLoader(Loader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
            assert key not in mapping, f"Found a duplicate key in the yaml. key={key}, line={node.start_mark.line}"
            mapping.append(key)
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore[no-untyped-call]
        return mapping

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
        try:
            with open(filename, 'r') as f:
                old_contents = f.read()
        except IOError:
            old_contents = None
        if contents != old_contents:
            with os.fdopen(os.open(filename, os.O_RDWR|os.O_CREAT, stat.S_IWUSR|stat.S_IRUSR), "w") as f:
                f.write(contents)

    def write_with_template(self, filename: str, template_fn: str,
                            env_callable: Callable[[], Union[str, Dict[str, Any]]]) -> None:
        filename = '{}/{}'.format(self.install_dir, filename)
        assert filename not in self.filenames, "duplicate file write {filename}"
        self.filenames.add(filename)
        if not self.dry_run:
            env = env_callable()
            if isinstance(env, dict):
                # TODO: Update the comment reference to the correct location
                if 'generated_comment' not in env:
                    comment = "@" + "generated by tools/codegen/gen.py"
                    comment += " from {}".format(os.path.basename(template_fn))
                    env['generated_comment'] = comment
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
                    assert isinstance(shard[key], list), "sharded keys in base_env must be a list"
                    shard[key] = shard[key].copy()
                else:
                    shard[key] = []


        def merge_env(into: Dict[str, List[str]], from_: Dict[str, List[str]]) -> None:
            for k, v in from_.items():
                assert k in sharded_keys, f"undeclared sharded key {k}"
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

    def write_outputs(self, variable_name: str, filename: str) -> None:
        """Write a file containing the list of all outputs which are
        generated by this script."""
        content = 'set({}\n    {})'.format(
            variable_name, '\n    '.join('"' + name + '"' for name in sorted(self.filenames)))
        self._write_if_changed(filename, content)
