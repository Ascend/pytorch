# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

import glob
import inspect
import multiprocessing
import multiprocessing.pool
import os
import re
import shutil
import subprocess
import sys
import site

import distutils.ccompiler
import distutils.command.clean
from setuptools.command.build_ext import build_ext
from setuptools import setup, find_packages, distutils, Extension


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION = '1.8.1'


def _get_build_mode():
    for i in range(1, len(sys.argv)):
        if not sys.argv[i].startswith('-'):
            return sys.argv[i]


def generate_bindings_code(base_dir):
    generate_code_cmd = ["sh", os.path.join(base_dir, 'scripts', 'generate_code.sh')]
    if subprocess.call(generate_code_cmd) != 0:
        print(
            'Failed to generate ATEN bindings: {}'.format(generate_code_cmd),
            file=sys.stderr)
        sys.exit(1)


def get_npu_sources(base_dir):
    npu_sources = []
    for cur_dir, _, filenames in os.walk(os.path.join(base_dir, 'torch_npu/csrc')):
        for filename in filenames:
            if not filename.endswith('.cpp'):
                continue
            npu_sources.append(os.path.join(cur_dir, filename))

    return npu_sources


def _compile_parallel(self,
                      sources,
                      output_dir=None,
                      macros=None,
                      include_dirs=None,
                      debug=0,
                      extra_preargs=None,
                      extra_postargs=None,
                      depends=None):
    # Those lines are copied from distutils.ccompiler.CCompiler directly.
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
            output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    def compile_one(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            raise Exception(f'KeyError: {obj} not exists!')
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    list(
        multiprocessing.pool.ThreadPool(multiprocessing.cpu_count()).imap(compile_one, objects))
    return objects


if (os.getenv('COMPILE_PARALLEL', default='1').upper() in ['ON', '1', 'YES', 'TRUE', 'Y'] and
    inspect.signature(distutils.ccompiler.CCompiler.compile) == inspect.signature(_compile_parallel)):
    distutils.ccompiler.CCompiler.compile = _compile_parallel


def CppExtension(name, sources, *args, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for C++.
    '''
    if '--user' in sys.argv:
        package_dir = site.getusersitepackages()
    else:
        py_version = f'{sys.version_info.major}.{sys.version_info.minor}'
        package_dir = f'{sys.prefix}/lib/python{py_version}/site-packages'

    temp_include_dirs = kwargs.get('include_dirs', [])
    temp_include_dirs.append(os.path.join(package_dir, 'torch/include'))
    temp_include_dirs.append(os.path.join(package_dir, 'torch/include/torch/csrc/api/include'))
    kwargs['include_dirs'] = temp_include_dirs

    temp_library_dirs = kwargs.get('library_dirs', [])
    temp_library_dirs.append(os.path.join(package_dir, 'torch/lib'))
    temp_library_dirs.append(os.path.join(BASE_DIR, "third_party/acl/libs"))
    kwargs['library_dirs'] = temp_library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')
    libraries.append('hccl')
    kwargs['libraries'] = libraries
    kwargs['language'] = 'c++'
    return Extension(name, sources, *args, **kwargs)


class Clean(distutils.command.clean.clean):

    def run(self):
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
            for wildcard in filter(None, ignores.split('\n')):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    for filename in glob.glob(wildcard):
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


class Build(build_ext, object):

    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        return super(Build, self).build_extensions()


build_mode = _get_build_mode()
if build_mode not in ['clean']:
    # Generate bindings code, including RegisterNPU.cpp & NPUNativeFunctions.h.
    generate_bindings_code(BASE_DIR)

# Fetch the sources to be built.
torch_npu_sources = get_npu_sources(BASE_DIR)

lib_path = os.path.join(BASE_DIR, 'torch_npu/lib')
library_dirs = []
library_dirs.append(lib_path)

# Setup include directories folders.
include_directories = [
    BASE_DIR,
    os.path.join(BASE_DIR, 'torch_npu/csrc/aten'),
    os.path.join(BASE_DIR, 'third_party/hccl/inc'),
    os.path.join(BASE_DIR, 'third_party/acl/inc')
]

extra_link_args = []

DEBUG = (os.getenv('DEBUG', default='').upper() in ['ON', '1', 'YES', 'TRUE', 'Y'])

extra_compile_args = [
    '-std=c++14',
    '-Wno-sign-compare',
    '-Wno-deprecated-declarations',
    '-Wno-return-type',
]

if re.match(r'clang', os.getenv('CC', '')):
    extra_compile_args += [
        '-Wno-macro-redefined',
        '-Wno-return-std-move',
    ]

if DEBUG:
    extra_compile_args += ['-O0', '-g']
    extra_link_args += ['-O0', '-g']
else:
    extra_compile_args += ['-DNDEBUG']

setup(
        name=os.environ.get('TORCH_NPU_PACKAGE_NAME', 'torch_npu'),
        version=VERSION,
        description='NPU bridge for PyTorch',
        url='https://gitee.com/ascend/pytorch',
        author='PyTorch/NPU Dev Team',
        author_email='pytorch-npu@huawei.com',
        # Exclude the build files.
        packages=find_packages(exclude=['build']),
        ext_modules=[
            CppExtension(
                'torch_npu._C',
                torch_npu_sources,
                include_dirs=include_directories,
                extra_compile_args=extra_compile_args,
                library_dirs=library_dirs,
                extra_link_args=extra_link_args + \
                        ['-Wl,-rpath,$ORIGIN/torch_npu/lib'],
            ),
        ],
        extras_require={
        },
        package_data={
            'torch_npu': [
                    'lib/*.so*',
            ],
        },
        cmdclass={
            'build_ext': Build,
            'clean': Clean,
        })
