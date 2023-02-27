# Copyright (c) 2022 Huawei Technologies Co., Ltd
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
import sys
import platform
import subprocess
import importlib
from typing import List, Optional, Union

import setuptools
import torch
import torch.utils.cpp_extension as TorchExtension

import torch_npu


BISHENG_CPP_HOME = os.environ.get('BISHENG_CPP_HOME', '/opt/BiShengCPP')
if not os.path.exists(BISHENG_CPP_HOME):
    raise RuntimeError(f"Dir {BISHENG_CPP_HOME} not exists, please export BISHENG_CPP_HOME.")
PYTORCH_INSTALL_PATH = os.path.dirname(os.path.abspath(torch.__file__))
PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
PYTHON_INCLUES = subprocess.getoutput("python3-config --includes")
ASCEND_HOME_PATH = os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/ascend-toolkit/latest')
if not os.path.exists(ASCEND_HOME_PATH):
    raise RuntimeError(f"Dir {ASCEND_HOME_PATH} not exist, please run 'source set_env.sh' in the CANN install path.")
CANN_ARCH = platform.machine() + "-linux"
SOC_VERSION_SYCL_TARGET_DICT = {
    100 : "ascend_910-cce",
    101 : "ascend_910-cce",
    102 : "ascend_910-cce",
    103 : "ascend_910-cce",
    104 : "ascend_910-cce",
    200 : "ascend_310-cce",
    201 : "ascend_310-cce",
    202 : "ascend_310-cce",
    203 : "ascend_310-cce",
    220 : "ascend_910-cce",
    221 : "ascend_910-cce",
    222 : "ascend_910-cce",
    223 : "ascend_910-cce",
    240 : "ascend_310-cce",
    241 : "ascend_310-cce",
    242 : "ascend_310-cce",
    250 : "ascend_910-cce",
    251 : "ascend_910-cce",
    252 : "ascend_910-cce",
    253 : "ascend_910-cce"
}

npu_device = os.environ.get('SET_NPU_DEVICE')
if npu_device is None:
    torch.npu.set_device("npu:0")
else:
    torch.npu.set_device(f"npu:{npu_device}")
soc_version = torch_npu.npu.utils.get_soc_version()
if soc_version in SOC_VERSION_SYCL_TARGET_DICT:
    SYCL_TARGET = SOC_VERSION_SYCL_TARGET_DICT[soc_version]
else:
    SYCL_TARGET = "ascend_920-cce"


def NpuExtension(name, sources, *args, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a C++ extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> from setuptools import setup
        >>> from torch_npu.utils.cpp_extension import NpuExtension
        >>> setup(
                name='extension',
                ext_modules=[
                    NpuExtension(
                        name='extension',
                        sources=['extension.cpp'],
                        extra_compile_args=['-g']),
                ],
                cmdclass={
                    'build_ext': BuildExtension
                })
    '''

    torch_npu_dir = os.path.dirname(os.path.abspath(torch_npu.__file__))
    include_dirs = kwargs.get('include_dirs', [])    
    include_dirs.append(os.path.join(torch_npu_dir, 'include'))
    include_dirs += TorchExtension.include_paths()
    kwargs['include_dirs'] = include_dirs

    library_dirs = kwargs.get('library_dirs', [])    
    library_dirs.append(os.path.join(torch_npu_dir, 'lib'))
    library_dirs += TorchExtension.library_paths()
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')
    libraries.append('torch_npu')
    kwargs['libraries'] = libraries

    kwargs['language'] = 'c++'
    return setuptools.Extension(name, sources, *args, **kwargs)


def BiShengExtension(name, sources, *args, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a C++ extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension
        >>> from torch_npu.utils.cpp_extension import BiShengExtension
        >>> setup(
                name='extension',
                ext_modules=[
                    BiShengExtension(
                        name='extension',
                        sources=['extension.cpp']),
                ],
                cmdclass={
                    'build_ext': BuildExtension
                })
    '''

    torch_npu_dir = os.path.dirname(os.path.abspath(torch_npu.__file__))
    include_dirs = kwargs.get('include_dirs', [])    
    include_dirs.append(os.path.join(torch_npu_dir, 'include'))
    include_dirs.append(f"{ASCEND_HOME_PATH}/{CANN_ARCH}/include")
    include_dirs += TorchExtension.include_paths()
    
    kwargs['include_dirs'] = include_dirs

    library_dirs = kwargs.get('library_dirs', [])    
    library_dirs.append(os.path.join(torch_npu_dir, 'lib'))
    library_dirs.append(f"{ASCEND_HOME_PATH}/{CANN_ARCH}/lib64")
    library_dirs += TorchExtension.library_paths()
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')
    libraries.append('torch_npu')
    libraries.append('sycl')
    libraries.append('ascendcl')
    kwargs['libraries'] = libraries

    extra_compile_args = kwargs.get('extra_compile_args', [])
    extra_compile_args.append('-fsycl')
    extra_compile_args.append(f'-fsycl-targets={SYCL_TARGET}')
    extra_compile_args.append('-std=c++17')
    kwargs['extra_compile_args'] = extra_compile_args

    extra_link_args = kwargs.get('extra_link_args', [])
    extra_link_args.append('-fsycl')
    extra_link_args.append(f'-fsycl-targets={SYCL_TARGET}')
    extra_link_args.append('-std=c++17')
    kwargs['extra_link_args'] = extra_link_args

    kwargs['language'] = 'c++'
    os.environ["CC"] = f"{BISHENG_CPP_HOME}/bin/clang"
    os.environ["CXX"] = f"{BISHENG_CPP_HOME}/bin/clang++"
    return setuptools.Extension(name, sources, *args, **kwargs)


def _build_source_file(name, source, build_directory, extra_cflags, extra_include_paths, verbose):
    extra_include_paths_str = ""
    if extra_include_paths is not None:
        extra_include_paths_str = " ".join(extra_include_paths)

    extra_cflags_str = ""
    if extra_cflags is not None:
        extra_cflags_str = " ".join(extra_cflags)

    obj_name = source.replace(".cpp", ".o")

    cmd = f"{BISHENG_CPP_HOME}/bin/clang++  \
    -I{PYTORCH_NPU_INSTALL_PATH}/include \
    -I{PYTORCH_INSTALL_PATH}/include \
    -I{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include \
    -I{ASCEND_HOME_PATH}/{CANN_ARCH}/include \
    {PYTHON_INCLUES} \
    {extra_include_paths_str} \
    -D_GLIBCXX_USE_CXX11_ABI=1 -fPIC -std=c++17 -fsycl -fsycl-targets={SYCL_TARGET} \
    {extra_cflags_str} \
    -c {source} -o {build_directory}/{obj_name}"

    if verbose:
        print(cmd)
    os.system(cmd)


def _link_target(name, sources, build_directory, extra_ldflags, verbose):
    obj_name_list = []
    for source in sources:
        obj_name_list.append(os.path.join(
            build_directory, source.replace(".cpp", ".o")))
    obj_str = " ".join(obj_name_list)

    extra_ldflags_str = ""
    if extra_ldflags is not None:
        extra_ldflags_str = " ".join(extra_ldflags)

    cmd = f"{BISHENG_CPP_HOME}/bin/clang++ {obj_str} -o {build_directory}/{name}.so \
    -L{PYTORCH_INSTALL_PATH}/lib \
    -L{PYTORCH_NPU_INSTALL_PATH}/lib \
    -L{ASCEND_HOME_PATH}/{CANN_ARCH}/lib64 \
    -lc10 -ltorch_cpu -ltorch -ltorch_python -lsycl -lascendcl -ltorch_npu\
    -v -shared -fsycl -fsycl-targets={SYCL_TARGET} {extra_ldflags_str}"

    if verbose:
        print(cmd)
    os.system(cmd)


def _import_module_from_library(module_name, path, is_python_module, verbose):
    file_path = os.path.join(path, f"{module_name}.so")
    if verbose:
        print("module file path:", file_path)
    if is_python_module:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert isinstance(spec.loader, importlib.abc.Loader)
        spec.loader.exec_module(module)
        return module
    else:
        torch.ops.load_library(file_path)


def _get_build_directory(name: str, verbose: bool) -> str:
    root_extensions_directory = os.environ.get('TORCH_EXTENSIONS_DIR')
    if root_extensions_directory is None:
        root_extensions_directory = torch.utils.cpp_extension.get_default_build_root()
    build_directory = os.path.join(root_extensions_directory, name)
    if verbose:
        print("build dir:", build_directory)
    if not os.path.exists(build_directory):
        if verbose:
            print(f'Creating extension directory {build_directory}...', file=sys.stderr)
        os.makedirs(build_directory, exist_ok=True)

    return build_directory


# is_standalone and keep_intermediates may be used in future
def load(name,
         sources: Union[str, List[str]],
         extra_cflags=None,
         extra_ldflags=None,
         extra_include_paths=None,
         build_directory=None,
         verbose=False,
         is_python_module=True,
         is_standalone=False,
         keep_intermediates=True):
    build_directory = build_directory or _get_build_directory(name, verbose)
    for source in sources:
        _build_source_file(name, source, build_directory, extra_cflags, extra_include_paths, verbose)

    _link_target(name, sources, build_directory, extra_ldflags, verbose)

    return _import_module_from_library(name, build_directory, is_python_module, verbose)
