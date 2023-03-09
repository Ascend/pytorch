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


def _find_bisheng_cpp_home(check_none=True) -> Optional[str]:
    '''Finds the BISHENG_CPP install path.'''
    bisheng_cpp_home = os.environ.get('BISHENG_CPP_HOME')
    if check_none and BISHENG_CPP_HOME is None:
        raise RuntimeError(
            "BISHENG_CPP_HOME not exist, "
            "Place make sure that you have installed BiShengCpp "
            "and run 'source set_env.sh' in the install path.")
    return bisheng_cpp_home

BISHENG_CPP_HOME = _find_bisheng_cpp_home(False)

def _find_cann_home() -> Optional[str]:
    cann_home = os.environ.get('ASCEND_HOME_PATH')
    if cann_home is None:
        raise RuntimeError(
            "ASCEND_HOME_PATH not exist, "
            "please run 'source set_env.sh' in the CANN install path.")
    return cann_home

PYTORCH_INSTALL_PATH = os.path.dirname(os.path.abspath(torch.__file__))
PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
PLATFORM_ARCH = platform.machine() + "-linux"

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

def _get_sycl_target():
    soc_version = torch_npu.npu.utils.get_soc_version()
    sycl_target = SOC_VERSION_SYCL_TARGET_DICT.get(soc_version, None)
    if sycl_target is None:
        raise RuntimeError(
            "Please make sure: "
            "a) the current device supports BiShengCPP. "
            "b) use torch_npu.npu.set_device(device) to set device before building.")
    return sycl_target


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

    torch_npu_dir = PYTORCH_NPU_INSTALL_PATH
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
    Creates a :class:`setuptools.Extension` for BISHENGCPP.

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
    cann_home = _find_cann_home()
    sycl_target = _get_sycl_target()
    bishengcpp_home = _find_bisheng_cpp_home()

    include_dirs = kwargs.get('include_dirs', [])    
    include_dirs.append(os.path.join(PYTORCH_NPU_INSTALL_PATH, 'include'))
    include_dirs.append(os.path.join(cann_home, PLATFORM_ARCH, 'include'))
    include_dirs += TorchExtension.include_paths()
    kwargs['include_dirs'] = include_dirs

    library_dirs = kwargs.get('library_dirs', [])    
    library_dirs.append(os.path.join(PYTORCH_NPU_INSTALL_PATH, 'lib'))
    library_dirs.append(os.path.join(cann_home, PLATFORM_ARCH, 'lib64'))
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
    extra_compile_args.append(f'-fsycl-targets={sycl_target}')
    extra_compile_args.append('-std=c++17')
    kwargs['extra_compile_args'] = extra_compile_args

    extra_link_args = kwargs.get('extra_link_args', [])
    extra_link_args.append('-fsycl')
    extra_link_args.append(f'-fsycl-targets={sycl_target}')
    extra_link_args.append('-std=c++17')
    kwargs['extra_link_args'] = extra_link_args

    kwargs['language'] = 'c++'
    os.environ["CC"] = f"{bishengcpp_home}/bin/clang"
    os.environ["CXX"] = f"{bishengcpp_home}/bin/clang++"
    return setuptools.Extension(name, sources, *args, **kwargs)

def load(name,
         sources: Union[str, List[str]],
         extra_cflags=None,
         extra_bishengcpp_cflags=None,
         extra_ldflags=None,
         extra_include_paths=None,
         build_directory=None,
         verbose=False,
         with_bishengcpp: Optional[bool] = None,
         is_python_module=True,
         is_standalone=False,
         keep_intermediates=True):
    r'''
    Loads a PyTorch C++ extension just-in-time (JIT).

    To load an extension, a Ninja build file is emitted, which is used to
    compile the given sources into a dynamic library. This library is
    subsequently loaded into the current Python process as a module and
    returned from this function, ready for use.

    By default, the directory to which the build file is emitted and the
    resulting library compiled to is ``<tmp>/torch_extensions/<name>``, where
    ``<tmp>`` is the temporary folder on the current platform and ``<name>``
    the name of the extension. This location can be overridden in two ways.
    First, if the ``TORCH_EXTENSIONS_DIR`` environment variable is set, it
    replaces ``<tmp>/torch_extensions`` and all extensions will be compiled
    into subfolders of this directory. Second, if the ``build_directory``
    argument to this function is supplied, it overrides the entire path, i.e.
    the library will be compiled into that folder directly.

    To compile the sources, the default system compiler (``c++``) is used,
    which can be overridden by setting the ``CXX`` environment variable. To pass
    additional arguments to the compilation process, ``extra_cflags`` or
    ``extra_ldflags`` can be provided. For example, to compile your extension
    with optimizations, pass ``extra_cflags=['-O3']``. You can also use
    ``extra_cflags`` to pass further include directories.

    To compile BiShengCPP kernel, the BiSheng compiler
    (BISHENG_CPP_HOME/bin/clang++) is used. You need to run ``set_env.sh`` to
    set ``BISHENG_CPP_HOME`` and ``ASCEND_HOME_PATH``. You can pass additional
    flags to clang++ via ``extra_bishengcpp_cflags``, just like with
    ``extra_cflags`` for C++. 

    Args:
        name: The name of the extension to build. This MUST be the same as the
            name of the pybind11 module!
        sources: A list of relative or absolute paths to C++ source files.
        extra_cflags: optional list of compiler flags to forward to the build.
        extra_bishengcpp_cflags: optional list of compiler flags to forward to
            clang++ when building BiShengCPP sources.
        extra_ldflags: optional list of linker flags to forward to the build.
        extra_include_paths: optional list of include directories to forward
            to the build.
        build_directory: optional path to use as build workspace.
        verbose: If ``True``, turns on verbose logging of load steps.
        with_bishengcpp: Determines whether to build with BiShengCPP.
            If set to ``None`` (default), this value is automatically determined
            based on the existence of BISHENG_CPP_HOME. Set it to `True`` to
            build with BiShengCPP.
        is_python_module: If ``True`` (default), imports the produced shared
            library as a Python module. ``False``, .
        is_standalone: only support ``False`` (default), loads the constructed extension
            into the process as a plain dynamic library.
    Returns:
        If ``is_python_module`` is ``True``:
            Returns the loaded PyTorch extension as a Python module.

        If ``is_python_module`` is ``False`` and ``is_standalone`` is ``False``:
            Returns nothing. (The shared library is loaded into the process as
            a side effect.)

        If ``is_standalone`` is ``True``.
            Returns nothing. (load does not support is_standalone = True)

    Example:
        >>> from torch_npu.utils.cpp_extension import load
        >>> module = load(
                name='extension',
                sources=['extension.cpp'],
                extra_cflags=['-g'],
                verbose=True)
    '''

    if with_bishengcpp is None and BISHENG_CPP_HOME is not None:
        with_bishengcpp = True

    if with_bishengcpp:
        cann_home = _find_cann_home()
        sycl_target = _get_sycl_target()
        bishengcpp_home = _find_bisheng_cpp_home()
        
        os.environ["CC"] = f"{bishengcpp_home}/bin/clang"
        os.environ["CXX"] = f"{bishengcpp_home}/bin/clang++"

        extra_cflags = extra_cflags or []
        if not isinstance(extra_cflags, list):
            raise RuntimeError("arg extra_cflags should be None or List[str], "
                f"not {type(extra_cflags)}")
        extra_cflags.append("-fsycl")
        extra_cflags.append(f'-fsycl-targets={sycl_target}')
        extra_cflags.append('-std=c++17')
        if extra_bishengcpp_cflags:
            extra_cflags.extend(extra_bishengcpp_cflags)

    extra_ldflags = extra_ldflags or []
    if not isinstance(extra_ldflags, list):
            raise RuntimeError("arg extra_ldflags should be None or List[str], "
                f"not {type(extra_ldflags)}")
    if with_bishengcpp:
        extra_ldflags.append("-lsycl")
        extra_ldflags.append("-lascendcl")
        extra_ldflags.append("-fsycl")
        extra_ldflags.append(f'-fsycl-targets={sycl_target}')
        extra_ldflags.append(f"-L{cann_home}/{PLATFORM_ARCH}/lib64")
    extra_ldflags.append("-ltorch_npu")
    extra_ldflags.append(f"-L{PYTORCH_NPU_INSTALL_PATH}/lib")

    extra_include_paths = extra_include_paths or []
    if not isinstance(extra_include_paths, list):
            raise RuntimeError("arg extra_include_paths should be None or List[str], " +
                f"not {type(extra_include_paths)}")
    extra_include_paths.append(os.path.join(PYTORCH_NPU_INSTALL_PATH, 'include'))
    if with_bishengcpp:
        extra_include_paths.append(os.path.join(cann_home, PLATFORM_ARCH, "include"))

    return TorchExtension.load(name,
                               sources,
                               extra_cflags,
                               None,
                               extra_ldflags,
                               extra_include_paths,
                               build_directory,
                               verbose,
                               False,
                               is_python_module,
                               is_standalone,
                               keep_intermediates)
