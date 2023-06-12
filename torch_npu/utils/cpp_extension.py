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


PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))

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

    Args:
        name: The name of the extension to build. This MUST be the same as the
            name of the pybind11 module!
        sources: A list of relative or absolute paths to C++ source files.
        extra_cflags: optional list of compiler flags to forward to the build.
        extra_ldflags: optional list of linker flags to forward to the build.
        extra_include_paths: optional list of include directories to forward
            to the build.
        build_directory: optional path to use as build workspace.
        verbose: If ``True``, turns on verbose logging of load steps.
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

    extra_ldflags = extra_ldflags or []
    if not isinstance(extra_ldflags, list):
            raise RuntimeError("arg extra_ldflags should be None or List[str], "
                f"not {type(extra_ldflags)}")

    extra_ldflags.append("-ltorch_npu")
    extra_ldflags.append(f"-L{PYTORCH_NPU_INSTALL_PATH}/lib")

    extra_include_paths = extra_include_paths or []
    if not isinstance(extra_include_paths, list):
            raise RuntimeError("arg extra_include_paths should be None or List[str], " +
                f"not {type(extra_include_paths)}")
    extra_include_paths.append(os.path.join(PYTORCH_NPU_INSTALL_PATH, 'include'))


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
