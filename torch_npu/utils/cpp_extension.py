import os
import setuptools

import torch
import torch.utils.cpp_extension as TorchExtension

import torch_npu

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.realpath(torch_npu.__file__))


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
