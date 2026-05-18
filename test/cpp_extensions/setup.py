import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

from torch_npu.utils.cpp_extension import NpuExtension
from torch_npu.testing.common_utils import set_npu_device

set_npu_device()

CXX_FLAGS = ['-g']

USE_NINJA = os.getenv('USE_NINJA') == '1'
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHIM_SOURCE = os.path.join(
    REPO_ROOT, "torch_npu", "csrc", "inductor", "aoti_torch", "shim_npu.cpp"
)

ext_modules = [
    NpuExtension(
        'torch_test_cpp_extension.npu', ['extension.cpp'],
        extra_compile_args=CXX_FLAGS),
    NpuExtension(
        'torch_test_cpp_extension.npu_from_blob', ['test_from_blob.cpp'],
        extra_compile_args=CXX_FLAGS),
    NpuExtension(
        'torch_test_cpp_extension.stable_libtorch', ['test_stable_libtorch.cpp'],
        extra_compile_args=CXX_FLAGS),
    NpuExtension(
        'torch_test_cpp_extension.npu_aoti_shim',['npu_aoti_shim_extension.cpp', SHIM_SOURCE],
        include_dirs=[REPO_ROOT],
        extra_compile_args=CXX_FLAGS),
]

setup(
    name='torch_test_cpp_extension',
    packages=['torch_test_cpp_extension'],
    ext_modules=ext_modules,
    include_dirs='self_compiler_include_dirs_test',
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=USE_NINJA)})
