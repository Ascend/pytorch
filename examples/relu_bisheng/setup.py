from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import BiShengExtension


torch_npu.npu.set_device(0)

setup(
    name='relu_bisheng',
    ext_modules=[
        BiShengExtension(
            name='relu_bisheng',
            sources=['relu_bisheng.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
