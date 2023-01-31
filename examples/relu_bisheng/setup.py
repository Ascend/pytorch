# Copyright (c) 2023 Huawei Technologies Co., Ltd
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

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

from torch_npu.utils.cpp_extension import BiShengExtension


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
