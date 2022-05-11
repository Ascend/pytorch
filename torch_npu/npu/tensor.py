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


import functools
import torch


NpuTensorDict = {
    "float": torch.FloatTensor,
    "int": torch.IntTensor,
    "double": torch.DoubleTensor,
    "long": torch.LongTensor,
    "short": torch.ShortTensor,
    "byte": torch.ByteTensor,
    "char": torch.CharTensor
}

def create_npu_float_tensor(*args, **kwargs):
    device = kwargs.pop("device", "npu")
    dtype = kwargs.pop("dtype", None)
    assert dtype and dtype in NpuTensorDict
    return NpuTensorDict[dtype](*args, **kwargs).to(device)


FloatTensor = functools.partial(create_npu_float_tensor, dtype="float")
IntTensor = functools.partial(create_npu_float_tensor, dtype="int")
DoubleTensor = functools.partial(create_npu_float_tensor, dtype="double")
LongTensor = functools.partial(create_npu_float_tensor, dtype="long")
ShortTensor = functools.partial(create_npu_float_tensor, dtype="short")
ByteTensor = functools.partial(create_npu_float_tensor, dtype="byte")
CharTensor = functools.partial(create_npu_float_tensor, dtype="char")
