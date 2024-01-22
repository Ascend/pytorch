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

import sys
import copy
from functools import wraps

import torch
from torch import distributed as dist

WRAP_DTYPE_DICT = {
    'torch.uint8': torch.int32,
    'torch.uint16': torch.int32,
    'torch.uint32': torch.int32,
    'torch.uint64': torch.int32,
    'torch.float64': torch.float32,
}

WRAP_DTYPE_FNNAME_LIST_ONE_INPUT = ['all_reduce', 'reduce']


def fn_replace(src, tar, prefix=''):
    for k in sys.modules:
        if k.startswith(prefix):
            if isinstance(tar, (list, tuple)):
                for target in tar:
                    if getattr(sys.modules[k], target, None):
                        setattr(sys.modules[k], target, src)
            else:
                if getattr(sys.modules[k], tar, None):
                    setattr(sys.modules[k], tar, src)


def wrapper_dist_dtype_one_input(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if len(args) > 0 and str(args[0].dtype) in WRAP_DTYPE_DICT.keys():
            new_args = [copy.deepcopy(args[0])] + list(args[1:])
            raw_type = args[0].dtype
            tar_type = WRAP_DTYPE_DICT[str(args[0].dtype)]
            new_args[0] = new_args[0].to(tar_type)
            output = fn(*new_args, **kwargs)
            if output is not None:
                output.wait()
            args[0].copy_(new_args[0].to(raw_type))
            return output
        elif 'tensor' in kwargs and str(kwargs['tensor'].dtype) in WRAP_DTYPE_DICT.keys():
            old_tensor = kwargs['tensor']
            raw_type = kwargs['tensor'].dtype
            tar_type = WRAP_DTYPE_DICT[str(raw_type)]
            kwargs['tensor'] = kwargs['tensor'].to(tar_type)
            output = fn(*args, **kwargs)
            if output is not None:
                output.wait()
            kwargs['tensor'] = old_tensor.copy_(kwargs['tensor'].to(raw_type))
            return output
        return fn(*args, **kwargs)

    return wrapper


def wrap_dtype_for_hccl():
    for fn_name in WRAP_DTYPE_FNNAME_LIST_ONE_INPUT:
        fn_replace(
            wrapper_dist_dtype_one_input(getattr(dist, fn_name)), fn_name,
            'torch.distributed')
