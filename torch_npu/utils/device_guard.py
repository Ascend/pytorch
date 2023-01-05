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

import collections
import torch_npu

device = collections.namedtuple('device', ['type', 'index'])

def torch_device_guard(func):
    # Parse args/kwargs from namedtuple(torch.device) to matched torch.device objects
    def wrapper(*args, **kwargs):
        if args:
            args_list = list(args)
            for index, arg in enumerate(args_list):
                if isinstance(arg, tuple) and "type='npu'" in str(arg):
                    args_list[index] = torch_npu.new_device(type=torch_npu.npu.native_device, index=arg.index)
                    break
            args = tuple(args_list)
        if kwargs and isinstance(kwargs.get("device"), tuple):
            namedtuple_device = kwargs.get("device")
            if "type='npu'" in str(namedtuple_device):
                kwargs['device'] = torch_npu.new_device(type=torch_npu.npu.native_device, index=namedtuple_device.index)
        if kwargs and 'device' in kwargs and not kwargs['device']:
            kwargs['device'] = 'cpu'
        return func(*args, **kwargs)
    return wrapper