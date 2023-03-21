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

import torch_npu


def torch_device_guard(func):
    torch_npu.npu._lazy_init()
    # Parse args/kwargs matched torch.device objects
    def wrapper(*args, **kwargs):
        if args:
            args_list = list(args)
            for index, arg in enumerate(args_list):
                if isinstance(arg, torch_npu._C.device):
                    args_list[index] = str(arg).replace("npu", torch_npu.npu.native_device)
                    break
                elif isinstance(arg, str) and "npu" in arg:
                    args_list[index] = arg.replace("npu", torch_npu.npu.native_device)
                    break
            args = tuple(args_list)
        if kwargs and kwargs.get("device"):
            device_kwarg = kwargs.get("device")
            if isinstance(device_kwarg, torch_npu._C.device):
                kwargs['device'] = str(kwargs['device']).replace("npu", torch_npu.npu.native_device)
            elif isinstance(device_kwarg, str) and "npu" in device_kwarg:
                kwargs['device'] = device_kwarg.replace("npu", torch_npu.npu.native_device)
        return func(*args, **kwargs)
    return wrapper
