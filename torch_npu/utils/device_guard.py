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

import re
import torch_npu


def torch_device_guard(func):
    # Parse args/kwargs matched torch.device objects
    def wrapper(*args, **kwargs):
        if args:
            args_list = list(args)
            for index, arg in enumerate(args_list):
                if isinstance(arg, torch_npu._C.device):
                    check_is_valid_ordinal(arg)
                    args_list[index] = str(arg).replace("npu", torch_npu.npu.native_device)
                    break
                elif isinstance(arg, str) and "npu" in arg:
                    check_is_valid_ordinal(arg)
                    args_list[index] = arg.replace("npu", torch_npu.npu.native_device)
                    break
            args = tuple(args_list)
        if kwargs and kwargs.get("device"):
            device_kwarg = kwargs.get("device")
            if isinstance(device_kwarg, torch_npu._C.device):
                check_is_valid_ordinal(device_kwarg)
                kwargs['device'] = str(kwargs['device']).replace("npu", torch_npu.npu.native_device)
            elif isinstance(device_kwarg, str) and "npu" in device_kwarg:
                check_is_valid_ordinal(device_kwarg)
                kwargs['device'] = device_kwarg.replace("npu", torch_npu.npu.native_device)
            elif isinstance(device_kwarg, int):
                check_is_valid_ordinal(device_kwarg)
        return func(*args, **kwargs)
    return wrapper


env_device_cnt = None


def check_is_valid_ordinal(arg):
    global env_device_cnt
    if env_device_cnt is None:
        env_device_cnt = torch_npu.npu.device_count()
    # When env_device_cnt equals to 0, the error message is stored in NPU LOG.
    if env_device_cnt == 0:
        return
    device_str_pattern = "^npu:([1-9]\d*|0)$"
    device_ofr_info = "Invalid NPU device ordinal. Valid device ordinal ranges from 0 - {}.".format(env_device_cnt - 1)
    if isinstance(arg, torch_npu._C.device):
        device_index = arg.index
        if device_index is not None:
            if 0 <= device_index < env_device_cnt:
                return
            else:
                raise RuntimeError(device_ofr_info)
    elif isinstance(arg, str):
        if arg == "npu":
            return
        elif re.match(device_str_pattern, arg):
            if arg.split(":")[-1] not in set([str(i) for i in range(env_device_cnt)]):
                raise RuntimeError(device_ofr_info)
        else:
            raise RuntimeError("Invalid device string: {}.".format(arg))
    elif isinstance(arg, int):
        if 0 <= arg < env_device_cnt:
            return
        raise RuntimeError(device_ofr_info)
