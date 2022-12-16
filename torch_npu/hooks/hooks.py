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

import os
import inspect
import json
import stat

from datetime import datetime, timezone

import torch

import torch_npu


def get_time_stamp():
    time_stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return time_stamp


def set_dump_path(fpath=None):
    if fpath is not None:
        dump_path = os.path.realpath(fpath)
        if os.path.isdir(dump_path):
            raise RuntimeError("set_dump_path '{}' error, please set a valid filename.".format(dump_path))
        else:
            dir_path = os.path.dirname(dump_path)
            if not dir_path and not os.path.isdir(dir_path):
                raise RuntimeError("set_dump_path error, the directory '{}' does not exist.".format(dir_path))
            filename = os.path.basename(dump_path)
            if os.path.exists(dump_path):
                os.remove(dump_path)
        new_dump_path = os.path.join(dir_path, filename)
        os.environ["DUMP_PATH"] = new_dump_path
    else:
        raise RuntimeError("set_dump_path '{}' error, please set a valid filename".format(fpath))


def get_dump_path():
    assert "DUMP_PATH" in os.environ, "Please set dump path for hook tools."
    return os.environ.get("DUMP_PATH")


def dump_tensor_for_acc_cmp(x, prefix="", sample=True):
    if "DUMP_PATH" not in os.environ:
        return

    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            dump_tensor_for_acc_cmp(item, prefix="{}.{}".format(prefix, i), sample=sample)
    elif isinstance(x, torch.Tensor):
        if len(x.shape) == 0 or not x.is_floating_point():
            return

        if hasattr(dump_tensor_for_acc_cmp, "call_number"):
            dump_tensor_for_acc_cmp.call_number = dump_tensor_for_acc_cmp.call_number + 1
        else:
            dump_tensor_for_acc_cmp.call_number = 0
        prefix = f"{dump_tensor_for_acc_cmp.call_number}_{prefix}"
        with os.fdopen(os.open(get_dump_path(), os.O_RDWR|os.O_CREAT, stat.S_IWUSR|stat.S_IRUSR), "a") as f:
            if sample:
                tensor_max = torch._C._VariableFunctionsClass.max(x).cpu().detach().float().numpy().tolist()
                tensor_min = torch._C._VariableFunctionsClass.min(x).cpu().detach().float().numpy().tolist()
                tensor_mean = torch._C._VariableFunctionsClass.mean(x).cpu().detach().float().numpy().tolist()
                save_tensor = x.contiguous().view(-1)[:10].cpu().detach().float().numpy().tolist() + [
                    tensor_max, tensor_min, tensor_mean
                ]
            else:
                save_tensor = x.contiguous().view(-1).cpu().detach().float().numpy().tolist()
            json.dump([prefix, save_tensor, str(x.dtype), tuple(x.shape)], f)
            f.write('\n')


def dump_tensor_for_overflow(x, dump_file_name, prefix=""):
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            dump_tensor_for_overflow(item, dump_file_name, prefix="{}.{}".format(prefix, i))
    else:
        with os.fdopen(os.open(dump_file_name, os.O_RDWR|os.O_CREAT, stat.S_IWUSR|stat.S_IRUSR), "a") as f:
            if isinstance(x, torch.Tensor):
                save_tensor = x.contiguous().view(-1).cpu().detach().float().numpy().tolist()
                json.dump([prefix, save_tensor, str(x.dtype), tuple(x.shape)], f)
            else:
                json.dump([prefix, x], f)
            f.write('\n')


def wrap_acc_cmp_hook(name, **kwargs):

    sample = kwargs.get('sample', True)
    pid = kwargs.get('pid')
    if not pid:
        return RuntimeError("Not get the specified process pid.")

    def acc_cmp_hook(module, in_feat, out_feat):
        if pid == os.getpid():
            name_template = f"{name}" + "_{}"
            dump_tensor_for_acc_cmp(in_feat, name_template.format("input"), sample)
            dump_tensor_for_acc_cmp(out_feat, name_template.format("output"), sample)

    return acc_cmp_hook


def wrap_checkoverflow_hook(name, **kwargs):

    def checkoverflow_hook(module, in_feat, out_feat):
        module_name = name
        module.has_overflow = torch_npu._C._check_overflow_npu()
        if module.has_overflow:
            name_template = f"{name}" + "_{}"
            dump_file_name = f"Overflow_info_{get_time_stamp()}.pkl"
            stack_str = [str(_) for _ in inspect.stack()[3:]]
            dump_tensor_for_overflow(stack_str, dump_file_name, name_template.format("stack_info"))
            dump_tensor_for_overflow(in_feat, dump_file_name, name_template.format("input"))
            dump_tensor_for_overflow(out_feat, dump_file_name, name_template.format("output"))
            raise ValueError("[check overflow]: module name :'{}' is overflow and dump file is saved in '{}'.".format(
                module_name, os.path.realpath(dump_file_name)))

    return checkoverflow_hook