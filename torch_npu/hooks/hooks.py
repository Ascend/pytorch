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

import inspect
import json
import os
import stat
import threading
import warnings
from datetime import datetime, timezone

import numpy
import torch

import torch_npu
from .initialize import step_schedule


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
        DumpUtil.set_dump_path(new_dump_path)
    else:
        raise RuntimeError("set_dump_path '{}' error, please set a valid filename".format(fpath))


class DumpUtil(object):
    dump_path = None
    dump_init_enable = False

    @staticmethod
    def set_dump_path(save_path):
        DumpUtil.dump_path = save_path
        DumpUtil.dump_init_enable = True

    @staticmethod
    def get_dump_path():
        assert DumpUtil.dump_path is not None, "Please set dump path for hook tools."
        return DumpUtil.dump_path


def dump_tensor_for_acc_cmp(x, prefix="", sample=True):
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            dump_tensor_for_acc_cmp(item, prefix="{}.{}".format(prefix, i), sample=sample)
    elif isinstance(x, torch.Tensor):
        if len(x.shape) == 0 or not x.is_floating_point():
            return

        if DumpUtil.dump_init_enable:
            dump_tensor_for_acc_cmp.call_number = 0
            DumpUtil.dump_init_enable = False
        else:
            dump_tensor_for_acc_cmp.call_number = dump_tensor_for_acc_cmp.call_number + 1
        prefix = f"{dump_tensor_for_acc_cmp.call_number}_{prefix}"
        with os.fdopen(os.open(DumpUtil.get_dump_path(), os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR),
                       "a") as f:
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
        with os.fdopen(os.open(dump_file_name, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), "a") as f:
            if isinstance(x, torch.Tensor):
                save_tensor = x.contiguous().view(-1).cpu().detach().float().numpy().tolist()
                json.dump([prefix, save_tensor, str(x.dtype), tuple(x.shape)], f)
            else:
                json.dump([prefix, x], f)
            f.write('\n')


def wrap_acc_cmp_hook(name, **kwargs):
    warnings.warn("wrap_acc_cmp_hook is not suggested to use,\
                  please use ptdbg_ascend precision comparison tool instead.")
    sample = kwargs.get('sample', True)
    pid = kwargs.get('pid')
    if not pid:
        return RuntimeError("Not get the specified process pid.")

    def acc_cmp_hook(module, in_feat, out_feat):
        if not step_schedule.is_step_enable():
            return
        if pid == os.getpid():
            name_template = f"{name}" + "_{}"
            dump_tensor_for_acc_cmp(in_feat, name_template.format("input"), sample)
            dump_tensor_for_acc_cmp(out_feat, name_template.format("output"), sample)

    return acc_cmp_hook


def wrap_checkoverflow_hook(name, **kwargs):
    pid = kwargs.get('pid')
    if not pid:
        return RuntimeError("Not get the specified process pid.")

    def checkoverflow_hook(module, in_feat, out_feat):
        if pid != os.getpid():
            return

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


datadump_deque_thread = None


def wrap_async_datadump_hook(name, **kwargs):
    pid = kwargs.get('pid')
    path = kwargs.get('path')
    capacity = kwargs.get('capacity')
    if not pid:
        raise RuntimeError("Not get the specified process pid.")
    if not capacity:
        capacity = 3
    if isinstance(capacity, int) and (capacity < 3 or capacity > 2048):
        raise RuntimeError("capacity range [3, 2048].")

    def async_datadump_hook(module, in_feat, out_feat):
        if pid != os.getpid():
            return
        if not step_schedule.is_step_enable():
            return
        if not datadump_deque_thread:
            start_datadump_deque_thread(path)
        name_template = f"{name}" + "_{}"
        datadump_enque(in_feat, name_template.format("input"), capacity)
        datadump_enque(out_feat, name_template.format("output"), capacity)

    return async_datadump_hook


def datadump_enque(input_tensors, prefix, capacity):
    if isinstance(input_tensors, (tuple, list)) and input_tensors:
        tensors = []
        for item in input_tensors:
            if isinstance(item, torch.Tensor) and item.device.type == 'npu':
                tensors.append(item)
        if not tensors:
            return
        if len(tensors) < 100:
            torch_npu.npu_enque_tensor(tensors, prefix, capacity)
        else:
            for index, item in enumerate([tensors[i:i + 100] for i in range(0, len(tensors), 100)]):
                torch_npu.npu_enque_tensor(item, prefix + str(index), capacity)
    elif isinstance(input_tensors, torch.Tensor) and input_tensors.device.type == 'npu':
        torch_npu.npu_enque_tensor([input_tensors], prefix, capacity)


def start_datadump_deque_thread(path):
    device_id = torch_npu._C._npu_getDevice()
    print("Start datadump deque thread. device id: " + str(device_id))
    global datadump_deque_thread
    datadump_deque_thread = threading.Thread(target=deque_and_dump, kwargs={'device_id': device_id, 'path': path})
    datadump_deque_thread.daemon = True
    datadump_deque_thread.start()


def deque_and_dump(device_id, path):
    torch_npu._C._npu_setDevice(device_id)
    index = 0
    if not path.endswith("/"):
        path = path + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    while True:
        try:
            tensorTuple = torch_npu._C._npu_deque_tensor()
            length = len(tensorTuple)
            info = str(tensorTuple[length - 1].decode())
            infos = info.split('|')
            opName = infos[0]
            for i in range(length - 1):
                t = tensorTuple[i].numpy()
                metas = infos[i + 1].split('#')
                savePath = path + str(index) + '_' + opName + str(i) + '_shape' + metas[0].replace(' ', '') \
                           + '_stride' + metas[1].replace(' ', '') + '_offset[' + metas[2] + \
                           ']_format[' + metas[3] + '].npy'
                numpy.save(savePath, t)
            print("Datadump: " + opName)
        except Exception as e:
            print("datadump deque thread exception", e)
        finally:
            index = index + 1
