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

import json
import os
import stat

import torch

import torch_npu


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


def dump_tensor(x, prefix=""):
    if "DUMP_PATH" not in os.environ:
        return

    f = os.fdopen(os.open(get_dump_path(), os.O_RDWR|os.O_CREAT, stat.S_IWUSR|stat.S_IRUSR), "a")
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            dump_tensor(item, prefix="{}.{}".format(prefix, i))
    elif isinstance(x, torch.Tensor):
        list_tensor = x.contiguous().view(-1).cpu().detach().float().numpy().tolist()
        json.dump([prefix, list_tensor, str(x.dtype), tuple(x.shape)], f)
        f.write('\n')
    f.close()


def wrap_acc_cmp_hook(name):

    def acc_cmp_hook(module, in_feat, out_feat):
        name_template = f"{name}" + "_{}"
        dump_tensor(in_feat, name_template.format("input"))
        dump_tensor(out_feat, name_template.format("output"))

    return acc_cmp_hook


def wrap_checkoverflow_hook(name):

    def checkoverflow_hook(module, in_feat, out_feat):
        module_name = name
        module.has_overflow = torch_npu._C._check_overflow_npu()
        if module.has_overflow:
            raise ValueError("[check overflow]:module name :'{}' is overflow!".format(module_name))

    return checkoverflow_hook