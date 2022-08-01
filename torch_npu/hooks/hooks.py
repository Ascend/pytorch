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
import json
import stat
import torch


def set_dump_path(fpath=None):
    assert fpath is not None
    os.environ["DUMP_PATH"] = fpath


def get_dump_path():
    assert "DUMP_PATH" in os.environ, "Please set dump path for hook tools."
    return os.environ.get("DUMP_PATH")


def dump_tensor(x, prefix=""):
    f = os.fdopen(os.open(get_dump_path(), os.O_RDWR|os.O_CREAT, stat.S_IWUSR|stat.S_IRUSR), "a")
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            dump_tensor(item, prefix="{}.{}".format(prefix, i))
    elif isinstance(x, torch.Tensor):
        list_tensor = x.contiguous().view(-1).cpu().detach().float().numpy().tolist()
        json.dump([prefix, list_tensor, str(x.dtype), tuple(x.shape)], f)
        f.write('\n')
    
    f.close()


def warp_acc_cmp_hook(name):
    def acc_cmp_hook(module, in_feat, out_feat):
        name_template = f"{name}_{module.__class__.__name__}"+ "_{}"
        dump_tensor(in_feat, name_template.format("input"))
        dump_tensor(out_feat, name_template.format("output"))

    return acc_cmp_hook
