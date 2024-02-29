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

import sys
import os
import time
import torch
import torch_npu
from torch_npu.utils.error_code import ErrCode, prof_error


def Singleton(cls):
    _instances = {}

    def _singleton(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]
    return _singleton


@Singleton
class Profile(object):
    def __init__(self,
                 total_steps: int = 10,
                 save_path: str = "./npu_profiling",
                 profile_type: str = None,
                 **kwargs):
        r"""Dump TORCH/CANN/GE profiling data

        Args:
            total_steps:    The step to dump profiling data
            save_path:      Save path of data
            profile_type:   The type of profile, Optional['TORCH', 'CANN', 'GE']

        note:
            When dumping GE profiling data, the configuration of 'GE_PROFILING_TO_STD_OUT'
            needs to be before model.to(device).

        Example:
            >>> model = Vgg16()
            >>> os.environ['GE_PROFILING_TO_STD_OUT'] = '1'
            >>> model.to(device)
        """
        self.step_count = 0
        self.save_path = save_path
        self.profile_type = profile_type
        self.enable = self.profile_type in ['TORCH', 'CANN', 'GE']
        self.total_steps = total_steps
        self.count = 0

        if self.enable:
            torch_args_set = set(["enabled", "use_cuda", "use_npu", "record_shapes", "with_flops", "profile_memory",
                                  "with_stack", "use_kineto", "use_cpu", "use_npu_simple"])
            cann_ge_args_set = set(["use_e2e_profiler", "config"])
            if self.profile_type == "TORCH":
                if not set(kwargs.keys()).issubset(torch_args_set):
                    raise ValueError("the args '%s' is invaild." % (kwargs.keys()) +
                                     prof_error(ErrCode.VALUE))
                self.prof = torch.autograd.profiler.profile(**kwargs)
            elif self.profile_type == "CANN":
                if not set(kwargs.keys()).issubset(cann_ge_args_set):
                    raise ValueError("the args '%s' is invaild." % (kwargs.keys()) +
                                     prof_error(ErrCode.VALUE))
                self.prof = torch.npu.profile(self.save_path, **kwargs)
            elif self.profile_type == "GE":
                if not set(kwargs.keys()).issubset(cann_ge_args_set):
                    raise ValueError("the args '%s' is invaild." % (kwargs.keys()) +
                                     prof_error(ErrCode.VALUE))
                self.prof = torch.npu.profile(self.save_path, **kwargs)

            try:
                os.makedirs(self.save_path, exist_ok=True)
            except Exception as e:
                raise ValueError("the path of '%s' is invaild." % (self.save_path) + prof_error(ErrCode.VALUE)) from e

    def __del__(self):
        if self.count != 0:
            self.end()

    def start(self):
        if self.count != 0:
            raise ValueError("start interface can only be called once" + prof_error(ErrCode.INTERNAL))
        self.count = self.count + 1

        if self.enable:
            self.step_count = self.step_count + 1
            if self.step_count == self.total_steps:
                self.prof.__enter__()

    def end(self):
        if self.count == 0:
            raise ValueError("start interface must be called after end" + prof_error(ErrCode.INTERNAL))
        self.count = 0

        if self.enable and self.step_count == self.total_steps:
            self.prof.__exit__(None, None, None)
            if self.profile_type == "TORCH":
                filename = "torch_prof_" + time.strftime("%Y%m%d%H%M%S") + ".json"
                self.prof.export_chrome_trace(os.path.join(self.save_path, filename))
            sys.exit()
