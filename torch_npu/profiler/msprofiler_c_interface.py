# Copyright (c) 2023, Huawei Technologies.
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

import torch

import torch_npu._C


class ProfilerActivity:
    CPU = torch_npu._C._profiler.ProfilerActivity.CPU
    NPU = torch_npu._C._profiler.ProfilerActivity.NPU


def supported_ms_activities() -> set:
    return torch_npu._C._profiler._supported_npu_activities()


class MsProfilerInterface:
    def __init__(self, config: list, activities: set) -> None:
        self.config = config
        self.activities = activities
        self.path = None
        self.msprof_config = None

    def stop_profiler(self):
        torch_npu._C._profiler._stop_profiler()

    @classmethod
    def finalize_profiler(cls):
        torch_npu._C._profiler._finalize_profiler()

    def set_config(self, path: str) -> None:
        self.path = path
        ms_config = [path] + self.config
        self.msprof_config = torch_npu._C._profiler.NpuProfilerConfig(*tuple(ms_config))

    def init_profiler(self):
        torch_npu._C._profiler._init_profiler(self.path, self.activities)

    def start_profiler(self):
        torch_npu._C._profiler._start_profiler(self.msprof_config, self.activities)

    def dump_info(self, total_info: dict):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_id = torch.distributed.get_rank()
            path = os.path.join(os.path.abspath(self.path), f'profiler_info_{rank_id}.json')
            total_info["rank_id"] = rank_id
        else:
            path = os.path.join(os.path.abspath(self.path), 'profiler_info.json')
        with open(path, "w") as f:
            json.dump(total_info, f, indent=4)
