import os

import torch

import torch_npu._C

from .analysis.prof_common_func.file_manager import FileManager


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
        self.syscnt_enable = False
        self.freq = 100
        self.start_cnt = 0
        self.start_monotonic = 0

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
        self.syscnt_enable = torch_npu._C._profiler._get_syscnt_enable()
        if self.syscnt_enable:
            self.freq = torch_npu._C._profiler._get_freq()
        self.start_cnt = torch_npu._C._profiler._get_syscnt()
        self.start_monotonic = torch_npu._C._profiler._get_monotonic()
        torch_npu._C._profiler._start_profiler(self.msprof_config, self.activities)

    def dump_info(self, total_info: dict):
        total_info['start_info'] = {}
        total_info['start_info']['syscnt_enable'] = self.syscnt_enable
        total_info['start_info']['freq'] = self.freq
        total_info['start_info']['start_cnt'] = self.start_cnt
        total_info['start_info']['start_monotonic'] = self.start_monotonic
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_id = torch.distributed.get_rank()
            path = os.path.join(os.path.realpath(self.path), f'profiler_info_{rank_id}.json')
            total_info["rank_id"] = rank_id
        else:
            path = os.path.join(os.path.realpath(self.path), 'profiler_info.json')
        FileManager.create_json_file_by_path(path, total_info, indent=4)
