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
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            os.mknod(os.path.join(os.path.abspath(self.path), 'profiler_info_{}.json'.format(torch.distributed.get_rank())))
        else:
            os.mknod(os.path.join(os.path.abspath(self.path), 'profiler_info.json'))
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
