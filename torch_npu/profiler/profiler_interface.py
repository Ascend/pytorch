import os
import time
from typing import Optional, Iterable, Callable, Dict

import torch
from torch_npu._C._profiler import (
    ProfilerActivity,
    NpuProfilerConfig,
    _supported_npu_activities,
    _init_profiler,
    _start_profiler,
    _stop_profiler,
    _finalize_profiler,
    _get_syscnt_enable,
    _get_freq,
    _get_syscnt,
    _get_monotonic
)
from torch_npu.npu import _lazy_init

from .profiler_path_creator import ProfPathCreator
from .scheduler import ProfilerAction
from .experimental_config import _ExperimentalConfig
from .analysis.prof_common_func.constant import Constant
from .analysis.npu_profiler import NpuProfiler
from .analysis.prof_common_func.constant import print_warn_msg
from .analysis.prof_common_func.file_manager import FileManager
from .analysis.prof_common_func.path_manager import ProfilerPathManager
from ..utils.path_manager import PathManager


class ProfInterface:
    def __init__(
        self,
        activities: Optional[Iterable[ProfilerActivity]] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        schedule: Optional[Callable[[int], ProfilerAction]] = None,
        metadata: Dict = None,
        experimental_config: Optional[_ExperimentalConfig] = None,
    ) -> None:
        self.prof_path = ""
        self.syscnt_enable = False
        self.freq = 100
        self.start_cnt = 0
        self.start_monotonic = 0
        self.activities = set(activities) if activities else supported_activities()
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules
        if experimental_config is None:
            experimental_config = _ExperimentalConfig()
        self.experimental_config = experimental_config
        self.schedule = schedule
        self.metadata = metadata
        self._check_params()
        _lazy_init()

    def init_trace(self):
        ProfPathCreator().create_prof_dir()
        self.prof_path = ProfPathCreator().get_prof_dir()
        _init_profiler(self.prof_path, self.activities)

    def start_trace(self):
        prof_config = [self.prof_path, self.record_shapes, self.profile_memory,
                       self.with_stack, self.with_flops, self.with_modules, self.experimental_config()]
        npu_prof_config = NpuProfilerConfig(*tuple(prof_config))
        self.syscnt_enable = _get_syscnt_enable()
        if self.syscnt_enable:
            self.freq = _get_freq()
            self.start_cnt = _get_syscnt()
        self.start_monotonic = _get_monotonic()
        _start_profiler(npu_prof_config, self.activities)

    def stop_trace(self):
        _stop_profiler()

    def finalize_trace(self):
        _finalize_profiler()
        self._dump_profiler_info()
        self._dump_metadata()
        ProfPathCreator().is_prof_inited = False

    def analyse(self, analysis_type: str = Constant.TENSORBOARD_TRACE_HANDLER, output_path: str = None, **kwargs):
        try:
            NpuProfiler.analyse(self.prof_path, analysis_type, output_path, **kwargs)
        except Exception:
            print_warn_msg("Profiling data parsing failed.")

    def _check_params(self):
        for activity in self.activities:
            if activity in supported_activities():
                continue
            print_warn_msg("Invalid activities, only CPU and NPU are supported, reset it to default.")
            self.activities = supported_activities()
            break

        if not isinstance(self.record_shapes, bool):
            print_warn_msg("Parameter record_shapes is of boolean type, reset it to False.")
            self.record_shapes = False
        if not isinstance(self.profile_memory, bool):
            print_warn_msg("Parameter profile_memory is of boolean type, reset it to False.")
            self.profile_memory = False
        if not isinstance(self.with_stack, bool):
            print_warn_msg("Parameter with_stack is of boolean type, reset it to False.")
            self.with_stack = False
        if not isinstance(self.with_flops, bool):
            print_warn_msg("Parameter with_flops is of boolean type, reset it to False.")
            self.with_flops = False
        if not isinstance(self.with_modules, bool):
            print_warn_msg("Parameter with_modules is of boolean type, reset it to False.")
            self.with_modules = False
        if not isinstance(self.experimental_config, _ExperimentalConfig):
            print_warn_msg("Parameter experimental_config is an instance of _ExperimentalConfig, "
                           "reset it to default.")
            self.experimental_config = _ExperimentalConfig()

        if ProfilerActivity.NPU not in self.activities and self.experimental_config is not None:
            print_warn_msg("Experimental config will not be uesd while ProfilerActivity.NPU is not set.")

    def _dump_profiler_info(self):
        def _trans_obj2cfg(obj):
            if not obj:
                return None
            obj_attr = getattr(obj, "__dict__", {})
            return obj_attr

        common_config = {"activities": list(map(str, list(self.activities))),
                         "schedule": _trans_obj2cfg(self.schedule),
                         "record_shapes": self.record_shapes,
                         "profile_memory": self.profile_memory,
                         "with_stack": self.with_stack,
                         "with_flops": self.with_flops,
                         "with_modules": self.with_modules}
        experimental_config = _trans_obj2cfg(self.experimental_config)
        config = {
            Constant.COMMON_CONFIG: common_config,
            Constant.EXPERIMENTAL_CONFIG: experimental_config}
        start_info = {
            Constant.SyscntEable: self.syscnt_enable,
            Constant.SysCntFreq: self.freq,
            Constant.StartCnt: self.start_cnt,
            Constant.StartMonotonic: self.start_monotonic}
        end_info = {
            Constant.FWK_END_TIME: time.time_ns(),
            Constant.FWK_END_MONOTONIC: time.monotonic_ns()}
        total_info = {
            Constant.CONFIG: config,
            Constant.START_INFO: start_info,
            Constant.END_INFO: end_info}
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_id = torch.distributed.get_rank()
            path = os.path.join(os.path.realpath(self.prof_path), f'profiler_info_{rank_id}.json')
            total_info["rank_id"] = rank_id
        else:
            path = os.path.join(os.path.realpath(self.prof_path), 'profiler_info.json')
        FileManager.create_json_file_by_path(path, total_info, indent=4)

    def _dump_metadata(self):
        if not self.metadata:
            return
        metadata_path = os.path.join(self.prof_path, "profiler_metadata.json")
        FileManager.create_json_file_by_path(metadata_path, self.metadata, indent=4)
        self.metadata.clear()


def supported_activities():
    return _supported_npu_activities()
