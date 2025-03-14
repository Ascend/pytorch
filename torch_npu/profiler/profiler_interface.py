import os
import time
import functools
from typing import Optional, Iterable, Callable, Dict

import torch
from torch.distributed.distributed_c10d import _world as distributed_world
from torch_npu._C._profiler import (
    ProfilerActivity,
    NpuProfilerConfig,
    _supported_npu_activities,
    _init_profiler,
    _warmup_profiler,
    _start_profiler,
    _stop_profiler,
    _finalize_profiler,
    _get_syscnt_enable,
    _get_freq,
    _get_syscnt,
    _get_monotonic
)
from torch_npu.npu import _lazy_init
from torch_npu.npu import Event
from torch_npu.utils.collect_env import get_torch_npu_version, get_cann_version

from ._profiler_path_creator import ProfPathCreator
from ._profiler_gc_detect import ProfGCDetector
from .scheduler import ProfilerAction
from .experimental_config import _ExperimentalConfig
from .analysis.prof_common_func._constant import Constant
from .analysis._npu_profiler import NpuProfiler
from .analysis.prof_common_func._constant import print_warn_msg
from .analysis.prof_common_func._file_manager import FileManager
from .analysis.prof_common_func._utils import collect_env_vars, no_exception_func
from .analysis.prof_common_func._path_manager import ProfilerPathManager
from .analysis.prof_common_func._log import ProfilerLogger
from ..utils._path_manager import PathManager

__all__ = ['supported_activities']


def _enable_event_record():
    def record_wrapper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with torch.profiler.record_function(f"Event::{func.__name__}"):
                out = func(*args, **kwargs)
                return out

        wrapper.origin_func = func
        return wrapper

    Event.record = record_wrapper(Event.record)
    Event.wait = record_wrapper(Event.wait)
    Event.query = record_wrapper(Event.query)
    Event.elapsed_time = record_wrapper(Event.elapsed_time)
    Event.synchronize = record_wrapper(Event.synchronize)


def _disable_event_record():
    Event.record = getattr(Event.record, "origin_func", Event.record)
    Event.wait = getattr(Event.wait, "origin_func", Event.wait)
    Event.query = getattr(Event.query, "origin_func", Event.query)
    Event.elapsed_time = getattr(Event.elapsed_time, "origin_func", Event.elapsed_time)
    Event.synchronize = getattr(Event.synchronize, "origin_func", Event.synchronize)


class _ProfInterface:
    PARALLEL_GROUP_KEY = "parallel_group_info"

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
        self.gc_detector = None
        self._check_params()
        _lazy_init()

    def init_trace(self):
        ProfPathCreator().create_prof_dir()
        self.prof_path = ProfPathCreator().get_prof_dir()
        _init_profiler(self.prof_path, self.activities)

    def warmup_trace(self):
        prof_config = [self.prof_path, self.record_shapes, self.profile_memory,
                       self.with_stack, self.with_flops, self.with_modules, self.experimental_config()]
        npu_prof_config = NpuProfilerConfig(*tuple(prof_config))
        _warmup_profiler(npu_prof_config, self.activities)

    def start_trace(self):
        prof_config = [self.prof_path, self.record_shapes, self.profile_memory,
                       self.with_stack, self.with_flops, self.with_modules, self.experimental_config()]
        npu_prof_config = NpuProfilerConfig(*tuple(prof_config))
        self.syscnt_enable = _get_syscnt_enable()
        if self.syscnt_enable:
            self.freq = _get_freq()
            self.start_cnt = _get_syscnt()
        self.start_monotonic = _get_monotonic()
        _enable_event_record()
        _start_profiler(npu_prof_config, self.activities)
        self.start_gc_detect()

    def stop_trace(self):
        _stop_profiler()
        self.stop_gc_detect()
        _disable_event_record()

    def finalize_trace(self):
        _finalize_profiler()
        self._dump_profiler_info()
        self._dump_metadata()
        ProfPathCreator().is_prof_inited = False
        ProfilerLogger.destroy()

    def delete_prof_dir(self):
        ProfPathCreator().delete_prof_dir()

    def analyse(self, analysis_type: str = Constant.TENSORBOARD_TRACE_HANDLER, output_path: str = None, **kwargs):
        try:
            NpuProfiler.analyse(self.prof_path, analysis_type, output_path, **kwargs)
        except Exception as e:
            print_warn_msg(f"Profiling data parsing failed, error: {e}")

    def check_gc_detect_enable(self):
        return ProfilerActivity.CPU in self.activities and self.experimental_config.with_gc

    def start_gc_detect(self):
        if self.check_gc_detect_enable():
            self.gc_detector = ProfGCDetector(self.experimental_config.gc_detect_threshold)
            self.gc_detector.start()

    def stop_gc_detect(self):
        if self.check_gc_detect_enable() and self.gc_detector is not None:
            self.gc_detector.stop()
            self.gc_detector = None

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

        if ProfilerActivity.CPU not in self.activities and self.experimental_config.with_gc:
            print_warn_msg("GC detect will not take effect while ProfilerActivity.CPU is not set.")

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
            Constant.END_INFO: end_info,
            Constant.TORCH_NPU_VERSION: get_torch_npu_version().replace("'", "").replace(" ", ""),
            Constant.CANN_VERSION: get_cann_version()}
        rank_id = os.environ.get('RANK')
        if rank_id is None and torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_id = torch.distributed.get_rank()
        if rank_id is None:
            path = os.path.join(os.path.realpath(self.prof_path), 'profiler_info.json')
        else:
            path = os.path.join(os.path.realpath(self.prof_path), f'profiler_info_{rank_id}.json')
            total_info["rank_id"] = rank_id
        FileManager.create_json_file_by_path(path, total_info, indent=4)

    def _dump_metadata(self):
        if Constant.Text in self.experimental_config.export_type:
            self.metadata.update(collect_env_vars())
        self._add_group_info_to_metadata()
        if not self.metadata:
            return
        if not ProfPathCreator().is_prof_inited:
            print_warn_msg("Profiler is not initialized. Skip this metadata.")
            return
        metadata_path = os.path.join(self.prof_path, Constant.PROFILER_META_DATA)
        FileManager.create_json_file_by_path(metadata_path, self.metadata)
        self.metadata.clear()

    def _add_group_info_to_metadata(self):
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                group_info = {}
                global_rank = torch.distributed.get_rank()
                for group, group_config in distributed_world.pg_map.items():
                    backend = str(group_config[0]).lower()
                    if backend != "hccl":
                        continue
                    hccl_group = group._get_backend(torch.device("npu"))
                    comm_name = hccl_group.get_hccl_comm_name(global_rank, init_comm=False)
                    if comm_name:
                        group_info[comm_name] = {
                            "group_name": hccl_group.options.hccl_config.get("group_name", ""),
                            "group_rank": torch.distributed.get_group_rank(group, global_rank),
                            "global_ranks": torch.distributed.get_process_group_ranks(group)
                        }
                default_group = torch.distributed.distributed_c10d._get_default_group()
                comm_name = default_group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank, init_comm=False)
                if comm_name:
                    group_info[comm_name] = {
                        "group_name": "default_group",
                        "group_rank": torch.distributed.get_group_rank(default_group, global_rank),
                        "global_ranks": torch.distributed.get_process_group_ranks(default_group)
                    }
                if group_info:
                    self.metadata.update({self.PARALLEL_GROUP_KEY: group_info})
        except Exception as err:
            print_warn_msg(f"Failed to get parallel group info, Exception: {str(err)}.")


@no_exception_func(set())
def supported_activities():
    return _supported_npu_activities()
