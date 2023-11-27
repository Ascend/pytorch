import os.path
import time
import json
from sys import getsizeof
from functools import partial
from typing import Optional, Iterable, Callable, Any, Dict, List, Tuple

from torch.futures import Future
from torch import distributed
import torch.autograd.profiler as prof
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

from .experimental_config import _ExperimentalConfig
from .prof_manager import ProfManager
from .scheduler import default_schedule_fn, ProfilerAction
from .analysis.prof_common_func.constant import Constant
from .analysis.prof_common_func.constant import print_warn_msg
from .analysis.npu_profiler import NpuProfiler
from .analysis.prof_common_func.file_manager import FileManager
from .analysis.prof_common_func.path_manager import ProfilerPathManager
from ..utils.path_manager import PathManager


def supported_activities():
    return _supported_npu_activities()


class _KinetoProfile:
    def __init__(
        self,
        *,
        activities: Optional[Iterable[ProfilerActivity]] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        experimental_config: Optional[_ExperimentalConfig] = None,
    ):
        self.activities = set(activities) if activities else supported_activities()
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules
        if experimental_config is None:
            experimental_config = _ExperimentalConfig()
        self.experimental_config = experimental_config
        self.prof_path = ""
        self.metadata = {}
        self.is_prof_inited = False
        self.max_meta_size = 50 * 1024
        self.max_str_len = 4096
        self.syscnt_enable = False
        self.freq = 100
        self.start_cnt = 0
        self.start_monotonic = 0
        _lazy_init()

    def start(self):
        ProfManager().init()
        self.init_trace()
        self.start_trace()

    def stop(self):
        self.stop_trace()
        self.finalize_trace()

    def init_trace(self):
        ProfManager().create_prof_dir()
        self.prof_path = ProfManager().get_prof_dir()
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
        ProfManager().is_prof_inited = False

    def _dump_config(self, total_info):
        def _trans_obj2cfg(obj):
            if not obj:
                return None
            obj_attr = getattr(obj, "__dict__", {})
            return obj_attr
        common_config = {"activities": list(map(str, list(self.activities))),
                         "record_shapes": self.record_shapes,
                         "profile_memory": self.profile_memory,
                         "with_stack": self.with_stack,
                         "with_flops": self.with_flops,
                         "with_modules": self.with_modules}
        experimental_config = _trans_obj2cfg(self.experimental_config)
        config = {
            Constant.COMMON_CONFIG: common_config,
            Constant.EXPERIMENTAL_CONFIG: experimental_config}
        total_info[Constant.CONFIG] = config

    def _dump_metadata(self):
        if not self.metadata:
            return
        fwk_path = ProfilerPathManager.get_fwk_path(self.prof_path)
        if not fwk_path:
            fwk_path = os.path.join(self.prof_path, Constant.FRAMEWORK_DIR)
            PathManager.make_dir_safety(fwk_path)
        metadata_path = os.path.join(fwk_path, "profiler_metadata.json")
        FileManager.create_json_file_by_path(metadata_path, self.metadata, indent=4)
        self.metadata = {}

    def _dump_profiler_info(self):
        start_info = {
            Constant.SyscntEable: self.syscnt_enable,
            Constant.SysCntFreq: self.freq,
            Constant.StartCnt: self.start_cnt,
            Constant.StartMonotonic: self.start_monotonic}
        end_info = {
            Constant.FWK_END_TIME: time.time_ns(),
            Constant.FWK_END_MONOTONIC: time.monotonic_ns()}
        total_info = {
            Constant.START_INFO: start_info,
            Constant.END_INFO: end_info}
        self._dump_config(total_info)
        if distributed.is_available() and distributed.is_initialized():
            rank_id = distributed.get_rank()
            path = os.path.join(os.path.abspath(self.prof_path), f'profiler_info_{rank_id}.json')
            total_info["rank_id"] = rank_id
        else:
            path = os.path.join(os.path.abspath(self.prof_path), 'profiler_info.json')
        FileManager.create_json_file_by_path(path, total_info, indent=4)

    def _analyse(self, analysis_type: str = Constant.TENSORBOARD_TRACE_HANDLER, output_path: str = None, **kwargs):
        try:
            NpuProfiler.analyse(self.prof_path, analysis_type, output_path, **kwargs)
        except Exception:
            print_warn_msg("Profiling data parsing failed.")

    def export_chrome_trace(self, output_path: str):
        output_path = ProfilerPathManager.get_realpath(output_path)
        PathManager.check_input_file_path(output_path)
        file_name = os.path.basename(output_path)
        if not file_name.endswith(".json"):
            raise RuntimeError("Invalid parameter output_path, which must be a json file.")
        if not self.prof_path:
            print_warn_msg("Invalid profiling path.")
            return
        self._analyse(Constant.EXPORT_CHROME_TRACE, output_path)

    def _check_str_valid(self, input_str: str):
        if len(input_str) > self.max_str_len:
            return False
        return True

    def _support_export_stacks_metrics(self):
        return [Constant.METRIC_CPU_TIME, Constant.METRIC_NPU_TIME]

    def add_metadata(self, key: str, value: str):
        if not ProfManager().is_prof_inited:
            print_warn_msg("Profiler is not initialized. Skip this metadata.")
            return
        if not isinstance(key, str) or not isinstance(value, str):
            print_warn_msg("The key and value of metadata must be string. Skip this metadata.")
            return
        if not self._check_str_valid(key) or not self._check_str_valid(value):
            print_warn_msg("Invalid input key or value. Skip this metadata.")
            return
        add_size = getsizeof(key) + getsizeof(value)
        if getsizeof(self.metadata) + add_size < self.max_meta_size:
            if key in self.metadata.keys():
                print_warn_msg(f"{key} is already saved as metadata, override it.")
            self.metadata[key] = value
        else:
            print_warn_msg("Too many metadata added. Skip this metadata")

    def add_metadata_json(self, key: str, value: str):
        if not ProfManager().is_prof_inited:
            print_warn_msg("Profiler is not initialized. Skip this metadata.")
            return
        if not isinstance(key, str) or not isinstance(value, str):
            print_warn_msg("The key and value of metadata must be string. Skip this metadata.")
            return
        if not self._check_str_valid(key) or not self._check_str_valid(value):
            print_warn_msg("Invalid input key or value. Skip this metadata.")
            return
        add_size = getsizeof(key) + getsizeof(value)
        if getsizeof(self.metadata) + add_size < self.max_meta_size:
            try:
                if key in self.metadata.keys():
                    print_warn_msg(f"{key} is already saved as metadata, override it.")
                self.metadata[key] = json.loads(value)
            except ValueError:
                print_warn_msg("The metadata value must be json format string. Skip this metadata")
        else:
            print_warn_msg("Too many metadata added. Skip this metadata")

    def export_stacks(self, output_path: str, metric: str = Constant.METRIC_CPU_TIME):
        if not self.with_stack:
            print_warn_msg("Function export_stacks() requires with_stack=True.")
            return
        if metric not in self._support_export_stacks_metrics():
            print_warn_msg("Metric should be self_cpu_time_total or self_npu_time_total."
                  "Here it is presumed to be self_cpu_time_total.")
            metric = Constant.METRIC_CPU_TIME
        if not self.prof_path:
            print_warn_msg("Invalid profiling path.")
            return
        self._analyse(Constant.EXPORT_STACK, output_path, metric=metric)


def tensorboard_trace_handler(dir_name: str = None, worker_name: str = None):
    ProfManager().init(worker_name=worker_name, dir_name=dir_name)

    def handler_fn(prof_inst) -> None:
        prof_inst._analyse()

    return handler_fn


class profile(_KinetoProfile):
    def __init__(
        self,
        *,
        activities: Optional[Iterable[ProfilerActivity]] = None,
        schedule: Optional[Callable[[int], ProfilerAction]] = None,
        on_trace_ready: Optional[Callable[..., Any]] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        experimental_config: Optional[_ExperimentalConfig] = None,
        # deprecated:
        use_cuda: Optional[bool] = None,
    ):
        activities_set = set(activities) if activities else supported_activities()
        super().__init__(
            activities=activities_set,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            experimental_config=experimental_config,
        )
        if schedule:
            self.schedule = schedule
            # add step markers into the trace and table view
            self.record_steps = True
        else:
            self.schedule = default_schedule_fn
            self.record_steps = False
        self.on_trace_ready = on_trace_ready
        self.step_num = 0
        self.current_action = self.schedule(self.step_num)
        self.step_rec_fn: Optional[prof.record_function] = None
        if use_cuda is not None:
            print_warn_msg("This is npu environment, use_cuda is invalid")
        self._check_params()
        self.action_map = self._init_action_map()
        self.stopped = False

    def _init_action_map(self):
        action_map = {
            (ProfilerAction.NONE, ProfilerAction.NONE): [],
            (ProfilerAction.NONE, ProfilerAction.WARMUP): [self.init_trace],
            (ProfilerAction.NONE, ProfilerAction.RECORD): [self.init_trace, self.start_trace],
            (ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE): [self.init_trace, self.start_trace],

            (ProfilerAction.WARMUP, ProfilerAction.NONE): [
                partial(print_warn_msg, "Incorrect schedule: WARMUP followed by NONE"),
                self.start_trace,
                self.stop_trace,
                self.finalize_trace],
            (ProfilerAction.WARMUP, ProfilerAction.WARMUP): [],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD): [self.start_trace],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD_AND_SAVE): [self.start_trace],

            (ProfilerAction.RECORD, ProfilerAction.NONE): [
                partial(print_warn_msg, "Incorrect schedule: RECORD followed by NONE"),
                self.stop_trace,
                self.finalize_trace],
            (ProfilerAction.RECORD, ProfilerAction.WARMUP): [
                partial(print_warn_msg, "Incorrect schedule: RECORD followed by WARMUP"),
                self.stop_trace,
                self.finalize_trace],
            (ProfilerAction.RECORD, ProfilerAction.RECORD): [],
            (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE): [],

            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE): [self.stop_trace, self.finalize_trace, self._trace_ready],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.WARMUP): [self.stop_trace, self.finalize_trace, self._trace_ready, self.init_trace],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD): [self.stop_trace, self.finalize_trace, self._trace_ready, self.init_trace,
                                                                      self.start_trace],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD_AND_SAVE): [self.stop_trace, self.finalize_trace, self._trace_ready,
                                                                               self.init_trace, self.start_trace],
            # used for exit action
            (ProfilerAction.WARMUP, None): [
                partial(print_warn_msg,
                        "Incorrect schedule: Stop profiler while current state is WARMUP "
                        "which will result in empty parsed data."),
                self.start_trace,
                self.stop_trace, self.finalize_trace],
            (ProfilerAction.RECORD, None): [
                partial(print_warn_msg,
                        "Incorrect schedule: Stop profiler while current state is RECORD "
                        "which may result in incomplete parsed data."),
                self.stop_trace, self.finalize_trace, self._trace_ready],
            (ProfilerAction.RECORD_AND_SAVE, None): [
                partial(print_warn_msg,
                        "Stop profiler while current state is RECORD_AND_SAVE, "
                        "perhaps the scheduling cycle has not yet completed."),
                self.stop_trace, self.finalize_trace, self._trace_ready],
        }

        return action_map

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exe_type, exe_val, exc_tb):
        self.stop()

    def start(self):
        self.stopped = False
        if not self.on_trace_ready:
            ProfManager().init()
        self._transit_action(ProfilerAction.NONE, self.current_action)
        if self.record_steps:
            self.step_rec_fn = prof.record_function("ProfilerStep#" + str(self.step_num))
            self.step_rec_fn.__enter__()

    def stop(self):
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        self._transit_action(self.current_action, None)
        self.stopped = True

    def step(self):
        if self.stopped:
            print_warn_msg("Profiler is stopped, step takes no effect!")
            return
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        prev_action = self.current_action
        self.step_num += 1
        self.current_action = self.schedule(self.step_num)
        self._transit_action(prev_action, self.current_action)
        if self.record_steps:
            self.step_rec_fn = prof.record_function("ProfilerStep#" + str(self.step_num))
            self.step_rec_fn.__enter__()

    def _trace_ready(self):
        if self.on_trace_ready:
            self.on_trace_ready(self)

    def _transit_action(self, prev_action, current_action):
        action_list = self.action_map.get((prev_action, current_action))
        if action_list:
            for action in action_list:
                action()

    def _dump_config(self, total_info):
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
        total_info[Constant.CONFIG] = config

    def _check_params(self):
        for activity in self.activities:
            if activity in supported_activities():
                continue
            print_warn_msg("Invalid activities, only CPU and NPU are supported, reset it to default.")
            self.activities = supported_activities()
            break

        if not isinstance(self.record_shapes, bool):
            print_warn_msg("record_shapes is of Boolean type, reset it to False.")
            self.record_shapes = False
        if not isinstance(self.profile_memory, bool):
            print_warn_msg("profile_memory is of boolean type, reset it to False.")
            self.profile_memory = False
        if not isinstance(self.with_stack, bool):
            print_warn_msg("with_stack is of boolean type, reset it to False.")
            self.with_stack = False
        if not isinstance(self.with_modules, bool):
            print_warn_msg("with_modules is of boolean type, reset it to False.")
            self.with_modules = False
        if not isinstance(self.with_flops, bool):
            print_warn_msg("with_flops is of boolean type, reset it to False.")
            self.with_flops = False
        if not isinstance(self.experimental_config, _ExperimentalConfig):
            print_warn_msg("experimental_config is an instance of _ExperimentalConfig, "
                           "reset it to default.")
            self.experimental_config = _ExperimentalConfig()

        if ProfilerActivity.NPU not in self.activities and self.experimental_config is not None:
            print_warn_msg("Experimental config will not be uesd while ProfilerActivity.NPU is not set.")


def analyse(profiler_path: str):
    NpuProfiler.analyse(profiler_path)
