import os.path
import json
from sys import getsizeof
from typing import Optional, Iterable, Callable, Any, Union

import torch.autograd.profiler as prof
import torch_npu.npu
from torch_npu._C._profiler import ProfilerActivity
from torch_npu.utils._error_code import ErrCode, prof_error

from .experimental_config import _ExperimentalConfig
from ._profiler_path_creator import ProfPathCreator
from .profiler_interface import _ProfInterface, supported_activities
from ._profiler_action_controller import ProfActionController
from .scheduler import _default_schedule_fn, ProfilerAction
from .analysis.prof_common_func._constant import Constant
from .analysis.prof_common_func._constant import print_warn_msg
from .analysis.prof_common_func._utils import no_exception_func
from .analysis._npu_profiler import NpuProfiler
from .analysis.prof_common_func._path_manager import ProfilerPathManager
from ..utils._path_manager import PathManager

__all__ = [
    'supported_activities',
    'analyse',
    'tensorboard_trace_handler',
    'profile'
]


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
        self.metadata = {}
        self.prof_if = _ProfInterface(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            experimental_config=experimental_config,
            metadata=self.metadata
        )
        self.max_meta_size = 50 * 1024
        self.max_str_len = 4096

    @no_exception_func()
    def __del__(self):
        ProfPathCreator().delete_export_only_prof()

    @no_exception_func()
    def start(self):
        ProfPathCreator().init(export_only_mode=True)
        self.prof_if.init_trace()
        self.prof_if.start_trace()

    @no_exception_func()
    def stop(self):
        self.prof_if.stop_trace()
        self.prof_if.finalize_trace()

    @no_exception_func()
    def export_chrome_trace(self, output_path: str):
        output_path = ProfilerPathManager.get_realpath(output_path)
        PathManager.check_input_file_path(output_path)
        file_name = os.path.basename(output_path)
        if not file_name.endswith(".json"):
            raise RuntimeError("Invalid parameter output_path, which must be a json file." + prof_error(ErrCode.VALUE))
        if not self.prof_if.prof_path:
            print_warn_msg("Invalid profiling path.")
            return
        self.prof_if.analyse(Constant.EXPORT_CHROME_TRACE, output_path)

    @no_exception_func()
    def add_metadata(self, key: str, value: str):
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

    @no_exception_func()
    def add_metadata_json(self, key: str, value: str):
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

    @no_exception_func()
    def export_stacks(self, output_path: str, metric: str = Constant.METRIC_CPU_TIME):
        output_path = ProfilerPathManager.get_realpath(output_path)
        PathManager.check_input_file_path(output_path)
        if not self.prof_if.with_stack:
            print_warn_msg("Function export_stacks() requires with_stack=True.")
            return
        if metric not in self._support_export_stacks_metrics():
            print_warn_msg("Metric should be self_cpu_time_total or self_npu_time_total."
                           "Here it is presumed to be self_cpu_time_total.")
            metric = Constant.METRIC_CPU_TIME
        if not self.prof_if.prof_path:
            print_warn_msg("Invalid profiling path.")
            return
        self.prof_if.analyse(Constant.EXPORT_STACK, output_path, metric=metric)

    @no_exception_func()
    def export_memory_timeline(self, output_path: str, device: Optional[str] = None) -> None:
        if device is None:
            device = "npu:0" if torch_npu.npu.is_available() else "cpu"
        missing = []
        if not self.prof_if.record_shapes:
            missing.append("record_shapes=True")
        if not self.prof_if.profile_memory:
            missing.append("profile_memory=True")
        if not (self.prof_if.with_stack or self.prof_if.with_modules):
            missing.append("with_stack=True or with_modules=True")
        if missing:
            print_warn_msg(f"{', '.join(missing)} required for memory profiling.")
            return

        if not self.prof_if.prof_path:
            print_warn_msg("Invalid profiling path.")
            return
        self.prof_if.analyse(Constant.EXPORT_MEMORY_TIMELINE, output_path, device=device)

    def _check_str_valid(self, input_str: str):
        if len(input_str) > self.max_str_len:
            return False
        return True

    def _support_export_stacks_metrics(self):
        return [Constant.METRIC_CPU_TIME, Constant.METRIC_NPU_TIME]


@no_exception_func()
def tensorboard_trace_handler(dir_name: str = None, worker_name: str = None,
                              analyse_flag: bool = True, async_mode: bool = False):
    ProfPathCreator().init(worker_name=worker_name, dir_name=dir_name)
    if not isinstance(analyse_flag, bool):
        print_warn_msg("analyse_flag is not bool, set by default.")
        analyse_flag = True
    if not isinstance(async_mode, bool):
        print_warn_msg("async_mode is not bool, set by default.")
        async_mode = False

    def handler_fn(prof_inst) -> None:
        if analyse_flag:
            prof_inst.prof_if.analyse(async_mode=async_mode)

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
        super().__init__(activities=activities,
                        record_shapes=record_shapes,
                        profile_memory=profile_memory,
                        with_stack=with_stack,
                        with_flops=with_flops,
                        with_modules=with_modules,
                        experimental_config=experimental_config)
        activities_set = set(activities) if activities else supported_activities()
        if schedule and isinstance(schedule, Callable):
            self.schedule = schedule
            # add step markers into the trace and table view
            self.record_steps = True
        else:
            if schedule:
                print_warn_msg("schedule is not Callable, set by default.")
            self.schedule = _default_schedule_fn
            self.record_steps = False
        if on_trace_ready and not isinstance(on_trace_ready, Callable):
            print_warn_msg("on_trace_ready is not Callable, set by default.")
            on_trace_ready = None
        self.prof_if = _ProfInterface(
            activities=activities_set,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            experimental_config=experimental_config,
            schedule=self.schedule,
            metadata=self.metadata
        )
        self.on_trace_ready = on_trace_ready
        self.step_num = 0
        self.current_action = self.schedule(self.step_num)
        self._step_num_offset = 0
        self.step_rec_fn: Optional[prof.record_function] = None
        if use_cuda is not None:
            print_warn_msg("This is npu environment, use_cuda is invalid")
        self.stopped = False
        self.action_controller = ProfActionController(self, self.prof_if, self.on_trace_ready)

    @no_exception_func()
    def __enter__(self):
        self.start()
        return self

    @no_exception_func()
    def __exit__(self, exe_type, exe_val, exc_tb):
        self.stop()

    @no_exception_func()
    def __del__(self):
        if self.stopped == False:
            self.stop()

    @no_exception_func()
    def _set_step_num_offset_for_dynamic_prof(self, step: int):
        self._step_num_offset = step

    @no_exception_func()
    def start(self):
        self.stopped = False
        if not self.on_trace_ready:
            ProfPathCreator().init(export_only_mode=True)
        self.action_controller.transit_action(ProfilerAction.NONE, self.current_action)
        if self.record_steps:
            self.step_rec_fn = prof.record_function("ProfilerStep#" + str(self.step_num + self._step_num_offset))
            self.step_rec_fn.__enter__()

    @no_exception_func()
    def stop(self):
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        self.action_controller.transit_action(self.current_action, None)
        self.stopped = True

    @no_exception_func()
    def step(self):
        if self.stopped:
            print_warn_msg("Profiler is stopped, step takes no effect!")
            return
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        prev_action = self.current_action
        self.step_num += 1
        self.current_action = self.schedule(self.step_num)
        self.action_controller.transit_action(prev_action, self.current_action)
        if self.record_steps:
            self.step_rec_fn = prof.record_function("ProfilerStep#" + str(self.step_num + self._step_num_offset))
            self.step_rec_fn.__enter__()


@no_exception_func()
def analyse(profiler_path: str, max_process_number: int = Constant.DEFAULT_PROCESS_NUMBER,
            export_type: Union[str, list] = None):
    if not isinstance(max_process_number, int) or max_process_number <= 0:
        max_process_number = Constant.DEFAULT_PROCESS_NUMBER
        print_warn_msg("Invalid max_process_number, reset it to default!")
    if max_process_number > os.cpu_count():
        max_process_number = os.cpu_count()
        print_warn_msg("max_process_number exceeds the number of cpu cores, reset it to the number of cpu cores!")
    if export_type is not None:
        if isinstance(export_type, str):
            export_type = [export_type]
        elif isinstance(export_type, list):
            export_type = list(set(export_type))
        else:
            print_warn_msg(f"Invalid parameter export_type: {export_type}, reset it to None.")
            export_type = None
        if export_type is not None:
            if not export_type or not all(_type in [Constant.Text, Constant.Db] for _type in export_type):
                print_warn_msg(f"Invalid parameter export_type: {export_type}, reset it to None.")
                export_type = None
    NpuProfiler.analyse(profiler_path, max_process_number=max_process_number, export_type=export_type)
