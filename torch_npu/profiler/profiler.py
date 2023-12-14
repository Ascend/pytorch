import os.path
import time
import json
from sys import getsizeof
from typing import Optional, Iterable, Callable, Any

from torch.futures import Future
import torch.autograd.profiler as prof
from torch_npu._C._profiler import ProfilerActivity

from .experimental_config import _ExperimentalConfig
from .profiler_path_creator import ProfPathCreator
from .profiler_interface import ProfInterface, supported_activities
from .profiler_action_controller import ProfActionController
from .scheduler import default_schedule_fn, ProfilerAction
from .analysis.prof_common_func.constant import Constant
from .analysis.prof_common_func.constant import print_warn_msg
from .analysis.npu_profiler import NpuProfiler
from .analysis.prof_common_func.path_manager import ProfilerPathManager
from ..utils.path_manager import PathManager


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
        self.prof_if = ProfInterface(
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

    def __del__(self):
        ProfPathCreator().delete_export_only_prof()

    def start(self):
        ProfPathCreator().init(export_only_mode=True)
        self.prof_if.init_trace()
        self.prof_if.start_trace()

    def stop(self):
        self.prof_if.stop_trace()
        self.prof_if.finalize_trace()

    def export_chrome_trace(self, output_path: str):
        output_path = ProfilerPathManager.get_realpath(output_path)
        PathManager.check_input_file_path(output_path)
        file_name = os.path.basename(output_path)
        if not file_name.endswith(".json"):
            raise RuntimeError("Invalid parameter output_path, which must be a json file.")
        if not self.prof_if.prof_path:
            print_warn_msg("Invalid profiling path.")
            return
        self.prof_if.analyse(Constant.EXPORT_CHROME_TRACE, output_path)

    def add_metadata(self, key: str, value: str):
        if not ProfPathCreator().is_prof_inited:
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
        if not ProfPathCreator().is_prof_inited:
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

    def _check_str_valid(self, input_str: str):
        if len(input_str) > self.max_str_len:
            return False
        return True

    def _support_export_stacks_metrics(self):
        return [Constant.METRIC_CPU_TIME, Constant.METRIC_NPU_TIME]


def tensorboard_trace_handler(dir_name: str = None, worker_name: str = None):
    ProfPathCreator().init(worker_name=worker_name, dir_name=dir_name)

    def handler_fn(prof_inst) -> None:
        prof_inst.analyse()

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
        super().__init__()
        activities_set = set(activities) if activities else supported_activities()
        if schedule:
            self.schedule = schedule
            # add step markers into the trace and table view
            self.record_steps = True
        else:
            self.schedule = default_schedule_fn
            self.record_steps = False
        self.prof_if = ProfInterface(
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
        self.step_rec_fn: Optional[prof.record_function] = None
        if use_cuda is not None:
            print_warn_msg("This is npu environment, use_cuda is invalid")
        self.stopped = False
        self.action_controller = ProfActionController(self.prof_if, self.on_trace_ready)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exe_type, exe_val, exc_tb):
        self.stop()

    def start(self):
        self.stopped = False
        if not self.on_trace_ready:
            ProfPathCreator().init(export_only_mode=True)
        self.action_controller.transit_action(ProfilerAction.NONE, self.current_action)
        if self.record_steps:
            self.step_rec_fn = prof.record_function("ProfilerStep#" + str(self.step_num))
            self.step_rec_fn.__enter__()

    def stop(self):
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        self.action_controller.transit_action(self.current_action, None)
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
        self.action_controller.transit_action(prev_action, self.current_action)
        if self.record_steps:
            self.step_rec_fn = prof.record_function("ProfilerStep#" + str(self.step_num))
            self.step_rec_fn.__enter__()


def analyse(profiler_path: str):
    NpuProfiler.analyse(profiler_path)
