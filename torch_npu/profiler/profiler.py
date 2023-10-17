import os.path
import shutil
import time

from torch_npu.npu import _lazy_init

from .analysis.npu_profiler import NpuProfiler
from .analysis.prof_common_func.constant import Constant, print_warn_msg
from .analysis.prof_common_func.path_manager import ProfilerPathManager
from .experimental_config import _ExperimentalConfig
from .profiler_action_controller import ActionController
from .profiler_action_controller import NpuProfCreator
from .msprofiler_c_interface import MsProfilerInterface, supported_ms_activities, ProfilerActivity
from .scheduler import CLOSE_STEP, ProfilerAction
from ..utils.path_manager import PathManager


def tensorboard_trace_handler(dir_name: str = None, worker_name: str = None, use_gzip: bool = False):
    return NpuProfCreator(worker_name, dir_name)


def analyse(profiler_path: str):
    NpuProfiler.analyse(profiler_path)


class profile:
    def __init__(
            self,
            activities=None,
            schedule=None,
            on_trace_ready=None,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=None,
            use_cuda=None
    ):
        self._activities = set(activities) if activities else supported_ms_activities()
        self._schedule = schedule
        self._record_shapes = record_shapes
        self._with_flops = with_flops
        if ProfilerActivity.NPU not in self._activities and experimental_config is not None:
            print_warn_msg("Experimental config will not be used while ProfilerActivity.NPU is not set.")
        if experimental_config is None:
            experimental_config = _ExperimentalConfig()
        self._experimental_config = experimental_config
        self._use_cuda = use_cuda
        self._on_trace_ready = on_trace_ready
        self._profile_memory = profile_memory
        self._with_stack = with_stack
        self._with_modules = with_modules
        self._check_params()
        self._record_shapes |= self._with_flops
        self._msprofiler_interface = MsProfilerInterface([self._record_shapes, self._profile_memory,
                                                          self._with_stack, self._with_flops, self._with_modules,
                                                          self._experimental_config()], self._activities)
        self._action_controller = ActionController(self._msprofiler_interface, self._schedule, self,
                                                   self._on_trace_ready)
        _lazy_init()

    def __enter__(self):
        self._action_controller.transit_action()
        return self

    def __exit__(self, exe_type, exe_val, exc_tb):
        prev_step = self._action_controller.next_step - 1
        self._action_controller.next_step = CLOSE_STEP
        self._action_controller.transit_action()
        if self._schedule and prev_step > 0:
            prev_action = self._schedule(prev_step)
            if prev_action == ProfilerAction.NONE:
                return
            PathManager.remove_path_safety(self._msprofiler_interface.path)

    def step(self):
        if self._schedule:
            self._action_controller.transit_action()

    def start(self):
        self._action_controller.init()
        self._msprofiler_interface.start_profiler()

    def stop(self):
        self._msprofiler_interface.stop_profiler()
        self._msprofiler_interface.finalize_profiler()
        self.dump_profiler_info()
        self._action_controller.trace_ready()

    def export_chrome_trace(self, output_path: str):
        output_path = ProfilerPathManager.get_realpath(output_path)
        PathManager.check_input_file_path(output_path)
        file_name = os.path.basename(output_path)
        if not file_name.endswith(".json"):
            raise RuntimeError("Invalid parameter output_path, which must be a json file.")
        if isinstance(self._action_controller._on_trace_ready, NpuProfCreator):
            print_warn_msg(
                "Already generate result files for TensorBoard, export_chrome_trace not producing any effect")
            return
        if not self._msprofiler_interface.path:
            print_warn_msg("Invalid profiling path.")
            return
        try:
            NpuProfiler.analyse(self._msprofiler_interface.path, Constant.EXPORT_CHROME_TRACE, output_path)
        except Exception:
            print_warn_msg("Profiling data parsing failed.")
            return
        try:
            shutil.rmtree(self._msprofiler_interface.path)
        except Exception:
            msg = f"Can't remove directory: {self._msprofiler_interface.path}"
            print_warn_msg(msg)

    def dump_profiler_info(self):
        if not self._msprofiler_interface:
            return

        def _trans_obj2cfg(obj):
            if not obj:
                return None
            obj_attr = getattr(obj, "__dict__", {})
            return obj_attr

        common_config = {"activities": list(map(str, list(self._activities))),
                         "schedule": _trans_obj2cfg(self._schedule),
                         "on_trace_ready": _trans_obj2cfg(self._on_trace_ready),
                         "record_shapes": self._record_shapes,
                         "profile_memory": self._profile_memory,
                         "with_stack": self._with_stack,
                         "with_flops": self._with_flops,
                         "with_modules": self._with_modules}
        experimental_config = _trans_obj2cfg(self._experimental_config)
        config = {Constant.COMMON_CONFIG: common_config, Constant.EXPERIMENTAL_CONFIG: experimental_config}
        end_info = {Constant.FWK_END_TIME: time.time_ns(), Constant.FWK_END_MONOTONIC: time.monotonic_ns()}
        total_info = {Constant.CONFIG: config, Constant.END_INFO: end_info}
        self._msprofiler_interface.dump_info(total_info)

    def _check_params(self):
        if self._use_cuda is not None:
            print_warn_msg("This is npu environment, use_cuda is invalid")
        if len(self._activities) > 2:
            print_warn_msg(
                "Invalid parameter activities, which must be a list less than 2 in length, reset it to default.")
            self._activities = supported_ms_activities()
        for activities in self._activities:
            if activities in (ProfilerActivity.CPU, ProfilerActivity.NPU):
                continue
            print_warn_msg("Invalid parameter activities, which only supported CPU and NPU, reset it to default.")
            self._activities = supported_ms_activities()
            break
        if not isinstance(self._record_shapes, bool):
            print_warn_msg("Invalid parameter record_shapes, which must be of boolean type, reset it to False.")
            self._record_shapes = False
        if not isinstance(self._profile_memory, bool):
            print_warn_msg("Invalid parameter profile_memory, which must be of boolean type, reset it to False.")
            self._profile_memory = False
        if not isinstance(self._with_stack, bool):
            print_warn_msg("Invalid parameter with_stack, which must be of boolean type, reset it to False.")
            self._with_stack = False
        if not isinstance(self._with_modules, bool):
            print_warn_msg("Invalid parameter with_modules, which must be of boolean type, reset it to False.")
            self._with_modules = False
        if not isinstance(self._with_flops, bool):
            print_warn_msg("Invalid parameter with_flops, which must be of boolean type, reset it to False.")
            self._with_flops = False
        if not isinstance(self._experimental_config, _ExperimentalConfig):
            print_warn_msg("Invalid parameter experimental_config, which must be instance of _ExperimentalConfig, "
                           "reset it to default.")
            self._experimental_config = _ExperimentalConfig()
