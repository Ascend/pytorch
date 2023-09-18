import os
import shutil
import time
from warnings import warn

from torch_npu.npu import _lazy_init

from .analysis.npu_profiler import NpuProfiler
from .analysis.prof_common_func.constant import Constant
from .analysis.prof_common_func.file_manager import FileManager
from .experimental_config import _ExperimentalConfig
from .profiler_action_controller import ActionController
from .profiler_action_controller import NpuProfCreator
from .msprofiler_c_interface import MsProfilerInterface, supported_ms_activities, ProfilerActivity
from .scheduler import CLOSE_STEP, ProfilerAction


def tensorboard_trace_handler(dir_name: str = None, worker_name: str = None, use_gzip: bool = False):
    if dir_name is None:
        dir_name = os.getenv(Constant.ASCEND_WORK_PATH, default=None)
        dir_name = os.path.join(os.path.abspath(dir_name), Constant.PROFILING_WORK_PATH) if dir_name else os.getcwd()
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
        self._record_shapes |= self._with_flops
        if ProfilerActivity.NPU not in self._activities and experimental_config is not None:
            warn("Experimental config will not be uesd while ProfilerActivity.NPU is not set.")
        if experimental_config is None:
            experimental_config = _ExperimentalConfig()
        self._experimental_config = experimental_config
        self._msprofiler_interface = MsProfilerInterface([self._record_shapes, profile_memory,
                                                          with_stack, self._with_flops, with_modules,
                                                          self._experimental_config()], self._activities)
        self._action_controller = ActionController(self._msprofiler_interface, schedule, self, on_trace_ready)
        self._use_cuda = use_cuda
        self._on_trace_ready = on_trace_ready
        self._profile_memory = profile_memory
        self._with_stack = with_stack
        self._with_modules = with_modules
        self._check_params()
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
            FileManager.remove_file_safety(self._msprofiler_interface.path)

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
        if isinstance(self._action_controller._on_trace_ready, NpuProfCreator):
            print(f"[WARNING] [{os.getpid()}] profiler.py: "
                  "Already generate result files for TensorBoard, export_chrome_trace not producing any effect")
            return
        if not self._msprofiler_interface.path:
            print(f"[WARNING] [{os.getpid()}] profiler.py: Invalid profiling path.")
            return
        try:
            NpuProfiler.analyse(self._msprofiler_interface.path, Constant.EXPORT_CHROME_TRACE, output_path)
        except Exception:
            print(f"[WARNING] [{os.getpid()}] profiler.py: Profiling data parsing failed.")
            return
        try:
            shutil.rmtree(self._msprofiler_interface.path)
        except Exception:
            print(f"[WARNING] [{os.getpid()}] profiler.py: Can't remove directory: {self._msprofiler_interface.path}")

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
            warn("This is npu environment, use_cuda is invalid")
