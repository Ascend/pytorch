import shutil
from warnings import warn

from torch_npu.npu import _lazy_init
from .analysis.npu_profiler import NpuProfiler
from .analysis.prof_common_func.constant import Constant
from .experimental_config import _ExperimentalConfig
from .profiler_action_controller import ActionController
from .profiler_action_controller import NpuProfCreator
from .msprofiler_c_interface import MsProfilerInterface, supported_ms_activities, ProfilerActivity
from .scheduler import CLOSE_STEP, ProfilerAction


def tensorboard_trace_handler(dir_name: str, worker_name: str = None, use_gzip: bool = False):
    return NpuProfCreator(worker_name, dir_name)


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
            use_cuda=None):
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
        self._check_params()

    def __enter__(self):
        _lazy_init()
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
            try:
                shutil.rmtree(self._msprofiler_interface.path)
            except Exception:
                warn(f"Can't remove directory: {self._msprofiler_interface.path}")

    def step(self):
        if self._schedule:
            self._action_controller.transit_action()

    def export_chrome_trace(self, output_path: str):
        if isinstance(self._action_controller._on_trace_ready, NpuProfCreator):
            warn("Already generate result files for TensorBoard, export_chrome_trace not producing any effect")
            return
        if self._action_controller.next_step == CLOSE_STEP + 1:
            level_config = {
                Constant.PROFILER_LEVEL: self._experimental_config.profiler_level(),
                Constant.AI_CORE_METRICS: self._experimental_config.aic_metrics(),
                Constant.L2_CACHE: self._experimental_config.l2_cache()
            }
            NpuProfiler.analyse(self._msprofiler_interface.path, level_config, output_path)
            try:
                shutil.rmtree(self._msprofiler_interface.path)
            except Exception:
                warn(f"Can't remove directory: {self._msprofiler_interface.path}")
        else:
            raise RuntimeError("Profiler didn't finish running")

    def _check_params(self):
        if self._use_cuda is not None:
            warn("This is npu environment, use_cuda is invalid")
