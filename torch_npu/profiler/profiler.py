import shutil
from warnings import warn

from .analysis.npu_profiler import NpuProfiler
from .profiler_action_controller import ActionController
from .profiler_action_controller import NpuProfCreator
from .msprofiler_c_interface import MsProfilerInterface, supported_ms_activities
from .scheduler import CLOSE_STEP


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
            use_cuda=None):
        self._activities = set(activities) if activities else supported_ms_activities()
        self._schedule = schedule
        self._record_shapes = record_shapes
        self._with_flops = with_flops
        self._record_shapes |= self._with_flops
        self._msprofiler_interface = MsProfilerInterface([self._record_shapes, profile_memory,
                                                          with_stack, self._with_flops, with_modules],
                                                         self._activities)
        self._action_controller = ActionController(self._msprofiler_interface, schedule, self, on_trace_ready)
        self._use_cuda = use_cuda
        self._check_params()

    def __enter__(self):
        self._action_controller.transit_action()
        return self

    def __exit__(self, exe_type, exe_val, exc_tb):
        self._action_controller.next_step = CLOSE_STEP
        self._action_controller.transit_action()

    def step(self):
        if self._schedule:
            self._action_controller.transit_action()

    def export_chrome_trace(self, output_path: str):
        if isinstance(self._action_controller._on_trace_ready, NpuProfCreator):
            warn("Already generate result files for TensorBoard, export_chrome_trace not producing any effect")
            return
        if self._action_controller.next_step == CLOSE_STEP + 1:
            NpuProfiler.analyse(self._msprofiler_interface.path, output_path)
            try:
                shutil.rmtree(self._msprofiler_interface.path)
            except Exception:
                warn(f"Can't remove directory: {self._msprofiler_interface.path}")
        else:
            raise RuntimeError("Profiler didn't finish running")

    def _check_params(self):
        if self._use_cuda is not None:
            warn("This is npu environment, use_cuda is invalid")
