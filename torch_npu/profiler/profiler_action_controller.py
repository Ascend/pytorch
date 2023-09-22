import os
import socket
import time
from warnings import warn

import torch.autograd.profiler as prof

from .analysis.npu_profiler import NpuProfiler
from .scheduler import default_schedule_fn, ProfilerAction
from .analysis.prof_common_func.constant import Constant
from .analysis.prof_common_func.file_manager import FileManager


class NpuProfCreator:
    DEFAULT_PROF_SUFFIX = "{}_{}".format(socket.gethostname(), str(os.getpid()))

    def __init__(self, worker_name: str = None, dir_name: str = "./") -> None:
        self._worker_name = worker_name
        self._dir_name = dir_name

    @classmethod
    def __call__(cls, instance: any) -> None:
        try:
            NpuProfiler.analyse(instance._msprofiler_interface.path)
        except Exception:
            print(f"[WARNING] [{os.getpid()}] profiler.py: Profiling data parsing failed.")

    def create_prof_dir(self) -> str:
        FileManager.check_input_path(self._dir_name)
        if not self._worker_name:
            self._worker_name = "{}_{}".format(socket.gethostname(), str(os.getpid()))
        worker_span_name = "{}_{}_ascend_pt".format(self._worker_name,
                                                    time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))

        total_path = os.path.join(self._dir_name, worker_span_name)
        FileManager.make_dir_safety(total_path)
        FileManager.check_directory_path_writable(total_path)
        return total_path

    def create_default_prof_dir(self) -> str:
        dir_name = os.getenv(Constant.ASCEND_WORK_PATH, default=None)
        dir_name = os.path.join(os.path.abspath(dir_name), Constant.PROFILING_WORK_PATH) if dir_name else os.getcwd()
        target_path = "{}_{}_ascend_pt".format(self.DEFAULT_PROF_SUFFIX,
                                               time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))
        target_path = os.path.join(dir_name, target_path)
        FileManager.make_dir_safety(target_path)
        FileManager.check_directory_path_writable(target_path)
        return target_path


class ActionController:
    def __init__(self, msprofiler_interface: any, schedule: any, instance: any, on_trace_ready: any) -> None:
        self._msprofiler_interface = msprofiler_interface
        self._current_action = ProfilerAction.NONE
        self._record_steps = True if schedule else False
        self._schedule = schedule if schedule else default_schedule_fn
        self._action_map = self._init_action_map()
        self._prev_action = ProfilerAction.NONE
        self._instance = instance
        self._on_trace_ready = on_trace_ready
        self.next_step = 0
        self.step_rec_fc = None

    @classmethod
    def _warn_none_follow_record(cls) -> None:
        warn("Incorrect schedule: RECORD followed by NONE")

    @classmethod
    def _warn_warmup_follow_record(cls) -> None:
        warn("Incorrect schedule: RECORD followed by WARMUP")

    @classmethod
    def _warn_warmup_follow_none(cls) -> None:
        warn("Incorrect schedule: WARMUP followed by NONE")

    def transit_action(self):
        self._prev_action = self._current_action
        self._current_action = self._schedule(self.next_step)

        action_list = self._action_map.get((self._prev_action, self._current_action), [])
        if action_list:
            for action in action_list:
                action()

        self.next_step += 1

    def init(self) -> None:
        if isinstance(self._on_trace_ready, NpuProfCreator):
            path = self._on_trace_ready.create_prof_dir()
        else:
            path = NpuProfCreator().create_default_prof_dir()
        self._msprofiler_interface.set_config(path)
        self._msprofiler_interface.init_profiler()

    def start_prof(self) -> None:
        self._msprofiler_interface.start_profiler()
        self._iteration_start()

    def stop_prof(self) -> None:
        self._iteration_end()
        self._msprofiler_interface.stop_profiler()
        self._msprofiler_interface.finalize_profiler()
        self._instance.dump_profiler_info()

    def trace_ready(self) -> None:
        if isinstance(self._on_trace_ready, NpuProfCreator):
            self._on_trace_ready(self._instance)

    def _iteration_start(self) -> None:
        if self._record_steps:
            self.step_rec_fc = prof.record_function("ProfilerStep#" + str(self.next_step))
            self.step_rec_fc.__enter__()

    def _iteration_end(self) -> None:
        if self._record_steps:
            if self.step_rec_fc:
                self.step_rec_fc.__exit__(None, None, None)

    def _init_action_map(self) -> dict:
        return {
            (ProfilerAction.NONE, ProfilerAction.NONE): [],
            (ProfilerAction.NONE, ProfilerAction.WARMUP): [self.init],
            (ProfilerAction.NONE, ProfilerAction.RECORD): [self.init, self.start_prof],
            (ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE): [self.init, self.start_prof],

            (ProfilerAction.WARMUP, ProfilerAction.NONE): [self._warn_warmup_follow_none, self.start_prof,
                                                           self.stop_prof],
            (ProfilerAction.WARMUP, ProfilerAction.WARMUP): [],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD): [self.start_prof],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD_AND_SAVE): [self.start_prof],

            (ProfilerAction.RECORD, ProfilerAction.NONE): [self._warn_none_follow_record, self.stop_prof],
            (ProfilerAction.RECORD, ProfilerAction.WARMUP): [self._warn_warmup_follow_record, self.stop_prof],
            (ProfilerAction.RECORD, ProfilerAction.RECORD): [self._iteration_end, self._iteration_start],
            (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE): [self._iteration_end, self._iteration_start],

            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE): [self.stop_prof, self.trace_ready],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.WARMUP): [self.stop_prof, self.trace_ready, self.init],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD): [self.stop_prof, self.trace_ready, self.init,
                                                                      self.start_prof],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD_AND_SAVE): [self.stop_prof, self.trace_ready,
                                                                               self.init, self.start_prof]
        }
