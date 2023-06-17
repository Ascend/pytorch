# Copyright (c) 2023, Huawei Technologies.
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import socket
import time
from warnings import warn

import torch.autograd.profiler as prof

from .analysis.npu_profiler import NpuProfiler
from .scheduler import default_schedule_fn, ProfilerAction
from .analysis.prof_common_func.constant import Constant

class NpuProfCreator:
    DEFAULT_PROF_SUFFIX = "./profiler"

    def __init__(self, worker_name: str = None, dir_name: str = "./") -> None:
        self._worker_name = worker_name
        self._dir_name = dir_name

    @classmethod
    def __call__(cls, instance: any) -> None:
        level_config = {
            Constant.PROFILER_LEVEL: instance._experimental_config.profiler_level(),
            Constant.AI_CORE_METRICS: instance._experimental_config.aic_metrics(),
            Constant.L2_CACHE: instance._experimental_config.l2_cache()
        }
        NpuProfiler.analyse(instance._msprofiler_interface.path, level_config)

    @classmethod
    def make_dir(cls, target_path: str) -> any:
        if not os.path.isdir(target_path):
            try:
                os.makedirs(target_path, exist_ok=True)
            except Exception:
                raise RuntimeError("Can't create directory: " + target_path)

    def create_prof_dir(self) -> str:
        if not self._worker_name:
            self._worker_name = "{}_{}".format(socket.gethostname(), str(os.getpid()))
        worker_span_name = "{}_{}_ascend_pt".format(self._worker_name,
                                                    time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))

        total_path = os.path.join(self._dir_name, worker_span_name)
        self.make_dir(total_path)
        return total_path

    def create_default_prof_dir(self) -> str:
        target_path = "{}_{}_ascend_pt".format(self.DEFAULT_PROF_SUFFIX,
                                               time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))
        self.make_dir(target_path)
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

    def _init_action_map(self) -> dict:
        return {
            (ProfilerAction.NONE, ProfilerAction.NONE): [],
            (ProfilerAction.NONE, ProfilerAction.WARMUP): [self._init],
            (ProfilerAction.NONE, ProfilerAction.RECORD): [self._init, self._start_prof],
            (ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE): [self._init, self._start_prof],

            (ProfilerAction.WARMUP, ProfilerAction.NONE): [self._warn_warmup_follow_none, self._start_prof,
                                                           self._stop_prof],
            (ProfilerAction.WARMUP, ProfilerAction.WARMUP): [],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD): [self._start_prof],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD_AND_SAVE): [self._start_prof],

            (ProfilerAction.RECORD, ProfilerAction.NONE): [self._warn_none_follow_record, self._stop_prof],
            (ProfilerAction.RECORD, ProfilerAction.WARMUP): [self._warn_warmup_follow_record, self._stop_prof],
            (ProfilerAction.RECORD, ProfilerAction.RECORD): [self._iteration_end, self._iteration_start],
            (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE): [self._iteration_end, self._iteration_start],

            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE): [self._stop_prof, self._trace_ready],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.WARMUP): [self._stop_prof, self._trace_ready, self._init],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD): [self._stop_prof, self._trace_ready, self._init,
                                                                      self._start_prof],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD_AND_SAVE): [self._stop_prof, self._trace_ready,
                                                                               self._init, self._start_prof]
        }

    def _init(self) -> None:
        if isinstance(self._on_trace_ready, NpuProfCreator):
            path = self._on_trace_ready.create_prof_dir()
        else:
            path = NpuProfCreator().create_default_prof_dir()
        self._msprofiler_interface.set_config(path)
        self._msprofiler_interface.init_profiler()

    def _start_prof(self) -> None:
        self._msprofiler_interface.start_profiler()
        self._iteration_start()

    def _stop_prof(self) -> None:
        self._iteration_end()
        self._msprofiler_interface.stop_profiler()
        self._msprofiler_interface.finalize_profiler()

    def _trace_ready(self) -> None:
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
