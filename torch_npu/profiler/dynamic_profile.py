import os
import json
import atexit
import time

from .profiler import tensorboard_trace_handler, profile
from .scheduler import Schedule as schedule

from .analysis.prof_common_func.singleton import Singleton
from ..utils.path_manager import PathManager
from .analysis.prof_common_func.constant import print_warn_msg
from .analysis.prof_common_func.constant import print_error_msg
from ._dynamic_profiler import logger, init_logger, DynamicProfilerMonitor

__all__ = [
    'init',
    'step',
]


@Singleton
class _DynamicProfile:
    RECORD_TIME_STEP = 10

    def __init__(self) -> None:
        self.config_path = None
        self.prof = None
        self.cfg_ctx = None
        self.cur_mtime = None
        self.step_num = 0
        self.cur_step = 0
        self._dynamic_monitor = None
        self.repeat_init = False
        self._step_record_time = None
        self._step_time = 0
        self._min_poll_interval = 1

    def init(self, path: str, buffer_size: int = 1024, poll_interval: int = 2):
        if self.repeat_init:
            print_warn_msg("Init dynamic profiling repeatedly")
            return
        self._dynamic_monitor = DynamicProfilerMonitor(path, buffer_size, poll_interval)
        self.repeat_init = True
        atexit.register(self._dynamic_monitor.clean_resource)

    def _dynamic_profiler_valid(self):
        prof_cfg_ctx = self._dynamic_monitor.shm_to_prof_conf_context()
        if prof_cfg_ctx is None:
            return None
        else:
            return prof_cfg_ctx

    def step(self):
        self.cur_step += 1
        if self.cur_step == self.RECORD_TIME_STEP:
            self._step_record_time = time.time()
        elif self.cur_step - self.RECORD_TIME_STEP == 1:
            self._step_time = max(self._min_poll_interval, int(time.time() - self._step_record_time))
            self._dynamic_monitor.modify_step_time(self._step_time)
        if self.prof:
            self.prof.step()
            self.step_num -= 1
            if 0 == self.step_num:
                self.prof.stop()
                self.prof = None
                logger.info(f"Stop Dynamic Profiler at {self.cur_step} step.")
        elif self.prof is None:
            self.cfg_ctx = self._dynamic_profiler_valid()
            if self.cfg_ctx is None:
                return
            self.step_num = self.cfg_ctx.active()
            self.enable_prof()
            self.cfg_ctx = None

    def enable_prof(self):
        self.prof = profile(
            activities=self.cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=self.cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_ctx.prof_path, analyse_flag=self.cfg_ctx.analyse()),
            record_shapes=self.cfg_ctx.record_shapes,
            profile_memory=self.cfg_ctx.profile_memory,
            with_stack=self.cfg_ctx.with_stack,
            with_flops=self.cfg_ctx.with_flops,
            with_modules=self.cfg_ctx.with_modules,
            experimental_config=self.cfg_ctx.experimental_config
        )
        self.prof.start()
        for key, value in self.cfg_ctx.meta_data().items():
            self.prof.add_metadata_json(str(key), json.dumps(value))
        logger.info(f"Start Dynamic Profiler at {self.cur_step} step.")


def init(path: str):
    try:
        PathManager.check_input_directory_path(path)
    except RuntimeError:
        print_error_msg(f"The path '{path}' is invalid, and profiler will not be enabled.")
        return
    dp_path = os.path.abspath(path)
    init_logger(logger, dp_path)
    _DynamicProfile().init(dp_path)


def step():
    _DynamicProfile().step()
