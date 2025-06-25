import os
import json
import atexit
import time

from ..npu import mstx, current_stream
from .profiler import tensorboard_trace_handler, profile
from .scheduler import Schedule as schedule

from .analysis.prof_common_func._singleton import Singleton
from ..utils._path_manager import PathManager
from .analysis.prof_common_func._utils import no_exception_func
from .analysis.prof_common_func._file_manager import FileManager
from ._dynamic_profiler._dynamic_profiler_utils import DynamicProfilerUtils
from ._dynamic_profiler._dynamic_profiler_monitor import DynamicProfilerMonitor
from ._dynamic_profiler._dynamic_profiler_config_context import ConfigContext
from ._dynamic_profiler._dynamic_monitor_proxy import PyDynamicMonitorProxySingleton


__all__ = [
    'init',
    'step',
    'start'
]


@Singleton
class _DynamicProfile:
    RECORD_TIME_STEP = 10

    def __init__(self) -> None:
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
        self._step_mstx_range_id = 0

    def init(self):
        if self.repeat_init:
            DynamicProfilerUtils.stdout_log("Init dynamic profiling repeatedly",
                                            DynamicProfilerUtils.LoggerLevelEnum.WARNING)
            return
        self._dynamic_monitor = DynamicProfilerMonitor()
        self.repeat_init = True
        atexit.register(self._clean_resource)
        atexit.register(self._finalize_dynolog)

    def _clean_resource(self):
        if self.prof is not None:
            self.prof.stop()
            self.prof = None
            DynamicProfilerUtils.stdout_log(
                "Profiler stop when process exit, check cfg json active whether over all step!",
                DynamicProfilerUtils.LoggerLevelEnum.WARNING)
        self._dynamic_monitor.clean_resource()

    def _finalize_dynolog(self):
        py_dyno_monitor = PyDynamicMonitorProxySingleton().get_proxy()
        if py_dyno_monitor:
            py_dyno_monitor.finalize_dyno()

    def _dynamic_profiler_valid(self):
        prof_cfg_ctx = self._dynamic_monitor.shm_to_prof_conf_context()
        return prof_cfg_ctx

    def step(self):
        self.cur_step += 1
        cfg_ctx = self._dynamic_profiler_valid()
        if cfg_ctx is not None:
            self.cfg_ctx = cfg_ctx
        if self.cur_step == self.RECORD_TIME_STEP:
            self._step_record_time = time.time()
        elif self.cur_step - self.RECORD_TIME_STEP == 1:
            self._step_time = max(self._min_poll_interval, int(time.time() - self._step_record_time))
            self._dynamic_monitor.modify_step_time(self._step_time)
        if self.prof:
            if self._step_mstx_range_id:
                mstx.range_end(self._step_mstx_range_id)
                self._step_mstx_range_id = mstx.range_start(f"step {self.cur_step}", current_stream())
            self.prof.step()
            self.step_num -= 1
            if 0 == self.step_num:
                self.prof.stop()
                self.prof = None
                DynamicProfilerUtils.out_log("Stop Dynamic Profiler at {} step.".format(
                    self.cur_step), DynamicProfilerUtils.LoggerLevelEnum.INFO)
        elif self.prof is None and self.cfg_ctx is not None and self.cur_step == self.cfg_ctx.start_step():
            self.step_num = self.cfg_ctx.active() + self.cfg_ctx.warmup()
            self.enable_prof()
            self.cfg_ctx = None

    def start(self, config_path: str):
        if self.prof:
            DynamicProfilerUtils.stdout_log("Profiler already started. "
                                            "Cannot call start interface while the profiler is active. ",
                                            DynamicProfilerUtils.LoggerLevelEnum.ERROR)
            return
        enable_config_path = ""
        if config_path:
            try:
                PathManager.check_input_file_path(config_path)
                PathManager.check_directory_path_readable(config_path)
                enable_config_path = config_path
            except Exception as err:
                DynamicProfilerUtils.stdout_log("The provided config_path is invalid: {}. Details: {}".format(
                    config_path, str(err)), DynamicProfilerUtils.LoggerLevelEnum.ERROR)
                enable_config_path = ""
        if not enable_config_path:
            enable_config_path = self._dynamic_monitor._config_path
        DynamicProfilerUtils.stdout_log("The start interface profiler enable config path is set to {}".format(
            enable_config_path), DynamicProfilerUtils.LoggerLevelEnum.INFO)
        try:
            json_data = FileManager.read_json_file(enable_config_path)
            if not json_data:
                DynamicProfilerUtils.stdout_log("The config data is empty from: {}. Please check the config file. ".format(
                    enable_config_path), DynamicProfilerUtils.LoggerLevelEnum.ERROR)
                return
        except RuntimeError:
            DynamicProfilerUtils.stdout_log("Failed to read config from : {}. Please check the config file. ".format(
                enable_config_path), DynamicProfilerUtils.LoggerLevelEnum.ERROR)
            return
        self.cfg_ctx = ConfigContext(json_data)
        self.step_num = self.cfg_ctx.active() + self.cfg_ctx.warmup()
        self.enable_prof()
        self.cfg_ctx = None

    def enable_prof(self):
        self.prof = profile(
            activities=self.cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=self.cfg_ctx.warmup(), active=self.cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_ctx.prof_path, analyse_flag=self.cfg_ctx.analyse(),
                                                     async_mode=self.cfg_ctx.async_mode()),
            record_shapes=self.cfg_ctx.record_shapes,
            profile_memory=self.cfg_ctx.profile_memory,
            with_stack=self.cfg_ctx.with_stack,
            with_flops=self.cfg_ctx.with_flops,
            with_modules=self.cfg_ctx.with_modules,
            experimental_config=self.cfg_ctx.experimental_config
        )
        self.prof._set_step_num_offset_for_dynamic_prof(self.cur_step)
        self.prof.start()
        self._step_mstx_range_id = mstx.range_start(f"step {self.cur_step}", current_stream())
        for key, value in self.cfg_ctx.meta_data().items():
            self.prof.add_metadata_json(str(key), json.dumps(value))
        DynamicProfilerUtils.out_log("Start Dynamic Profiler at {} step.".format(
            self.cur_step), DynamicProfilerUtils.LoggerLevelEnum.INFO)


@no_exception_func()
def init(path: str):
    if DynamicProfilerUtils.is_dyno_model():
        _DynamicProfile().init()
        return
    try:
        PathManager.check_input_directory_path(path)
    except RuntimeError:
        DynamicProfilerUtils.stdout_log("The path '{}' is invalid, and profiler will not be enabled.".format(
            path), DynamicProfilerUtils.LoggerLevelEnum.ERROR)
        return
    DynamicProfilerUtils.CFG_CONFIG_PATH = os.path.abspath(path)
    DynamicProfilerUtils.init_logger()
    _DynamicProfile().init()


@no_exception_func()
def step():
    _DynamicProfile().step()


@no_exception_func()
def start(config_path: str = None):
    _DynamicProfile().start(config_path)
