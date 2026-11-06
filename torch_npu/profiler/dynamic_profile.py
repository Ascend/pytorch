import os
import json
import atexit
import time

from ..npu import mstx, current_stream
from .profiler import tensorboard_trace_handler, profile
from .scheduler import Schedule as schedule

from .analysis.prof_common_func._singleton import Singleton
from .analysis.prof_common_func._constant import print_warn_msg
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
    'start',
    'set_state'
]


@Singleton
class _DynamicProfile:
    RECORD_TIME_STEP = 10
    MAX_CONFIG_QUEUE_SIZE = 5

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
        self._max_poll_interval = 50
        self._step_mstx_range_id = 0
        self._cfg_ctx_queue = []
        self._seen_trace_ids = {}  # used as an ordered set of seen trace_ids
        self._max_seen_trace_ids = 10000
        self._running_trace_id = ""
        self._rejected_cfg_count = 0

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
            self._running_trace_id = ""
            DynamicProfilerUtils.stdout_log(
                "Profiler stop when process exit, check cfg json active whether over all step!",
                DynamicProfilerUtils.LoggerLevelEnum.WARNING)
        self._dynamic_monitor.clean_resource()

    def _finalize_dynolog(self):
        py_dyno_monitor = PyDynamicMonitorProxySingleton().get_proxy()
        if py_dyno_monitor:
            py_dyno_monitor.finalize_dyno()

    def _dynamic_profiler_valid(self):
        return self._dynamic_monitor.shm_to_prof_conf_context()

    def set_state(self, state_step: dict):
        if not isinstance(state_step, dict):
            DynamicProfilerUtils.stdout_log("Dynamic profiler state_step must be dict.",
                                            DynamicProfilerUtils.LoggerLevelEnum.ERROR)
            return
        cur_step = state_step.get("cur_step", 0)
        if not isinstance(cur_step, int) or cur_step < 0:
            DynamicProfilerUtils.stdout_log("Dynamic profiler cur_step must be a non-negative integer.",
                                            DynamicProfilerUtils.LoggerLevelEnum.ERROR)
            return
        self.cur_step = cur_step

    def _remember_trace_id(self, trace_id: str):
        if trace_id not in self._seen_trace_ids:
            if len(self._seen_trace_ids) >= self._max_seen_trace_ids:
                self._seen_trace_ids.pop(next(iter(self._seen_trace_ids)))
            self._seen_trace_ids[trace_id] = None

    def _append_cfg_ctxs(self, cfg_ctx_list: list):
        skipped_trace_ids = []
        expired_trace_ids = []
        rejected_trace_ids = []
        for ctx in cfg_ctx_list:
            trace_id = ctx.trace_id()
            if trace_id and trace_id in self._seen_trace_ids:
                skipped_trace_ids.append(trace_id)
                continue
            if not ctx.start() and self.cur_step > ctx.start_step():
                expired_trace_ids.append(trace_id or "N/A")
                continue
            if len(self._cfg_ctx_queue) >= self.MAX_CONFIG_QUEUE_SIZE:
                self._rejected_cfg_count += 1
                rejected_trace_ids.append(trace_id or "N/A")
                continue
            if trace_id:
                self._remember_trace_id(trace_id)
            ctx.enqueued_step = self.cur_step
            self._cfg_ctx_queue.append(ctx)
        if expired_trace_ids:
            DynamicProfilerUtils.out_log(
                f"Drop {len(expired_trace_ids)} expired config(s) on arrival, trace ids = {expired_trace_ids[:10]}",
                DynamicProfilerUtils.LoggerLevelEnum.WARNING)
        if rejected_trace_ids:
            DynamicProfilerUtils.out_log(
                f"Config queue is full (limit {self.MAX_CONFIG_QUEUE_SIZE}), "
                f"reject {len(rejected_trace_ids)} config(s), "
                f"trace ids = {rejected_trace_ids[:10]}",
                DynamicProfilerUtils.LoggerLevelEnum.WARNING)
        if skipped_trace_ids:
            DynamicProfilerUtils.out_log(
                f"Skip {len(skipped_trace_ids)} duplicate config context(s), trace ids = {skipped_trace_ids[:10]}",
                DynamicProfilerUtils.LoggerLevelEnum.INFO)

    def _apply_latest_cfg_ctxs(self, cfg_ctx_list: list):
        skipped_trace_ids = []
        latest_ctx = None
        for ctx in cfg_ctx_list:
            trace_id = ctx.trace_id()
            if trace_id and trace_id in self._seen_trace_ids:
                skipped_trace_ids.append(trace_id)
                continue
            if trace_id:
                self._remember_trace_id(trace_id)
            ctx.enqueued_step = self.cur_step
            latest_ctx = ctx
        if skipped_trace_ids:
            DynamicProfilerUtils.out_log(
                f"Skip {len(skipped_trace_ids)} duplicate config context(s), trace ids = {skipped_trace_ids[:10]}",
                DynamicProfilerUtils.LoggerLevelEnum.INFO)
        if latest_ctx is None:
            return
        if self.cfg_ctx is not None:
            DynamicProfilerUtils.out_log(
                f"Replace pending config (start_step={self.cfg_ctx.start_step()}) with latest "
                f"config (start_step={latest_ctx.start_step()}), trace id = {latest_ctx.trace_id() or 'N/A'}",
                DynamicProfilerUtils.LoggerLevelEnum.INFO)
        self.cfg_ctx = latest_ctx

    def _status_trace_id(self) -> str:
        if self.prof is not None:
            return self._running_trace_id
        if self.cfg_ctx is not None:
            return self.cfg_ctx.trace_id()
        if self._cfg_ctx_queue:
            return self._cfg_ctx_queue[0].trace_id()
        return ""

    def _status_queue_depth(self) -> int:
        return len(self._cfg_ctx_queue) + (1 if self.cfg_ctx is not None else 0)

    @staticmethod
    def _queued_while_valid(cfg_ctx, start_step: int) -> bool:
        enqueued_step = getattr(cfg_ctx, "enqueued_step", None)
        return enqueued_step is not None and enqueued_step <= start_step

    def step(self):
        self.cur_step += 1
        new_cfg_ctx_list = self._dynamic_profiler_valid()
        if new_cfg_ctx_list:
            if DynamicProfilerUtils.is_dyno_model():
                self._append_cfg_ctxs(new_cfg_ctx_list)
            else:
                self._apply_latest_cfg_ctxs(new_cfg_ctx_list)
        if self.cur_step == self.RECORD_TIME_STEP:
            self._step_record_time = time.time()
        elif self.cur_step - self.RECORD_TIME_STEP == 1 and self._step_record_time is not None:
            self._step_time = min(self._max_poll_interval, max(
                self._min_poll_interval, int(time.time() - self._step_record_time)))
            self._dynamic_monitor.modify_step_time(self._step_time)
        if self._step_mstx_range_id:
            mstx.range_end(self._step_mstx_range_id)
            self._step_mstx_range_id = mstx.range_start(f"step {self.cur_step}", current_stream())
        if self.prof:
            self.prof.step()
            self.step_num -= 1
            if 0 == self.step_num:
                self.prof.stop()
                self._dynamic_monitor.update_profiler_status(DynamicProfilerUtils.ProfilerStatus.IDLE)
                self.prof = None
                self._running_trace_id = ""
                DynamicProfilerUtils.out_log("Stop Dynamic Profiler at {} step.".format(
                    self.cur_step), DynamicProfilerUtils.LoggerLevelEnum.INFO)
        elif self.prof is None:
            if self.cfg_ctx is None and self._cfg_ctx_queue:
                self.cfg_ctx = self._cfg_ctx_queue.pop(0)
            if self.cfg_ctx is not None:
                step_num = self.cfg_ctx.active() + self.cfg_ctx.warmup()
                start_step = self.cfg_ctx.start_step()
                instant = self.cfg_ctx.start()
                delayed = (self.cur_step > start_step and not instant
                           and self._queued_while_valid(self.cfg_ctx, start_step))
                expired = self.cur_step > start_step and not instant and not delayed
                if expired:
                    print_warn_msg(f"Dynamic Profiler config is not effective. The start_step={start_step}, "
                                   f"current_step={self.cur_step}")
                    self._dynamic_monitor.update_profiler_status(DynamicProfilerUtils.ProfilerStatus.IDLE)
                    self.cfg_ctx = None
                elif instant or self.cur_step >= start_step:
                    if delayed:
                        print_warn_msg(f"Dynamic Profiler config is delayed by the config queue, start at "
                                       f"current_step={self.cur_step}, start_step={start_step}")
                    self.step_num = step_num
                    self._dynamic_monitor.update_step_info(self.cur_step, DynamicProfilerUtils.START_STEP)
                    self._dynamic_monitor.update_step_info(self.cur_step + step_num,
                                                           DynamicProfilerUtils.STOP_STEP)
                    self._dynamic_monitor.update_profiler_status(DynamicProfilerUtils.ProfilerStatus.RUNNING)
                    self.enable_prof()
                    self.cfg_ctx = None
                else:
                    self._dynamic_monitor.update_step_info(start_step, DynamicProfilerUtils.START_STEP)
                    self._dynamic_monitor.update_step_info(start_step + step_num,
                                                           DynamicProfilerUtils.STOP_STEP)
                    self._dynamic_monitor.update_profiler_status(DynamicProfilerUtils.ProfilerStatus.READY)
        self._dynamic_monitor.update_cfg_status(
            self._status_trace_id(), self._status_queue_depth(), self._rejected_cfg_count)
        self._dynamic_monitor.update_step_info(self.cur_step, DynamicProfilerUtils.CURRENT_STEP)
        self._dynamic_monitor.report_profiler_status()

        if not self._step_mstx_range_id:
            self._step_mstx_range_id = mstx.range_start(f"step {self.cur_step}", current_stream())

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
        if self.cfg_ctx.trace_id():
            self._remember_trace_id(self.cfg_ctx.trace_id())
        self.step_num = self.cfg_ctx.active() + self.cfg_ctx.warmup()
        self.enable_prof()
        self.cfg_ctx = None
        self._step_mstx_range_id = mstx.range_start(f"step {self.cur_step}", current_stream())

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
        self._running_trace_id = self.cfg_ctx.trace_id()
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
    DynamicProfilerUtils.init_logger(log_path=os.path.abspath(path))
    _DynamicProfile().init()


@no_exception_func()
def step():
    _DynamicProfile().step()


@no_exception_func()
def start(config_path: str = None):
    _DynamicProfile().start(config_path)


@no_exception_func()
def set_state(state_step: dict):
    _DynamicProfile().set_state(state_step)
