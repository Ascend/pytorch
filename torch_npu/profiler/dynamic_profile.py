from datetime import datetime
import logging
import logging.handlers
import json
from json import JSONDecodeError
import os
import socket

from torch_npu._C._profiler import ProfilerActivity
from .profiler import tensorboard_trace_handler, profile
from .experimental_config import _ExperimentalConfig, ProfilerLevel, AiCMetrics
from .scheduler import Schedule as schedule
from .analysis.prof_common_func._singleton import Singleton
from ..utils.path_manager import PathManager
from .analysis.prof_common_func._file_manager import FileManager
from .analysis.prof_common_func._constant import print_warn_msg
from .analysis.prof_common_func._constant import print_error_msg

__all__ = [
    'init',
    'step',
]

logger = logging.getLogger(__name__)


def _init_logger(path: str):
    worker_name = "{}".format(socket.gethostname())
    log_name = "dp_{}_{}.log".format(worker_name, os.getpid())
    log_file = os.path.join(path, log_name)
    if not os.path.exists(log_file):
        PathManager.create_file_safety(log_file)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(process)d] %(filename)s: %(message)s")
    handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)


class _ConfigContext():
    DEFAULT_ACTIVE_NUM = 1

    def __init__(self, path: str):
        self.activity_set = set()
        self.prof_path = str()
        self.analyse = False
        self.record_shapes = False
        self.profile_memory = False
        self.with_stack = False
        self.with_flops = False
        self.with_modules = False
        self.experimental_config = None
        self.active = 1
        self.parse(path)

    def parse(self, path: str):
        config_data = FileManager.file_read_all(path)
        if not config_data:
            return
        try:
            json_data = json.loads(config_data)
        except JSONDecodeError as e:
            logger.error(f"profiler_config.json parse failed. {e}")
            return
        activities = json_data.get('activities')
        if activities and isinstance(activities, list):
            for entry in activities:
                activity = getattr(ProfilerActivity, entry.upper(), None)
                if activity:
                    self.activity_set.add(activity)
        self.prof_path = json_data.get('prof_dir')
        self.analyse = json_data.get('analyse', False)
        self.record_shapes = json_data.get('record_shapes', False)
        self.profile_memory = json_data.get('profile_memory', False)
        self.with_stack = json_data.get('with_stack', False)
        self.with_flops = json_data.get('with_flops', False)
        self.with_modules = json_data.get('with_modules', False)
        self.active = json_data.get('active', self.DEFAULT_ACTIVE_NUM)
        exp_config = json_data.get('experimental_config')
        if not exp_config:
            self.experimental_config = None
        else:
            profiler_level = exp_config.get('profiler_level', 'Level0')
            profiler_level = getattr(ProfilerLevel, profiler_level)
            aic_metrics = exp_config.get('aic_metrics', 'AiCoreNone')
            aic_metrics = getattr(AiCMetrics, aic_metrics)
            l2_cache = exp_config.get('l2_cache', False)
            op_attr = exp_config.get('op_attr', False)
            gc_detect_threshold = exp_config.get('gc_detect_threshold', None)
            data_simplification = exp_config.get('data_simplification', True)
            record_op_args = exp_config.get('record_op_args', False)
            export_type = exp_config.get('export_type', 'text')
            msprof_tx = exp_config.get('msprof_tx', False)
            self.experimental_config = _ExperimentalConfig(
                profiler_level=profiler_level,
                aic_metrics=aic_metrics,
                l2_cache=l2_cache,
                op_attr=op_attr,
                gc_detect_threshold=gc_detect_threshold,
                data_simplification=data_simplification,
                record_op_args=record_op_args,
                export_type=export_type,
                msprof_tx=msprof_tx
            )

    def activities(self) -> list:
        return list(self.activity_set)

    def prof_path(self) -> str:
        return self.prof_path

    def analyse(self) -> bool:
        return self.analyse

    def record_shapes(self) -> bool:
        return self.record_shapes

    def profile_memory(self) -> bool:
        return self.profile_memory

    def with_stack(self) -> bool:
        return self.with_stack

    def with_flops(self) -> bool:
        return self.with_flops

    def with_modules(self) -> bool:
        return self.with_modules

    def active(self) -> int:
        if not isinstance(self.active, int) or self.active <= 0:
            return self.DEFAULT_ACTIVE_NUM
        return self.active

    def experimental_config(self) -> _ExperimentalConfig:
        return self.experimental_config


@Singleton
class _DynamicProfile():
    def __init__(self) -> None:
        self.config_path = None
        self.prof = None
        self.cfg_ctx = None
        self.cur_mtime = None
        self.step_num = 0
        self.cur_step = 0
        self.repeat_init = False

    @staticmethod
    def init_default_config(path: str):
        json_data = {
            "activities": ["CPU", "NPU"],
            "prof_dir": "./",
            "analyse": False,
            "record_shapes": False,
            "profile_memory": False,
            "with_stack": False,
            "with_flops": False,
            "with_modules": False,
            "active": 1,
            "experimental_config": {
                "profiler_level": "Level0",
                "aic_metrics": "AiCoreNone",
                "l2_cache": False,
                "op_attr": False,
                "gc_detect_threshold": None,
                "data_simplification": True,
                "record_op_args": False,
                "export_type": "text",
                "msprof_tx": False
            }
        }
        FileManager.create_json_file_by_path(path, json_data, indent=4)

    def is_repeat_init(self):
        return self.repeat_init

    def init(self, path: str):
        if self.config_path:
            print_warn_msg("Init dynamic profiling repeatedly")
            self.repeat_init = True
        log_path = os.path.join(path, 'log')
        if not os.path.exists(log_path):
            PathManager.make_dir_safety(log_path)
        _init_logger(log_path)
        self.config_path = os.path.join(path, 'profiler_config.json')
        if not os.path.exists(self.config_path):
            logger.info("Create profiler_config.json default.")
            self.init_default_config(self.config_path)
        file_stat = os.stat(self.config_path)
        self.cur_mtime = file_stat.st_mtime

    def step(self):
        self.cur_step += 1
        if self.prof:
            self.prof.step()
            self.step_num += 1
            if self.step_num == self.cfg_ctx.active:
                self.prof.stop()
                self.prof = None
                self.cfg_ctx = None
                logger.info(f"Stop Dynamic Profiler at {self.cur_step} step.")
        elif os.path.exists(self.config_path) and self.file_changed():
            self.step_num = 0
            self.cfg_ctx = _ConfigContext(self.config_path)
            self.enable_prof()

    def file_changed(self) -> bool:
        file_stat = os.stat(self.config_path)
        if file_stat.st_mtime == self.cur_mtime:
            return False
        self.cur_mtime = file_stat.st_mtime
        return True

    def enable_prof(self):
        self.prof = profile(
            activities=self.cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=self.cfg_ctx.active, repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_ctx.prof_path, analyse_flag=self.cfg_ctx.analyse),
            record_shapes=self.cfg_ctx.record_shapes,
            profile_memory=self.cfg_ctx.profile_memory,
            with_stack=self.cfg_ctx.with_stack,
            with_flops=self.cfg_ctx.with_flops,
            with_modules=self.cfg_ctx.with_modules,
            experimental_config=self.cfg_ctx.experimental_config
        )
        self.prof.start()
        logger.info(f"Start Dynamic Profiler at {self.cur_step} step.")


def init(path: str):
    try:
        PathManager.check_input_directory_path(path)
    except RuntimeError:
        print_error_msg(f"The path '{path}' is invalid, and profiler will not be enabled.")
        return
    dp_path = os.path.abspath(path)
    if not os.path.exists(dp_path):
        PathManager.make_dir_safety(dp_path)
    _DynamicProfile().init(dp_path)


def step():
    _DynamicProfile().step()
