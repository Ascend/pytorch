import json
import os
import re
from json import JSONDecodeError

from .prof_common_func.file_manager import FileManager
from .prof_common_func.path_manager import PathManager
from .prof_common_func.singleton import Singleton
from .prof_common_func.constant import Constant
from .prof_bean.ai_cpu_bean import AiCpuBean
from .prof_parse.cann_file_parser import CANNDataEnum, CANNFileParser
from .prof_bean.l2_cache_bean import L2CacheBean


@Singleton
class ProfilerConfig:
    LEVEL_PARSER_CONFIG = {
        Constant.LEVEL0: [],
        Constant.LEVEL1: [],
        Constant.LEVEL2: [(CANNDataEnum.AI_CPU, AiCpuBean)]
    }
    LEVEL_TRACE_PRUNE_CONFIG = {
        Constant.LEVEL0: ['CANN', 'AscendCL', 'Runtime', 'GE', 'Node', 'Model', 'Hccl', 'acl_to_npu'],
        Constant.LEVEL1: ['Runtime', 'GE', 'Node', 'Model', 'Hccl'],
        Constant.LEVEL2: []
    }

    def __init__(self):
        self._profiler_level = Constant.LEVEL0
        self._ai_core_metrics = Constant.AicMetricsNone
        self._l2_cache = False
        self._data_simplification = None
        self._is_cluster = False
        self._localtime_diff = 0

    @property
    def data_simplification(self):
        return self._data_simplification

    @classmethod
    def _get_profiler_info_json(cls, profiler_path: str):
        info_file_path = PathManager.get_info_file_path(profiler_path)
        if not info_file_path:
            print(f"[WARNING] [{os.getpid()}] profiler.py: Failed to get profiler info file.")
            return {}
        try:
            return json.loads(FileManager.file_read_all(info_file_path, "rt"))
        except JSONDecodeError:
            print(f"[WARNING] [{os.getpid()}] profiler.py: Failed to get profiler info json from file.")
            return {}

    def load_info(self, profiler_path: str):
        self.load_is_cluster(profiler_path)
        info_json = self._get_profiler_info_json(profiler_path)
        self.load_experimental_cfg_info(info_json)
        self.load_timediff_info(profiler_path, info_json)

    def load_is_cluster(self, profiler_path: str):
        info_file_path = PathManager.get_info_file_path(profiler_path)
        if info_file_path:
            self._is_cluster = re.match(r"^profiler_info_\d+\.json", os.path.basename(info_file_path))

    def load_timediff_info(self, profiler_path: str, info_json: dict):
        self._localtime_diff = CANNFileParser(profiler_path).get_localtime_diff()
        end_info = info_json.get(Constant.END_INFO, {})
        if not self._localtime_diff and end_info:
            self._localtime_diff = (end_info.get(Constant.FWK_END_TIME, 0) - end_info.get(Constant.FWK_END_MONOTONIC,
                                                                                          0)) / Constant.NS_TO_US

    def load_experimental_cfg_info(self, info_json: dict):
        experimental_config = info_json.get(Constant.CONFIG, {}).get(Constant.EXPERIMENTAL_CONFIG, {})
        self._profiler_level = experimental_config.get(Constant.PROFILER_LEVEL, self._profiler_level)
        self._ai_core_metrics = experimental_config.get(Constant.AI_CORE_METRICS, self._ai_core_metrics)
        self._l2_cache = experimental_config.get(Constant.L2_CACHE, self._l2_cache)
        self._data_simplification = experimental_config.get(Constant.DATA_SIMPLIFICATION, self._data_simplification)
        if self._data_simplification is None:
            self._data_simplification = self._get_default_state()

    def get_parser_bean(self):
        return self.LEVEL_PARSER_CONFIG.get(self._profiler_level) + self._get_l2_cache_bean()

    def get_prune_config(self):
        return self.LEVEL_TRACE_PRUNE_CONFIG.get(self._profiler_level)

    def is_all_kernel_headers(self):
        if self._ai_core_metrics != Constant.AicMetricsNone:
            return True
        else:
            return False

    def get_local_time(self, monotonic_time: float):
        return monotonic_time + self._localtime_diff

    def _get_l2_cache_bean(self):
        return [(CANNDataEnum.L2_CACHE, L2CacheBean)] if self._l2_cache else []

    def _get_default_state(self):
        return self._is_cluster
