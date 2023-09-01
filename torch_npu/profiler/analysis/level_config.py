import json
import os
import re
from json import JSONDecodeError

from .prof_common_func.file_manager import FileManager
from .prof_common_func.path_manager import PathManager
from .prof_common_func.singleton import Singleton
from .prof_common_func.constant import Constant
from .prof_bean.ai_cpu_bean import AiCpuBean
from .prof_parse.cann_file_parser import CANNDataEnum
from .prof_bean.l2_cache_bean import L2CacheBean


@Singleton
class LevelConfig:
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

    def load_info(self, profiler_path: str):
        info_file_path = PathManager.get_info_file_path(profiler_path)
        if not info_file_path:
            print(f"[WARNING] [{os.getpid()}] profiler.py: Failed to get profiler config info.")
            return
        try:
            info_json = json.loads(FileManager.file_read_all(info_file_path, "rt"))
        except JSONDecodeError:
            print(f"[WARNING] [{os.getpid()}] profiler.py: Failed to get profiler config info.")
            return
        experimental_config = info_json.get(Constant.CONFIG, {}).get(Constant.EXPERIMENTAL_CONFIG, {})
        self._profiler_level = experimental_config.get(Constant.PROFILER_LEVEL, self._profiler_level)
        self._ai_core_metrics = experimental_config.get(Constant.AI_CORE_METRICS, self._ai_core_metrics)
        self._l2_cache = experimental_config.get(Constant.L2_CACHE, self._l2_cache)

    def get_parser_bean(self):
        return self.LEVEL_PARSER_CONFIG.get(self._profiler_level) + self._get_l2_cache_bean()

    def get_prune_config(self):
        return self.LEVEL_TRACE_PRUNE_CONFIG.get(self._profiler_level)

    def is_all_kernel_headers(self):
        if self._ai_core_metrics != Constant.AicMetricsNone:
            return True
        else:
            return False

    def _get_l2_cache_bean(self):
        return [(CANNDataEnum.L2_CACHE, L2CacheBean)] if self.l2_cache else []

    @property
    def profiler_level(self):
        return self._profiler_level

    @property
    def l2_cache(self):
        return self._l2_cache
