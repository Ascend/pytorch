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
        Constant.LEVEL0: ['CANN', 'AscendCL', 'Runtime', 'GE', 'acl_to_npu'],
        Constant.LEVEL1: ['Runtime', 'GE'],
        Constant.LEVEL2: []
    }

    def __init__(self, profiler_level: int = Constant.LEVEL0,
                 ai_core_metrics: str = Constant.AicMetricsNone, l2_cache: bool = False):
        self._profiler_level = profiler_level
        self._ai_core_metrics = ai_core_metrics
        self._l2_cache = l2_cache

    def load_info(self, config_dict: dict):
        self._profiler_level = config_dict.get(Constant.PROFILER_LEVEL, self._profiler_level)
        self._ai_core_metrics = config_dict.get(Constant.AI_CORE_METRICS, self._ai_core_metrics)
        self._l2_cache = config_dict.get(Constant.L2_CACHE, self._l2_cache)

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
