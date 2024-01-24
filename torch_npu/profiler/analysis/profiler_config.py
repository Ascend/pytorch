import json
import os
import re
from json import JSONDecodeError
from configparser import ConfigParser

from .prof_common_func.file_manager import FileManager
from .prof_common_func.path_manager import ProfilerPathManager
from .prof_common_func.singleton import Singleton
from .prof_common_func.constant import Constant, print_warn_msg
from .prof_bean.ai_cpu_bean import AiCpuBean
from .prof_parse.cann_file_parser import CANNDataEnum, CANNFileParser
from .prof_bean.l2_cache_bean import L2CacheBean
from .prof_bean.op_statistic_bean import OpStatisticBean
from .prof_bean.npu_module_mem_bean import NpuModuleMemoryBean


@Singleton
class ProfilerConfig:
    LEVEL_PARSER_CONFIG = {
        Constant.LEVEL0: [(CANNDataEnum.NPU_MODULE_MEM, NpuModuleMemoryBean)],
        Constant.LEVEL1: [(CANNDataEnum.OP_STATISTIC, OpStatisticBean),
                          (CANNDataEnum.NPU_MODULE_MEM, NpuModuleMemoryBean)],
        Constant.LEVEL2: [(CANNDataEnum.AI_CPU, AiCpuBean), (CANNDataEnum.OP_STATISTIC, OpStatisticBean),
                          (CANNDataEnum.NPU_MODULE_MEM, NpuModuleMemoryBean)]
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
        self._data_simplification = True
        self._is_cluster = False
        self._localtime_diff = 0
        self._syscnt_enable = False
        self._freq = 100.0
        self._time_offset = 0
        self._start_cnt = 0

    @property
    def data_simplification(self):
        return self._data_simplification

    def is_number(self, string):
        pattern = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
        return bool(pattern.match(string))

    def get_timestamp_from_syscnt(self, syscnt: int, time_fmt: int = 1000):
        if self._syscnt_enable == False:
            return syscnt
        else:
            ratio = time_fmt / self._freq
            timestamp = int((syscnt - self._start_cnt) * ratio) + self._time_offset
            return timestamp

    def load_syscnt_info(self, profiler_path: str, info_json: dict):
        start_info = info_json.get(Constant.START_INFO, {})
        info_path = ProfilerPathManager.get_info_path(profiler_path)
        self._syscnt_enable = start_info.get(Constant.SyscntEable, False)
        if info_path:
            try:
                jsondata = json.loads(FileManager.file_read_all(info_path, "rt"))
                config_freq = jsondata.get("CPU")[0].get("Frequency")
                if (self.is_number(str(config_freq))):
                    self._freq = float(config_freq)
            except JSONDecodeError:
                print_warn_msg("Failed to get profiler info json from file.")
        else:
            # 驱动频率
            self._freq = start_info.get(Constant.SysCntFreq, self._freq)
        host_start_log_path = ProfilerPathManager.get_host_start_log_path(profiler_path)
        if host_start_log_path:
            start_log = ConfigParser()
            start_log.read(host_start_log_path)
            config_offset = start_log.get("Host", "clock_monotonic_raw")
            config_start_cnt = start_log.get("Host", "cntvct")
            if self.is_number(str(config_offset)) and self.is_number(str(config_start_cnt)):
                self._time_offset = int(config_offset)
                self._start_cnt = int(config_start_cnt)
        else:
            # 保存的offset 和 cnt
            self._time_offset = start_info.get(Constant.StartMonotonic, self._time_offset)
            self._start_cnt = start_info.get(Constant.StartCnt, self._start_cnt)

    @classmethod
    def _get_profiler_info_json(cls, profiler_path: str):
        info_file_path = ProfilerPathManager.get_info_file_path(profiler_path)
        if not info_file_path:
            print_warn_msg("Failed to get profiler info file.")
            return {}
        try:
            return json.loads(FileManager.file_read_all(info_file_path, "rt"))
        except JSONDecodeError:
            print_warn_msg("Failed to get profiler info json from file.")
            return {}

    def load_info(self, profiler_path: str):
        self.load_is_cluster(profiler_path)
        info_json = self._get_profiler_info_json(profiler_path)
        self.load_experimental_cfg_info(info_json)
        self.load_timediff_info(profiler_path, info_json)
        self.load_syscnt_info(profiler_path, info_json)

    def load_is_cluster(self, profiler_path: str):
        info_file_path = ProfilerPathManager.get_info_file_path(profiler_path)
        if info_file_path:
            self._is_cluster = re.match(r"^profiler_info_\d+\.json", os.path.basename(info_file_path))

    def load_timediff_info(self, profiler_path: str, info_json: dict):
        self._localtime_diff = CANNFileParser(profiler_path).get_localtime_diff()
        end_info = info_json.get(Constant.END_INFO, {})
        if not self._localtime_diff and end_info:
            self._localtime_diff = int(end_info.get(Constant.FWK_END_TIME, 0)) - int(
                end_info.get(Constant.FWK_END_MONOTONIC, 0))

    def load_experimental_cfg_info(self, info_json: dict):
        experimental_config = info_json.get(Constant.CONFIG, {}).get(Constant.EXPERIMENTAL_CONFIG, {})
        self._profiler_level = experimental_config.get(Constant.PROFILER_LEVEL, self._profiler_level)
        self._ai_core_metrics = experimental_config.get(Constant.AI_CORE_METRICS, self._ai_core_metrics)
        self._l2_cache = experimental_config.get(Constant.L2_CACHE, self._l2_cache)
        self._data_simplification = experimental_config.get(Constant.DATA_SIMPLIFICATION, self._data_simplification)
    
    def get_parser_bean(self):
        return self.LEVEL_PARSER_CONFIG.get(self._profiler_level) + self._get_l2_cache_bean()

    def get_prune_config(self):
        return self.LEVEL_TRACE_PRUNE_CONFIG.get(self._profiler_level)

    def is_all_kernel_headers(self):
        if self._ai_core_metrics != Constant.AicMetricsNone:
            return True
        else:
            return False

    def get_local_time(self, monotonic_time: int):
        return int(monotonic_time + self._localtime_diff)

    def _get_l2_cache_bean(self):
        return [(CANNDataEnum.L2_CACHE, L2CacheBean)] if self._l2_cache else []
