import json
import os
import re
from json import JSONDecodeError
from configparser import ConfigParser

from .prof_common_func._file_manager import FileManager
from .prof_common_func._path_manager import ProfilerPathManager
from .prof_common_func._singleton import Singleton
from .prof_common_func._constant import Constant, print_warn_msg, print_error_msg
from .prof_bean._ai_cpu_bean import AiCpuBean
from .prof_parse._cann_file_parser import CANNDataEnum, CANNFileParser
from .prof_bean._l2_cache_bean import L2CacheBean
from .prof_bean._api_statistic_bean import ApiStatisticBean
from .prof_bean._op_statistic_bean import OpStatisticBean
from .prof_bean._npu_module_mem_bean import NpuModuleMemoryBean

__all__ = []


@Singleton
class ProfilerConfig:
    LEVEL_PARSER_CONFIG = {
        Constant.LEVEL_NONE: [],
        Constant.LEVEL0: [(CANNDataEnum.NPU_MODULE_MEM, NpuModuleMemoryBean)],
        Constant.LEVEL1: [(CANNDataEnum.API_STATISTIC, ApiStatisticBean),
                          (CANNDataEnum.OP_STATISTIC, OpStatisticBean),
                          (CANNDataEnum.NPU_MODULE_MEM, NpuModuleMemoryBean)],
        Constant.LEVEL2: [(CANNDataEnum.AI_CPU, AiCpuBean), 
                          (CANNDataEnum.API_STATISTIC, ApiStatisticBean),
                          (CANNDataEnum.OP_STATISTIC, OpStatisticBean),
                          (CANNDataEnum.NPU_MODULE_MEM, NpuModuleMemoryBean)]
    }
    LEVEL_TRACE_PRUNE_CONFIG = {
        Constant.LEVEL_NONE: [],
        Constant.LEVEL0: ['CANN', 'AscendCL', 'Runtime', 'GE', 'Node', 'Model', 'Hccl', 'acl_to_npu'],
        Constant.LEVEL1: [],
        Constant.LEVEL2: []
    }

    def __init__(self):
        self._profiler_level = Constant.LEVEL0
        self._ai_core_metrics = Constant.AicMetricsNone
        self._l2_cache = False
        self._msprof_tx = False
        self._op_attr = False
        self._data_simplification = True
        self._is_cluster = False
        self._localtime_diff = 0
        self._syscnt_enable = False
        self._freq = 100.0
        self._time_offset = 0
        self._start_cnt = 0
        self._export_type = Constant.Text
        self._rank_id = -1

    @property
    def data_simplification(self):
        return self._data_simplification

    @property
    def export_type(self):
        return self._export_type

    @export_type.setter
    def export_type(self, export_type: str):
        self._export_type = export_type

    @property
    def msprof_tx(self):
        return self._msprof_tx

    @property
    def rank_id(self):
        return self._rank_id

    def is_number(self, string):
        if not isinstance(string, str):
            print_error_msg(f"Input string is not str, get type: {type(string)}")
            return False

        pattern = re.compile(r'^[-+]?[0-9]{0,20}\.?[0-9]{1,20}([eE][-+]?[0-9]{1,20})?$')
        return bool(pattern.match(string))

    def get_timestamp_from_syscnt(self, syscnt: int, time_fmt: int = 1000):
        if self._syscnt_enable == False:
            return syscnt
        else:
            if abs(self._freq) < 1e-15:
                msg = "The frequency value is too small to be close to zero, please check."
                raise RuntimeError(msg)
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
    def _get_json_data(cls, json_file_path: str):
        if not json_file_path:
            print_warn_msg(f"Failed to get file, expect path is {json_file_path}.")
            return {}
        try:
            return json.loads(FileManager.file_read_all(json_file_path, "rt"))
        except JSONDecodeError:
            print_warn_msg(f"Failed to get json from file, path is {json_file_path}")
            return {}

    def load_info(self, profiler_path: str):
        self.load_is_cluster(profiler_path)
        info_json = self._get_json_data(ProfilerPathManager.get_info_file_path(profiler_path))
        self.load_rank_info(info_json)
        self.load_experimental_cfg_info(info_json)
        self.load_timediff_info(profiler_path, info_json)
        self.load_syscnt_info(profiler_path, info_json)

    def load_is_cluster(self, profiler_path: str):
        info_file_path = ProfilerPathManager.get_info_file_path(profiler_path)
        if info_file_path:
            self._is_cluster = re.match(r"^profiler_info_\d+\.json", os.path.basename(info_file_path))

    def load_rank_info(self, info_json: dict):
        self._rank_id = info_json.get(Constant.RANK_ID, -1)

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
        self._msprof_tx = experimental_config.get(Constant.MSPROF_TX, self._msprof_tx)
        self._op_attr = experimental_config.get(Constant.OP_ATTR, self._op_attr)
        self._data_simplification = experimental_config.get(Constant.DATA_SIMPLIFICATION, self._data_simplification)
        self._export_type = experimental_config.get(Constant.EXPORT_TYPE, self._export_type)

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

    def get_level(self):
        return self._profiler_level

    def _get_l2_cache_bean(self):
        return [(CANNDataEnum.L2_CACHE, L2CacheBean)] if self._l2_cache else []
