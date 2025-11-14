import json
import os
import re
from json import JSONDecodeError
from configparser import ConfigParser
from unittest.mock import patch

from torch_npu.profiler.analysis.prof_common_func._file_manager import FileManager
from torch_npu.profiler.analysis.prof_common_func._path_manager import ProfilerPathManager
from torch_npu.profiler.analysis.prof_common_func._singleton import Singleton
from torch_npu.profiler.analysis.prof_common_func._constant import Constant, print_warn_msg, print_error_msg

from torch_npu.profiler.analysis.prof_bean._ai_cpu_bean import AiCpuBean
from torch_npu.profiler.analysis.prof_bean._l2_cache_bean import L2CacheBean
from torch_npu.profiler.analysis.prof_bean._api_statistic_bean import ApiStatisticBean
from torch_npu.profiler.analysis.prof_bean._op_statistic_bean import OpStatisticBean
from torch_npu.profiler.analysis.prof_bean._npu_module_mem_bean import NpuModuleMemoryBean
from torch_npu.profiler.analysis.prof_bean._nic_bean import NicBean
from torch_npu.profiler.analysis.prof_bean._roce_bean import RoCEBean
from torch_npu.profiler.analysis.prof_bean._pcie_bean import PcieBean
from torch_npu.profiler.analysis.prof_bean._hccs_bean import HccsBean

import torch_npu.profiler.analysis._profiler_config as profiler_config
from torch_npu.testing.testcase import TestCase, run_tests


class TestProfilerConfig(TestCase):
    def test_load_is_cluster_with_non_cluster_file(self):
        config = profiler_config.ProfilerConfig()
        with patch.object(ProfilerPathManager, 'get_info_file_path', return_value='/path/profiler_info_normal.json'):
            config.load_is_cluster('/mock/path')
            self.assertFalse(config._is_cluster)

    def test_get_timestamp_from_syscnt_disabled(self):
        config = profiler_config.ProfilerConfig()
        config._syscnt_enable = False
        result = config.get_timestamp_from_syscnt(1000)
        self.assertEqual(result, 1000)

    def test_get_export_type_from_profiler_info_invalid(self):
        config = profiler_config.ProfilerConfig()
        experimental_config = {"export_type": "invalid_type"}
        result = config._get_export_type_from_profiler_info(experimental_config)
        self.assertEqual(result, [Constant.Text])

        experimental_config = {"export_type": ["invalid_type"]}
        result = config._get_export_type_from_profiler_info(experimental_config)
        self.assertEqual(result, [Constant.Text])

    def test_is_number_valid(self):
        config = profiler_config.ProfilerConfig()
        self.assertTrue(config.is_number("123"))
        self.assertTrue(config.is_number("-123"))
        self.assertTrue(config.is_number("123.456"))
        self.assertTrue(config.is_number("-123.456"))
        self.assertTrue(config.is_number("1.23e10"))
        self.assertTrue(config.is_number("1.23E-10"))

    def test_profiler_config_initialization(self):
        config = profiler_config.ProfilerConfig()
        self.assertEqual(config._profiler_level, Constant.LEVEL0)
        self.assertEqual(config._ai_core_metrics, Constant.AicMetricsNone)
        self.assertFalse(config._l2_cache)
        self.assertFalse(config._msprof_tx)
        self.assertFalse(config._op_attr)
        self.assertTrue(config._data_simplification)
        self.assertFalse(config._is_cluster)
        self.assertEqual(config._localtime_diff, 0)
        self.assertFalse(config._syscnt_enable)
        self.assertFalse(config._sys_io)
        self.assertFalse(config._sys_interconnection)
        self.assertEqual(config._freq, 100.0)
        self.assertEqual(config._time_offset, 0)
        self.assertEqual(config._start_cnt, 0)
        self.assertEqual(config._export_type, [Constant.Text])
        self.assertEqual(config._rank_id, -1)
        self.assertEqual(config._activities, [])


    def test_load_syscnt_info_json_decode_error(self):
        config = profiler_config.ProfilerConfig()
        with patch.object(FileManager, 'file_read_all', side_effect=JSONDecodeError('error', 'doc', 0)):
            with patch.object(ProfilerPathManager, 'get_info_path', return_value='/mock/info.json'):
                with patch.object(ProfilerPathManager, 'get_host_start_log_path', return_value=None):
                    config.load_syscnt_info('/mock/path', {})



if __name__ == "__main__":
    run_tests()

