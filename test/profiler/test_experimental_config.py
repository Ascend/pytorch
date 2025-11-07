import unittest
import warnings

from torch_npu.profiler.experimental_config import supported_ai_core_metrics
from torch_npu.profiler.experimental_config import supported_profiler_level
from torch_npu.profiler.experimental_config import supported_export_type
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.experimental_config import _ExperimentalConfig
from torch_npu._C._profiler import _ExperimentalConfig as Cpp_ExperimentalConfig
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.profiler.analysis.prof_common_func._constant import Constant, print_warn_msg, print_info_msg
from torch_npu.profiler.analysis.prof_common_func._cann_package_manager import CannPackageManager


class TestExperimentalConfig(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.profile_levels = set([Constant.LEVEL0, Constant.LEVEL1, Constant.LEVEL2])
        cls.ai_core_metrics = set([
            Constant.AicPipeUtilization,
            Constant.AicArithmeticUtilization,
            Constant.AicMemory,
            Constant.AicMemoryL0,
            Constant.AicMemoryUB,
            Constant.AicResourceConflictRatio,
            Constant.AicL2Cache,
        ])
        cls.export_type = set([Constant.Db, Constant.Text])

    @unittest.skip("Skip test_supported_profiler_level now!")
    def test_supported_profiler_level(self):
        profile_levels = supported_profiler_level()
        self.assertEqual(self.profile_levels, profile_levels)

    @unittest.skip("Skip test_supported_ai_core_metrics now!")
    def test_supported_ai_core_metrics(self):
        ai_core_metrics = supported_ai_core_metrics()
        self.assertEqual(self.ai_core_metrics, ai_core_metrics)

    @unittest.skip("Skip test_supported_export_type now!")
    def test_supported_export_type(self):
        export_type = supported_export_type()
        self.assertEqual(self.export_type, export_type)

    def test_call_experimental_config(self):
        experimental_config = _ExperimentalConfig()
        self.assertTrue(isinstance(experimental_config(), Cpp_ExperimentalConfig))

    def test_mstx_domain_switches_will_reset_when_msproftx_and_mstx_not_enabled(self):
        experimental_config = _ExperimentalConfig(msprof_tx=False,
                                                  mstx=False,
                                                  mstx_domain_include=['x'],
                                                  mstx_domain_exclude=['y'])
        self.assertEqual([], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)

    def test_mstx_domain_switches_will_reset_when_input_invaild_msproftx_and_mstx(self):
        experimental_config = _ExperimentalConfig(msprof_tx=1,
                                                  mstx=2)
        self.assertEqual(False, experimental_config._msprof_tx)
        self.assertEqual(False, experimental_config._mstx)

    def test_mstx_domain_switches_will_save_empty_list_when_not_set_domain_switches(self):
        experimental_config = _ExperimentalConfig(msprof_tx=True)
        self.assertEqual([], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)

        experimental_config = _ExperimentalConfig(mstx=True)
        self.assertEqual([], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)


    def test_mstx_domain_switches_will_reset_when_input_invalid_domain_switches(self):
        experimental_config = _ExperimentalConfig(msprof_tx=True,
                                                  mstx_domain_include=1,
                                                  mstx_domain_exclude=1)
        self.assertEqual([], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)

        experimental_config = _ExperimentalConfig(msprof_tx=True,
                                                  mstx_domain_include=[1],
                                                  mstx_domain_exclude=[1])
        self.assertEqual([], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)

        experimental_config = _ExperimentalConfig(mstx=True,
                                                  mstx_domain_include=1,
                                                  mstx_domain_exclude=1)
        self.assertEqual([], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)

        experimental_config = _ExperimentalConfig(mstx=True,
                                                  mstx_domain_include=[1],
                                                  mstx_domain_exclude=[1])
        self.assertEqual([], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)

        experimental_config = _ExperimentalConfig(msprof_tx=True,
                                                  mstx=True,
                                                  mstx_domain_include=1,
                                                  mstx_domain_exclude=1)
        self.assertEqual([], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)

        experimental_config = _ExperimentalConfig(msprof_tx=True,
                                                  mstx=True,
                                                  mstx_domain_include=[1],
                                                  mstx_domain_exclude=[1])
        self.assertEqual([], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)

    def test_mstx_domain_switches_will_reset_exclude_domain_when_both_set_domain_switches(self):
        experimental_config = _ExperimentalConfig(msprof_tx=True,
                                                  mstx_domain_include=['x'],
                                                  mstx_domain_exclude=['y'])
        self.assertEqual(['x'], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)

        experimental_config = _ExperimentalConfig(mstx=True,
                                                  mstx_domain_include=['x'],
                                                  mstx_domain_exclude=['y'])
        self.assertEqual(['x'], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)

        experimental_config = _ExperimentalConfig(msprof_tx=True,
                                                  mstx=True,
                                                  mstx_domain_include=['x'],
                                                  mstx_domain_exclude=['y'])
        self.assertEqual(['x'], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)

    def test_mstx_domain_switches_will_save_when_input_valid_domain_switches(self):
        experimental_config = _ExperimentalConfig(msprof_tx=True,
                                                  mstx_domain_include=['x'])
        self.assertEqual(['x'], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)

        experimental_config = _ExperimentalConfig(msprof_tx=True,
                                                  mstx_domain_exclude=['y'])
        self.assertEqual([], experimental_config._mstx_domain_include)
        self.assertEqual(['y'], experimental_config._mstx_domain_exclude)

        experimental_config = _ExperimentalConfig(mstx=True,
                                                  mstx_domain_include=['x'])
        self.assertEqual(['x'], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)

        experimental_config = _ExperimentalConfig(mstx=True,
                                                  mstx_domain_exclude=['y'])
        self.assertEqual([], experimental_config._mstx_domain_include)
        self.assertEqual(['y'], experimental_config._mstx_domain_exclude)

        experimental_config = _ExperimentalConfig(msprof_tx=True,
                                                  mstx=True,
                                                  mstx_domain_include=['x'])
        self.assertEqual(['x'], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)

        experimental_config = _ExperimentalConfig(msprof_tx=True,
                                                  mstx=True,
                                                  mstx_domain_exclude=['y'])
        self.assertEqual([], experimental_config._mstx_domain_include)
        self.assertEqual(['y'], experimental_config._mstx_domain_exclude)

    def test_host_sys_switches_will_save_empty_list_when_not_set_host_sys(self):
        experimental_config = _ExperimentalConfig()
        self.assertEqual([], experimental_config._host_sys)

    def test_host_sys_switches_will_save_when_set_valid_host_sys(self):
        experimental_config = _ExperimentalConfig(host_sys=[Constant.CPU])
        self.assertEqual(["cpu"], experimental_config._host_sys)

    def test_sys_switches_will_save_empty_list_when_not_set_sys(self):
        experimental_config = _ExperimentalConfig()
        self.assertEqual(False, experimental_config._sys_io)
        self.assertEqual(False, experimental_config._sys_interconnection)

    def test_sys_switches_will_save_when_set_valid_sys(self):
        experimental_config = _ExperimentalConfig(sys_io=True, sys_interconnection=True)
        self.assertEqual(True, experimental_config._sys_io)
        self.assertEqual(True, experimental_config._sys_interconnection)

    def test_check_params_reset_data_simplification(self):
        experimental_config = _ExperimentalConfig(data_simplification="invalid")
        self.assertEqual(True, experimental_config._data_simplification)

    def test_check_params_reset_l2_cache(self):
        experimental_config = _ExperimentalConfig(l2_cache="invalid")
        self.assertEqual(False, experimental_config._l2_cache)

    def test_check_params_reset_invalid_profiler_level(self):
        experimental_config = _ExperimentalConfig(profiler_level=999)
        self.assertEqual(Constant.LEVEL0, experimental_config._profiler_level)

    def test_check_params_reset_aic_metrics_level0(self):
        experimental_config = _ExperimentalConfig(profiler_level=Constant.LEVEL0, aic_metrics=Constant.AicMemory)
        self.assertEqual(Constant.AicMetricsNone, experimental_config._aic_metrics)

    def test_convert_export_type_string(self):
        experimental_config = _ExperimentalConfig(export_type="text")
        self.assertEqual(["text"], experimental_config._export_type)

    def test_check_params_invalid_gc_detect_threshold(self):
        experimental_config = _ExperimentalConfig(gc_detect_threshold=-1.0)
        self.assertIsNone(experimental_config._gc_detect_threshold)

    def test_check_host_sys_params_invalid_elements(self):
        experimental_config = _ExperimentalConfig(host_sys=[Constant.CPU, "invalid"])
        self.assertEqual([], experimental_config._host_sys)

    def test_check_params_invalid_record_op_args(self):
        experimental_config = _ExperimentalConfig(record_op_args="invalid")
        self.assertEqual(False, experimental_config.record_op_args)

    def test_get_proxy_returns_none_no_failure(self):
        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {'IPCMonitor': None}):
            from torch_npu.profiler._dynamic_profiler._dynamic_monitor_proxy import PyDynamicMonitorProxySingleton
            singleton = PyDynamicMonitorProxySingleton()
            singleton._proxy = None
            singleton._load_success = True
            result = singleton.get_proxy()
            self.assertIsNone(result)

    def test_load_proxy_import_failure(self):
        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {'IPCMonitor': None}):
            from torch_npu.profiler._dynamic_profiler._dynamic_monitor_proxy import PyDynamicMonitorProxySingleton
            singleton = PyDynamicMonitorProxySingleton()
            singleton._proxy = None
            singleton._load_success = True
            singleton._load_proxy()
            self.assertFalse(singleton._load_success)
            self.assertIsNone(singleton._proxy)

    def test_load_proxy_initialization_success(self):
        import sys
        from unittest.mock import MagicMock, patch

        mock_proxy_class = MagicMock()
        mock_proxy_instance = MagicMock()
        mock_proxy_class.return_value = mock_proxy_instance

        with patch.dict(sys.modules, {'IPCMonitor': MagicMock()}):
            with patch('IPCMonitor.PyDynamicMonitorProxy', mock_proxy_class):
                from torch_npu.profiler._dynamic_profiler._dynamic_monitor_proxy import PyDynamicMonitorProxySingleton
                singleton = PyDynamicMonitorProxySingleton()
                singleton._proxy = None
                singleton._load_success = True
                singleton._load_proxy()
                self.assertTrue(singleton._load_success)
                self.assertEqual(singleton._proxy, mock_proxy_instance)

    def test_gc_detector_stop(self):
        from torch_npu.profiler._profiler_gc_detect import ProfGCDetector
        from unittest.mock import patch, MagicMock
        import gc
        detector = ProfGCDetector(1.0)
        detector.start()
        detector.save_info = [(1, 2, 3)]
        original_callbacks = gc.callbacks[:]

        try:
            with patch('torch_npu.profiler._profiler_path_creator.ProfPathCreator') as mock_creator, \
                    patch(
                        'torch_npu.profiler._profiler_gc_detect.ProfilerPathManager.get_fwk_path') as mock_get_fwk_path:
                mock_creator_instance = MagicMock()
                mock_creator_instance.get_prof_dir.return_value = '/mock/path'
                mock_creator.return_value = mock_creator_instance
                mock_get_fwk_path.return_value = '/mock/path'
                detector.stop()
                self.assertEqual(detector.time_info, {})
                self.assertEqual(detector.save_info, [])
                self.assertNotIn(detector.gc_callback, gc.callbacks)
        finally:
            if detector.gc_callback in gc.callbacks:
                gc.callbacks.remove(detector.gc_callback)

    def test_gc_detector_save_file_creation_failure(self):
        from torch_npu.profiler._profiler_gc_detect import ProfGCDetector
        from unittest.mock import patch, MagicMock
        import os
        detector = ProfGCDetector(1.0)
        detector.save_info = [(1, 2, 3)]

        with patch('torch_npu.profiler._profiler_path_creator.ProfPathCreator') as mock_creator, \
                patch('torch_npu.profiler._profiler_gc_detect.ProfilerPathManager.get_fwk_path') as mock_get_fwk_path, \
                patch('torch_npu.profiler._profiler_gc_detect.FileManager.create_bin_file_by_path') as mock_create_file:
            mock_creator_instance = MagicMock()
            mock_creator_instance.get_prof_dir.return_value = '/mock/path'
            mock_creator.return_value = mock_creator_instance
            mock_get_fwk_path.return_value = '/mock/path'
            mock_create_file.side_effect = Exception("File creation failed")
            detector.save()

    def test_gc_callback_valid_phases(self):
        from torch_npu.profiler._profiler_gc_detect import ProfGCDetector
        import gc
        import os

        detector = ProfGCDetector(0.001)
        detector.start()

        try:
            detector.gc_callback(detector.START_PHASE, {})
            pid = os.getpid()
            self.assertIn(pid, detector.time_info)
            detector.gc_callback(detector.STOP_PHASE, {})
            self.assertEqual(len(detector.save_info), 1)
            detector.gc_callback("invalid", {})
        finally:
            if detector.gc_callback in gc.callbacks:
                gc.callbacks.remove(detector.gc_callback)

    def test_gc_detector_start(self):
        from torch_npu.profiler._profiler_gc_detect import ProfGCDetector
        import gc

        detector = ProfGCDetector(1.0)
        original_callbacks = gc.callbacks[:]

        try:
            detector.start()
            self.assertIn(detector.gc_callback, gc.callbacks)
        finally:
            if detector.gc_callback in gc.callbacks:
                gc.callbacks.remove(detector.gc_callback)

    def test_gc_detector_init(self):
        from torch_npu.profiler._profiler_gc_detect import ProfGCDetector

        detector = ProfGCDetector(1.0)
        self.assertEqual(detector.threshold, 1.0 * Constant.NS_TO_MS)
        self.assertEqual(detector.time_info, {})
        self.assertEqual(detector.save_info, [])
        self.assertIsNotNone(detector.get_cur_ts)


if __name__ == "__main__":
    run_tests()
