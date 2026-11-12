import json
import pickle
import unittest
from unittest.mock import PropertyMock, patch

from torch._C._profiler import _ExperimentalConfig as KinetoExperimentalConfig
from torch_npu.profiler import _NPUExperimentalConfig
from torch_npu.profiler.experimental_config import supported_ai_core_metrics
from torch_npu.profiler.experimental_config import supported_profiler_level
from torch_npu.profiler.experimental_config import supported_export_type
from torch_npu.profiler.experimental_config import _ExperimentalConfig
from torch_npu.profiler.experimental_config import ExportType, ProfilerLevel
from torch_npu._C._profiler import _ExperimentalConfig as Cpp_ExperimentalConfig
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.analysis.prof_common_func._cann_package_manager import CannPackageManager


class TestNPUExperimentalConfig(unittest.TestCase):
    def test_backend_only_compatibility(self):
        self.assertEqual(str(_NPUExperimentalConfig()), '{"backend":"mspti"}')
        for backend in ["mspti", "aclprof", "invalid", None, 1]:
            with self.subTest(backend=backend):
                config = _NPUExperimentalConfig(backend=backend)
                self.assertEqual(str(config), json.dumps({"backend": backend}, separators=(",", ":")))
                self.assertEqual(config.__getstate__(), KinetoExperimentalConfig(
                    custom_profiler_config=str(config)).__getstate__())

    def test_configuration_combinations(self):
        for backend in [None, "mspti", "aclprof"]:
            for options in [{"profiler_level": ProfilerLevel.Level1}, {"export_type": ExportType.Db},
                            {"profiler_level": ProfilerLevel.Level1, "export_type": [ExportType.Db]}]:
                with self.subTest(backend=backend, options=options):
                    kwargs = {} if backend is None else {"backend": backend}
                    data = json.loads(str(_NPUExperimentalConfig(**kwargs, **options)))
                    expected = {"backend": backend or "mspti", **options}
                    if "export_type" in options:
                        expected["export_type"] = [ExportType.Db]
                    self.assertEqual(data, expected)

    def test_supported_levels(self):
        for level in supported_profiler_level():
            with self.subTest(level=level):
                self.assertEqual(json.loads(str(_NPUExperimentalConfig(profiler_level=level)))["profiler_level"], level)

    def test_export_normalization_and_determinism(self):
        scalar = _NPUExperimentalConfig(export_type=ExportType.Db)
        self.assertEqual(str(scalar), str(_NPUExperimentalConfig(export_type=[ExportType.Db])))
        exports = [ExportType.Text, ExportType.Db, ExportType.Text]
        config = _NPUExperimentalConfig(export_type=exports)
        self.assertEqual(str(config), '{"backend":"mspti","export_type":["db","text"]}')
        exports.clear()
        self.assertEqual(str(config), str(_NPUExperimentalConfig(export_type=[ExportType.Db, ExportType.Text])))
        self.assertEqual(str(_NPUExperimentalConfig(profiler_level=None, export_type=None)),
                         str(_NPUExperimentalConfig()))

    def test_invalid_profiler_level(self):
        for level in ["invalid", "Level3", "", 1, True, 1.0, [], {}]:
            with self.subTest(level=level), self.assertRaisesRegex(ValueError, "profiler_level"):
                _NPUExperimentalConfig(profiler_level=level)

    def test_invalid_export_type(self):
        for exports in ["invalid", "db,text", "", 1, True, [], {}, (ExportType.Db,), [ExportType.Db, 1], [[]]]:
            with self.subTest(exports=exports), self.assertRaisesRegex(ValueError, "export_type"):
                _NPUExperimentalConfig(export_type=exports)

    def test_standard_channel_and_pickle_roundtrip(self):
        config = _NPUExperimentalConfig(backend="aclprof", profiler_level=ProfilerLevel.Level1,
                                        export_type=ExportType.Db, verbose=True)
        payload = str(config)
        self.assertEqual(payload, '{"backend":"aclprof","profiler_level":"Level1","export_type":["db"]}')
        standard = KinetoExperimentalConfig(custom_profiler_config=payload, verbose=True)
        self.assertEqual(config.__getstate__(), standard.__getstate__())
        for value in [config, standard]:
            restored = pickle.loads(pickle.dumps(value))
            self.assertEqual(restored.__getstate__(), value.__getstate__())
        self.assertEqual(str(pickle.loads(pickle.dumps(config))), payload)
        # Newer PyTorch versions append unrelated fields after the custom config.
        with patch.object(_NPUExperimentalConfig, "__getstate__", return_value=(*config.__getstate__(), False, False)):
            self.assertEqual(str(config), payload)

    def test_str_uses_direct_attribute_without_serializing(self):
        config = _NPUExperimentalConfig()
        payload = '{"backend":"aclprof"}'
        with patch.object(_NPUExperimentalConfig, "custom_profiler_config", payload, create=True), \
                patch.object(_NPUExperimentalConfig, "__getstate__", side_effect=AssertionError("unexpected fallback")) as state:
            self.assertEqual(str(config), payload)
            state.assert_not_called()

    def test_str_falls_back_when_attribute_is_unavailable(self):
        config = _NPUExperimentalConfig(backend="aclprof")
        payload = '{"backend":"aclprof"}'
        # Emulate the older PyTorch state used when the direct attribute is unavailable.
        legacy_state = ([], False, False, [], False, False, False, False, False, False, False, payload)
        with patch.object(_NPUExperimentalConfig, "custom_profiler_config", create=True,
                          new_callable=PropertyMock, side_effect=AttributeError):
            with patch.object(_NPUExperimentalConfig, "__getstate__", return_value=legacy_state) as state:
                self.assertEqual(str(config), payload)
                state.assert_called_once_with()
        self.assertEqual(str(pickle.loads(pickle.dumps(config))), payload)

    def test_json_escaping(self):
        backend = 'aclprof",\n"profiler_level":"Level2'
        self.assertEqual(json.loads(str(_NPUExperimentalConfig(backend=backend))), {"backend": backend})


class TestExperimentalConfig(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.profile_levels = {Constant.LEVEL0, Constant.LEVEL1, Constant.LEVEL2}
        cls.ai_core_metrics = {
            Constant.AicPipeUtilization,
            Constant.AicArithmeticUtilization,
            Constant.AicMemory,
            Constant.AicMemoryL0,
            Constant.AicMemoryUB,
            Constant.AicResourceConflictRatio,
            Constant.AicL2Cache,
        }
        cls.export_type = {Constant.Db, Constant.Text}

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

    def test_kineto_backend_config(self):
        standard_config = pickle.dumps(KinetoExperimentalConfig())
        aclprof_config = pickle.dumps(_NPUExperimentalConfig(backend="aclprof"))
        self.assertNotIn(b'{"backend":"aclprof"}', standard_config)
        self.assertIn(b'{"backend":"aclprof"}', aclprof_config)

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

    def test_op_attr_with_default_db_export(self):
        with patch.object(CannPackageManager, "is_support_default_export_db", return_value=True):
            experimental_config = _ExperimentalConfig(op_attr=True, export_type="text")
        self.assertTrue(experimental_config._op_attr)

    def test_op_attr_without_db_export(self):
        with patch.object(CannPackageManager, "is_support_default_export_db", return_value=False):
            experimental_config = _ExperimentalConfig(op_attr=True, export_type="text")
        self.assertFalse(experimental_config._op_attr)

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
        from unittest.mock import MagicMock, patch

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


class TestExperimentalConfigPickle(TestCase):
    """Reviewer follow-up: pickle states written by older torch_npu releases stay loadable."""

    def test_pickle_round_trip_keeps_ftrace_fields(self):
        config = Cpp_ExperimentalConfig(ftrace=["sched/sched_switch"], ftrace_period=5, ftrace_bufsize=8192)
        state = config.__getstate__()
        self.assertEqual(len(state), 14)

        restored = Cpp_ExperimentalConfig()
        restored.__setstate__(state)
        self.assertEqual(restored.__getstate__(), state)

    def test_state_without_ftrace_fields_falls_back_to_defaults(self):
        state = Cpp_ExperimentalConfig().__getstate__()

        restored = Cpp_ExperimentalConfig()
        restored.__setstate__(state[:11])
        self.assertEqual(restored.__getstate__()[11:], ([], 0, 0))

if __name__ == "__main__":
    run_tests()
