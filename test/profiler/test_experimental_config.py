import unittest

from torch_npu.profiler.experimental_config import supported_ai_core_metrics
from torch_npu.profiler.experimental_config import supported_profiler_level
from torch_npu.profiler.experimental_config import supported_export_type
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.experimental_config import _ExperimentalConfig
from torch_npu._C._profiler import _ExperimentalConfig as Cpp_ExperimentalConfig
from torch_npu.testing.testcase import TestCase, run_tests


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

    def test_mstx_domain_switches_will_reset_when_msproftx_not_enabled(self):
        experimental_config = _ExperimentalConfig(msprof_tx=False,
                                                  mstx_domain_include=['x'],
                                                  mstx_domain_exclude=['y'])
        self.assertEqual([], experimental_config._mstx_domain_include)
        self.assertEqual([], experimental_config._mstx_domain_exclude)

    def test_mstx_domain_switches_will_save_empty_list_when_not_set_domain_switches(self):
        experimental_config = _ExperimentalConfig(msprof_tx=True)
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

    def test_mstx_domain_switches_will_reset_exclude_domain_when_both_set_domain_switches(self):
        experimental_config = _ExperimentalConfig(msprof_tx=True,
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


if __name__ == "__main__":
    run_tests()
