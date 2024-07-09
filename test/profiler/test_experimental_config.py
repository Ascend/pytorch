import unittest

from torch_npu.profiler.experimental_config import supported_ai_core_metrics
from torch_npu.profiler.experimental_config import supported_profiler_level
from torch_npu.profiler.experimental_config import supported_export_type
from torch_npu.profiler.analysis.prof_common_func.constant import Constant
from torch_npu.profiler.experimental_config import _ExperimentalConfig
from torch_npu._C._profiler import _ExperimentalConfig as Cpp_ExperimentalConfig
from torch_npu.testing._testcase import TestCase, run_tests


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


if __name__ == "__main__":
    run_tests()
