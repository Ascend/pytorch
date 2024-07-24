from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.analysis.prof_common_func._feature_manager import FeatureManager
from torch_npu.testing.testcase import TestCase, run_tests


class TestFeatureManager(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_is_supported_feature(self):
        feature_info = {
            "attr": {
                Constant.Compatibility: "1",
                Constant.FeatureVersion: "1",
                Constant.AffectedComponent: "PTA",
                Constant.AffectedComponentVersion: "all",
                Constant.InfoLog: "error",
            },
            "mindsporeTest": {
                Constant.Compatibility: "1",
                Constant.FeatureVersion: "0",
                Constant.AffectedComponent: "MindSpore",
                Constant.AffectedComponentVersion: "all",
                Constant.InfoLog: "error",
            },
        }
        featureMgr = FeatureManager()
        featureMgr.load_feature_info(feature_info)
        self.assertEqual(featureMgr.is_supported_feature("attr"), True)
        self.assertEqual(featureMgr.is_supported_feature("mindsporeTest"), False)
        featureMgr.clear()



if __name__ == "__main__":
    run_tests()
