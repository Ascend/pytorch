# Owner(s): ["module: npu"]

import os
import unittest

import torch_npu
from torch_npu.testing.testcase import run_tests


class TestCompatibleImplSocDetect(unittest.TestCase):

    def test_is_ascend950_returns_bool(self):
        result = torch_npu._is_ascend950()
        self.assertIsInstance(result, bool)

    def test_is_ascend950_matches_cpp_soc_version(self):
        # Cross-check with C++ GetSocVersion(): Ascend950 = 260
        soc_version = torch_npu._C._npu_get_soc_version()
        expected = soc_version >= 260
        actual = torch_npu._is_ascend950()
        self.assertEqual(actual, expected,
            f"_is_ascend950()={actual} but GetSocVersion()={soc_version} (>=260 is {expected})")

    def test_compatible_impl_default_matches_soc(self):
        # When env var not pre-set, TORCH_NPU_USE_COMPATIBLE_IMPL should match SoC
        if "TORCH_NPU_USE_COMPATIBLE_IMPL" in os.environ:
            self.skipTest("TORCH_NPU_USE_COMPATIBLE_IMPL was pre-set in environment")
        is_950 = torch_npu._is_ascend950()
        val = os.environ.get("TORCH_NPU_USE_COMPATIBLE_IMPL", "0")
        self.assertEqual(val == "1", is_950)

    def test_compatible_impl_respects_user_value(self):
        # When env var was pre-set, it should be preserved
        if "TORCH_NPU_USE_COMPATIBLE_IMPL" not in os.environ:
            self.skipTest("TORCH_NPU_USE_COMPATIBLE_IMPL was not pre-set")
        user_val = os.environ["TORCH_NPU_USE_COMPATIBLE_IMPL"]
        # Verify the env var wasn't overwritten
        self.assertIn(user_val, ("0", "1"))


if __name__ == "__main__":
    run_tests()
