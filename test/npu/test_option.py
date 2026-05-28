import sys
import subprocess

import torch
import torch_npu

import torch_npu.npu.utils as utils

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestOption(TestCase):

    def test_option_pm(self):
        option = {"ACL_PRECISION_MODE": "allow_fp32_to_fp16"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_osim(self):
        option = {"ACL_OP_SELECT_IMPL_MODE": "high_precision"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_ofi(self):
        option = {"ACL_OPTYPELIST_FOR_IMPLMODE": "Conv2d"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_odl(self):
        option = {"ACL_OP_DEBUG_LEVEL": "2"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_occm(self):
        option = {"ACL_OP_COMPILER_CACHE_MODE": "enable"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_dd(self):
        option = {"ACL_DEBUG_DIR": "test1"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_occd(self):
        option = {"ACL_OP_COMPILER_CACHE_DIR": "test"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_an(self):
        option = {"ACL_AICORE_NUM": "1"}
        self.assertIsNone(torch.npu.set_option(option))

    def test_option_pme(self):
        option = {"ACL_PRECISION_MODE": "500"}
        with self.assertRaises(ValueError):
            torch.npu.set_option(option)

    def test_option_osime(self):
        option = {"ACL_OP_SELECT_IMPL_MODE": "100"}
        with self.assertRaises(ValueError):
            torch.npu.set_option(option)

    def test_option_dle(self):
        option = {"ACL_OP_DEBUG_LEVEL": "300"}
        with self.assertRaises(ValueError):
            torch.npu.set_option(option)

    def test_option_occme(self):
        option = {"ACL_OP_COMPILER_CACHE_MODE": "2"}
        with self.assertRaises(ValueError):
            torch.npu.set_option(option)

    def test_option_ane(self):
        option = {"ACL_AICORE_NUM": "at"}
        with self.assertRaises(ValueError):
            torch.npu.set_option(option)

    def test_option_fa(self):
        option = {"FORCE_ACLNN_OP_LIST": "index"}
        self.assertIsNone(torch.npu.set_option(option))


class TestAclOpInitMode(TestCase):

    ACLNN_WARN = "ACL_OP_INIT_MODE=0 or 1 is not supported on this device."
    INVALID_VALUE_WARN = "Get env ACL_OP_INIT_MODE not in [0, 1, 2]"

    def _run_subprocess(self, env_val):
        env_line = f"os.environ['ACL_OP_INIT_MODE']='{env_val}'" if env_val is not None else ""
        test_script = f"import os; {env_line}; import torch; import torch_npu; torch_npu.npu.set_device(0)"
        result = subprocess.run(
            [sys.executable, '-c', test_script],
            capture_output=True, text=True
        )
        return result.stderr

    @SupportedDevices(['Ascend950'])
    def test_mode0_on_ascend950(self):
        stderr = self._run_subprocess("0")
        self.assertIn(self.ACLNN_WARN, stderr,
                      "ACL_OP_INIT_MODE=0 should be auto-corrected to 2 on Ascend950")

    @SupportedDevices(['Ascend950'])
    def test_mode1_on_ascend950(self):
        stderr = self._run_subprocess("1")
        self.assertIn(self.ACLNN_WARN, stderr,
                      "ACL_OP_INIT_MODE=1 should be auto-corrected to 2 on Ascend950")

    @SupportedDevices(['Ascend910B'])
    def test_mode0_on_non_ascend950(self):
        stderr = self._run_subprocess("0")
        self.assertNotIn(self.ACLNN_WARN, stderr,
                         "ACL_OP_INIT_MODE=0 should stay 0 on non-Ascend950 device")

    def test_mode_invalid(self):
        stderr = self._run_subprocess("999")
        self.assertIn(self.INVALID_VALUE_WARN, stderr)


if __name__ == "__main__":
    run_tests()
