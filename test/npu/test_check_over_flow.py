import os
import subprocess
import sys

import torch
import torch_npu
import torch_npu.npu.utils as utils
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests

# Module-level env (same pattern as test/custom_ops/test_float_status.py),
# set before any NPU op so the C++ stream-creation path also observes it.
os.environ["FORCE_OVERFLOW_CHECK"] = "1"
os.environ["INF_NAN_MODE_ENABLE"] = "1"
os.environ["INF_NAN_MODE_FORCE_DISABLE"] = "0"

# Devices where inf-nan (non-saturation) mode is the default.
NON_SATURATION_DEVICES = ["Ascend910B", "Ascend910C", "Ascend910_93", "Ascend950"]


class TestCheckOverFlow(TestCase):

    def test_check_over_flow(self):
        # Saturation: original float_status path; inf-nan: register path via FORCE_OVERFLOW_CHECK=1.
        a = torch.Tensor([65535]).npu().half()
        a = a + a
        ret = utils.npu_check_overflow(a)
        self.assertTrue(ret)


class TestForceOverflowCheck(TestCase):

    @SupportedDevices(NON_SATURATION_DEVICES)
    def test_float_status_detection(self, device="npu"):
        # Register-level assertions, mirroring test/custom_ops/test_float_status.py.
        self.assertTrue(utils.is_support_inf_nan())
        self.assertTrue(utils.is_force_overflow_check())

        input1 = torch.zeros(8).npu()
        float_status = torch_npu.npu_alloc_float_status(input1)
        torch_npu.npu_clear_float_status(float_status)

        a = torch.tensor([40000.0], dtype=torch.float16).npu()
        a = a + a  # 80000 overflows fp16 -> inf, float_status register must be set
        torch_npu.npu.synchronize()
        local_float_status = torch_npu.npu_get_float_status(float_status)
        self.assertTrue(local_float_status.cpu()[0] != 0)

        torch_npu.npu_clear_float_status(float_status)
        torch_npu.npu.synchronize()
        local_float_status = torch_npu.npu_get_float_status(float_status)
        self.assertTrue(local_float_status.cpu()[0] == 0)

    @SupportedDevices(NON_SATURATION_DEVICES)
    def test_npu_check_overflow_register_path(self, device="npu"):
        self.assertTrue(utils.is_support_inf_nan())
        self.assertTrue(utils.is_force_overflow_check())

        utils.clear_npu_overflow_flag()
        a = torch.Tensor([65535]).npu().half()
        a = a + a
        torch_npu.npu.synchronize()
        self.assertTrue(utils.npu_check_overflow(a))
        self.assertFalse(utils.get_npu_overflow_flag())  # auto-cleared


class TestForceOverflowCheckEnvCases(TestCase):
    # Negative env scenarios run in subprocesses (test_inf_nan_mode.py pattern).

    @SupportedDevices(NON_SATURATION_DEVICES)
    def test_default_behavior_unchanged(self):
        code = (
            "import torch_npu.npu.utils as utils\n"
            "assert not utils.is_force_overflow_check(), 'is_force_overflow_check should be False by default'\n"
            "try:\n"
            "    utils.get_npu_overflow_flag()\n"
            "    assert False, 'get_npu_overflow_flag should raise by default'\n"
            "except RuntimeError:\n"
            "    pass\n"
        )
        env = os.environ.copy()
        env.pop("FORCE_OVERFLOW_CHECK", None)  # remove the module-level setting
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True)
        self.assertEqual(
            result.returncode, 0,
            f"Default behavior should be unchanged.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}")

    def test_version_mismatch_warning(self):
        # Below 9.1.0: env ignored and LOGW printed; >= 9.1.0: no warning.
        code = (
            "import torch;"  # trigger lazy init and stream creation for the C++ LOGW
            "import torch_npu;"
            "x = torch.zeros(1).npu();"
            "torch_npu.npu.synchronize();"
        )
        env = os.environ.copy()
        env["FORCE_OVERFLOW_CHECK"] = "1"
        # ASCEND_LOGW is gated by ASCEND_GLOBAL_LOG_LEVEL (defaults to 3=error,
        # which suppresses WARNING) and CANN slog writes to plog files by
        # default; both env vars are needed for the LOGW to reach stdout/stderr.
        env["ASCEND_GLOBAL_LOG_LEVEL"] = "2"
        env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True)
        self.assertEqual(
            result.returncode, 0,
            f"stdout: {result.stdout}\nstderr: {result.stderr}")
        warning = "FORCE_OVERFLOW_CHECK=1 requires CANN"
        if utils._is_gte_cann_version("9.1.0"):
            self.assertNotIn(
                warning, result.stderr + result.stdout,
                "no LOGW expected on CANN >= 9.1.0")
        else:
            self.assertIn(
                warning, result.stderr + result.stdout,
                "LOGW about the ignored env var is expected")

    def test_invalid_env_value(self):
        code = (
            "import torch_npu.npu.utils as utils;"
            "assert not utils.is_force_overflow_check(), 'non-1 values should be treated as unset';"
        )
        env = os.environ.copy()
        env["FORCE_OVERFLOW_CHECK"] = "2"
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True)
        self.assertEqual(
            result.returncode, 0,
            f"FORCE_OVERFLOW_CHECK=2 should be treated as unset.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}")


if __name__ == "__main__":
    run_tests()
