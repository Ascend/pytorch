import os
import subprocess
import sys

from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestInfNanMode(TestCase):

    @SupportedDevices(['Ascend950'])
    def test_is_support_inf_nan_always_enabled(self):
        import torch_npu.npu.utils as utils
        self.assertTrue(utils.is_support_inf_nan())

    @SupportedDevices(['Ascend950'])
    def test_env_vars_ignored(self):
        code = (
            "import torch_npu.npu.utils as utils;"
            "assert utils.is_support_inf_nan()"
        )
        env_cases = [
            ({'INF_NAN_MODE_ENABLE': '1'}, "INF_NAN_MODE_ENABLE=1"),
            ({'INF_NAN_MODE_ENABLE': '0'}, "INF_NAN_MODE_ENABLE=0"),
            ({'INF_NAN_MODE_FORCE_DISABLE': '0'}, "INF_NAN_MODE_FORCE_DISABLE=0"),
            ({'INF_NAN_MODE_FORCE_DISABLE': '1'}, "INF_NAN_MODE_FORCE_DISABLE=1"),
            ({'INF_NAN_MODE_ENABLE': '0', 'INF_NAN_MODE_FORCE_DISABLE': '1'},
             "INF_NAN_MODE_ENABLE=0 + INF_NAN_MODE_FORCE_DISABLE=1"),
        ]
        for env_vars, desc in env_cases:
            env = os.environ.copy()
            env.update(env_vars)
            result = subprocess.run(
                [sys.executable, '-c', code],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode, 0,
                f"{desc} should be ignored on Ascend950.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )


if __name__ == "__main__":
    run_tests()
