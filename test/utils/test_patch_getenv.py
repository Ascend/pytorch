import os
import sys
import subprocess

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


def _run_in_subprocess(enable_env: bool):
    code = r"""
import os

os.environ["FOO_TEST"] = "bar"

import torch_npu._logging  # 按 env 初始化
from torch_npu.utils import patch_getenv  # 触发 patch

_ = os.getenv("FOO_TEST")
_ = os.environ.get("FOO_TEST")
"""

    env = os.environ.copy()
    env.pop("TORCH_NPU_LOGS", None)
    env.pop("TORCH_LOGS", None)
    env["FOO_TEST"] = "bar"

    if enable_env:
        env["TORCH_NPU_LOGS"] = "env"

    p = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


class TestPatchGetenv(TestCase):
    def test_env_log_when_enabled_env(self):
        rc, out = _run_in_subprocess(enable_env=True)
        self.assertTrue(rc == 0, f"subprocess failed rc={rc}\n{out}")
        self.assertIn("get env FOO_TEST = bar", out)

    def test_no_env_log_when_disabled_env(self):
        rc, out = _run_in_subprocess(enable_env=False)
        self.assertTrue(rc == 0, f"subprocess failed rc={rc}\n{out}")
        self.assertNotIn("FOO_TEST = bar", out)
        self.assertNotIn("get env", out)


if __name__ == "__main__":
    run_tests()