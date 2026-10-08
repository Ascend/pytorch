import logging
import os
import sys
import subprocess

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torch_npu.utils.patch_getenv as patch_getenv


def _run_in_subprocess(extra_env=None):
    code = r"""
import os

os.environ["FOO_TEST"] = "bar"

import torch_npu._logging  # 按 env 初始化
from torch_npu.utils import patch_getenv  # 触发 patch

installed = os.getenv is patch_getenv._patched_getenv
print("INSTALLED=%s" % installed)

_ = os.getenv("FOO_TEST")
_ = os.environ.get("FOO_TEST")
"""

    env = os.environ.copy()
    env.pop("TORCH_NPU_LOGS", None)
    env.pop("TORCH_LOGS", None)
    env["FOO_TEST"] = "bar"
    if extra_env:
        env.update(extra_env)

    p = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def _installed_flag(out):
    for line in out.splitlines():
        if line.startswith("INSTALLED="):
            return line.split("=", 1)[1] == "True"
    raise AssertionError(f"INSTALLED flag missing from output:\n{out}")


class TestPatchGetenvSubprocess(TestCase):
    def test_env_log_when_enabled_env(self):
        rc, out = _run_in_subprocess({"TORCH_NPU_LOGS": "env"})
        self.assertTrue(rc == 0, f"subprocess failed rc={rc}\n{out}")
        self.assertTrue(_installed_flag(out))
        self.assertIn("get env FOO_TEST = bar", out)

    def test_no_env_log_when_disabled_env(self):
        rc, out = _run_in_subprocess()
        self.assertTrue(rc == 0, f"subprocess failed rc={rc}\n{out}")
        self.assertFalse(_installed_flag(out))
        self.assertNotIn("FOO_TEST = bar", out)
        self.assertNotIn("get env", out)

    def test_installed_with_torch_logs(self):
        rc, out = _run_in_subprocess({"TORCH_LOGS": "+all"})
        self.assertTrue(rc == 0, f"subprocess failed rc={rc}\n{out}")
        self.assertTrue(_installed_flag(out))


class TestPatchGetenvLocal(TestCase):
    def setUp(self):
        self._saved_getenv = os.getenv
        self._saved_had_instance_get = "get" in vars(os.environ)
        self._saved_instance_get = vars(os.environ).get("get")
        self._saved_installed = patch_getenv._installed

    def tearDown(self):
        os.getenv = self._saved_getenv
        if self._saved_had_instance_get:
            os.environ.get = self._saved_instance_get
        elif "get" in vars(os.environ):
            del os.environ.get
        patch_getenv._installed = self._saved_installed

    def test_should_install_gate(self):
        logger = patch_getenv.loggerEnv
        old_level = logger.level
        saved_env = {k: os.environ.get(k) for k in ("TORCH_NPU_LOGS", "TORCH_LOGS")}
        try:
            os.environ.pop("TORCH_NPU_LOGS", None)
            os.environ.pop("TORCH_LOGS", None)
            logger.setLevel(logging.WARNING)
            self.assertFalse(patch_getenv._should_install())
            logger.setLevel(logging.INFO)
            self.assertTrue(patch_getenv._should_install())
            logger.setLevel(logging.WARNING)
            os.environ["TORCH_NPU_LOGS"] = "acl"
            self.assertTrue(patch_getenv._should_install())
            os.environ.pop("TORCH_NPU_LOGS", None)
            os.environ["TORCH_LOGS"] = "+all"
            self.assertTrue(patch_getenv._should_install())
        finally:
            for key, val in saved_env.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val
            logger.setLevel(old_level)

    def test_patched_getenv_semantics(self):
        patch_getenv._install()
        try:
            os.environ["FOO_LOCAL"] = "val"
            os.environ["FOO_LOCAL_EMPTY"] = ""
            self.assertEqual(os.getenv("FOO_LOCAL"), "val")
            self.assertEqual(os.environ.get("FOO_LOCAL"), "val")
            self.assertEqual(os.getenv("FOO_LOCAL_MISSING", "dflt"), "dflt")
            self.assertIsNone(os.getenv("FOO_LOCAL_MISSING"))
            self.assertEqual(os.getenv("FOO_LOCAL_EMPTY"), "")
            self.assertEqual(os.getenv("FOO_LOCAL", default="dflt"), "val")
        finally:
            os.environ.pop("FOO_LOCAL", None)
            os.environ.pop("FOO_LOCAL_EMPTY", None)

    def test_single_real_lookup_per_call(self):
        os.environ["FOO_LOOKUP"] = "val"
        orig_getitem = os._Environ.__getitem__
        calls = []

        def counting_getitem(self, key):
            calls.append(key)
            return orig_getitem(self, key)

        patch_getenv._install()
        os._Environ.__getitem__ = counting_getitem
        try:
            calls.clear()
            self.assertEqual(os.getenv("FOO_LOOKUP"), "val")
            self.assertEqual(len(calls), 1)
            calls.clear()
            self.assertEqual(os.getenv("FOO_LOOKUP_MISSING", "dflt"), "dflt")
            self.assertEqual(len(calls), 1)
        finally:
            os._Environ.__getitem__ = orig_getitem


if __name__ == "__main__":
    run_tests()