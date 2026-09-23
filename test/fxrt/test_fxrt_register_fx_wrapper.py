import importlib.util
import subprocess
import sys
import unittest
from unittest import mock

import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401


# Look the package up without importing it: importing torch_npu does not load fxrt.
_HAS_FXRT = importlib.util.find_spec("torch_npu.fxrt") is not None

_REGISTER_PROBE = """
import torch._inductor.config as inductor_config
from torch._inductor.codegen.common import device_codegens

import torch_npu
import torch_npu.fxrt as fxrt
from torch_npu.fxrt.fx_wrapper import FxrtFxWrapper

print("before", inductor_config.fx_wrapper)
print("returned", fxrt.register_fx_wrapper())
codegen = device_codegens.get("npu")
print("after", inductor_config.fx_wrapper)
print("size_asserts", inductor_config.size_asserts)
print("alignment_asserts", inductor_config.alignment_asserts)
print("codegen", getattr(codegen, "fx_wrapper_codegen", None) is FxrtFxWrapper)
"""

_IMPORT_TORCH_NPU_PROBE = """
import sys

import torch_npu

print("fxrt", "fxrt" in sys.modules)
print("torch_npu.fxrt", "torch_npu.fxrt" in sys.modules)
print("npu_inductor", "torch_npu._inductor" in sys.modules)
print("npu_initialized", torch_npu.npu.is_initialized())
"""

_IMPORT_FXRT_PROBE = """
import sys

import torch._inductor.config as inductor_config
from torch._inductor.codegen.common import device_codegens

import torch_npu
import torch_npu.fxrt
import fxrt
from fxrt.fx_wrapper import FxrtFxWrapper

print("same_module", torch_npu.fxrt is fxrt)
print("config", inductor_config.fx_wrapper)
codegen = device_codegens.get("npu")
print("codegen", getattr(codegen, "fx_wrapper_codegen", None) is FxrtFxWrapper)
print("npu_inductor", "torch_npu._inductor" in sys.modules)
print("npu_initialized", torch_npu.npu.is_initialized())
"""


@unittest.skipUnless(_HAS_FXRT, "torch_npu is built without fxrt")
class TestFxrtRegisterFxWrapper(TestCase):
    """fxrt registers its fx_wrapper codegen only when asked to."""

    def setUp(self):
        super().setUp()
        for name in ("fx_wrapper", "size_asserts", "alignment_asserts"):
            self.addCleanup(setattr, inductor_config, name, getattr(inductor_config, name))

    @property
    def fx_wrapper_module(self):
        import torch_npu.fxrt.fx_wrapper as module

        return module

    def _run_probe(self, probe):
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout

    def test_import_torch_npu_does_not_load_fxrt(self):
        """Importing torch_npu leaves fxrt, inductor and the NPU untouched."""
        stdout = self._run_probe(_IMPORT_TORCH_NPU_PROBE)
        self.assertIn("fxrt False", stdout)
        self.assertIn("torch_npu.fxrt False", stdout)
        self.assertIn("npu_inductor False", stdout)
        self.assertIn("npu_initialized False", stdout)

    def test_import_fxrt_registers_nothing(self):
        """Importing torch_npu.fxrt must not take over inductor or initialize the NPU."""
        stdout = self._run_probe(_IMPORT_FXRT_PROBE)
        self.assertIn("same_module True", stdout)
        self.assertIn("config False", stdout)
        self.assertIn("codegen False", stdout)
        self.assertIn("npu_inductor False", stdout)
        self.assertIn("npu_initialized False", stdout)

    def _reset_config(self):
        inductor_config.fx_wrapper = False
        inductor_config.size_asserts = True
        inductor_config.alignment_asserts = True

    def _assert_config_untouched(self):
        self.assertFalse(inductor_config.fx_wrapper)
        self.assertTrue(inductor_config.size_asserts)
        self.assertTrue(inductor_config.alignment_asserts)

    def test_returns_false_and_keeps_config_when_install_fails(self):
        module = self.fx_wrapper_module
        self._reset_config()
        with mock.patch.object(
            module, "_install_fx_wrapper_codegen", return_value=False
        ):
            self.assertFalse(module.register_fx_wrapper())
        self._assert_config_untouched()

    def test_patch_config_false_keeps_official_switches(self):
        module = self.fx_wrapper_module
        self._reset_config()
        with mock.patch.object(
            module, "_install_fx_wrapper_codegen", return_value=True
        ) as install:
            self.assertTrue(module.register_fx_wrapper(patch_config=False))
        install.assert_called_once_with("npu")
        self._assert_config_untouched()

    def test_patch_config_applies_fx_form_settings(self):
        module = self.fx_wrapper_module
        self._reset_config()
        with mock.patch.object(
            module, "_install_fx_wrapper_codegen", return_value=True
        ):
            self.assertTrue(module.register_fx_wrapper())
        self.assertTrue(inductor_config.fx_wrapper)
        self.assertFalse(inductor_config.size_asserts)
        self.assertFalse(inductor_config.alignment_asserts)

    def test_register_installs_codegen_and_enables_config(self):
        stdout = self._run_probe(_REGISTER_PROBE)
        self.assertIn("before False", stdout)
        self.assertIn("returned True", stdout)
        self.assertIn("after True", stdout)
        self.assertIn("size_asserts False", stdout)
        self.assertIn("alignment_asserts False", stdout)
        self.assertIn("codegen True", stdout)


if __name__ == "__main__":
    run_tests()
