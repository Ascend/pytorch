import importlib.util
import subprocess
import sys
import unittest

from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401


# Look the package up without importing it: importing torch_npu does not load fxrt.
_HAS_FXRT = importlib.util.find_spec("torch_npu.fxrt") is not None

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
    """FXRT imports leave Inductor registration untouched."""

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


if __name__ == "__main__":
    run_tests()
