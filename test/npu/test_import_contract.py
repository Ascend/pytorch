# Owner(s): ["module: dynamo"]

"""Guard the torch_npu import contract: ``import torch_npu`` must not preload
``torch._dynamo`` / ``torch._inductor``.
"""

import subprocess
import sys

from torch_npu.testing.testcase import TestCase, run_tests

# Runs in a subprocess: this test module already imports torch and torch_npu,
# so the contract can only be observed from a fresh interpreter.
GUARD_CODE = """
import sys

def loaded(prefix):
    return sorted(
        name for name in sys.modules
        if name == prefix or name.startswith(prefix + ".")
    )

import torch_npu

# Modules that must stay unloaded until the user really enters the compile
# stack. Preloading any of them makes `import torch_npu` much heavier and
# breaks the lazy dynamo/inductor setup.
for prefix in ("torch._dynamo", "torch._inductor", "torch_npu._inductor"):
    preloaded = loaded(prefix)
    assert not preloaded, (
        f"import torch_npu preloaded {prefix}: "
        f"{len(preloaded)} module(s), e.g. {preloaded[:5]}"
    )
"""


class TestTorchNpuImportContract(TestCase):
    def test_import_torch_npu_keeps_compile_stack_unloaded(self):
        result = subprocess.run(
            [sys.executable, "-c", GUARD_CODE],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Subprocess failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )


if __name__ == "__main__":
    run_tests()
