"""DVM online accuracy: enabled/disabled checks and dynamic input coverage."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


def _run_case(case):
    # Import-time switches need fresh processes; each process owns its caches.
    enabled = case != "disabled"
    dynamic = case == "dynamic_bf16"
    os.environ.update(
        INDUCTOR_ASCEND_CHECK_ACCURACY="1" if enabled else "0",
        INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL="rtol=0,atol=0",
        TORCHINDUCTOR_NPU_BACKEND="dvm",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        TORCH_WARM_POOL="0",
        AUTOTUNE="0",
        DISABLE_MP_COMPILE="1",
    )
    with tempfile.TemporaryDirectory(prefix="dvm-online-accuracy-") as cache:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache + "/inductor"
        os.environ["TRITON_CACHE_DIR"] = cache + "/triton"
        import torch
        import torch_npu

        from torch_npu._inductor import dvm, npu_compare

        dtype = torch.bfloat16 if dynamic else torch.float32
        comparisons = []
        error = [0.0]
        original_compare = npu_compare.compare_outputs

        def compare(*args, **kwargs):
            passed = original_compare(*args, **kwargs)
            comparisons.append(passed)
            return passed

        def check_comparisons(start, *, mismatch=False):
            recent = comparisons[start:]
            if enabled:
                assert recent, "Online accuracy did not compare this invocation"
                assert (False in recent) == mismatch, recent
            else:
                assert not recent, recent

        class FaultInjection:
            def __init__(self, factory):
                self.native = factory()

            def __getattr__(self, name):
                return getattr(self.native, name)

            def run(self, *args):
                self.native.run(*args)
                if error[0]:
                    args[-1].add_(error[0])

        def fn(x, y):
            value = (x + y) * 2
            return (value, x - y) if dynamic else value

        def inputs(n):
            x = torch.ones(n, 32, dtype=dtype, device="npu")
            y = torch.ones_like(x)
            return (x.t(), y.t()) if dynamic else (x, y)

        factories = (
            {
                key: (lambda factory=factory: FaultInjection(factory))
                for key, factory in dvm.KERNEL_FACTORY.items()
            }
            if not dynamic
            else {}
        )
        with (
            mock.patch.dict(dvm.KERNEL_FACTORY, factories),
            mock.patch.object(npu_compare, "compare_outputs", side_effect=compare),
        ):
            compiled = torch.compile(
                fn, fullgraph=True, dynamic=dynamic, options={"npu_backend": "dvm"}
            )
            args = inputs(16)
            actual = compiled(*args)
            torch.testing.assert_close(actual, fn(*args))
            check_comparisons(0)
            start = len(comparisons)
            torch.testing.assert_close(compiled(*args), fn(*args))
            check_comparisons(start)
            start = len(comparisons)
            if not dynamic:
                # Corrupt a real kernel output: only the enabled checker should
                # compare it against eager and report the mismatch.
                error[0] = 0.25
                actual = compiled(*args)
                torch.testing.assert_close(actual, fn(*args) + error[0])
                check_comparisons(start, mismatch=True)
            else:
                args = inputs(24)
                torch.testing.assert_close(compiled(*args), fn(*args))
                check_comparisons(start)
            torch.npu.synchronize()
        print(
            f"DVM_ONLINE_RESULT case={case} comparisons={comparisons} package={torch_npu.__file__}"
        )


class TestDvmOnlineAccuracy(unittest.TestCase):
    def _check_case(self, case):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--case=" + case],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("DVM_ONLINE_RESULT case=" + case, result.stdout)
        print(result.stdout, end="")

    def test_enabled(self):
        self._check_case("enabled")

    def test_disabled(self):
        self._check_case("disabled")

    def test_dynamic_bf16_noncontiguous_multioutput(self):
        self._check_case("dynamic_bf16")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].startswith("--case="):
        _run_case(sys.argv[1].split("=", 1)[1])
    else:
        unittest.main()
