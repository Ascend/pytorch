import os
import subprocess
import sys
import textwrap

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestInductorBackendColdStart(TestCase):
    @parametrize(
        "backend_case",
        [
            ("triton", "default", "0", "triton_"),
            ("mlir", "mlir", "0", "mlir_fused"),
            ("dvm", "dvm", "0", "dvm_"),
            ("akg", "mlir", "1", "akg_auto_fallback"),
        ],
    )
    def test_first_compile_cold_process(self, backend_case):
        backend_name, backend_env, use_akg, code_marker = backend_case
        script = textwrap.dedent(
            """
            import importlib
            import os
            import sys

            import torch
            import torch_npu
            from torch._inductor.utils import run_and_get_code

            backend_name = os.environ["COLD_START_BACKEND_NAME"]
            backend_env = os.environ["TORCHINDUCTOR_NPU_BACKEND"]

            if backend_name == "triton":
                if importlib.util.find_spec("triton_ascend") is None:
                    print("__SKIP__: triton_ascend is not installed")
                    sys.exit(0)

            if backend_name in ("mlir", "akg"):
                if importlib.util.find_spec("torch_mlir") is None:
                    print("__SKIP__: torch_mlir is not installed")
                    sys.exit(0)

            if backend_name == "dvm":
                try:
                    importlib.import_module("torch_npu._C.dvm")
                except ImportError as exc:
                    print(f"__SKIP__: dvm is not available: {exc}")
                    sys.exit(0)

            if backend_name == "akg" and importlib.util.find_spec("akg") is None:
                print("__SKIP__: akg is not installed")
                sys.exit(0)

            import torch_npu._inductor

            assert torch_npu._inductor._get_backend() == backend_env

            def fn(x, y):
                return x + y * 2

            x = torch.randn(2, 2, device="npu")
            y = torch.randn(2, 2, device="npu")
            actual, codes = run_and_get_code(
                torch.compile(fn, backend="inductor"), x, y
            )
            torch.testing.assert_close(actual, x + y * 2)
            code = "\\n".join(codes)
            assert os.environ["COLD_START_CODE_MARKER"] in code, code
            """
        )
        env = os.environ.copy()
        env["COLD_START_BACKEND_NAME"] = backend_name
        env["COLD_START_CODE_MARKER"] = code_marker
        env["TORCHINDUCTOR_NPU_BACKEND"] = backend_env
        env["TORCHINDUCTOR_USE_AKG"] = use_akg
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if "__SKIP__:" in proc.stdout:
            self.skipTest(proc.stdout.strip())
        self.assertEqual(proc.returncode, 0, proc.stdout)


instantiate_parametrized_tests(TestInductorBackendColdStart)

if __name__ == "__main__":
    run_tests()
