import os
from unittest.mock import patch

os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "mlir"
os.environ["TORCHINDUCTOR_USE_AKG"] = "1"

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.mlir_compiler import AkgCompiler


class TestMlirKernelNotFallback(TestCase):
    def test_basic_op_uses_non_fallback_kernel(self):
        def fusion_func(x, y):
            return x + y + 1

        x_cpu = torch.randn((64, 128), dtype=torch.float32)
        y_cpu = torch.randn((64, 128), dtype=torch.float32)

        expected = fusion_func(x_cpu, y_cpu)

        x_npu = x_cpu.npu()
        y_npu = y_cpu.npu()

        has_fallback_kernel = False
        original_run = AkgCompiler.run

        def run_and_check_fallback(compiler, *args, **kwargs):
            nonlocal has_fallback_kernel
            result = original_run(compiler, *args, **kwargs)
            launcher_idx = compiler.get_primary_launcher_index()
            has_fallback_kernel |= compiler.is_fallback_kernels[launcher_idx]
            return result

        compiled = torch.compile(fusion_func, backend="inductor")

        with patch.object(AkgCompiler, "run", run_and_check_fallback):
            actual = compiled(x_npu, y_npu)

        self.assertFalse(has_fallback_kernel)
        self.assertEqual(expected, actual.cpu(), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
